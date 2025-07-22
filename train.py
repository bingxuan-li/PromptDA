import os, cv2, torch, wandb, numpy as np, torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from promptda.promptda import PromptDA
from promptda.utils.logger import Log
from dataset import SimulatedDataset, RealDataset
from pynvml import *
import time
from visualize import get_visualization

nvmlInit()
def log_all_gpu_stats():
    stats = {}
    for i in range(torch.cuda.device_count()):
        handle = nvmlDeviceGetHandleByIndex(i)
        mem = nvmlDeviceGetMemoryInfo(handle)
        util = nvmlDeviceGetUtilizationRates(handle)
        temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

        stats[f"gpu/{i}/mem_used_MB"] = mem.used / 1024**2
        stats[f"gpu/{i}/util_percent"] = util.gpu
        stats[f"gpu/{i}/temp_C"] = temp
    return stats


def gradient_loss(pred, target):
    pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
    return torch.mean(torch.abs(pred_dx - target_dx)) + torch.mean(torch.abs(pred_dy - target_dy))

def compute_metrics(pred, target):
    mask = target > 1e-5  # ignore invalid depth (e.g., zero or negative)

    abs_diff = torch.abs(pred - target)[mask]
    mae = abs_diff.mean()
    abs_rel = (abs_diff / target[mask]).mean()
    
    return {
        'mae': mae.item(),
        'abs_rel': abs_rel.item()
    }

def to_target_range(depth, min_val=0.2, max_val=1.5):
    # input depth is expected to be in [0, 1]
    depth = depth * (max_val - min_val) + min_val
    return depth

def get_aspect_ratio(aspect_ratio_str):
    if aspect_ratio_str == "1_1":
        return (1, 1)
    elif aspect_ratio_str == "4_3":
        return (3, 4)
    elif aspect_ratio_str == "16_9":
        return (9, 16)
    else:
        None

def forward(model, rgb, gray, prompt_depth, mode):
    zeros = torch.zeros_like(prompt_depth, dtype=prompt_depth.dtype, device=prompt_depth.device)
    if mode == "mono":
        return model.forward(gray, zeros)
    elif mode == "mono_fusion":
        return model.forward(gray, prompt_depth)
    elif mode == "rgb_fusion":
        return model.forward(rgb, prompt_depth)
    elif mode == "rgb":
        return model.forward(rgb, zeros)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def validate(model, val_loaders, step, device, args, minmax, tag="val"):
    model.eval()
    Log.info(f"Validating at step {step}...")

    all_metrics = []
    total_time_start = time.time()

    with torch.no_grad():
        for val_idx, val_loader in enumerate(val_loaders):
            metrics_dict = {'l1': 0.0, 'grad': 0.0, 'mae': 0.0, 'abs_rel': 0.0}
            for rgb, gray, prompt_depth, target_depth in val_loader:
                rgb = rgb.to(device)
                gray = gray.to(device)
                prompt_depth = prompt_depth.to(device)
                target_depth = target_depth.to(device)

                val_pred = forward(model, rgb, gray, prompt_depth, mode=args.mode)
                val_pred = torch.clamp(val_pred, min=0.0, max=1.0)
                
                val_pred = to_target_range(val_pred, min_val=minmax[0], max_val=minmax[1])
                val_target = to_target_range(target_depth, min_val=minmax[0], max_val=minmax[1])

                l1 = nn.functional.l1_loss(val_pred, val_target)
                grad = gradient_loss(val_pred, val_target)
                metrics = compute_metrics(val_pred, val_target)

                metrics_dict['l1'] += l1.item()
                metrics_dict['grad'] += grad.item()
                metrics_dict['mae'] += metrics['mae']
                metrics_dict['abs_rel'] += metrics['abs_rel']

            # Average per dataset
            metrics_dict = {k: v / len(val_loader) for k, v in metrics_dict.items()}
            all_metrics.append(metrics_dict)

            Log.info(f"[Validation Step {step}] Dataset {val_idx} | "
                     f"L1: {metrics_dict['l1']:.4f} | Grad: {metrics_dict['grad']:.4f} | "
                     f"MAE: {metrics_dict['mae']:.4f} | Abs Rel: {metrics_dict['abs_rel']:.4f}")

            wandb.log({
                f"{tag}{val_idx}/loss": metrics_dict['l1'] + 0.5 * metrics_dict['grad'],
                f"{tag}{val_idx}/l1": metrics_dict['l1'],
                f"{tag}{val_idx}/grad": metrics_dict['grad'],
                f"{tag}{val_idx}/mae": metrics_dict['mae'],
                f"{tag}{val_idx}/abs_rel": metrics_dict['abs_rel'],
            }, step=step)

    # Compute average across datasets
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    Log.info(f"[Validation Step {step}] AVERAGE | "
             f"L1: {avg_metrics['l1']:.4f} | Grad: {avg_metrics['grad']:.4f} | "
             f"MAE: {avg_metrics['mae']:.4f} | Abs Rel: {avg_metrics['abs_rel']:.4f}")

    wandb.log({
        f"{tag}_avg/loss": avg_metrics['l1'] + 0.5 * avg_metrics['grad'],
        f"{tag}_avg/l1": avg_metrics['l1'],
        f"{tag}_avg/grad": avg_metrics['grad'],
        f"{tag}_avg/mae": avg_metrics['mae'],
        f"{tag}_avg/abs_rel": avg_metrics['abs_rel'],
        f"{tag}_avg/time_sec": time.time() - total_time_start,
    }, step=step)

    return all_metrics, avg_metrics


def test(model, test_loader, step, device, args, minmax, tag="val"):
    model.eval()
    Log.info(f"Testing at step {step}...")

    time_spent = []
    with torch.no_grad(): 
        for idx, (rgb, gray, prompt_depth, fname) in enumerate(test_loader):
            rgb = rgb.to(device)
            gray = gray.to(device)
            prompt_depth = prompt_depth.to(device)
            
            time_start = time.time()
            val_pred = forward(model, rgb, gray, prompt_depth, mode=args.mode)
            val_pred = torch.clamp(val_pred, min=0.0, max=1.0)
            val_pred = to_target_range(val_pred, min_val=minmax[0], max_val=minmax[1])
            time_spent.append(time.time() - time_start)

            # Visualization
            input_img = rgb if "rgb" in args.mode else gray
            input_img = input_img.squeeze().cpu().numpy().transpose(1, 2, 0)
            val_pred = val_pred.squeeze().cpu().numpy()
            vis_img = get_visualization(val_pred, input_img, min_depth=minmax[0], max_depth=minmax[1])

            # Log to wandb
            wandb.log({
                f"{tag}/{idx:03d}_{fname}": wandb.Image(vis_img, caption=f"{tag} Sample {idx}: {fname}"),
            })

    avg_time = sum(time_spent) / len(time_spent)
    Log.info(f"[Test Step {step}] Avg inference time per sample: {avg_time:.4f} seconds")
    wandb.log({f"{tag}_avg/inference_time_sec": avg_time}, step=step)
    wandb.finish()
    

def train(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR_VIT_WARMUP = 5e-6
    LR_VIT = 5e-6
    LR_OTHER = 5e-5
    MINMAX = (args.d_min, args.d_max)

    train_txt = args.train_txt
    val_txt = args.val_txt
    ts = None
    if args.input_res[0] > -1 and args.input_res[1] > -1:
        ts = (args.input_res[0], args.input_res[1])
    ar = get_aspect_ratio(args.input_aspect_ratio)
    train_sets = [SimulatedDataset(txt, aspect_ratio=ar, random_crop=args.random_crop, target_size=ts) for txt in train_txt]
    val_sets   = [SimulatedDataset(txt, aspect_ratio=ar, random_crop=args.random_crop, target_size=ts) for txt in val_txt]
    train_loaders = [DataLoader(train_set, batch_size=args.batch_size, shuffle=True) for train_set in train_sets]
    val_loaders   = [DataLoader(val_set, batch_size=args.batch_size, shuffle=False) for val_set in val_sets]
    train_iters = [iter(loader) for loader in train_loaders]
    train_weights = np.array(args.sample_weights, dtype=np.float32)
    train_probs = train_weights / np.sum(train_weights)

    if args.test_txt:
        test_set = RealDataset(args.test_txt, aspect_ratio=ar, target_size=ts)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    else:
        test_loader = None

    model = PromptDA(encoder=args.backbone, output_act=args.output_act)
    model = torch.nn.DataParallel(model)
    pretrained = torch.load(args.pretrained, map_location=DEVICE)
    model.module.load_state_dict(pretrained, strict=False)
    model = model.to(DEVICE).train()

    vit_params = []
    other_params = []
    for name, param in model.module.named_parameters():
        (vit_params if "pretrained" in name else other_params).append(param)
    for param in other_params:
        param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': vit_params,   'lr': LR_VIT_WARMUP},
        {'params': other_params, 'lr': 0.0}
    ])
    # No scheduling is used according to authors' reply
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.5)

    wandb.init(project="PromptDA-Training", name=args.exp_name)
    wandb.config.update(vars(args))

    step = 0
    total_steps = args.train_steps + args.warm_up_steps
    while step < total_steps:
        step_start_time = time.time()
        dataset_idx = np.random.choice(len(train_iters), p=train_probs)
        train_iter = train_iters[dataset_idx]
        try:
            data_start = time.time()
            rgb, gray, prompt_depth, target_depth = next(train_iter)
            data_time = time.time() - data_start
        except StopIteration:
            train_iters[dataset_idx] = iter(train_loaders[dataset_idx])
            train_iter = train_iters[dataset_idx]
            rgb, gray, prompt_depth, target_depth = next(train_iter)
            data_time = 0.0

        step += 1
        rgb          = rgb.to(DEVICE)
        gray         = gray.to(DEVICE)
        prompt_depth = prompt_depth.to(DEVICE)
        target_depth = target_depth.to(DEVICE)

        if step == args.warm_up_steps:
            Log.info(f"Warm-up completed at step {step}. Switching to full training mode.")
            for param in model.module.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW([
                {'params': vit_params,   'lr': LR_VIT},
                {'params': other_params, 'lr': LR_OTHER}
            ])
            # No scheduling is used according to authors' reply
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.5)

        model.train()
        optimizer.zero_grad()

        fwd_start = time.time()
        pred_depth = forward(model, rgb, gray, prompt_depth, args.mode)
        fwd_time = time.time() - fwd_start

        # !!! WE ASSUME LABELS ARE IN THE RANGE [0, 1] !!!
        pred_depth = to_target_range(pred_depth, min_val=MINMAX[0], max_val=MINMAX[1])
        target_depth = to_target_range(target_depth, min_val=MINMAX[0], max_val=MINMAX[1])
        # !!! ---------------------------------------- !!!
        
        loss_l1 = nn.functional.l1_loss(pred_depth, target_depth)
        loss_grad = gradient_loss(pred_depth, target_depth)
        loss = loss_l1 + 0.5 * loss_grad

        bwd_start = time.time()
        loss.backward()
        optimizer.step()
        bwd_time = time.time() - bwd_start
        scheduler.step()
        gpu_stats = log_all_gpu_stats()
        
        if step % 100 == 0:
            pred_depth = torch.clamp(pred_depth, min=MINMAX[0], max=MINMAX[1])
            train_metrics = compute_metrics(pred_depth, target_depth)
            wandb.log({
                "train/loss": loss.item(),
                "train/loss_l1": loss_l1.item(),
                "train/loss_grad": loss_grad.item(),
                "train/mae": train_metrics['mae'],
                "train/abs_rel": train_metrics['abs_rel'],
                "lr": scheduler.get_last_lr()[0],
                "timing/data_time_sec": data_time,
                "timing/fwd_time_sec": fwd_time,
                "timing/bwd_time_sec": bwd_time,
                "timing/total_step_time_sec": time.time() - step_start_time,
                **gpu_stats
            }, step=step)
            Log.info(f"[Step {step}] "
                     f"Loss: {loss.item():.4f} | "
                     f"L1: {loss_l1.item():.4f} | "
                     f"Grad: {loss_grad.item():.4f} | "
                     f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                     f"FWD: {fwd_time:.3f}s | BWD: {bwd_time:.3f}s | "
                     f"Total: {time.time() - step_start_time:.3f}s")

        if step % args.validate_every == 0:
            validate(model, val_loaders, step, DEVICE, args, MINMAX, tag="val")

        if test_loader and step % args.test_every == 0:
            test(model, test_loader, step, DEVICE, args, MINMAX, tag="test")

        if step % args.save_step == 0:
            ckpt_dir = os.path.join(args.ckpt_dir, f"{args.exp_name}-{args.backbone}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"step_{step}.pth")
            torch.save({
                'step': step,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': vars(args),  # optional: saves hyperparams too
            }, ckpt_path)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Train PromptDA on HyperSim")
    parser.add_argument("--backbone", type=str, default="vitb", help="Backbone architecture")
    parser.add_argument("--pretrained", type=str, default="/scratch/bl3912/checkpoints/PromptDA/depth_anything_v2_metric_hypersim_vitb.pth", help="Pretrained backbone model")
    parser.add_argument("--exp-name", type=str, default="default", help="Experiment name")
    parser.add_argument("--save-step", type=int, default=10000, help="Save model every N steps")
    parser.add_argument("--warm-up-steps", type=int, default=10000, help="Warm-up steps")
    parser.add_argument("--train-steps", type=int, default=200000, help="Total training steps")
    parser.add_argument("--validate-every", type=int, default=500, help="Validation frequency")
    parser.add_argument("--test-every", type=int, default=1000, help="Testing frequency")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--ckpt-dir", type=str, default="/scratch/bl3912/checkpoints/PromptDA", help="Directory to save checkpoints")
    
    parser.add_argument("--train-txt", type=str, nargs='+', default=["/vast/bl3912/hypersim-random-10k/train.txt"], help="Path to training text file")
    parser.add_argument("--sample-weights", type=float, nargs='+', default=[1.0], help="Sample weights for training datasets")
    parser.add_argument("--val-txt", type=str, nargs='+', default=["/vast/bl3912/hypersim-random-10k/val.txt"], help="Path to validation text file")
    parser.add_argument("--test-txt", type=str, default=None, help="Path to test text file")

    parser.add_argument("--disparity", action='store_true', help="Use disparity instead of depth")
    parser.add_argument("--normalize", action='store_true', help="Whether to normalize input images")
    parser.add_argument("--random-crop", action='store_true', help="Whether to use random cropping in training")

    parser.add_argument("--d-min", type=float, default=0.2, help="Minimum depth value")
    parser.add_argument("--d-max", type=float, default=1.5, help="Maximum depth value")
    parser.add_argument("--output-act", type=str, default="identity", help="Last layer type (e.g., Identity, Conv2d, etc.)")
    parser.add_argument("--mode", type=str, default="rgb_fusion", choices=["mono", "mono_fusion", "rgb_fusion", "rgb"], help="Mode of operation")
    parser.add_argument("--input-aspect-ratio", type=str, default="None", choices=["1_1", "4_3", "16_9"], help="Aspect ratio of input images")
    parser.add_argument("--input-res", type=int, nargs='+', default=[-1, -1], help="Input image size (height and width)")
    parser.add_argument("--pol-imbalance", action='store_true', help="Use polarization imbalance augmentation")
    parser.add_argument("--max-gs-kernels", type=int, default=0, help="Number of Gaussian kernels for augmentation")
    parser.add_argument("--max-gs-intensity", type=float, default=0.0, help="Maximum intensity for Gaussian kernels")

    args = parser.parse_args()

    train(args)
