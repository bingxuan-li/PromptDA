import os, cv2, torch, wandb, numpy as np, torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from promptda.promptda import PromptDA
from promptda.utils.logger import Log
from dataset import SimulatedDataset, RealDataset, Augmentation
from pynvml import *
import time, glob
from visualize import visualize_depth, overlay_text_on_image

def finalize(model, ckpt, test_loader, tag, args):
    with torch.no_grad():
        ckpt_list = sorted(glob.glob(os.path.join(ckpt, '*.pth')))
        for flip in [True, False]:
            image_v = []
            for ckpt in ckpt_list:
                params = torch.load(ckpt, map_location='cuda')
                model.module.load_state_dict(params['model_state_dict'], strict=False)
                model = model.to('cuda').eval()
                image_h = []
                for idx, (rgb, gray, prompt_depth, fname) in enumerate(test_loader):
                    if flip:
                        img1, img2 = img2, img1
                    rgb = rgb.to('cuda')
                    gray = gray.to('cuda')
                    prompt_depth = prompt_depth.to('cuda')

                    pred = forward(model, rgb, gray, gray, prompt_depth, mode=args.mode)
                    pred = torch.clamp(pred, min=0.3, max=1.2)

                    pred = pred.squeeze().cpu().numpy()
                    if 'rgb' in args.mode:
                        img = rgb
                    else:
                        img = gray
                    img = img.squeeze().cpu().numpy().transpose(1, 2, 0)
                    img = (img * 255.0).astype(np.uint8)
                    if idx == 0:
                        img = overlay_text_on_image(img, f"{params['step']}")
                    else:
                        img = overlay_text_on_image(img, f"{fname}")
                    depth_vis = visualize_depth(pred, depth_min=0.3, depth_max=1.2, cmap='Spectral')
                    combined = np.hstack((img, depth_vis))
                    combined = cv2.resize(combined, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
                    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                    image_h.append(combined)
                image_h = np.hstack(image_h)
                image_v.append(image_h)
            image_v = np.vstack(image_v)
            fname = f"{args.exp_name}_{tag}_flip_{flip}"
            prefix = f'./tmp/0806/flip_{flip}/{args.exp_name}'
            os.makedirs(prefix, exist_ok=True)
            cv2.imwrite(os.path.join(f'{prefix}', f"{fname}.jpg"), image_v, [cv2.IMWRITE_JPEG_QUALITY, 80])


def count_parameters_in_millions(model):
    """Returns the number of trainable parameters in a PyTorch model, in millions."""
    num_params = sum(p.numel() for p in model.parameters())
    return num_params/1e6

def has_nested_attr(obj, attr_path):
    for attr in attr_path.split('.'):
        if not hasattr(obj, attr):
            return False
        obj = getattr(obj, attr)
    return True

def count_model_params(model):
    total_params = count_parameters_in_millions(model)
    vit_params = count_parameters_in_millions(model.pretrained) + count_parameters_in_millions(model.depth_head)
    encoder_params = count_parameters_in_millions(model.pretrained)
    decoder_params = vit_params - encoder_params
    fusion_params = count_parameters_in_millions(model.depth_head.scratch)
    vit_params -= fusion_params
    if has_nested_attr(model, "depth_head.scratch.depth_encoder"):
        depth_encoder_params = count_parameters_in_millions(model.depth_head.scratch.depth_encoder)
    else:
        depth_encoder_params = 0.0
    return {'total': total_params, 'vit': vit_params, 'encoder': encoder_params, 'decoder': decoder_params, 'fusion': fusion_params, 'depth_encoder': depth_encoder_params}

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

def forward(model, rgb, sim_gray, true_gray, prompt_depth, mode):
    zeros = torch.zeros_like(prompt_depth, dtype=prompt_depth.dtype, device=prompt_depth.device)
    if mode == "mono":
        return model.forward(sim_gray, zeros)
    elif mode == "true_mono":
        return model.forward(true_gray, zeros)
    elif mode == "mono_fusion":
        return model.forward(sim_gray, prompt_depth)
    elif mode == "true_mono_fusion":
        return model.forward(true_gray, prompt_depth)
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
            for rgb, sim_gray, true_gray, prompt_depth, target_depth in val_loader:
                rgb = rgb.to(device)
                sim_gray = sim_gray.to(device)
                true_gray = true_gray.to(device)
                prompt_depth = prompt_depth.to(device)
                val_target = target_depth.to(device)

                val_pred = forward(model, rgb, sim_gray, true_gray, prompt_depth, mode=args.mode)
                val_pred = torch.clamp(val_pred, min=minmax[0], max=minmax[1])

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


def test(model, test_loader, step, device, args, minmax, tag="test"):
    model.eval()
    Log.info(f"Testing at step {step}...")

    time_spent = []
    with torch.no_grad(): 
        combined_images = []
        for idx, (rgb, gray, prompt_depth, fname) in enumerate(test_loader):
            rgb = rgb.to(device)
            gray = gray.to(device)
            prompt_depth = prompt_depth.to(device)
            
            time_start = time.time()
            val_pred = forward(model, rgb, gray, gray, prompt_depth, mode=args.mode)
            val_pred = torch.clamp(val_pred, min=minmax[0], max=minmax[1])

            time_spent.append(time.time() - time_start)
        
            try:
                # Visualization
                # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                # success, encoded_img = cv2.imencode('.jpg', vis_img, encode_param)
                # os.makedirs('tmp', exist_ok=True)
                # if success:
                #     with open('tmp/tmp.jpg', 'wb') as f:
                #         f.write(encoded_img.tobytes())
                # Log to wandb
                input_img = rgb if "rgb" in args.mode else gray
                input_img = input_img.squeeze().cpu().numpy().transpose(1, 2, 0)
                val_pred = val_pred.squeeze().cpu().numpy()

                img = (input_img * 255.0).astype(np.uint8)
                img = overlay_text_on_image(img, f"{fname}")
                depth_vis = visualize_depth(val_pred, depth_min=minmax[0], depth_max=minmax[1], cmap='Spectral')
                combined = np.hstack([img, depth_vis])
                # vis_img = get_visualization(val_pred, input_img, min_depth=minmax[0], max_depth=minmax[1])
                combined = cv2.resize(combined, (0, 0), fx=0.2, fy=0.2)  # Downsample for visualization
                combined_images.append(combined)
            except Exception as e:
                Log.error(f"Error visualizing sample {idx} ({fname}): {e}")
        combined_images = np.hstack(combined_images) if combined_images else None
        if combined_images is not None:
            wandb.log({
                f"{tag}": wandb.Image(combined_images)
            }, step=step)


    avg_time = sum(time_spent) / len(time_spent)
    Log.info(f"[Test Step {step}] Avg inference time per sample: {avg_time:.4f} seconds")
    wandb.log({f"timing/inference_time_sec": avg_time}, step=step)
    

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
    pc = args.prompt_channels
    bs = args.batch_size
    td = args.test_downsample
    tr = args.test_res
    train_sets = [SimulatedDataset(txt, aspect_ratio=ar, random_crop=args.random_crop, target_size=ts, prompt_channels=pc, confuse_lens_z=args.confuse_lens_z) for txt in train_txt]
    val_sets   = [SimulatedDataset(txt, aspect_ratio=ar, random_crop=args.random_crop, target_size=ts, prompt_channels=pc, confuse_lens_z=args.confuse_lens_z) for txt in val_txt]
    test_sets =  [RealDataset(txt, downsample=td[i], aspect_ratio=ar, target_size=(tr[2*i], tr[2*i+1]), prompt_channels=pc) for i, txt in enumerate(args.test_txt)]

    train_loaders = [DataLoader(train_set, batch_size=bs, num_workers=min(os.cpu_count(), args.num_workers), shuffle=True) for train_set in train_sets]
    val_loaders   = [DataLoader(val_set,   batch_size=bs, num_workers=min(os.cpu_count(), args.num_workers), shuffle=False) for val_set in val_sets]
    test_loaders =  [DataLoader(test_set,  batch_size=1, shuffle=False) for test_set in test_sets]

    train_iters = [iter(loader) for loader in train_loaders]
    train_weights = np.array(args.sample_weights, dtype=np.float32)
    train_probs = train_weights / np.sum(train_weights)

    model = PromptDA(encoder=args.backbone, output_act=args.output_act, prompt_channels=pc, resnet_enabled=args.resnet_enabled, resnet_blocks_per_stage=args.resnet_blocks_per_stage)

    param_counts = count_model_params(model)
    Log.info(
        f"Model has {param_counts['total']:.2f}M parameters. "
        f"ViT encoder: {param_counts['vit']:.2f}M, "
        f"Encoder: {param_counts['encoder']:.2f}M, "
        f"Decoder: {param_counts['decoder']:.2f}M, "
        f"Fusion head: {param_counts['fusion']:.2f}M, "
        f"Depth encoder: {param_counts['depth_encoder']:.2f}M."
    )
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.8)
    
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
            rgb, sim_gray, true_gray, prompt_depth, target_depth = next(train_iter)
            data_time = time.time() - data_start
        except StopIteration:
            train_iters[dataset_idx] = iter(train_loaders[dataset_idx])
            train_iter = train_iters[dataset_idx]
            rgb, sim_gray, true_gray, prompt_depth, target_depth = next(train_iter)
            data_time = 0.0

        step += 1
        rgb          = rgb.to(DEVICE)
        sim_gray     = sim_gray.to(DEVICE)
        true_gray    = true_gray.to(DEVICE)
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
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.8)
        if step == args.augment_start_steps:
            Log.info(f"Starting augmentation at step {step}.")
            augment = Augmentation(args.augment_probability, args.max_p_noise, args.max_g_noise, args.max_gs_blur,
                                   args.max_global_imbalance, args.max_local_imbalance, args.max_gs_kernels)
            for train_set in train_sets:
                train_set.set_augmentation(augment)

        model.train()
        optimizer.zero_grad()

        fwd_start = time.time()
        pred_depth = forward(model, rgb, sim_gray, true_gray, prompt_depth, args.mode)
        fwd_time = time.time() - fwd_start

        loss_l1 = nn.functional.l1_loss(pred_depth, target_depth)
        loss_grad = gradient_loss(pred_depth, target_depth)
        loss = loss_l1 + 0.5 * loss_grad

        bwd_start = time.time()
        loss.backward()
        optimizer.step()
        bwd_time = time.time() - bwd_start
        scheduler.step()
        gpu_stats = log_all_gpu_stats()

        if step % args.log_every == 0:
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

        if step % args.test_every == 0:
            for i, test_loader in enumerate(test_loaders):
                test(model, test_loader, step, DEVICE, args, MINMAX, tag=f"test_{i}")

        if step % args.save_every == 0:
            ckpt_dir = os.path.join(args.ckpt_dir, f"{args.exp_name}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"step_{step}.pth")
            torch.save({
                'step': step,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': vars(args),  # optional: saves hyperparams too
            }, ckpt_path)
    for idx, test_loader in enumerate(test_loaders):
        Log.info(f"Finalizing test for dataset {idx}...")
        ckpt_dir = os.path.join(args.ckpt_dir, f"{args.exp_name}")
        finalize(model, ckpt_dir, test_loader, f"{idx}", args)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Train PromptDA on HyperSim")

    # Model Definition
    parser.add_argument("--backbone", type=str, default="vitb", help="Backbone architecture")
    parser.add_argument("--prompt-channels", type=int, default=2, help="Number of prompt channels")
    parser.add_argument("--output-act", type=str, default="identity", help="Last layer type (e.g., Identity, Conv2d, etc.)")
    parser.add_argument("--mode", type=str, default="rgb_fusion", choices=["true_mono", "mono", "true_mono_fusion", "mono_fusion", "rgb_fusion", "rgb"], help="Mode of operation")
    parser.add_argument("--resnet-enabled", action='store_true', help="Enable ResNet encoder")
    parser.add_argument("--resnet-blocks-per-stage", type=int, default=3, help="Number of blocks per stage in ResNet")

    # Training Parameters    
    parser.add_argument("--exp-name", type=str, default="default", help="Experiment name")
    parser.add_argument("--pretrained", type=str, default="/scratch/bl3912/checkpoints/PromptDA/depth_anything_v2_metric_hypersim_vitb.pth", help="Pretrained backbone model")
    parser.add_argument("--ckpt-dir", type=str, default="/scratch/bl3912/checkpoints/PromptDA", help="Directory to save checkpoints")
    parser.add_argument("--warm-up-steps", type=int, default=10000, help="Warm-up steps")
    parser.add_argument("--train-steps", type=int, default=200000, help="Total training steps")
    parser.add_argument("--save-every", type=int, default=10000, help="Save model every N steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log frequency")
    parser.add_argument("--validate-every", type=int, default=500, help="Validation frequency")
    parser.add_argument("--test-every", type=int, default=1000, help="Testing frequency")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    
    # Data Definition
    parser.add_argument("--d-min", type=float, default=0.2, help="Minimum depth value")
    parser.add_argument("--d-max", type=float, default=1.5, help="Maximum depth value")
    parser.add_argument("--random-crop", action='store_true', help="Whether to use random cropping in training")
    parser.add_argument("--train-txt", type=str, nargs='+', default=["/vast/bl3912/hypersim-random-10k/train.txt"], help="Path to training text file")
    parser.add_argument("--sample-weights", type=float, nargs='+', default=[1.0], help="Sample weights for training datasets")
    parser.add_argument("--input-aspect-ratio", type=str, default="None", choices=["1_1", "4_3", "16_9"], help="Aspect ratio of input images")
    parser.add_argument("--input-res", type=int, nargs='+', default=[-1, -1], help="Input image size (height and width)")
    parser.add_argument("--val-txt", type=str, nargs='+', default=["/vast/bl3912/hypersim-random-10k/val.txt"], help="Path to validation text file")
    parser.add_argument("--test-txt", type=str, nargs='+', default=[], help="Path to test text file")
    parser.add_argument("--test-downsample", type=float, nargs='+', default=[1.0], help="Downsample factor for test images")
    parser.add_argument("--test-res", type=int, nargs='+', default=[], help= "Resolution for test images, if empty, uses original size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--confuse-lens-z", type=str, nargs='+', default=['z180.0'], help="Weights for training text files")

    # Augmentation
    parser.add_argument("--augment-start-steps", type=int, default=1000000, help="Steps after which augmentation starts")
    parser.add_argument("--augment-probability", type=float, default=0.5, help="Probability of applying augmentation")
    parser.add_argument("--max-global-imbalance", type=float, default=0.0, help="Maximum global imbalance factor")
    parser.add_argument("--max-local-imbalance", type=float, default=0.0, help="Maximum local imbalance factor")
    parser.add_argument("--max-gs-blur", type=float, default=0, help="Maximum Gaussian blur")
    parser.add_argument("--max-gs-kernels", type=int, default=0, help="Maximum number of Gaussian kernels for local imbalance")
    parser.add_argument("--max-p-noise", type=float, default=0.0, help="Maximum Poisson noise factor")
    parser.add_argument("--max-g-noise", type=float, default=0.0, help="Maximum Gaussian noise factor")

    args = parser.parse_args()

    train(args)
