import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth
from promptda.utils.logger import Log
import wandb


class HypersimDepthDataset(Dataset):
    def __init__(self, txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        self.pairs = [line.strip().split(',') for line in lines]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path = self.pairs[idx][0].strip()
        img2_path = self.pairs[idx][1].strip()
        depth_path = self.pairs[idx][2].strip()

        img1 = load_image(img1_path)
        img2 = load_image(img2_path)
        rgb = torch.cat([img1, img2, (img1 + img2) / 2], dim=1)
        prompt_depth = torch.cat([img1, img2], dim=1)

        target_depth = load_depth(depth_path)
        target_depth = cv2.resize(
            target_depth.squeeze().numpy(),
            (img1.shape[3], img1.shape[2]),
            interpolation=cv2.INTER_AREA
        )
        target_depth = torch.from_numpy(target_depth).unsqueeze(0)
        return rgb.squeeze(0), prompt_depth.squeeze(0), target_depth

from pynvml import *
import time

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

def train(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR_VIT_WARMUP = 5e-6
    LR_VIT = 5e-6
    LR_OTHER = 5e-5

    train_txt = '/vast/bl3912/hypersim-random-10k/train.txt'
    val_txt = '/vast/bl3912/hypersim-random-10k/val.txt'

    train_set = HypersimDepthDataset(train_txt)
    val_set = HypersimDepthDataset(val_txt)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    train_iter = iter(train_loader)

    model = PromptDA(encoder=args.backbone)
    model = torch.nn.DataParallel(model)
    pretrained = torch.load(f"checkpoints/depth_anything_v2_metric_hypersim_{args.backbone}.pth")
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

    wandb.init(project="PromptDA-Training", name=args.exp_name+'-'+args.backbone)

    step = 0
    total_steps = args.train_steps + args.warm_up_steps
    while step < total_steps:
        step_start_time = time.time()
        try:
            data_start = time.time()
            rgb, prompt_depth, target_depth = next(train_iter)
            data_time = time.time() - data_start
        except StopIteration:
            train_iter = iter(train_loader)
            rgb, prompt_depth, target_depth = next(train_iter)
            data_time = 0.0

        step += 1
        rgb          = rgb.to(DEVICE)
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
        pred_depth = model.forward(rgb, prompt_depth)
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
        
        if step % 10 == 0 or step == 1:
            wandb.log({
                "train/loss": loss.item(),
                "train/loss_l1": loss_l1.item(),
                "train/loss_grad": loss_grad.item(),
                "lr": scheduler.get_last_lr()[0],
                "timing/data_time_sec": data_time,
                "timing/fwd_time_sec": fwd_time,
                "timing/bwd_time_sec": bwd_time,
                "timing/total_step_time_sec": time.time() - step_start_time,
                **gpu_stats
            }, step=step)

        if step % 100 == 0 or step == 1:
            Log.info(f"[Step {step}] "
                     f"Loss: {loss.item():.4f} | "
                     f"L1: {loss_l1.item():.4f} | "
                     f"Grad: {loss_grad.item():.4f} | "
                     f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                     f"FWD: {fwd_time:.3f}s | BWD: {bwd_time:.3f}s | "
                     f"Total: {time.time() - step_start_time:.3f}s")

        if step % args.validate_every == 0:
            model.eval()
            metrics_dict = {'l1': 0.0, 'grad': 0.0, 'mae': 0.0, 'abs_rel': 0.0}
            with torch.no_grad():
                for val_rgb, val_prompt, val_target in val_loader:
                    val_rgb = val_rgb.to(DEVICE)
                    val_prompt = val_prompt.to(DEVICE)
                    val_target = val_target.to(DEVICE)
                    val_pred = model.forward(val_rgb, val_prompt)
                    l1 = nn.functional.l1_loss(val_pred, val_target)
                    grad = gradient_loss(val_pred, val_target)
                    metrics = compute_metrics(val_pred, val_target)
                    metrics_dict['l1'] += l1.item()
                    metrics_dict['grad'] += grad.item()
                    metrics_dict['mae'] += metrics['mae']
                    metrics_dict['abs_rel'] += metrics['abs_rel']

                metrics_dict = {k: v / len(val_loader) for k, v in metrics_dict.items()}
                wandb.log({
                    "val/loss": metrics_dict['l1'] + 0.5 * metrics_dict['grad'],
                    "val/l1": metrics_dict['l1'],
                    "val/grad": metrics_dict['grad'],
                    "val/mae": metrics_dict['mae'],
                    "val/abs_rel": metrics_dict['abs_rel']
                }, step=step)
                Log.info(f"[Validation Step {step}] "
                         f"Loss: {metrics_dict['l1']:.4f} | "
                         f"Grad: {metrics_dict['grad']:.4f} | "
                         f"MAE: {metrics_dict['mae']:.4f} | "
                         f"Abs Rel: {metrics_dict['abs_rel']:.4f}")

        if step % args.save_step == 0:
            ckpt_dir = f"checkpoints/{args.exp_name}-{args.backbone}"
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
    parser.add_argument("--exp-name", type=str, default="default", help="Experiment name")
    parser.add_argument("--save-step", type=int, default=10000, help="Save model every N steps")
    parser.add_argument("--warm-up-steps", type=int, default=10000, help="Warm-up steps")
    parser.add_argument("--train-steps", type=int, default=200000, help="Total training steps")
    parser.add_argument("--validate-every", type=int, default=500, help="Validation frequency")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")

    args = parser.parse_args()

    train(args)
