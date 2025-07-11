from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth, save_depth
import os, cv2, torch, torch.nn as nn, tqdm
from promptda.utils.logger import Log
from torch.utils.data import Dataset, DataLoader
import numpy as np

def center_crop(img, crop_size):
    h, w = img.shape
    ch, cw = crop_size
    start_y = max((h - ch) // 2, 0)
    start_x = max((w - cw) // 2, 0)
    return img[start_y:start_y + ch, start_x:start_x + cw]

def process_real_img(img_src, scale_factor=0.5, crop_size=(768, 1024)):
    img = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    img = center_crop(img, crop_size)

    return img.astype(np.float32)  # Convert to float32 for consistency with model input

class RealStereoDataset(Dataset):
    def __init__(self, txt_path, scale_factor=0.5, crop_size=(1200, 1600), mutliple_of=14):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        self.pairs = [line.strip().split(',') for line in lines]
        self.scale_factor = scale_factor
        self.crop_size = crop_size  # Should be passed as (H, W)
        h = crop_size[0] // mutliple_of * mutliple_of
        w = crop_size[1] // mutliple_of * mutliple_of
        self.crop_size = (h, w)  # Ensure crop size is a multiple of
        self.multiple_of = mutliple_of

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path = self.pairs[idx][0].strip()
        img2_path = self.pairs[idx][1].strip()
        depth_path = self.pairs[idx][2].strip()

        img1 = process_real_img(img1_path, self.scale_factor, self.crop_size)
        img2 = process_real_img(img2_path, self.scale_factor, self.crop_size)
        depth = process_real_img(depth_path, 1, self.crop_size)

        img1_tensor = torch.from_numpy(img1 / 255.0).unsqueeze(0)  # [1, H, W]
        img2_tensor = torch.from_numpy(img2 / 255.0).unsqueeze(0)
        depth_tensor = torch.from_numpy(depth / 100.0).unsqueeze(0)
        
        avg = (img1_tensor + img2_tensor) / 2
        rgb = torch.cat([img1_tensor, img2_tensor, avg], dim=0)  # [3, H, W]
        
        prompt = torch.cat([img1_tensor, img2_tensor], dim=0)  # [2, H, W]

        return rgb, prompt, depth_tensor


model = PromptDA(encoder='vitl')
# model = torch.nn.DataParallel(model)

val_txt = '/vast/bl3912/real_capture/data.txt'
val_set = RealStereoDataset(val_txt, crop_size=(1200, 1600))
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

DEVICE = 'cuda'
for step in [130000]:
    ckpt_path = f"checkpoints/first_try_hypersim_10k_0709-vitb/step_{step}.pth"
    ckpt_path = f"checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt, strict=False)
        Log.info(f"Loaded checkpoint from {ckpt_path}")
        model = model.to(DEVICE).eval()
        l1_e = 0
        abs_rel_e = 0
        cnt = 0
        with torch.no_grad():
            with tqdm.tqdm(total=len(val_loader), desc="Validating") as progress:
                for val_rgb, val_prompt, val_target in val_loader:
                    progress.refresh()
                    val_rgb = val_rgb.to(DEVICE)
                    val_prompt = val_prompt.to(DEVICE)
                    val_target = val_target.to(DEVICE)
                    val_pred = model.predict(val_rgb, val_prompt)
                    l1 = nn.functional.l1_loss(val_pred, val_target)
                    abs_rel = torch.mean(torch.abs(val_pred - val_target) / (val_target + 1e-6))
                    l1_e += l1
                    abs_rel_e += abs_rel
                    progress.update(1)
                    progress.set_postfix({'L1 Loss': l1.item(), 'Abs Rel': abs_rel.item()})
                    save_depth(
                        val_pred, gt_depth=val_target, image=val_rgb, 
                        output_path=f'real-vitl', name=f'{cnt:03d}', in_one_plot=True
                    )
                    cnt += 1

            l1_e /= len(val_loader)
            abs_rel_e /= len(val_loader)
            Log.info(f"L1 = {l1_e.item():.4f}, Abs Rel = {abs_rel_e.item():.4f}")
            # depth = depth.squeeze(0).cpu().numpy()
            # print(np.min(depth), np.max(depth), depth.shape)
            # save_depth(
            #     depth, gt_depth=gt_depth, image=image, 
            #     output_path=f'results-{step}', name=f'{data_id:05d}', in_one_plot=True
            # )
            # validate(model, val_loader)