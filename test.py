from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth, save_depth
import os, cv2, torch

ds_txt = '/vast/bl3912/hypersim-random-10k'
with open(os.path.join(ds_txt, 'val.txt'), 'r') as f:
    lines = f.readlines()
data = lines[0]
img1 = data.split(',')[0].strip()
img2 = data.split(',')[1].strip()
depth = data.split(',')[2].strip()

DEVICE = 'cuda'
# image_path = "/vast/bl3912/hypersim-random-10k/val/00066-1.png"
# prompt_depth_path = "assets/example_images/arkit_depth.png"

# image_path = "assets/example_images/image.jpg"
image_path = "/vast/bl3912/hypersim-random-10k/val/color-png/00066.png"
image = load_image(image_path).to(DEVICE)
# image = image.repeat(1, 3, 1, 1) # BCHW, RGB image

# image2 = load_image(img2).to(DEVICE)
# image2 = image2.repeat(1, 3, 1, 1) #
# image = (image + image2) / 2

image1 = load_image(img1).to(DEVICE)
image2 = load_image(img2).to(DEVICE)
image3 = (image1 + image2) / 2
# image = torch.cat([image1, image2, image3], dim=1)  # BCHW, RGB image

prompt_depth = torch.cat([image1, image2], dim=1)
model = PromptDA(encoder='vitb')
model = model.from_depth_anything("checkpoints/depth_anything_v2_metric_hypersim_vitb.pth").to(DEVICE).eval()

gt_depth = load_depth(depth) # 192x256, ARKit LiDAR depth in meters
gt_depth = cv2.resize(gt_depth.squeeze().numpy(), (image.shape[3], image.shape[2]), interpolation=cv2.INTER_AREA)
gt_depth = torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0).to(DEVICE)  # B1HW, single channel depth
# model = PromptDA.from_pretrained("checkpoints/promptda_vits.ckpt").to(DEVICE).eval()

depth = model.predict(image, prompt_depth) # HxW, depth in meters

save_depth(depth, image=image)