from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth, save_depth

DEVICE = 'cuda'
image_path = "assets/example_images/image.jpg"
prompt_depth_path = "assets/example_images/arkit_depth.png"
image = load_image(image_path).to(DEVICE)
prompt_depth = load_depth(prompt_depth_path).to(DEVICE) # 192x256, ARKit LiDAR depth in meters

# model = PromptDA(encoder='vits')
# model = model.from_depth_anything("checkpoints/depth_anything_v2_metric_hypersim_vits.pth").to(DEVICE).eval()
model = PromptDA.from_pretrained("checkpoints/promptda_vits.ckpt").to(DEVICE).eval()

depth = model.predict(image, prompt_depth) # HxW, depth in meters

save_depth(depth, prompt_depth=prompt_depth, image=image)