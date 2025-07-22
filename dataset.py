import os, cv2, torch, numpy as np
from torch.utils.data import Dataset
from promptda.utils.io_wrapper import load_depth

def center_crop(img, crop_size):
    h, w = img.shape
    ch, cw = crop_size
    start_y = max((h - ch) // 2, 0)
    start_x = max((w - cw) // 2, 0)
    return img[start_y:start_y + ch, start_x:start_x + cw]

def target_crop(img, frame):
    """Crop from a fixed (y, x) origin."""
    h, w = img.shape
    start_x, start_y, sx, sy = frame
    assert 0 <= start_y <= h - sy and 0 <= start_x <= w - sx, "Invalid crop origin or size"
    return img[start_y:start_y + sy, start_x:start_x + sx]

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Failed to load image at {path}"
    return img.astype(np.float32) / 255.0  # Normalize to [0, 1]

def process_img(shape, multiple_of=14, aspect_ratio=None, target_size=None, random_crop=False):
    """
    Determine crop frame (x, y, sx, sy) without cropping the image.
    """
    h, w = shape

    if target_size is not None:
        tar_h = (target_size[0] // multiple_of) * multiple_of
        tar_w = (target_size[1] // multiple_of) * multiple_of
    elif aspect_ratio is not None:
        aspect_h, aspect_w = aspect_ratio
        scale_h = h // multiple_of
        scale_w = w // multiple_of
        max_scale_w = (scale_h * aspect_w) // aspect_h
        max_scale_h = (scale_w * aspect_h) // aspect_w
        tar_h = min(scale_h, max_scale_h) * multiple_of
        tar_w = min(scale_w, max_scale_w) * multiple_of
    else:
        tar_h = (h // multiple_of) * multiple_of
        tar_w = (w // multiple_of) * multiple_of

    if random_crop:
        start_y = np.random.randint(0, h - tar_h + 1)
        start_x = np.random.randint(0, w - tar_w + 1)
    else:
        start_y = max((h - tar_h) // 2, 0)
        start_x = max((w - tar_w) // 2, 0)

    return start_x, start_y, tar_w, tar_h

class SimulatedDataset(Dataset):
    def __init__(self, txt_path, multiple_of=14, aspect_ratio=None, random_crop=False, target_size=None):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        self.src = txt_path
        self.pairs = [line.strip().split(',') for line in lines]
        self.disparity = True if 'disparity' in txt_path.lower() else False
        self.normalize = True if 'normalize' in txt_path.lower() else False
        self.random_crop = random_crop
        self.multiple_of = multiple_of
        self.aspect_ratio = aspect_ratio
        self.target_size = target_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path = self.pairs[idx][0].strip()
        img2_path = self.pairs[idx][1].strip()
        depth_path = self.pairs[idx][2].strip()

        img1, img2 = read_img(img1_path), read_img(img2_path)
        frame = process_img(img1.shape, self.multiple_of, self.aspect_ratio, self.target_size, self.random_crop)
        img1, img2 = target_crop(img1, frame), target_crop(img2, frame)
        img1, img2 = torch.from_numpy(img1).unsqueeze(0), torch.from_numpy(img2).unsqueeze(0)

        rgb = torch.cat([img1, img2, (img1 + img2) / 2], dim=0)
        gray = torch.cat([img1, img1, img1], dim=0)
        prompt_depth = torch.cat([img1, img2], dim=0)

        target_depth = load_depth(depth_path, to_tensor=False)
        target_depth = target_crop(target_depth, frame)
        target_depth = torch.from_numpy(target_depth)
        if self.disparity:
            target_depth = 1.0 / (target_depth + 1e-6)
        if self.normalize:
            target_depth = (target_depth - target_depth.min()) / (target_depth.max() - target_depth.min())
        return rgb, gray, prompt_depth, target_depth.unsqueeze(0)


class RealDataset(Dataset):
    def __init__(self, txt_path, multiple_of=14, aspect_ratio=None, target_size=None):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        self.src = txt_path
        self.pairs = [line.strip().split(',') for line in lines]
        self.multiple_of = multiple_of
        self.aspect_ratio = aspect_ratio
        self.target_size = target_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path = self.pairs[idx][0].strip()
        img2_path = self.pairs[idx][1].strip()

        img1 = read_img(img1_path)
        img2 = read_img(img2_path)

        frame = process_img(img1.shape, self.multiple_of, self.aspect_ratio, self.target_size)
        img1, img2 = target_crop(img1, frame), target_crop(img2, frame)

        img1 = torch.from_numpy(img1).unsqueeze(0)
        img2 = torch.from_numpy(img2).unsqueeze(0)

        rgb = torch.cat([img1, img2, (img1 + img2) / 2], dim=0)
        gray = torch.cat([img1, img1, img1], dim=0)
        prompt_depth = torch.cat([img1, img2], dim=0)

        return rgb, gray, prompt_depth, self.pairs[idx][0].strip().split('/')[-1].rsplit('_', 1)[0]  # Return filename without extension
