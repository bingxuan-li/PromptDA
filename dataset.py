import os, cv2, torch, numpy as np
import random
from torch.utils.data import Dataset
from promptda.utils.io_wrapper import load_depth
from scipy.ndimage import gaussian_filter

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
    # assert 0 <= start_y <= h - sy and 0 <= start_x <= w - sx, "Invalid crop origin or size"
    end_x = min(start_x + sx, w)
    end_y = min(start_y + sy, h)
    return img[start_y:end_y, start_x:end_x]

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

class Augmentation():
    def __init__(self, augment_prob, max_p_noise=0.0, max_g_noise=0.0, max_gs_blur= 0.0,
                  max_global_imbalance=0.0, max_local_imbalance=0.0, max_gs_kernels=0):
        self.ph_num = 1000
        self.augment_prob = augment_prob
        self.max_p_noise = max_p_noise
        self.max_g_noise = max_g_noise
        self.max_gs_blur = max_gs_blur
        self.max_global_imbalance = max_global_imbalance
        self.max_local_imbalance = max_local_imbalance
        self.max_gs_kernels = max_gs_kernels
        self.alpha = 1
        self.beta = 3

    def augment(self, img, gray=False):
        if random.random() > self.augment_prob:
            return img

        H, W = img.shape[-2:]
        if self.max_gs_blur and not gray> 0:
            max_gs_blur = self.max_gs_blur
            gs_blur = np.random.uniform(0, max_gs_blur)
            img = torch.from_numpy(gaussian_filter(img.numpy(), sigma=gs_blur)).to(img.dtype)
        
        # Add global imbalance to the output images
        if self.max_global_imbalance > 0:
            global_imbalance  = np.random.beta(self.alpha, self.beta) * self.max_global_imbalance
            if np.random.random() < 0.5:
                global_imbalance = -global_imbalance
            img = img * (1 + global_imbalance)
        # Local imbalance
        if self.max_local_imbalance > 0 and self.max_gs_kernels > 0:
            mask = torch.zeros_like(img)
            y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            x_grid = x_grid.float()
            y_grid = y_grid.float()
            n_kernels = random.randint(1, self.max_gs_kernels)
            for _ in range(n_kernels):
                # Random center
                cx = random.uniform(0, W - 1)
                cy = random.uniform(0, H - 1)
                # Random anisotropic Gaussian parameters
                log_aspect_ratio = random.uniform(-1, 1)  # log(r), r in [0.1, 10]
                aspect_ratio = np.exp(log_aspect_ratio)  # r
                log_sigma_y = random.uniform(np.log(1), np.log(256))
                sigma_y = np.exp(log_sigma_y)
                sigma_x = sigma_y * aspect_ratio  # sigma_x = r * sigma_y
                theta = random.uniform(0, np.pi)  # orientation
                # Rotation matrix
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                # Apply rotation
                x_shift = x_grid - cx
                y_shift = y_grid - cy
                x_rot = cos_t * x_shift + sin_t * y_shift
                y_rot = -sin_t * x_shift + cos_t * y_shift
                gauss = torch.exp(-((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2) / 2)
                if np.random.random() < 0.5:
                    gauss = -gauss  # Randomly flip the sign
                mask += gauss
            if mask.abs().max() > 1e-6:
                mask = mask / mask.abs().max()  # Normalize the mask
            local_imbalance  = np.random.beta(self.alpha, self.beta) * self.max_local_imbalance
            mask = mask * local_imbalance
            img = img * (1 + mask)
        img = torch.clamp(img, min=0)  # Ensure non-negative value
        # Add shot noise to the output images
        if self.max_p_noise > 0:
            min_noise = 1e-3 * self.max_p_noise  # avoid divide-by-zero or huge `times`
            p_noise = np.random.beta(self.alpha, self.beta)*(self.max_p_noise-min_noise) + min_noise
            times = self.ph_num / p_noise
            img = torch.poisson(img * times) / times
        # Add gaussian noise to the output images
        if self.max_g_noise > 0:
            g_noise = np.random.beta(self.alpha, self.beta) * self.max_g_noise
            img = img + torch.randn_like(img) * g_noise
        # ensure non-negative value
        img = torch.clamp(img, min=0, max=1)
        return img

class SimulatedDataset(Dataset):
    def __init__(self, txt_path, multiple_of=14, aspect_ratio=None, random_crop=False, 
                 target_size=None, prompt_channels=2, confuse_lens_z=None):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        self.src = txt_path
        self.pairs = [line.strip().split(',') for line in lines]
        self.random_crop = random_crop
        self.multiple_of = multiple_of
        self.aspect_ratio = aspect_ratio
        self.target_size = target_size
        self.prompt_channels = prompt_channels
        self.augmentation = None
        self.confuse_lens_z = confuse_lens_z

    def set_augmentation(self, augmentation):
        """
        Set the augmentation object for this dataset.
        """
        if isinstance(augmentation, Augmentation):
            self.augmentation = augmentation
        elif augmentation is None:
            self.augmentation = None
        else:
            raise ValueError("Augmentation must be an instance of Augmentation class.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path = self.pairs[idx][0].strip()
        img2_path = self.pairs[idx][1].strip()
        if self.confuse_lens_z is not None and len(self.confuse_lens_z) >= 1:
            rand_lens_z = random.choice(self.confuse_lens_z)
            img1_path = img1_path.replace('z180.0', rand_lens_z)
            img2_path = img2_path.replace('z180.0', rand_lens_z)

        depth_path = self.pairs[idx][2].strip()
        gray_path = self.pairs[idx][3].strip() if len(self.pairs[idx]) > 3 else img1_path

        img1, img2, gray = read_img(img1_path), read_img(img2_path), read_img(gray_path)
        frame = process_img(img1.shape, self.multiple_of, self.aspect_ratio, self.target_size, self.random_crop)
        img1, img2, gray = target_crop(img1, frame), target_crop(img2, frame), target_crop(gray, frame)
        img1, img2 = torch.from_numpy(img1), torch.from_numpy(img2)

        
        # Always apply Gaussian blur to gray image
        gs_blur = np.random.uniform(1, 5)
        gray = torch.from_numpy(gaussian_filter(gray, sigma=gs_blur)).to(dtype=img1.dtype)

        # Ensure img1 and gray is augmented together
        if self.augmentation is not None:
            img1 = self.augmentation.augment(img1)
            img2 = self.augmentation.augment(img2)
            gray = self.augmentation.augment(gray, gray=True)

        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        gray = gray.unsqueeze(0)

        rgb = torch.cat([img1, img2, (img1 + img2) / 2], dim=0)
        true_gray = torch.cat([gray, gray, gray], dim=0)
        sim_gray = torch.cat([img1, img1, img1], dim=0)

        if self.prompt_channels == 2:
            prompt_depth = torch.cat([img1, img2], dim=0)
        elif self.prompt_channels == 3:
            prompt_depth = torch.cat([img1, img2, (img1-img2)], dim=0)

        target_depth = load_depth(depth_path, to_tensor=False)
        target_depth = target_crop(target_depth, frame)
        target_depth = torch.from_numpy(target_depth)
        return rgb, sim_gray, true_gray, prompt_depth, target_depth.unsqueeze(0)


class RealDataset(Dataset):
    def __init__(self, txt_path, multiple_of=14, downsample=1.0, aspect_ratio=None, target_size=None, prompt_channels=2):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        self.src = txt_path
        self.pairs = [line.strip().split(',') for line in lines]
        self.multiple_of = multiple_of
        self.downsample = downsample
        self.aspect_ratio = aspect_ratio
        self.target_size = target_size
        self.prompt_channels = prompt_channels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path = self.pairs[idx][0].strip()
        img2_path = self.pairs[idx][1].strip()

        img1 = read_img(img1_path)
        img2 = read_img(img2_path)
        if self.downsample != 1.0:
            img1 = cv2.resize(img1, (0, 0), fx=self.downsample, fy=self.downsample)
            img2 = cv2.resize(img2, (0, 0), fx=self.downsample, fy=self.downsample)

        frame = process_img(img1.shape, self.multiple_of, self.aspect_ratio, self.target_size)
        img1, img2 = target_crop(img1, frame), target_crop(img2, frame)

        img1 = torch.from_numpy(img1).unsqueeze(0)
        img2 = torch.from_numpy(img2).unsqueeze(0)

        rgb = torch.cat([img1, img2, (img1 + img2) / 2], dim=0)
        gray = torch.cat([img1, img1, img1], dim=0)
        
        if self.prompt_channels == 2:
            prompt_depth = torch.cat([img1, img2], dim=0)
        elif self.prompt_channels == 3:
            prompt_depth = torch.cat([img1, img2, (img1-img2)], dim=0)

        return rgb, gray, prompt_depth, self.pairs[idx][0].strip().split('/')[-1].rsplit('.', 1)[0]  # Return filename without extension
