import numpy as np
import imageio
import torch
import os
import matplotlib.pyplot as plt
import cv2, h5py
from promptda.utils.logger import Log
from promptda.utils.depth_utils import visualize_depth
from promptda.utils.parallel_utils import async_call



# DEVICE = 'cuda' if torch.cuda.is_available(
# ) else 'mps' if torch.backends.mps.is_available() else 'cpu'


def to_tensor_func(arr):
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_numpy_func(tensor):
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def ensure_multiple_of(x, multiple_of=14):
    return int(x // multiple_of * multiple_of)


def load_image(image_path, to_tensor=True, max_size=1008, multiple_of=14):
    '''
    Load image from path and convert to tensor
    max_size // 14 = 0
    '''
    image = np.asarray(imageio.imread(image_path)).astype(np.float32)
    image = image / 255.

    max_size = max_size // multiple_of * multiple_of
    if max(image.shape) > max_size:
        h, w = image.shape[:2]
        scale = max_size / max(h, w)
        tar_h = ensure_multiple_of(h * scale)
        tar_w = ensure_multiple_of(w * scale)
        image = cv2.resize(image, (tar_w, tar_h), interpolation=cv2.INTER_AREA)
    if to_tensor:
        return to_tensor_func(image)
    return image


def read_pfm(file):
    with open(file, 'rb') as f:
        header = f.readline().rstrip()
        color = header == b'PF'
        if not color and header != b'Pf':
            raise Exception('Not a PFM file.')

        dim_line = f.readline().decode('utf-8')
        while dim_line.startswith('#'):  # skip comments
            dim_line = f.readline().decode('utf-8')
        width, height = map(int, re.findall(r'\d+', dim_line))

        scale = float(f.readline().rstrip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        return np.flipud(np.reshape(data, shape))

def load_depth(depth_path, to_tensor=True):
    '''
    Load depth from path and convert to tensor
    '''
    if depth_path.endswith('.png') or depth_path.endswith('.jpg'):
        # depth = np.asarray(imageio.imread(depth_path)).astype(np.float32)
        # depth = depth / 1000.
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth.ndim == 3 and depth.shape[2] == 3:
            depth = np.mean(depth, axis=2)
        depth = depth / 255.0  # Normalize to [0, 1]
    elif depth_path.endswith('.npz'):
        depth = np.load(depth_path)['depth']
    elif depth_path.endswith('.npy'):
        depth = np.load(depth_path)
    elif depth_path.endswith('.hdf5'):
        with h5py.File(depth_path, 'r') as f:
            depth = f['depth'][()]
        depth = depth.astype(np.float32)
    elif depth_path.endswith('.pfm'):
        depth = read_pfm(depth_path)
        if depth.ndim == 3 and depth.shape[2] == 3:
            depth = np.mean(depth, axis=2)
        depth = depth.astype(np.float32)
    else:
        raise ValueError(f"Unsupported depth format: {depth_path}")
    depth = depth.astype(np.float32)
    if to_tensor:
        return to_tensor_func(depth)
    return depth

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbar
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

def render_colorbar(height, depth_min, depth_max, cmap='Spectral', width_px=120):
    """
    Renders a vertical colorbar with labeled ticks using the specified colormap and depth range.

    Args:
        height (int): Desired height of the colorbar in pixels.
        depth_min (float): Minimum depth value.
        depth_max (float): Maximum depth value.
        cmap (str): Colormap name (must match the one used in visualization).
        width_px (int): Width of the colorbar in pixels.

    Returns:
        np.ndarray: Colorbar image (H, W, 3) in uint8 format.
    """
    dpi = 100
    fig_height = height / dpi
    fig_width = width_px / dpi

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.subplots_adjust(left=0.25, right=0.75, top=1.0, bottom=0.0)

    norm = Normalize(vmin=depth_min, vmax=depth_max)
    cmap_obj = matplotlib.colormaps[cmap]
    cb = ColorbarBase(ax, cmap=cmap_obj, norm=norm, orientation='vertical')
    cb.set_label("Depth", fontsize=10)
    cb.ax.tick_params(labelsize=20)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    img = buf[:, :, :3]  # Drop alpha
    plt.close(fig)

    return img




@async_call
def save_depth(depth,
               gt_depth=None,
               image=None,
               output_path='results',
               name='',
               in_one_plot=False,
               min_depth=None,
               max_depth=None,):

    os.makedirs(output_path, exist_ok=True)
    if isinstance(depth, torch.Tensor):
        depth = to_numpy_func(depth)
    vis_depth = visualize_depth(depth)
    depth_path = os.path.join(output_path, f'{name}_depth.jpg')
    cv2.imwrite(depth_path, cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR))
    Log.info(f"Saved predicted depth to {depth_path}", tag="save_depth")
    
    # Save RGB image
    if image is not None:
        if isinstance(image, torch.Tensor):
            image = to_numpy_func(image)
        img_path = os.path.join(output_path, f'{name}_image.jpg')
        img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(img_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        Log.info(f"Saved input image to {img_path}", tag="save_depth")

    # Save gt_depth visualization
    if gt_depth is not None:
        if isinstance(gt_depth, torch.Tensor):
            gt_depth = to_numpy_func(gt_depth)
        gt_depth_path = os.path.join(output_path, f'{name}_gt_depth.jpg')
        vis_gt = visualize_depth(gt_depth)
        cv2.imwrite(gt_depth_path, cv2.cvtColor(vis_gt, cv2.COLOR_RGB2BGR))
        Log.info(f"Saved ground truth depth to {gt_depth_path}", tag="save_depth")

    if in_one_plot:
        if min_depth is None:
            min_depth = np.min(gt_depth)
        if max_depth is None:
            max_depth = np.max(gt_depth)
        vis_depth = visualize_depth(depth, depth_min=min_depth, depth_max=max_depth)

        if gt_depth is not None and image is not None:
            vis_gt = visualize_depth(gt_depth, depth_min=min_depth, depth_max=max_depth)
            combined = np.concatenate([img_uint8, vis_gt, vis_depth], axis=1)  # horizontal
        elif image is not None:
            combined = np.concatenate([img_uint8, vis_depth], axis=1)
        else:
            combined = vis_depth
        # Add colorbar
        colorbar = render_colorbar(combined.shape[0], min_depth, max_depth, width_px=200)
        colorbar = cv2.resize(colorbar, (200, combined.shape[0]))  # resize to match height
        vis_with_cb = np.concatenate([combined, colorbar], axis=1)

        out_path = os.path.join(output_path, f'{name}_concat_cb.jpg')
        cv2.imwrite(out_path, cv2.cvtColor(vis_with_cb, cv2.COLOR_RGB2BGR))
        Log.info(f"Saved concatenated image with colorbar to {out_path}", tag="save_depth")



