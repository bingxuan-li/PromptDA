import os, cv2
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

def visualize_depth(depth: np.ndarray, 
                    depth_min=None, 
                    depth_max=None, 
                    percentile=2, 
                    ret_minmax=False,
                    cmap='Spectral'):
    if depth_min is None: depth_min = np.percentile(depth, percentile)
    if depth_max is None: depth_max = np.percentile(depth, 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    else:
        return img_colored_np

def get_visualization(depth, image, min_depth=0.2, max_depth=1.5):
    vis_depth = visualize_depth(depth, depth_min=min_depth, depth_max=max_depth)
    img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    vis_depth = visualize_depth(depth, depth_min=0.2, depth_max=1.5)
    combined = np.concatenate([img_uint8, vis_depth], axis=1)
    # Add colorbar
    colorbar = render_colorbar(combined.shape[0], min_depth, max_depth, width_px=200)
    colorbar = cv2.resize(colorbar, (200, combined.shape[0]))  # resize to match height
    vis_with_cb = np.concatenate([combined, colorbar], axis=1)
    return vis_with_cb

def save_depth(depth, image, output_path, name='depth', min_depth=0.2, max_depth=1.5):
    os.makedirs(output_path, exist_ok=True)
    vis_with_cb = get_visualization(depth, image, min_depth=min_depth, max_depth=max_depth)
    out_path = os.path.join(output_path, f'{name}_concat_cb.jpg')
    cv2.imwrite(out_path, cv2.cvtColor(vis_with_cb, cv2.COLOR_RGB2BGR))