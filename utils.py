import numpy as np
import matplotlib.pyplot as plt

def overlay_mask(mask, ax, color=(0, 1, 0, 0.4)):
    """Applies a semi-transparent color mask to a matplotlib axis."""
    # Ensure mask is 2D
    if len(mask.shape) == 3:
        mask = mask[0]
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)

def get_bbox_from_mask(mask):
    """Calculates [x, y, w, h] from a binary mask."""
    if len(mask.shape) == 3:
        mask = mask[0]
    pos = np.where(mask)
    if len(pos[0]) == 0: return None
    xmin, xmax = np.min(pos[1]), np.max(pos[1])
    ymin, ymax = np.min(pos[0]), np.max(pos[0])
    return [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
