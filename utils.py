import numpy as np


# =========================
# OVERLAY MASK
# =========================
def overlay_mask(mask, ax, rgba_color=(0, 1, 0, 0.5)):
    if len(mask.shape) == 3:
        mask = mask[0]

    h, w = mask.shape

    mask_image = np.zeros((h, w, 4), dtype=np.float32)

    mask_image[mask > 0] = rgba_color

    ax.imshow(mask_image)


# =========================
# GET BBOX
# =========================
def get_bbox_from_mask(mask):
    if len(mask.shape) == 3:
        mask = mask[0]

    pos = np.where(mask)

    if len(pos[0]) == 0:
        return None

    x1 = int(np.min(pos[1]))
    y1 = int(np.min(pos[0]))
    x2 = int(np.max(pos[1]))
    y2 = int(np.max(pos[0]))

    return [x1, y1, x2 - x1, y2 - y1]