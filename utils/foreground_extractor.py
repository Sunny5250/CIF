import numpy as np
from scipy.ndimage import convolve


def get_neighbor_mean(img, p):
    n_neighbors = np.sum(img[p[0] - 1: p[0] + 2, p[1] - 1: p[1] + 2] > 0)
    if n_neighbors == 0:
        return None
    nb_mean = np.sum(img[p[0] - 1: p[0] + 2, p[1] - 1: p[1] + 2], axis=(0, 1)) / n_neighbors

    return nb_mean


def fill_gaps(img):
    mask = (img > 0).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    neighbor_count = convolve(mask, kernel, mode='constant', cval=0.0)
    masked_img = img * mask
    neighbor_sum = convolve(masked_img, kernel, mode='constant', cval=0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        neighbor_mean = neighbor_sum / neighbor_count
        neighbor_mean[np.isnan(neighbor_mean)] = 0
    filled_img = np.copy(img)
    filled_img[(img == 0) & (neighbor_count > 0)] = neighbor_mean[(img == 0) & (neighbor_count > 0)]

    return filled_img


def get_corner_points(img):
    upper_left = np.sum(img[:2, :2]) / np.sum(img[:2, :2] > 0)
    upper_right = np.sum(img[-2:, :2]) / np.sum(img[-2:, :2] > 0)
    lower_left = np.sum(img[:2, -2:]) / np.sum(img[:2, -2:] > 0)
    lower_right = np.sum(img[-2:, -2:]) / np.sum(img[-2:, -2:] > 0)

    return upper_left, upper_right, lower_left, lower_right


def remove_background(img, bg_thresh):
    w, h = img.shape[:2]
    upper_left, upper_right, lower_left, lower_right = get_corner_points(img)
    x_top = np.linspace(upper_left, upper_right, w)
    x_bottom = np.linspace(lower_left, lower_right, w)
    top_ratio = np.linspace(1, 0, h)[None]
    bottom_ratio = np.linspace(0, 1, h)[None]
    background = x_top[:, None] * top_ratio + x_bottom[:, None] * bottom_ratio
    foreground = np.zeros_like(img)
    foreground[np.abs(background - img) > bg_thresh] = 1
    foreground[img == 0] = 0
    
    return foreground
