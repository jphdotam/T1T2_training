import os
import cv2
import numpy as np
import skimage.transform

import torch

from lib.windows import normalize_data


def pad_if_needed(img, min_height, min_width):
    input_height, input_width = img.shape[:2]
    new_shape = list(img.shape)
    new_shape[0] = max(input_height, min_height)
    new_shape[1] = max(input_width, min_width)
    row_from, col_from = 0, 0
    if input_height < min_height:
        row_from = (min_height - input_height) // 2
    if input_width < min_width:
        col_from = (min_width - input_width) // 2
    out = np.zeros(new_shape, dtype=img.dtype)
    out[row_from:row_from+input_height, col_from:col_from+input_width] = img
    return out


def center_crop(img, crop_height, crop_width, centre=None):
    """Either crops by the center of the image, or around a supplied point.
    Does not pad; if the supplied centre is towards the egde of the image, the padded
    area is shifted so crops start at 0 and only go up to the max row/col
    Returns both the new crop, and the top-left coords as a row,col tuple"""
    input_height, input_width = img.shape[:2]
    if centre is None:
        row_from = (input_height - crop_height)//2
        col_from = (input_width - crop_width)//2
    else:
        row_centre, col_centre = centre
        row_from = max(row_centre - (crop_height//2), 0)
        if (row_from + crop_height) > input_height:
            row_from -= (row_from + crop_height - input_height)
        col_from = max(col_centre - (crop_width//2), 0)
        if (col_from + crop_width) > input_width:
            col_from -= (col_from + crop_width - input_width)
    return img[row_from:row_from+crop_height, col_from:col_from+crop_width], (row_from, col_from)


def get_original_npy_path_from_exported_npz_path(npz_path, peter_dir):
    date, study, file, _end = os.path.basename(npz_path).split('__')
    peter_path = os.path.join(peter_dir, date, study, file)
    return peter_path


def get_normalized_channel_stack(t1, t2, t1w, t2w, pd, data_stack_format=None):
    t1_pre = normalize_data(t1, window_centre=1300.0, window_width=1300.0)
    t1_post = normalize_data(t1, window_centre=500.0, window_width=1000.0)
    t2 = normalize_data(t2, window_centre=60.0, window_width=120.0)
    t1w = t1w - t1w.min()
    t1w /= t1w.max()
    t2w = t2w - t2w.min()
    t2w /= t2w.max()
    pd = pd - pd.min()
    pd /= pd.max()
    t1_pre = (t1_pre*255).astype(np.uint8)
    t1_post = (t1_post*255).astype(np.uint8)
    t2 = (t2*255).astype(np.uint8)
    t1w = (t1w*255).astype(np.uint8)
    t2w = (t2w*255).astype(np.uint8)
    pd = (pd*255).astype(np.uint8)

    if data_stack_format is None:
        t1_t2 = None
    elif data_stack_format == 'all':
        t1_t2 = np.dstack((t1w, t2w, pd, t1_pre, t1_post, t2))
    elif data_stack_format == 't1':
        t1_t2 = np.dstack((t1_pre, t1_post))
    elif data_stack_format == 't2':
        t1_t2 = np.expand_dims(t2, axis=-1)
    else:
        raise ValueError()

    return t1_pre, t1_post, t2, t1w, t2w, pd, t1_t2


def prep_normalized_stack_for_inference(t1_t2, fov, as_tensor, tensor_device=None):
    t1_t2_crop, _top_left = center_crop(pad_if_needed(t1_t2, min_height=fov, min_width=fov), crop_height=fov, crop_width=fov)
    t1_t2_double = skimage.transform.rescale(t1_t2_crop, 2, order=3, multichannel=True)
    t1_t2_in = t1_t2_double.transpose((2, 0, 1))
    img_batch = np.expand_dims(t1_t2_in, 0).astype(np.float32)
    if as_tensor:
        img_batch = torch.from_numpy(img_batch).float().to(tensor_device)
    return img_batch


def tta(model, x):
    flips = [[-1], [-2], [-2, -1]]
    pred_batch = model(x)
    for f in flips:
        xf = torch.flip(x, f)
        p_b = model(xf)
        p_b = torch.flip(p_b, f)
        pred_batch += p_b

    pred_batch = pred_batch/len(flips)

    return pred_batch


def paths_to_ridge_polygons(xs_epi, ys_epi, xs_end, ys_end, fov):
    mask_lvcav = np.zeros((fov, fov), dtype=np.uint8)
    mask_lvwall = np.zeros_like(mask_lvcav)
    points_end = np.array([list(zip(xs_end, ys_end))])
    points_epi = np.array([list(zip(xs_epi, ys_epi))])
    color = np.uint8(np.ones(3) * 1).tolist()
    cv2.fillPoly(mask_lvcav, points_end, color)
    cv2.fillPoly(mask_lvwall, points_epi, color)
    return mask_lvcav, mask_lvwall
