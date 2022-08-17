import cv2
import numpy as np
import skimage.transform
from collections import OrderedDict

import torch

from lib.tracing import get_epi_end_paths_from_heatmap_and_landmarks as get_paths
from lib.landmarks import load_landmark_model, perform_cmr_landmark_detection
from lib.bullseye import compute_bullseye_sector_mask_for_slice
from lib.inference import center_crop, pad_if_needed, get_original_npy_path_from_exported_npz_path

import wandb


def vis_pose(dataloader, model, epoch, cfg):
    def resize_for_vis(img, vis_res, is_mask):
        return skimage.transform.resize(img, vis_res, order=0 if is_mask else 1)

    def stick_posemap_on_frame(frame, posemap):
        posemap = posemap.transpose((1, 2, 0))
        if posemap.shape[:2] != frame.shape[:2]:
            posemap = resize_for_vis(posemap, frame.shape[:2], is_mask=True)
        img = np.dstack((frame, frame, frame))
        img[:, :, 0] = img[:, :, 0] + posemap[:, :, 0]
        img[:, :, 1] = img[:, :, 1] + posemap[:, :, 1]
        img = np.clip(img, 0, 1)
        return img

    if epoch % cfg['output']['vis_every']:
        return

    vis_n = cfg['output']['vis_n']
    vis_res = cfg['output']['vis_res']
    device = cfg['training']['device']
    landmark_model_path = cfg['export']['landmark_model_path']
    mask_classes = cfg['output']['mask_classes']

    batch_x, batch_y_true, batch_filepaths = next(iter(dataloader))
    with torch.no_grad():
        batch_y_pred = model(batch_x.to(device))
        if type(batch_y_pred) == OrderedDict:
            batch_y_pred = batch_y_pred['out']

    images = []
    masks = []

    landmark_model = load_landmark_model(landmark_model_path)
    for i, (frame, _y_true, y_pred, filepath) in enumerate(zip(batch_x, batch_y_true, batch_y_pred, batch_filepaths)):

        pred_np = y_pred.cpu().numpy()

        frame_t1_pre = resize_for_vis(frame[0], vis_res, False)
        frame_t1_post = resize_for_vis(frame[1], vis_res, False)
        frame_t2 = resize_for_vis(frame[2], vis_res, False)

        # heatmaps
        img_t1pre_pred = stick_posemap_on_frame(frame_t1_pre, pred_np)
        img_t1post_pred = stick_posemap_on_frame(frame_t1_post, pred_np)
        img_t2_pred = stick_posemap_on_frame(frame_t2, pred_np)
        img_plain = cv2.cvtColor(np.concatenate((frame_t1_pre, frame_t1_post, frame_t2), axis=1), cv2.COLOR_GRAY2RGB)
        img_heatmaps = np.concatenate((img_t1pre_pred, img_t1post_pred, img_t2_pred), axis=1)
        img_withoutmask = np.concatenate((img_plain, img_heatmaps), axis=0)

        # landmark detection
        orig_npy_path = get_original_npy_path_from_exported_npz_path(filepath, cfg['export']['dicom_path_trainval'])
        t1w, t2w, pd, t1, t2 = np.transpose(np.load(orig_npy_path), (2, 0, 1))
        t2w_landmark, _top_left_landmark = center_crop(pad_if_needed(t2w, min_height=256, min_width=256),
                                                       crop_height=256, crop_width=256)
        landmark_points, landmark_probs = perform_cmr_landmark_detection(t2w_landmark, model=landmark_model)

        # rv masks
        rvi1_xy, rvi2_xy, lv_xy = landmark_points
        rvimid_xy = 0.5 * (rvi1_xy + rvi2_xy)
        rv_xy = lv_xy + 2 * (rvimid_xy - lv_xy)
        mask_rvi1 = np.zeros_like(t2w_landmark)
        mask_rvi1[int(round(rvi1_xy[1])), int(round(rvi1_xy[0]))] = 1
        mask_rvmid = np.zeros_like(t2w_landmark)
        mask_rvmid[int(round(rv_xy[1])), int(round(rv_xy[0]))] = 1

        # Lv ridge tracing using landmarks
        if np.all(landmark_points == -1):
            print(f"Was unable to find landmarks on sample {i}")
            (xs_epi, ys_epi), (xs_end, ys_end) = [[], []]
        else:
            (xs_epi, ys_epi), (xs_end, ys_end) = get_paths(pred_np, landmark_points)

        # ridges to masks
        mask_lvcav, mask_lvwall = np.zeros_like(t2w_landmark, dtype=np.uint8), np.zeros_like(t2w_landmark,
                                                                                             dtype=np.uint8)
        points_end = np.array([list(zip(xs_end, ys_end))])
        points_epi = np.array([list(zip(xs_epi, ys_epi))])
        color = np.uint8(np.ones(3) * 1).tolist()
        cv2.fillPoly(mask_lvcav, points_end, color)
        cv2.fillPoly(mask_lvwall, points_epi, color)

        # sectors
        sectors, sectors_32 = compute_bullseye_sector_mask_for_slice(mask_lvcav, mask_lvwall, mask_rvmid, mask_rvi1, 6)
        sectors = resize_for_vis(sectors, vis_res, is_mask=True)
        sector_row = np.concatenate((sectors, sectors, sectors), axis=1)
        img_mask = np.concatenate((sector_row, sector_row), axis=0)

        images.append(img_withoutmask)
        masks.append(img_mask)

        if i >= vis_n - 1:
            break

    # WandB
    wandb_images = []
    for image, mask in zip(images, masks):
        wandb_img = wandb.Image(image, masks={
            "prediction": {
                "mask_data": mask,
                "class_labels": mask_classes,
            }
        })
        wandb_images.append(wandb_img)
    wandb.log({"epoch": epoch, "images": wandb_images})

    return images, masks


def iou(sectors1, sectors2, group_sectors=True):
    if not group_sectors:
        raise NotImplementedError()
    sectors1 = sectors1.astype(np.bool)
    sectors2 = sectors2.astype(np.bool)
    intersect = sectors1 & sectors2
    union = sectors1 | sectors2
    return intersect.sum() / union.sum()


def dice(sectors1, sectors2, group_sectors=True):
    if not group_sectors:
        raise NotImplementedError()
    sectors1 = sectors1.astype(np.bool)
    sectors2 = sectors2.astype(np.bool)
    intersect = sectors1 & sectors2
    d = (intersect.sum() * 2) / (sectors1.sum() + sectors2.sum())
    return d