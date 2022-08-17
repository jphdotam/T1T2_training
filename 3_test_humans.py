import os
import cv2
import json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from collections import defaultdict

from lib.landmarks import load_landmark_model, perform_cmr_landmark_detection
from lib.cfg import load_config
from lib.transforms import get_segmentation_transforms
from lib.inference import center_crop, pad_if_needed
from lib.vis import compute_bullseye_sector_mask_for_slice

CONFIG = "./experiments/001.yaml"
cfg, model_dir = load_config(CONFIG)

TEST_DICOM_DIR = cfg['export']['dicom_path_test']
LANDMARK_MODELPATH = cfg['export']['landmark_model_path']
LABEL_ROOT_DIR = cfg['export']['label_path_test']
FOV = 256
WRITE_PNGS = True

# Load config
sequences = cfg['export']['source_channels']
label_classes = cfg['export']['label_classes']
gaussian_sigma = cfg['export']['gaussian_sigma']
n_channels_keep_img = len(sequences)  # May have exported more channels to make PNG
n_channels_keep_lab = len(label_classes)

# landmark model
landmark_model = load_landmark_model(LANDMARK_MODELPATH)
_, transforms_test = get_segmentation_transforms(cfg)

label_dirs = [f for f in glob(os.path.join(LABEL_ROOT_DIR, "*")) if os.path.isdir(f)]

for label_dir in label_dirs:

    out_dir = os.path.join("./data", f"predictions")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output_dict = defaultdict(dict)
    label_paths_all = sorted(glob(os.path.join(label_dir, "**", "*.pickle"), recursive=True))

    label_paths = [p for p in label_paths_all if not os.path.exists(p+".invalid")]
    print(f"{len(label_paths)} - {len(label_paths_all) - len(label_paths)} excluded as labelled but also marked invalid")


    for i, label_path in tqdm(enumerate(label_paths), total=len(label_paths)):
        # load mask
        try:
            npy_label = np.load(label_path, allow_pickle=True)
        except OSError as e:
            print(f"Skipping file {label_path} as {e}")
            continue
        w, h = npy_label['dims'][:2]
        mask_lvwall, mask_lvcav = np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
        color = np.uint8(np.ones(3) * 1).tolist()
        points_end = [[x, h - y] for x, y in npy_label['endo']]
        points_epi = [[x, h - y] for x, y in npy_label['epi']]
        points_end = np.rint([points_end]).astype(np.int32)
        points_epi = np.rint([points_epi]).astype(np.int32)
        if points_end.size == 0 or points_epi.size == 0:
            print(f"Skipping {label_path} as missing points for >= 1 surface")
            continue
        cv2.fillPoly(mask_lvcav, points_end, color)
        cv2.fillPoly(mask_lvwall, points_epi, color)

        # crop mask
        mask_lvcav, _top_left = center_crop(pad_if_needed(mask_lvcav, min_height=FOV, min_width=FOV),
                                            crop_height=FOV, crop_width=FOV)
        mask_lvwall, _ = center_crop(pad_if_needed(mask_lvwall, min_height=FOV, min_width=FOV),
                                     crop_height=FOV, crop_width=FOV)

        # Find RV points so we can do the segmentation
        label_filename, seq_dir, date_dir = label_path.split(os.sep)[::-1][:3]
        image_path = os.path.join(TEST_DICOM_DIR, date_dir, seq_dir, label_filename.split('_HUMAN',1)[0])
        t1w, t2w, pd, t1, t2 = np.load(image_path, allow_pickle=True).transpose((2, 0, 1))
        t2w, _ = center_crop(pad_if_needed(t2w, min_height=FOV, min_width=FOV),
                             crop_height=FOV, crop_width=FOV)
        landmark_points, _landmark_probs = perform_cmr_landmark_detection(t2w, model=landmark_model)

        if np.all(landmark_points == -1):
            print(f"Skipping {label_path} - unable to identify landmarks")
            continue

        vector_ant = landmark_points[[0, 2]]
        vector_post = landmark_points[[1, 2]]

        # rv masks
        try:
            rvi1_xy, rvi2_xy, lv_xy = landmark_points
            rvimid_xy = 0.5 * (rvi1_xy + rvi2_xy)
            rv_xy = lv_xy + 2 * (rvimid_xy - lv_xy)
            mask_rvi1 = np.zeros_like(t2w)
            mask_rvi1[int(round(rvi1_xy[1])), int(round(rvi1_xy[0]))] = 1
            mask_rvmid = np.zeros_like(t2w)
            mask_rvmid[int(round(rv_xy[1])), int(round(rv_xy[0]))] = 1
        except IndexError as e:
            print(f"WARNING: Skipping {label_path} as {e}")
            continue

        # sectors
        sectors, sectors_32 = compute_bullseye_sector_mask_for_slice(mask_lvcav, mask_lvwall, mask_rvmid, mask_rvi1, 6)
        sector_row = np.concatenate((sectors, sectors, sectors), axis=1)
        img_mask = np.concatenate((sector_row, sector_row), axis=0)

        # window maps
        t1, _ = center_crop(pad_if_needed(t1, min_height=FOV, min_width=FOV), crop_height=FOV, crop_width=FOV)
        t2, _ = center_crop(pad_if_needed(t2, min_height=FOV, min_width=FOV), crop_height=FOV, crop_width=FOV)

        # wc, ww = SEQUENCE_WINDOWS['T2'][0]['wc'], SEQUENCE_WINDOWS['T2'][0]['ww']
        # t2 = window_numpy(t2, wc, ww)
        #
        # plt.imshow(t2, cmap=default_cmap)
        # plt.show()
        #
        # plt.imshow(sectors)
        # plt.show()Vk

        for seq_name, seq_img in zip(('t1', 't2'), (t1, t2)):
            for segment_id in list(range(1, 6+1)):
                seg_values = seq_img[sectors == segment_id]
                study_id = f"{os.path.basename(os.path.dirname(image_path))}__{os.path.basename(image_path)}"
                output_dict[study_id][f"{seq_name}_{segment_id}"] = {
                    'count': len(seg_values),
                    'mean': np.mean(seg_values),
                    'median': np.median(seg_values)
                }
                if WRITE_PNGS:
                    png_path = os.path.join(out_dir, study_id+".png")
                    skimage.io.imsave(png_path, sectors.astype(np.uint8), check_contrast=False)

    with open(os.path.join(out_dir, "segments.json"), 'w') as f:
        json.dump(output_dict, f, indent=4)