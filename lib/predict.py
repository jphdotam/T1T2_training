import skimage.io
import numpy as np
from lib.windows import *
from lib.inference import center_crop, pad_if_needed, prep_normalized_stack_for_inference
from lib.landmarks import perform_cmr_landmark_detection
from lib.tracing import get_epi_end_paths_from_heatmap_and_landmarks as get_paths


def predict_pose(pose_session, t1, t2, t1w, t2w, pd, fov, device="cuda"):
    t1_pre = normalize_data(t1, window_centre=WC_T1_PRE, window_width=WW_T1_PRE)
    t1_post = normalize_data(t1, window_centre=WC_T1_POST, window_width=WW_T1_POST)
    t2 = normalize_data(t2, window_centre=WC_T2, window_width=WW_T2)
    t1w = t1w - t1w.min()
    t1w /= t1w.max()
    t2w = t2w - t2w.min()
    t2w /= t2w.max()
    pd = pd - pd.min()
    pd /= pd.max()
    t1_pre = (t1_pre * 255).astype(np.uint8)
    t1_post = (t1_post * 255).astype(np.uint8)
    t2 = (t2 * 255).astype(np.uint8)
    t1w = (t1w * 255).astype(np.uint8)
    t2w = (t2w * 255).astype(np.uint8)
    pd = (pd * 255).astype(np.uint8)

    t1_t2 = np.dstack((t1w, t2w, pd, t1_pre, t1_post, t2))

    img_batch = prep_normalized_stack_for_inference(t1_t2, fov, False)

    return pose_session.run([output_name], {input_name: img_batch})[0]  # Returns a list of len 1


def get_points(pose_session, landmark_model, npy, fov):
    """If failure of landmark detection, returns [[], []]"""
    input_name = pose_session.get_inputs()[0].name
    output_name = pose_session.get_outputs()[0].name

    t1w, t2w, pd, t1, t2 = np.transpose(npy, (2, 0, 1))
    t1_raw, t2_raw = t1.copy(), t2.copy()

    t2w_landmark, _top_left_landmark = center_crop(pad_if_needed(t2w, min_height=fov, min_width=fov), crop_height=fov,
                                                   crop_width=fov)
    landmark_points, landmark_probs = perform_cmr_landmark_detection(t2w_landmark, model=landmark_model)

    if np.any(landmark_points == -1):
        print(f"Skipping - unable to identify all landmarks")
        return [[], []]

    # POSE MODEL
    pred_batch = predict_pose(pose_session, t1, t2, t1w, t2w, pd, fov)

    # rv masks
    rvi1_xy, rvi2_xy, lv_xy = landmark_points
    rvimid_xy = 0.5 * (rvi1_xy + rvi2_xy)
    rv_xy = lv_xy + 2 * (rvimid_xy - lv_xy)
    mask_rvi1 = np.zeros_like(t2w)
    mask_rvi1[int(round(rvi1_xy[1])), int(round(rvi1_xy[0]))] = 1
    mask_rvmid = np.zeros_like(t2w)
    mask_rvmid[int(round(rv_xy[1])), int(round(rv_xy[0]))] = 1

    # Lv ridge tracing using landmarks
    if np.all(landmark_points == -1):
        print(f"Was unable to find landmarks on sample {i}")
        (xs_epi, ys_epi), (xs_end, ys_end) = [[], []]
    else:
        (xs_epi, ys_epi), (xs_end, ys_end) = get_paths(pred_batch[0], landmark_points)

    return (xs_epi, ys_epi), (xs_end, ys_end)
