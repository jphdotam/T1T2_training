import os
import numpy as np
import torch
import torch.functional as F
from scipy import ndimage


def load_landmark_model(path):
    return torch.jit.load(path)


def adaptive_thresh(probs, p_thresh=0.5, p_thresh_max=0.988):
    p_thresh_incr = 0.01

    RO = probs.shape[0]
    E1 = probs.shape[1]

    try:
        number_of_blobs = float("inf")
        blobs = np.zeros((RO, E1))
        while number_of_blobs > 1 and p_thresh < p_thresh_max:
            mask = (probs > torch.max(probs) * p_thresh).float()
            blobs, number_of_blobs = ndimage.label(mask)
            p_thresh += p_thresh_incr  # <-- Note this line can lead to float drift.

        if (number_of_blobs == 1):
            return mask

        if (number_of_blobs == 0):
            mask = np.zeros((RO, E1))
            print("adaptive_thresh, did not find any blobs ... ")
            return mask

        ## If we are here then we cannot isolate a singular blob as the LV.
        ## Select the largest blob as the final mask.
        biggest_blob = (0, torch.zeros(RO, E1))
        for i in range(number_of_blobs):
            one_blob = torch.tensor((blobs == i + 1).astype(int), dtype=torch.uint8)
            area = torch.sum(one_blob)
            if (area > biggest_blob[0]):
                biggest_blob = (area, one_blob)

        return biggest_blob[1]

    except Exception as e:
        print("Error happened in adaptive_thresh ...")
        print(e)
        mask = np.zeros((RO, E1))

    return mask


def cpad_2d(data, RO, E1):
    '''
    data: [dRO, dE1, N], padded it round center to [RO, E1, N]
    return value: (s_ro, s_e1), starting point of data in padded array
    '''

    try:
        dRO, dE1, N = data.shape
    except:
        data = np.expand_dims(data, axis=2)
        dRO, dE1, N = data.shape

    s_ro = int((RO - dRO) / 2)
    s_e1 = int((E1 - dE1) / 2)

    # print(data.shape, RO, E1, s_ro, s_e1)
    if (s_ro >= 0):
        data_padded = np.zeros((RO, dE1, N))
        if (dRO >= RO):
            data_padded = data[s_ro:s_ro + RO, :, :]
        else:
            data_padded[s_ro:s_ro + dRO, :, :] = data
        data = data_padded
    else:
        data_padded = np.zeros((RO, dE1, N))
        if (dRO + s_ro + s_ro > RO):
            data_padded = data[-s_ro:(dRO + s_ro - 1), :, :]
        else:
            data_padded = data[-s_ro:(dRO + s_ro), :, :]
        data = data_padded

    # print(data.shape)

    if (s_e1 >= 0):
        data_padded = np.zeros((RO, E1, N))
        if (dE1 >= E1):
            data_padded = data[:, s_e1:s_e1 + E1, :]
        else:
            data_padded[:, s_e1:s_e1 + dE1, :] = data
        data = data_padded
    else:
        data_padded = np.zeros((RO, E1, N))
        if (dE1 + s_e1 + s_e1 > E1):
            data_padded = data[:, -s_e1:(dE1 + s_e1 - 1), :]
        else:
            data_padded = data[:, -s_e1:(dE1 + s_e1), :]
        data = data_padded

    return data_padded, s_ro, s_e1


def get_landmark_from_prob(prob, thres=0.5, mode="mean", binary_mask=False):
    """
    Compute landmark from prob
    prob : [RO, E1]
    mode : mean or max
    return pt : [x, y]
    """

    pt = None

    if (binary_mask):
        ind = np.where(prob == thres)
    else:
        if (thres > 0 and np.max(prob) < thres):
            return pt
        else:
            mask = adaptive_thresh_cpu(torch.from_numpy(prob), p_thresh=np.max(prob) / 2)
            ind = np.where(mask > 0)

    if (np.size(ind[0]) == 0):
        return pt

    pt = np.zeros(2)
    if (mode == "mean"):
        pt[0] = np.mean(ind[1].astype(np.float32))
        pt[1] = np.mean(ind[0].astype(np.float32))
    else:
        v = np.unravel_index(np.argmax(prob), prob.shape)
        pt[0] = v[1]
        pt[1] = v[0]

    return pt


def get_landmark_from_prob_fast(prob, thres=0.1):
    """
    Compute landmark from prob
    prob : [RO, E1]
    mode : mean or max
    return pt : [x, y]
    """

    pt = None

    if (thres > 0 and np.max(prob) < thres):
        return pt
    else:
        v = np.unravel_index(np.argmax(prob), prob.shape)
        pt = np.zeros(2)
        pt[0] = v[1]
        pt[1] = v[0]

        return pt

    return pt


def adaptive_thresh_cpu(probs, p_thresh=0.5, p_thresh_max=0.988):
    # Try regular adaptive thresholding first
    # p_thresh_max  = 0.988 # <-- Should not be too close to 1 to ensure while loop does not go over.

    p_thresh_incr = 0.01
    # p_thresh = 0.5

    RO = probs.shape[0]
    E1 = probs.shape[1]

    try:
        number_of_blobs = float("inf")
        blobs = np.zeros((RO, E1))
        while number_of_blobs > 1 and p_thresh < p_thresh_max:
            mask = (probs > torch.max(probs) * p_thresh).float()
            blobs, number_of_blobs = ndimage.label(mask)
            p_thresh += p_thresh_incr  # <-- Note this line can lead to float drift.

        if (number_of_blobs == 1):
            return mask.numpy()

        if (number_of_blobs == 0):
            mask = np.zeros((RO, E1))
            return mask

        ## If we are here then we cannot isolate a singular blob as the LV.
        ## Select the largest blob as the final mask.
        biggest_blob = (0, torch.zeros(RO, E1))
        for i in range(number_of_blobs):
            one_blob = torch.tensor((blobs == i + 1).astype(int), dtype=torch.uint8)
            area = torch.sum(one_blob)
            if (area > biggest_blob[0]):
                biggest_blob = (area, one_blob)

        return biggest_blob[1]

    except Exception as e:
        print("Error happened in adaptive_thresh_cpu ...")
        print(e)
        mask = np.zeros((RO, E1))

    return mask


def perform_cmr_landmark_detection(im, model, fast_mode=0.0, batch_size=8, p_thresh=0.1, oper_RO=352, oper_E1=352):
    """
    Perform CMR landmark detection
    Input :
    im : [RO, E1, N],image
    model : loaded model
    p_thres: if max(prob)<p_thres, then no landmark is found
    fast_mode : if True, use the max peak as landmark, faster
    oper_RO, oper_E1: expected array size of model. Image will be padded.
    Output:
    pts: [N_pts, 2, N], landmark points, if no landmark, it is -1
    probs: [RO, E1, N_pts+1, N], probability of detected landmark points
    """

    ori_shape = im.shape
    RO = im.shape[0]
    E1 = im.shape[1]

    try:
        im = np.reshape(im, (RO, E1, np.prod(ori_shape[2:])))
        N = im.shape[2]
    except:
        im = np.expand_dims(im, axis=2)
        RO, E1, N = im.shape

    im_used, s_ro, s_e1 = cpad_2d(im, oper_RO, oper_E1)

    for n in range(N):
        im_used[:, :, n] = im_used[:, :, n] / np.max(im_used[:, :, n])

    im_used = np.transpose(im_used, (2, 0, 1))
    im_used = np.expand_dims(im_used, axis=1).astype(np.float32)
    images = torch.from_numpy(im_used).float()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = model.to(device=device)
    images = images.to(device=device, dtype=torch.float32)

    if (device == torch.device('cpu')):
        batch_size = 16

    ind = np.arange(0, N, batch_size)
    scores = torch.zeros((N, 4, images.shape[2], images.shape[3]))

    model.eval()
    with torch.no_grad():
        for b in range(ind.shape[0] - 1):
            scores[ind[b]:ind[b + 1], :, :, :] = model(images[ind[b]:ind[b + 1], :, :, :]).to(
                device=torch.device('cpu'))

        scores[ind[-1]:N, :, :, :] = model(images[ind[-1]:N, :, :, :]).to(device=torch.device('cpu'))

    probs = torch.softmax(scores, dim=1)
    probs = probs.to(device=torch.device('cpu'))

    probs = probs.numpy()
    probs = probs.astype(np.float32)
    probs = np.transpose(probs, (2, 3, 1, 0))

    C = probs.shape[2]

    probs_used = np.reshape(probs, (probs.shape[0], probs.shape[1], C * N))
    probs_used, s_ro_p, s_e1_p = cpad_2d(probs_used, RO, E1)
    probs = np.reshape(probs_used, (probs_used.shape[0], probs_used.shape[1], C, N))

    N_pts = C - 1

    pts = np.zeros((N_pts, 2, N)) - 1.0

    for n in range(N):
        for p in range(N_pts):
            prob = probs[:, :, p + 1, n]
            if (fast_mode == 1.0):
                pt = get_landmark_from_prob_fast(prob, thres=p_thresh)
            else:
                pt = get_landmark_from_prob(prob, thres=p_thresh, mode="mean", binary_mask=False)
            if pt is not None:
                pts[p, 0, n] = pt[0]
                pts[p, 1, n] = pt[1]

    probs = probs.astype(np.float32)
    pts = pts.astype(np.float32)

    probs = np.reshape(probs, (RO, E1, C) + ori_shape[2:])
    pts = np.reshape(pts, (N_pts, 2) + ori_shape[2:])

    return pts, probs


def extend_landmarks(landmarks, fov):
    points_ant, points_post, point_mid = landmarks
    landmarks[0] = points_ant + (points_ant - point_mid)
    landmarks[1] = points_post + (points_post - point_mid)
    landmarks = np.clip(landmarks, a_min=0, a_max=fov)  # Prevent going off edges
    return landmarks


def get_t1_from_lvcav(t1_raw, lv_xy):
    lv_x, lv_y = lv_xy.astype(np.uint)
    t1 = t1_raw[lv_y, lv_x]
    return t1

