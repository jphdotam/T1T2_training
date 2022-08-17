import copy
import math
import scipy
import scipy.spatial
import numpy as np
from skimage import measure


def mask2sectors(endo_mask, epi_mask, rv_mask, rvi_mask, num_sectors):
    """
    Split myocardium to num_sectors sectors
    Input :
    endo_mask : [RO, E1], mask for endo
    epi_mask : [RO, E1], mask for epi
    rv_mask : [RO, E1], mask for rv
    rvi_mask : [RO, E1], mask for rv insertion mask, can be None; if not None, rv_mask is not used
    Output :
    sectors : [RO, E1] sector mask, sector 1 is labelled as value 1
    """

    def get_angle(a, b):
        # angle from a to b (rotate a to b)
        # positve angle for counter-clock wise
        # 0-360 degrees

        v1_theta = math.atan2(a[1], a[0])
        v2_theta = math.atan2(b[1], b[0])

        r = (v2_theta - v1_theta) * (180.0 / math.pi)

        if r < 0:
            r += 360.0

        return r

    def img_to_xy(rvi_, _, e1_):
        return rvi_[1], e1_ - 1 - rvi_[0]

    img_height, img_width = endo_mask.shape

    # find lv center
    endo_pts = np.argwhere(endo_mask > 0)
    lv_center = np.mean(endo_pts, axis=0)
    lv_center2 = img_to_xy(lv_center, img_height, img_width)

    # find rv center
    if rv_mask is not None:
        rv_pts = np.argwhere(rv_mask > 0)
        rv_center = np.mean(rv_pts, axis=0)
    else:
        if rvi_mask is None:
            raise ValueError("Both rv_mask and rvi_mask are None")

        rvi_pts = np.argwhere(rvi_mask > 0)
        rvi_pt = np.mean(rvi_pts, axis=0)
        dist = np.linalg.norm(rvi_pt - lv_center)
        if rvi_pt[1] < lv_center[1]:
            rv_center = lv_center
            rv_center[1] -= 2 * dist
            rv_center[0] += dist
        else:
            rv_center = lv_center
            rv_center[0] -= 2 * dist
            rv_center[1] -= dist

    rv_center2 = img_to_xy(rv_center, img_height, img_width)
    rv_vec = (rv_center2[0] - lv_center2[0], rv_center2[1] - lv_center2[1])

    # find rvi
    if rvi_mask is None:

        num_rv_pts = rv_pts.shape[0]

        rvi = np.zeros((1, 2))
        max_angle = 0

        for pt in range(num_rv_pts):
            pt2 = img_to_xy((rv_pts[pt, 0], rv_pts[pt, 1]), img_height, img_width)
            rv_pt_vec = (pt2[0] - lv_center2[0], pt2[1] - lv_center2[1])
            rv_rvi_angle = get_angle(rv_pt_vec, rv_vec)

            if 180 >= rv_rvi_angle > max_angle:
                max_angle = rv_rvi_angle
                rvi[0, 0] = rv_pts[pt, 0]
                rvi[0, 1] = rv_pts[pt, 1]
    else:
        rvi = np.argwhere(rvi_mask > 0)

    rvi2 = img_to_xy((rvi[0, 0], rvi[0, 1]), img_height, img_width)

    # split endo/epi to sectors
    rvi_vec = (rvi2[0] - lv_center2[0], rvi2[1] - lv_center2[1])
    rv_rvi_angle = get_angle(rv_vec, rvi_vec)

    delta_rvi_angle = 360 / num_sectors

    sectors = np.zeros(endo_mask.shape)

    myo_mask = epi_mask - endo_mask
    myo_pts = np.argwhere(myo_mask > 0)
    n_myo_pts = myo_pts.shape[0]
    angle_myo_pts = np.zeros(n_myo_pts)

    for n in range(n_myo_pts):
        myo_pts_xy = img_to_xy(myo_pts[n, :], img_height, img_width)
        angle_myo_pts[n] = get_angle(rvi_vec, (myo_pts_xy[0] - lv_center2[0], myo_pts_xy[1] - lv_center2[1]))
        if rv_rvi_angle >= 180:  # rotate rvi clock wise
            angle_myo_pts[n] = 360 - angle_myo_pts[n]

        sector_no = np.floor(angle_myo_pts[n] / delta_rvi_angle) + 1

        if sector_no == 1:
            sectors[myo_pts[n, 0], myo_pts[n, 1]] = sector_no
        else:
            sectors[myo_pts[n, 0], myo_pts[n, 1]] = num_sectors + 2 - sector_no

    return sectors


def smooth_contours(contour_x, contour_y, n_components=24, circularise=False, n_pts=2000):
    """ takes contour_x,contour_y the cartesian coordinates of a contour,
        then procdues a smoothed more circular contour smoothed_contour_x,smoothed_contour_y"""

    if n_components is None:
        n_components = 12  # slightly arbitary number,  but seems to work well

    npts = n_pts + 1
    contour_pts = np.transpose(np.stack([contour_x, contour_y]))

    if circularise:
        # get the contour points that form a convex hull
        hull = scipy.spatial.ConvexHull(contour_pts)
        to_sample = hull.vertices
    else:
        to_sample = range(0, len(contour_x))

    # wrap around cirlce
    to_sample = np.hstack([to_sample, to_sample[0]])
    sample_pts = contour_pts[to_sample, :]

    # sample each curve at uniform distances according to arc length parameterisation
    dist_between_pts = np.diff(sample_pts, axis=0)
    cumulative_distance = np.sqrt(dist_between_pts[:, 0] ** 2 + dist_between_pts[:, 1] ** 2)
    cumulative_distance = np.insert(cumulative_distance, 0, 0, axis=0)
    cumulative_distance = np.cumsum(cumulative_distance)
    cumulative_distance = cumulative_distance / cumulative_distance[-1]
    contour_x = np.interp(np.linspace(0, 1, npts), cumulative_distance, sample_pts[:, 0], period=360)
    contour_y = np.interp(np.linspace(0, 1, npts), cumulative_distance, sample_pts[:, 1], period=360)
    contour_x = contour_x[:-1]
    contour_y = contour_y[:-1]

    # smooth out contour by keeping the lowest nkeep Fourier components
    n = len(contour_x)
    n_filt = n - n_components - 1
    f = np.fft.fft(contour_x)
    f[int(n / 2 + 1 - n_filt / 2):int(n / 2 + n_filt / 2)] = 0.0
    smoothed_contour_x = np.abs(np.fft.ifft(f))
    f = np.fft.fft(contour_y)
    f[int(n / 2 + 1 - n_filt / 2):int(n / 2 + n_filt / 2)] = 0.0
    smoothed_contour_y = np.abs(np.fft.ifft(f))

    return smoothed_contour_x, smoothed_contour_y


def extract_contours(preds, thres=0.75, smoothing=True, num_components_smoothing=24, circular=False, n_pts=2000):
    """Extract contours from segmentation mask or probability map
    Inputs:
        preds : [RO E1], input mask or probablity map
        thres : threshold to extract contours, a 2D marching cube extration is performed
        smoothing : True or False, if true, contours are smoothed
        num_components_smoothing : number of fft components kept after smoothing
        circular : True or False, if true, contours are kept to approx. circle
    Outputs:
        contours : a list of contours, every contour is a nx2 numpy array
    """

    contours = measure.find_contours(preds, thres)
    len_contours = list()

    for n, contour in enumerate(contours):
        len_contours.append(contours[n].shape[0])

    if smoothing:
        s_c = copy.deepcopy(contours)
        for n, contour in enumerate(contours):
            sc_x, sc_y = smooth_contours(contour[:, 0],
                                         contour[:, 1],
                                         n_components=num_components_smoothing,
                                         circularise=circular,
                                         n_pts=n_pts)

            s_c[n] = np.zeros((sc_x.shape[0], 2))
            s_c[n][:, 0] = sc_x
            s_c[n][:, 1] = sc_y

        contours = copy.deepcopy(s_c)

    return contours, len_contours


def extract_epi_contours(preds, thres=0.75, smoothing=True, num_components_smoothing=24, circular=False, n_pts=2000):
    """Extract myocardium epi contours from segmentation mask or probability map
    Inputs:
        preds : [RO E1], input mask or probablity map
        thres : threshold to extract contours, a 2D marching cube extration is performed
        smoothing : True or False, if true, contours are smoothed
        num_components_smoothing : number of fft components kept after smoothing
        circular : True or False, if true, contours are kept to approx. circle
    Outputs:
        epi : a nx2 numpy array for epi contour
    """

    contours, len_contour = extract_contours(preds, thres, smoothing, num_components_smoothing, circular, n_pts)
    num_c = len(contours)
    epi = None

    if num_c == 0:
        return epi

    if num_c == 1:
        epi = contours[0]
        return epi

    if num_c > 1:
        # find the longest contours as epi
        c_len = np.zeros([num_c])
        for n, contour in enumerate(contours):
            c_len[n] = len_contour[n]
        c_ind = np.argsort(c_len)
        epi = contours[c_ind[-1]]

    return epi


def compute_bullseye_sector_mask_for_slice(endo_mask, epi_mask, rv_mask, rvi_mask, num_sectors=None):
    """
    Compute sector masks for single slice
    Input :
    endo_mask, epi_mask, rv_mask, rvi_mask : [RO, E1]
    rvi_mask can be all zeros. In this case, rv_mask is used
    num_sectors : 6, but should be for 4 apex
    Output :
    sectors : [RO, E1], sector mask. For 6 sectors, its values are 1, 2, 3, 4, 5, 6. background is 0.
    sectors_32 : [RO, E1], sector mask for endo and epi.
            For 6 EPI sectors, its values are 1-6. background is 0.
            For ENDO sectors, it is 7-12
    """

    rvi_pt = np.argwhere(rvi_mask > 0)
    has_rvi = True
    if (rvi_pt is None) or (rvi_pt.shape[0] == 0):
        print("Cannot find rvi point, image must be in CMR view ... ")
        endo_mask = np.transpose(endo_mask, [1, 0, 2])
        epi_mask = np.transpose(epi_mask, [1, 0, 2])
        rv_mask = np.transpose(rv_mask, [1, 0, 2])
        has_rvi = False

    img_height, img_width = endo_mask.shape

    # refine epi
    m = np.zeros((img_height, img_width))
    m[np.where(epi_mask > 0)] = 1
    m[np.where(endo_mask > 0)] = 1
    epi_mask_2 = m

    # get contours
    contours_endo = extract_epi_contours(endo_mask,
                                         thres=0.5,
                                         smoothing=True,
                                         num_components_smoothing=36,
                                         circular=False,
                                         n_pts=2000)
    contours_epi = extract_epi_contours(epi_mask_2,
                                        thres=0.95,
                                        smoothing=True,
                                        num_components_smoothing=36,
                                        circular=False,
                                        n_pts=2000)

    # split sectors
    rvi_pt = np.argwhere(rvi_mask > 0)
    if rvi_pt is None:
        raise ValueError("Cannot find rv insertion point")

    # split 16 sectors
    sectors = mask2sectors(endo_mask, epi_mask, rv_mask, rvi_mask, num_sectors)

    # split 32 sectors
    endo_kd = scipy.spatial.KDTree(contours_endo)
    epi_kd = scipy.spatial.KDTree(contours_epi)

    myo = np.copy(sectors)

    max_myo = np.max(myo)
    pts = np.where(myo > 0)
    n_pts = pts[0].shape[0]

    pts_2 = np.zeros((n_pts, 2))
    pts_2[:, 0] = pts[0]
    pts_2[:, 1] = pts[1]

    d_endo, i_endo = endo_kd.query(pts_2)
    d_epi, i_epi = epi_kd.query(pts_2)

    for p in range(n_pts):
        if d_epi[p] > d_endo[p]:
            myo[pts[0][p], pts[1][p]] = myo[pts[0][p], pts[1][p]] + max_myo

    sectors_32 = myo

    if (rvi_pt is None) or (rvi_pt.shape[0] == 0):
        sectors = np.transpose(sectors, [1, 0, 2])
        sectors_32 = np.transpose(sectors_32, [1, 0, 2])

    return sectors, sectors_32
