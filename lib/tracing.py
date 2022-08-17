import math
import skimage.graph
import numpy as np

from skimage.graph import route_through_array
from skimage.draw import circle as disk


def _get_max_of_image_between_two_points(im, x0, y0, x1, y1):
    n_points = int(np.hypot(x1 - x0, y1 - y0))
    x_between, y_between = np.linspace(x0, x1, n_points), np.linspace(y0, y1, n_points)
    z_between = im[y_between.astype(np.uint8), x_between.astype(np.uint8)]
    z_argmax = np.argmax(z_between)
    x_max, y_max, z_max = x_between[z_argmax], y_between[z_argmax], np.max(z_between)
    return z_max, (y_max, x_max)


def _get_cost_arrays_for_each_route(heatmap, landmarks, raise_to_power=4, block_cost=10, use_smaller_radius=False):
    heatmap_epi, heatmap_end = heatmap
    lv_x, lv_y = landmarks[2]
    (rv_ant_x, rv_ant_y), (rv_inf_x, rv_inf_y) = landmarks[:2]
    cost_epi = (1 - heatmap_epi) ** raise_to_power
    cost_end = (1 - heatmap_end) ** raise_to_power

    # outer paths (inner blocked)
    rv_mid_x, rv_mid_y = int(rv_ant_x + rv_inf_x) // 2, int(rv_ant_y + rv_inf_y) // 2
    double_rv_mid_x, double_rv_mid_y = lv_x + (rv_mid_x - lv_x) * 2, lv_y + (rv_mid_y - lv_y) * 3
    # print("inner")
    _, (inner_circle_y, inner_circle_x) = _get_max_of_image_between_two_points(heatmap_epi,
                                                                               double_rv_mid_x, double_rv_mid_y,
                                                                               lv_x, lv_y)

    inner_circle_radius = math.sqrt((rv_ant_x - rv_inf_x) ** 2 + (rv_ant_y - rv_inf_y) ** 2) * 0.3
    rr, cc = disk(inner_circle_y, inner_circle_x, inner_circle_radius//2 if use_smaller_radius else inner_circle_radius, shape=cost_epi.shape)

    # ended up dividing by 2 as sometimes so large it interferes... this seems to work, ugly!
    # See 'T1T2_141613_57381537_57381545_528_20201116-134335__T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy' for e.g.

    cost_outer_epi = cost_epi.copy()
    cost_outer_epi[rr, cc] = block_cost
    cost_outer_end = cost_end.copy()
    cost_outer_end[rr, cc] = block_cost

    # inner paths (outer blocked)
    double_opposite_x, double_opposite_y = lv_x - (rv_mid_x - lv_x) * 2, lv_y - (rv_mid_y - lv_y) * 3
    # print("outer")
    _, (outer_circle_y, outer_circle_x) = _get_max_of_image_between_two_points(heatmap_epi,
                                                                               double_opposite_x, double_opposite_y,
                                                                               lv_x, lv_y)

    outer_circle_radius = math.sqrt((outer_circle_x - lv_x) ** 2 + (outer_circle_y - lv_y) ** 2)
    rr, cc = disk(outer_circle_y, outer_circle_x, outer_circle_radius, shape=cost_epi.shape)
    cost_inner_epi = cost_epi.copy()
    cost_inner_epi[rr, cc] = block_cost
    cost_inner_end = cost_end.copy()
    cost_inner_end[rr, cc] = block_cost

    # import matplotlib.pyplot as plt
    # plt.imshow(cost_outer_epi)
    # plt.show()
    # plt.imshow(cost_inner_epi)
    # plt.show()

    return cost_outer_epi, cost_inner_epi, cost_outer_end, cost_inner_end


def _get_landmarks_on_epi_and_end(heatmap, landmarks):
    lv_x, lv_y = landmarks[2]
    # print("epi ant")
    _epi_ant_conf, (epi_ant_y, epi_ant_x) = _get_max_of_image_between_two_points(heatmap[0],
                                                                                 *landmarks[0],
                                                                                 lv_x,
                                                                                 lv_y)
    # print("epi inf")
    _epi_inf_conf, (epi_inf_y, epi_inf_x) = _get_max_of_image_between_two_points(heatmap[0],
                                                                                 *landmarks[1],
                                                                                 lv_x,
                                                                                 lv_y)
    # print("end ant")
    _end_ant_conf, (end_ant_y, end_ant_x) = _get_max_of_image_between_two_points(heatmap[1],
                                                                                 *landmarks[0],
                                                                                 lv_x,
                                                                                 lv_y)
    # print("end inf")
    _end_inf_conf, (end_inf_y, end_inf_x) = _get_max_of_image_between_two_points(heatmap[1],
                                                                                 *landmarks[1],
                                                                                 lv_x,
                                                                                 lv_y)

    return (epi_ant_x, epi_ant_y), (epi_inf_x, epi_inf_y), (end_ant_x, end_ant_y), (end_inf_x, end_inf_y)


def _get_path_for_cost_array_and_coords(cost_array, coord_from, coord_to):
    coord_from = [int(round(c)) for c in coord_from]
    coord_to = [int(round(c)) for c in coord_to]
    path, _cost = route_through_array(cost_array, coord_from, coord_to)
    return path


def get_epi_end_paths_from_heatmap_and_landmarks(heatmap, landmarks):
    (epi_ant_x, epi_ant_y), (epi_inf_x, epi_inf_y), (end_ant_x, end_ant_y), (end_inf_x, end_inf_y) = \
        _get_landmarks_on_epi_and_end(heatmap, landmarks)

    cost_outer_epi, cost_inner_epi, cost_outer_end, cost_inner_end = _get_cost_arrays_for_each_route(heatmap, landmarks)

    path_epi_inner = _get_path_for_cost_array_and_coords(cost_inner_epi, (epi_ant_y, epi_ant_x), (epi_inf_y, epi_inf_x))
    path_epi_outer = _get_path_for_cost_array_and_coords(cost_outer_epi, (epi_inf_y, epi_inf_x), (epi_ant_y, epi_ant_x))
    path_end_inner = _get_path_for_cost_array_and_coords(cost_inner_end, (end_ant_y, end_ant_x), (end_inf_y, end_inf_x))
    path_end_outer = _get_path_for_cost_array_and_coords(cost_outer_end, (end_inf_y, end_inf_x), (end_ant_y, end_ant_x))
    path_epi = path_epi_inner + path_epi_outer
    path_end = path_end_inner + path_end_outer

    ys_epi, xs_epi = np.array(path_epi).transpose((1, 0))
    ys_end, xs_end = np.array(path_end).transpose((1, 0))

    return (xs_epi, ys_epi), (xs_end, ys_end)
