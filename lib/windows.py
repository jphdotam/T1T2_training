from typing import Optional, Union

import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import matplotlib.pyplot as plt

T1_CUTOFF = 580

SEQUENCE_WINDOWS = {"T1w": [{'wc': 1300, 'ww': 1300}, {'wc': 500, 'ww': 1000}],
                    "T2w": [{'wc': 60, 'ww': 120}],
                    "PDw": None,
                    "T1": [{'wc': 1300, 'ww': 1300}, {'wc': 500, 'ww': 1000}],
                    "T2": [{'wc': 60, 'ww': 120}]}

WW_T1_PRE = SEQUENCE_WINDOWS['T1'][0]['ww']
WC_T1_PRE = SEQUENCE_WINDOWS['T1'][0]['wc']

WW_T1_POST = SEQUENCE_WINDOWS['T1'][1]['ww']
WC_T1_POST = SEQUENCE_WINDOWS['T1'][1]['wc']

WW_T2 = SEQUENCE_WINDOWS['T2'][0]['ww']
WC_T2 = SEQUENCE_WINDOWS['T2'][0]['wc']

def window_numpy(img: np.ndarray,
                 window_centre: Union[float, int],
                 window_width: Union[float, int],
                 cmap: Optional[LinearSegmentedColormap] = None,
                 rescale_255: Optional[bool] = True) -> np.ndarray:
    """
    :param img: image array
    :param window_centre: DICOM window center (level)
    :param window_width: DICOM window width
    :param cmap: matplotlib cmap, useful for e.g. T1/T2 maps
    :param rescale_255: return image as a uint8 rather than an array of floats
    :return: numpy image array (ndarray)
    """
    v_min = window_centre - (window_width // 2)
    v_max = window_centre + (window_width // 2)
    img = np.clip(img, v_min, v_max)
    img = img - v_min
    img = img / window_width
    if rescale_255:
        img = (img * 255).astype(np.uint8)
    if cmap:
        img = cmap(img)
    return img


def normalize_data(data, window_centre, window_width):
    window_min = max(0, window_centre - window_width // 2)
    frame = data - window_min
    frame = frame / window_width
    frame = np.clip(frame, 0, 1)
    frame = frame.astype(np.float32)
    return frame
