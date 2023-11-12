import cv2
import numpy as np
from copy import deepcopy

def convert_gray_to_rgb_matrix(target):
    width, height = target.shape
    ctarget = np.zeros((width, height, 3), dtype=np.uint8)
    ctarget[:, :, 0] = target
    ctarget[:, :, 1] = target
    ctarget[:, :, 2] = target
    return ctarget

#for gray image with format of RGB matrix
def setColorScheme(target):

    target_res = deepcopy(target)

    hue_range = (120, 260)
    sat_range = (70, 255)
    default_br = 245

    # Calculate h for the entire array in one go
    h_values = hue_range[0] + (hue_range[1] - hue_range[0]) * (target[:, :, 0] / 255)
    s_values = sat_range[0] + (sat_range[1] - sat_range[0]) * (target[:, :, 0] / 255)
    # Set the values for v for the entire array
    v_values = np.full_like(target_res[:, :, 2], default_br)

    # Assign calculated values to target_res
    target_res[:, :, 0] = h_values
    target_res[:, :, 1] = s_values
    target_res[:, :, 2] = v_values

    target_res = cv2.cvtColor(target_res, cv2.COLOR_HSV2BGR)
    return target_res.astype(np.uint8)