import cv2
import numpy as np


def desaturate(img: np.ndarray, factor=1.) -> np.ndarray:
    """
    Converts RGB image to HSV then multiplies the S channel by factor before
    converting back to RGB
    """
    assert 0 <= factor and factor <= 1, "`factor` should lie in [0, 1]"
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[..., 1] = (img[..., 1] * factor).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img