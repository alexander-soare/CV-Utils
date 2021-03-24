import numpy as np
import cv2


def resize_shortest_edge(img: np.ndarray, length: int, interpolation=None):
    """
    resize image with locked aspect ratio such that its shortest side is `length`
     pixels long. `interpolation` specifies the cv2 interpolation type
    """
    f = length/np.min(img.shape[:2])
    return cv2.resize(img, (0,0), fx=f, fy=f, interpolation=interpolation)