from typing import Union

import numpy as np
import cv2


def resize_shortest_edge(
        img: np.ndarray, length: int,
        interpolation: Union[int, str]=cv2.INTER_LINEAR):
    """
    Resize image with locked aspect ratio such that its shortest side is `length`
    pixels long.
    `interpolation` specifies the cv2 interpolation type and defaults to
    cv2.INTER_LINERAR It may be specified as 'auto' in which case either
    cv2.INTER_AREA or cv2.INTERCUBIC is used depnding on whether we are
    downsizing or upsizing (respectively)
    """
    f = length/np.min(img.shape[:2])
    if isinstance(interpolation, str):
        assert interpolation == 'auto', \
            "If `interpolation` is a str it can only be 'auto'"
        interpolation = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (0,0), fx=f, fy=f, interpolation=interpolation)


def resize_longest_edge(
        img: np.ndarray, length: int,
        interpolation: Union[int, str]=cv2.INTER_LINEAR,
        min_factor=np.float('-inf'), max_factor=np.float('inf')):
    """
    Resize image with locked aspect ratio such that its longest side is `length`
    pixels long BUT we keep into account min factor and max factor.
    `interpolation` specifies the cv2 interpolation type and defaults to
    cv2.INTER_LINERAR It may be specified as 'auto' in which case either
    cv2.INTER_AREA or cv2.INTERCUBIC is used depnding on whether we are
    downsizing or upsizing (respectively)
    """
    f = length/np.max(img.shape[:2])
    f = min(max_factor, f)
    f = max(min_factor, f)
    if isinstance(interpolation, str):
        assert interpolation == 'auto', \
            "If `interpolation` is a str it can only be 'auto'"
        interpolation = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (0,0), fx=f, fy=f, interpolation=interpolation)


def resize_to_area(
        img: np.ndarray, area: int, interpolation=cv2.INTER_LINEAR,
        min_factor=np.float('-inf'), max_factor=np.float('inf')) -> np.ndarray:
    """
    resize an image such that:
     - aspect ratio is maintained
     - the area ~ `area` but we keep into account min factor and max factor
    `interpolation` specifies the cv2 interpolation type and defaults to
    cv2.INTER_LINEAR. It may be specified as 'auto' in which case either
    cv2.INTER_AREA or cv2.INTERCUBIC is used depnding on whether we are
    downsizing or upsizing (respectively)
    """
    original_area = np.prod(img.shape[:2])
    f = np.sqrt(area/original_area)
    f = min(max_factor, f)
    f = max(min_factor, f)
    if isinstance(interpolation, str):
        assert interpolation == 'auto', \
            "If `interpolation` is a str it can only be 'auto'"
        interpolation = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (0,0), fx=f, fy=f, interpolation=interpolation)