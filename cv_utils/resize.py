from typing import Union

import numpy as np
import cv2


def resize_shortest_edge(
        img: np.ndarray, length: int,
        interpolation: Union[int, str]=cv2.INTER_LINEAR):
    """
    Resize an image such that it's shortest edge is some length while maintaining aspect ratio.
    Args:
        - img: Numpy uint8 array for the image to resize.
        - length: Longest edge length for the output. The image will be resized to as close to this as possible while 
            respecting `min_factor` and `max_factor`.
        - min_factor, max_factor: It might be desirable to specify a minimum or maximum resize factor in terms of 
            ratio between input height (width) vs output height (width).
        - interpolation: specifies the cv2 interpolation type and defaults to cv2.INTER_LINEAR. It may be specified as
            'auto' in which case either cv2.INTER_AREA or cv2.INTERCUBIC is used depending on whether we are downsizing
            or upsizing (respectively).
    Returns:
        - Resized image.
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
    Resize an image such that it's longest edge is some length while maintaining aspect ratio.
    Args:
        - img: Numpy uint8 array for the image to resize.
        - length: Longest edge length for the output. The image will be resized to as close to this as possible while 
            respecting `min_factor` and `max_factor`.
        - min_factor, max_factor: It might be desirable to specify a minimum or maximum resize factor in terms of 
            ratio between input height (width) vs output height (width).
        - interpolation: specifies the cv2 interpolation type and defaults to cv2.INTER_LINEAR. It may be specified as
            'auto' in which case either cv2.INTER_AREA or cv2.INTERCUBIC is used depending on whether we are downsizing
            or upsizing (respectively).
    Returns:
        - Resized image.
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
    Resize an image to some total area (approximately) and with locked aspect ratio.
    Args:
        - img: Numpy uint8 array for the image to resize.
        - area: Area we want for the output image. The image will be resized to as close to this as possible while 
            maintaining the aspect ratio, and respecting `min_factor` and `max_factor`.
        - min_factor, max_factor: It might be desirable to specify a minimum or maximum resize factor in terms of 
            ratio between input height (width) vs output height (width).
        - interpolation: specifies the cv2 interpolation type and defaults to cv2.INTER_LINEAR. It may be specified as
            'auto' in which case either cv2.INTER_AREA or cv2.INTERCUBIC is used depending on whether we are downsizing
            or upsizing (respectively).
    Returns:
        - Resized image.
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


def resize_vertical_edge(img: np.ndarray, height: int, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize an image to some vertical height and with locked aspect ratio.
    Args:
        - img: Numpy uint8 array for the image to resize.
        - length: Height of the desired output in pixels.
        - interpolation: specifies the cv2 interpolation type and defaults to cv2.INTER_LINEAR. It may be specified as
            'auto' in which case either cv2.INTER_AREA or cv2.INTERCUBIC is used depending on whether we are downsizing
            or upsizing (respectively).
    Returns:
        - Resized image.
    """
    f = height / img.shape[0]
    if isinstance(interpolation, str):
        assert interpolation == 'auto', \
            "If `interpolation` is a str it can only be 'auto'"
        interpolation = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (0, 0), fx=f, fy=f, interpolation=interpolation)


def resize_horizontal_edge(img: np.ndarray, width: int, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize an image to some horizontal width and with locked aspect ratio.
    Args:
        - img: Numpy uint8 array for the image to resize.
        - length: Width of the desired output in pixels.
        - interpolation: specifies the cv2 interpolation type and defaults to cv2.INTER_LINEAR. It may be specified as
            'auto' in which case either cv2.INTER_AREA or cv2.INTERCUBIC is used depending on whether we are downsizing
            or upsizing (respectively).
    Returns:
        - Resized image.
    """
    f = width / img.shape[1]
    if isinstance(interpolation, str):
        assert interpolation == 'auto', \
            "If `interpolation` is a str it can only be 'auto'"
        interpolation = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (0, 0), fx=f, fy=f, interpolation=interpolation)
