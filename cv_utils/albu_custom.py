"""
Just my custom albumentations addons
NOTE TO SELF: Be careful with these as you've been bitten by albumentations code changes before!
"""

from typing import Tuple, Union
import random

from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations.augmentations.functional as F
import cv2


class Grid(ImageOnlyTransform):
    """
    Helpful for visualizing the effect of some geometric transforms
    """
    def __init__(self, grid_size):
        super().__init__(True, 1)
        self.grid_size = grid_size

    def apply(self, img, **_):
        img[self.grid_size::self.grid_size] *= 0
        img[:, self.grid_size::self.grid_size] *= 0
        return img


class ResizeWidth(ImageOnlyTransform):
    """
    Resize by specifying only target width. Can lock aspect ratio if desired
    """
    def __init__(self, width, lock_aspect_ratio=True):
        super().__init__(True, 1)
        self.width = width
        self.lock_ar = lock_aspect_ratio

    def apply(self, img, **_):
        new_height, width = img.shape[:2]
        if self.lock_ar:
            new_height = int(round(self.width/width*new_height))
        img = F.resize(img, new_height, self.width)
        return img


class RandomScale(ImageOnlyTransform):
    """
    Like Albumentations random scale but we have control over height and width
    separately
    Output image is different size from input image
    """
    def __init__(self, height_scale_limit: Union[float, Tuple[float, float]],
                 width_scale_limit: Union[float, Tuple[float, float]],
                 interpolation=cv2.INTER_LINEAR, p: float = 0.5):
        """
        Args:
            height_scale_limit - Scaling factor range. If it is a single float
                value, the range will be (1 - height_scale_limit,
                1 + height_scale_limit). If it is a tuple of two float values,
                the range will be (height_scale_limit[0], height_scale_limit[1])
            width_scale_limit - Same concept as height_scale_limit but for
                width
            interpolation - cv2 interpolation flag.
            p - probability of applying the transform
        """
        super().__init__(True, 1)
        if isinstance(height_scale_limit, float):
            height_scale_limit = (1 - height_scale_limit, 1 + height_scale_limit)
        self.height_scale_limit = height_scale_limit
        if isinstance(width_scale_limit, float):
            width_scale_limit = (1 - width_scale_limit, 1 + width_scale_limit)
        self.width_scale_limit = width_scale_limit
        self.interpolation = interpolation

    def apply(self, img, **_):
        if random.random() > self.p:
            return img
        img = cv2.resize(
            img, (0, 0), fx=random.uniform(*self.width_scale_limit),
            fy=random.uniform(*self.height_scale_limit),
            interpolation=self.interpolation)
        return img