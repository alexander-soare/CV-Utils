"""
Just my custom albumentations addons
NOTE TO SELF: Be careful with these as you've been bitten by albumentations code changes before!
"""

from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations.augmentations.functional as F


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
    resize by specifying only target width. Can lock aspect ratio if desired
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
