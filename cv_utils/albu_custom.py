"""
Just my custom albumentations addons
NOTE TO SELF: Be careful with these as you've been bitten by albumentations code changes before!
"""

from albumentations.core.transforms_interface import ImageOnlyTransform


class Grid(ImageOnlyTransform):
    """
    Helpful for visualizing the effect of some geometric transforms
    """
    def __init__(self, grid_size):
        super().__init__(True, 1)
        self.grid_size = grid_size

    def apply(self, img, **params):
        img[self.grid_size::self.grid_size] *= 0
        img[:, self.grid_size::self.grid_size] *= 0
        return img
