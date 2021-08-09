from typing import Sequence
import warnings

import numpy as np


def pad_to_shape(img: np.ndarray, shape: Sequence, mode:str='symmetric'):
    """
    Takes an `img` and does symmetrical padding to get it into the desired shape.
    Resolves odd number padding by padding diff//2 to start and the remainder to the end.
    Padding is with zeros
    Resolves shape < orig shape by ignoring that dimension
    `mode` specifies whether it's symmetric, start, or end padding
    """
    n_dims = img.ndim
    assert n_dims in [2, 3], "Can only deal with 2D or 3D images"

    implemented_modes = ['symmetric', 'start', 'end']
    assert mode in implemented_modes, f"mode {mode} not implemented"

    if any((np.array(shape) - img.shape[:2]) < 0 ):
        warnings.warn("Image is larger than desired shape in at least one "
                      "dimension. Ignoring those dimensions.")
    diff = np.maximum(0, np.array(shape) - img.shape[:2])
    if mode == 'symmetric':
        pad_start = diff // 2
        pad_end = diff - diff // 2
    elif mode == 'start':
        pad_start = diff
        pad_end = np.zeros_like(diff)
    elif mode == 'end':
        pad_start = np.zeros_like(diff)
        pad_end = diff

    pad_vals = list(zip(pad_start, pad_end))
    if n_dims == 3:
        pad_vals.append((0, 0))

    return np.pad(img, pad_vals)