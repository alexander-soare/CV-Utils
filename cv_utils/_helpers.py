from typing import Sequence
import numpy as np


def adapt_to_dims(f):
    """
    Many bbox and polygon utilities should be able to work with a single input
    (1D) or with multiple (2D). Using this decorator we can make 1D inputs 
    2D then flatten the result before returning it.
    The decorated function should take bboxes/polys as the first argument and
    must also return bboxes/polys.
    """
    def wrapper(*args, **kwargs):
        inp, args = args[0], args[1:]
        is_1d = False
        if len(inp.shape) == 1:
            is_1d = True
            inp = np.expand_dims(inp, axis=0)
        out = f(inp, *args, **kwargs)
        if is_1d:
            out = out.flatten()
        return out
    return wrapper


# Taken from
# https://github.com/open-mmlab/mmocr/blob/b8f7ead74cb0200ad5c422e82724ca6b2eb1c543/mmocr/datasets/pipelines/box_utils.py
def sort_vertices(vertices: np.ndarray, direction: str = 'clockwise'):
    """
    Sort (N, 2) array of N vertices with xy coords such that the top-left
    vertex is first, and they are in clockwise or counter-clockwise order
    """
    valid_directions = ["clockwise", "counter-clockwise"]
    assert direction in valid_directions, \
        f"`direction` not in valid directions: {', '.join(valid_directions)}"

    assert vertices.ndim == 2
    assert vertices.shape[-1] == 2
    N = vertices.shape[0]
    if N == 0:
        return vertices

    # Sort vertices in clockwise order starting from 3 o'clock about the
    # centroid
    centroid = np.mean(vertices, axis=0)
    directions = vertices - centroid
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    sort_idx = np.argsort(-angles)
    if direction == 'counter-clockwise':
        sort_idx = sort_idx[::-1]
    vertices = vertices[sort_idx]

    # Find the top left (closest to an axis-aligned bounding box top-left point)
    left_top = np.min(vertices, axis=0)
    # Rotate vertex indices such that the first is the top-left one
    dists = np.linalg.norm(left_top - vertices, axis=-1, ord=2)
    topleft_ix = np.argmin(dists)
    indices = (np.arange(N, dtype=np.int) + topleft_ix) % N
    return vertices[indices]


def to_2tuple(x):
    if not isinstance(x, Sequence):
        x = (x, x)
    return x