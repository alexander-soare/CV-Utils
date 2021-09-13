from typing import Sequence, Callable

import numpy as np
from numba import jit
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

from ._helpers import adapt_to_dims

# BC
from .render import draw_bboxes


def bounding_bbox(bboxes: Sequence[np.ndarray]) -> np.ndarray:
    """
    takeas an array of bboxes in xyxy format and returns one bbox which contains
    them all
    """
    return np.concatenate([bboxes[:, :2].min(0), bboxes[:, 2:].max(0)], -1)


def bbox_crop(img: np.ndarray, bbox: Sequence):
    """
    takes img and returns contents of bbox (x0, y0, x1, y1)
    performs rounding on the bbox co-ordinates
    """
    bbox = np.round(np.array(bbox)).astype(int)
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()


def xyxy_to_xywh(bboxes: np.ndarray) -> np.ndarray:
    """
    in xywh:
        x, y are center coordinates
        w, h are full width and full height
    in xyxy:
        xy__ is the top left coordinate
        __xy is the top right coordinate
    """
    is_1d = False
    if len(bboxes.shape) == 1:
        is_1d = True
        bboxes = np.expand_dims(bboxes, axis=0)
    xc = bboxes[:, [0, 2]].mean(axis=1, keepdims=True)
    yc = bboxes[:, [1, 3]].mean(axis=1, keepdims=True)
    w = bboxes[:, 2:3] - bboxes[:, 0:1]
    h = bboxes[:, 3:4] - bboxes[:, 1:2]
    bboxes = np.concatenate([xc, yc, w, h], axis=1)
    if is_1d:
        bboxes = bboxes.flatten()
    return bboxes


def xywh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """
    in xywh:
        x, y are center coordinates
        w, h are full width and full height
    in xyxy:
        xy__ is the top left coordinate
        __xy is the top right coordinate
    """
    is_1d = False
    if len(bboxes.shape) == 1:
        is_1d = True
        bboxes = np.expand_dims(bboxes, axis=0)
    x0 = bboxes[:, 0:1] - bboxes[:, 2:3]/2
    x1 = x0 + bboxes[:, 2:3]
    y0 = bboxes[:, 1:2] - bboxes[:, 3:4]/2
    y1 = y0 + bboxes[:, 3:4]
    bboxes = np.concatenate([x0, y0, x1, y1], axis=1)
    if is_1d:
        bboxes = bboxes.flatten()
    return bboxes


def scale_bboxes(bboxes: np.ndarray, f, img_shape: Sequence):
    """
    centerwise scaling of bboxes, clipping everything to stay within image bounds
    bbox should be a list or a numpy array
    assumes bboxes are in xyxy format
    f is in xy fromat
    """
    is_1d = False
    if len(bboxes.shape) == 1:
        is_1d = True
        bboxes = np.expand_dims(bboxes, axis=0)
    bboxes = xyxy_to_xywh(bboxes)
    bboxes[:, 2:] = np.round(bboxes[:, 2:] * f)
    bboxes = xywh_to_xyxy(bboxes)
    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, img_shape[1])
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, img_shape[0])
    if is_1d:
        bboxes = bboxes.flatten()
    return bboxes


@adapt_to_dims
def norm_to_abs(bboxes: np.ndarray, img_shape: Sequence, inplace:bool=False) -> np.ndarray:
    """
    Convert from normalised bounding boxes (all coordinates in [0, 1] normalised
    to image dimensions) to absolute bounding boxes
    """
    if not inplace:
        bboxes = bboxes.copy()
    bboxes[:, [0,2]] = np.round(bboxes[:, [0,2]] * img_shape[1])
    bboxes[:, [1,3]] = np.round(bboxes[:, [1,3]] * img_shape[0])
    return bboxes


@adapt_to_dims
def abs_to_norm(bboxes: np.ndarray, img_shape: Sequence, inplace:bool=False) -> np.ndarray:
    """
    convert from absolute bounding boxes (coordinate measured in array indices)
     to normalised bounding boxes (all coordinates in [0, 1] normalised to image
     dimensions)
    """
    if not inplace:
        bboxes = bboxes.astype(float)
    else:
        assert 'int' not in bboxes.dtype, "Input array is an int type. Output" \
            + " needs to be a float type but you have set `inplace` to `True`"
    bboxes[:, [0,2]] = bboxes[:, [0,2]] / img_shape[1]
    bboxes[:, [1,3]] = bboxes[:, [1,3]] / img_shape[0]
    return bboxes


@jit(nopython=True)
def get_intersection(A: Sequence, B: Sequence) -> float:
    """
    expect A and B to be xyxy bboxes
    """
    vertical_intersection = max(0, min(A[3], B[3]) - max(A[1], B[1]))
    horizontal_intersection = max(0, min(A[2], B[2]) - max(A[0], B[0]))
    return horizontal_intersection * vertical_intersection


@jit(nopython=True)
def get_iou(A: Sequence, B: Sequence) -> float:
    """ Intersection over union
    Expect A and B to be xyxy bboxes
    """
    area_inter = get_intersection(A, B)

    if area_inter == 0:
        return 0

    area_A = (A[2] - A[0]) * (A[3] - A[1])
    area_B = (B[2] - B[0]) * (B[3] - B[1])

    iou = area_inter / float(area_A + area_B - area_inter)

    return iou


@jit(nopython=True)
def get_viou(A: Sequence, B: Sequence) -> float:
    """ Vertical intersection over union
    Expect A and B to be xyxy bboxes
    """
    v_inter = max(0, min(A[3], B[3]) - max(A[1], B[1]))
    viou = v_inter / float(A[3] - A[1] + B[3] - B[1] - v_inter)
    return viou


@jit(nopython=True)
def get_hiou(A: Sequence, B: Sequence) -> float:
    """ Horizontal intersection over union
    Expect A and B to be xyxy bboxes
    """
    h_inter = max(0, min(A[2], B[2]) - max(A[0], B[0]))
    viou = h_inter / float(A[2] - A[0] + B[2] - B[0] - h_inter)
    return viou


@jit(nopython=True)
def get_giou(A: Sequence, B: Sequence) -> float:
    """ Generalized intersection over union
    Following notation from https://arxiv.org/pdf/1902.09630.pdf
    Expect A and B to be xyxy bboxes
    """
    # First get iou and union
    area_interAB = get_intersection(A, B)
    area_A = (A[2] - A[0]) * (A[3] - A[1])
    area_B = (B[2] - B[0]) * (B[3] - B[1])
    area_unionAB = float(area_A + area_B - area_interAB)
    iou = area_interAB / area_unionAB

    # Box that contains both A and B
    C = [min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3])]

    area_C = (C[2] - C[0]) * (C[3] - C[1])
    giou = iou - (area_C - area_unionAB) /area_C
    return giou


def combine_bboxes(
        bboxes: np.ndarray, metric: Callable[[Sequence, Sequence], float],
        thresh: float, merge_method='union') -> np.ndarray:
    """
    Takes a set of bounding boxes in xyxy format
    Calculates the metric pairwise among all boxes.
    Uses agglomerative clustering to merge bboxes according to the threshold
    distance.
    """
    agglom = AgglomerativeClustering(
        n_clusters=None, distance_threshold=thresh, affinity='precomputed',
        linkage='single')
    merge_methods = ['union']
    assert merge_method in merge_methods, 'Provided `merge_method` is not implemented'
    affinity = squareform(pdist(bboxes, metric))
    labels = agglom.fit_predict(affinity)
    combined_bboxes = []
    for label in np.unique(labels):
        if merge_method == 'union':
            combined_bboxes.append(bounding_bbox(bboxes[labels == label]))
        else:
            assert False, 'Provided `merge_method` is not implemented'
    return np.stack(combined_bboxes, 0)