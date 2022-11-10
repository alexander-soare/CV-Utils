from typing import Sequence, Union, Optional, Tuple, List
import warnings

import numpy as np
from rasterio.features import rasterize
from shapely.geometry import Polygon
import shapely
import cv2

try:
    import pycocotools.mask as mask_util
    has_pycocotools = True
except ImportError:
    has_pycocotools = False

from .bbox_utils import bbox_crop
from ._helpers import adapt_to_dims, to_2tuple
from .render import draw_polygons  # BC


def mask_to_polys(mask: np.ndarray) -> List[np.ndarray]:
    """
    Takes a binary mask and return the polygons within it. Does not yet handle interior polys, ie mask regions must
    not have holes in them (an error will be raised if they do).
    Polys are returned in absolute xy... format
    """
    if mask.sum() == 0:  # Empty mask returns empty list and raises warning
        warnings.warn("WARNING (in mask_to_polys): An empty mask was given.")
        return []
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Check that none of the contours have parents
    if np.all(hier[..., -1] == -1):
        # Clue here https://github.com/facebookresearch/detectron2/blob/e3ed623d0c7bddeb01a84ecad85083c09e596d83/detectron2/utils/visualizer.py#L119
        # as to how to do this, but even if I do it, might not make much sense unless there's a way to make use of it
        # downstream. In the current representation (list of np.ndarrays) there's nothing to say which polygons
        # represent holes
        warnings.warn(
            "WARNING (in mask_to_polys): There are holes present in the mask. These will be converted to polygons but "
            "the fact that they are holes will be lost.")
    return [cnt.flatten() for cnt in cnts]


def polys_to_mask(polys: Union[List[np.ndarray], np.ndarray], out_shape: Tuple[int, int]) -> np.ndarray:
    """
    Take a list of polys and convert them to a binary mask with `out_shape`. A single poly may also be provided.
    NOTE: This does not yet support polys within polys (holes), so `polys` should not contain any overlapping polys.
    """
    if not isinstance(polys, list):
        polys = [polys]
    if has_pycocotools:  # I think this is way faster
        rles = mask_util.frPyObjects(polys, *out_shape[:2])
        rle = mask_util.merge(rles)
        return mask_util.decode(rle).astype(np.bool)
    else:
        warnings.warn("Using rasterio but pycocotools could be much faster. Install it with `pip install pycocotools`")
        sh_polys = [Polygon(poly.reshape(-1, 2)) for poly in polys]
        mask = rasterize(sh_polys, out_shape=out_shape)
        return mask


def poly_crop(img: np.ndarray, poly: np.ndarray, mask_val: Union[int, Sequence, None] = None) -> np.ndarray:
    """
    Gets a square crop around a provided polygon
    If `mask_val` is provided, the area not covered by the polygon is masked away
    with the color represented by masks. `mask_val` should be a tuple if the 
    image has multiple channels or an integer if the image is grayscale
    """
    if mask_val is not None:
        mask = polys_to_mask(poly)
        img = img.copy()
        img[mask == 0] *= 0
        img[mask == 0] = np.array(mask_val)
    bbox = poly_to_bbox(poly)
    return bbox_crop(img, bbox)


def scale_poly(
        poly: np.ndarray, f: Union[float, Tuple[float, float]], img_shape: Optional[Sequence] = None,
        shapely_origin: str = 'centroid') -> np.ndarray:
    """
    Take an xyxyxyxy polygon and scale it (centerwise) by factor f.
    f may be a tuple in which case the first component refers to x and the
    second to y.
    img_shape may be provided to make sure we clip the result to the image
    shape. NOTE that this would change the shape of the polygon.
    shapely_origin is the origin kwarg for shapely.affinity.scale
    """
    f = to_2tuple(f)
    sh_poly = Polygon(poly.reshape(-1, 2))
    sh_poly = shapely.affinity.scale(
        sh_poly, xfact=f[0], yfact=f[1], origin=shapely_origin)
    poly = np.array(*[sh_poly.exterior.xy]).T.flatten()[:8]
    if img_shape is not None:
        poly[0::2] = np.clip(poly[0::2], 0, img_shape[1])
        poly[1::2] = np.clip(poly[1::2], 0, img_shape[0])
    return poly


@adapt_to_dims
def bbox_to_poly(bboxes: np.ndarray) -> np.ndarray:
    """
    Expects bboxes in xyxy format. Turns each into a 1D array with 8 entries,
    every consecutive pair being for one vertex (starting from top left and
    going around clockwise)
    Works with single bboxes (shape is (4, )) or multiple bboxes (shape is
    (N, 4)).
    """
    polys = np.concatenate([bboxes[:, :2], bboxes[:, 0:1], bboxes[:, 3:4],
                            bboxes[:, 2:], bboxes[:, 2:3], bboxes[:, 1:2]],
                           axis=1)
    return polys


def poly_to_bbox(poly: np.ndarray) -> np.ndarray:
    """
    Expects poly in xy sequence format. Only works for a single poly at a time
    Turns each into the smallest axis-aligned bbox returned in xyxy format
    """
    bbox = np.array([poly[0::2].min(), poly[1::2].min(),
                     poly[0::2].max(), poly[1::2].max()])
    return bbox


@adapt_to_dims
def norm_to_abs(polys: np.ndarray, img_shape: Sequence, inplace:bool=False) -> np.ndarray:
    """
    Convert from normalised polygons (all coordinates in [0, 1] normalised
    to image dimensions) to absolute polygons
    Works on 1D and 2D inputs
    """
    if not inplace:
        polys = polys.copy()
    polys[:, 0::2] *= img_shape[1]
    polys[:, 1::2] *= img_shape[0]
    return polys


@adapt_to_dims
def abs_to_norm(polys: np.ndarray, img_shape: Sequence, inplace:bool=False) -> np.ndarray:
    """
    Convert from absolute polygons (coordinate measured in array indices)
     to normalised polygons (all coordinates in [0, 1] normalised to image
     dimensions)
    Works on 1D and 2D inputs
    """
    if not inplace:
        polys = polys.astype(float)
    else:
        assert 'int' not in polys.dtype, "Input array is an int type. Output" \
            + " needs to be a float type but you have set `inplace` to `True`"
    polys[:, 0::2] /= img_shape[1]
    polys[:, 1::2] /= img_shape[0]
    return polys
