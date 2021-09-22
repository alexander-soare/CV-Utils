from typing import Sequence, Union, Optional, Tuple

import numpy as np
from rasterio.features import rasterize
from shapely.geometry import Polygon
import shapely

from .bbox_utils import bbox_crop
from ._helpers import adapt_to_dims

# BC
from .render import draw_polygons
from ._helpers import to_2tuple


def poly_crop(img: np.ndarray, poly: np.ndarray,
              mask_val: Union[int, Sequence, None] = None) -> np.ndarray:
    """
    Gets a square crop around a provided polygon
    If `mask_val` is provided, the area not covered by the polygon is masked away
    with the color represented by masks. `mask_val` should be a tuple if the 
    image has multiple channels or an integer if the image is grayscale
    """
    if mask_val is not None:
        sh_poly = Polygon(poly.astype(int).reshape(-1, 2))
        mask = rasterize([sh_poly], out_shape=img.shape[:2])
        img = img.copy()
        img[mask == 0] *= 0
        img[mask == 0] = np.array(mask_val)
    bbox = poly_to_bbox(poly)
    return bbox_crop(img, bbox)


def scale_poly(poly: np.ndarray, f: Union[float, Tuple[float, float]],
               img_shape: Optional[Sequence] = None,
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
    poly = np.array(*[sh_poly.exterior.xy]).T.flatten()
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
