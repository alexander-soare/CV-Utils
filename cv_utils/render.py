from typing import Sequence
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt

from ._helpers import sort_vertices


def show_images(ls_img, titles=[], imsize=(7, 5), cmap=None, per_row=2,
                keep_ticks=False, font_size=16):
    """makes a figure with enough subplots to show the images of `ls_img`
    """                
    # make sure ls_img is a list
    if not isinstance(ls_img, Sequence):
        ls_img = [ls_img]

    # make sure titles is a list
    if not isinstance(titles, Sequence):
        titles = [titles]

    # make sure titles is same length as ls_img
    if len(titles):
        assert len(titles) == len(
            ls_img), "Please provide as many titles as there are images"
    else:
        titles = [''] * len(ls_img)

    # prepare figure
    num_rows = len(ls_img) // per_row + ((len(ls_img) % per_row) > 0)
    fig, ax = plt.subplots(num_rows, per_row, figsize=(
        imsize[0] * per_row, imsize[1] * num_rows))
    if type(ax) == np.ndarray:
        ax = ax.flatten()
    else:
        ax = np.array([ax])

    # populate figure
    for i, img in enumerate(ls_img):
        this_cmap = cmap
        if this_cmap is None and (len(img.shape) == 2 or img.shape[-1] == 1):
            this_cmap = 'gray'
        ax[i].imshow(img, cmap=this_cmap, vmin=0, vmax=255)
        ax[i].set_title(titles[i], fontdict={'fontsize': font_size})
        if not keep_ticks:
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    plt.tight_layout()
    return fig, ax


def _draw_outlines(
        outline_type: str, img: np.ndarray, outlines: Sequence[np.ndarray],
        thickness: int = 2, colors: Sequence[Sequence[int]] = [(0,255,0)],
        labels: Sequence[str] = [], label_font_size: float = 1.) -> np.ndarray:
    """
    Draw either axis-aligned boxes (`outline_type == 'aab'`) or polygons
    (`outline_type == 'poly'`). The boxes would be expected in xyxy format and
    the polygons would be expected in xy... format.
    Both are expected to be in absolute co-ordinates relative to the image and
    with int dtype.
    `colors` may be a single color tuple, or a list of tuples, one for each
    outline
    """
    # Handle single color tuple of colors
    if len(colors) == 1 and len(outlines) > 1:
        colors = colors*len(outlines)
    if len(colors) == 3 and isinstance(colors[0], int):
        colors = [colors]*len(outlines)
    for i, (color, outline) in enumerate(zip(colors, outlines)):
        if outline_type == 'poly':
            # Sort vertices is just for making sure the label is on the top left
            if len(labels):
                outline = sort_vertices(outline.reshape(-1, 2)).flatten()
            cv2.polylines(
                img, [outline.reshape(-1, 1, 2)], isClosed=True, color=color,
                thickness=thickness)
        elif outline_type == 'aab':
            cv2.rectangle(
                img, (outline[0], outline[1]), (outline[2], outline[3]),
                color, thickness=thickness)
        if len(labels):
            label = f'{i:02d}' if len(labels) == 0 else labels[i]
            cv2.putText(
                img, label, (outline[0], outline[1] - 3), cv2.FONT_HERSHEY_SIMPLEX,
                label_font_size, color, thickness)
    return img


def draw_bboxes(
        img: np.ndarray, bboxes: Sequence[np.ndarray],
        thickness: int = 2, colors: Sequence[Sequence[int]] = [(0,255,0)],
        labeled: bool = False, labels: Sequence[str] = [],
        label_font_size: float = 1.) -> np.ndarray:
    if labeled:
        warnings.warn(
            "Deprecation warning: `labeled` is now deprecated. Instead, if "
            "`labels` has non-zero length, we render labels.")
    return _draw_outlines(
        'aab', img, bboxes, thickness=thickness, colors=colors, labels=labels,
        label_font_size=label_font_size)


def draw_polygons(
        img: np.ndarray, polys: Sequence[np.ndarray],
        thickness: int = 2, colors: Sequence[Sequence[int]] = [(0,255,0)],
        labeled: bool = False, labels: Sequence[str] = [],
        label_font_size: float = 1.) -> np.ndarray:
    if labeled:
        warnings.warn(
            "Deprecation warning: `labeled` is now deprecated. Instead, if "
            "`labels` has non-zero length, we render labels.")
    return _draw_outlines(
        'poly', img, polys, thickness=thickness, colors=colors, labels=labels,
        label_font_size=label_font_size)


def draw_masks(
        img: np.ndarray, masks: Sequence[np.ndarray], alpha=1.,
        colors: Sequence[Sequence[int]] = [(0,255,0)],
        labels: Sequence[str] = [], label_thickness: int = 1,
        label_font_size: float = 1.) -> np.ndarray:
    """
    Draw masks onto image with some level of transparency (alpha)
    All maks should have the same spatial dimensions as the image. Masks
    are expected to be either False/True or 0/1
    """
    img = img.copy()
    # Handle single color tuple of colors
    if len(colors) == 1 and len(masks) > 1:
        colors = colors*len(masks)
    if len(colors) == 3 and isinstance(colors[0], int):
        colors = [colors]*len(masks)
    for color, mask in zip(colors, masks):
        mask = mask.astype(bool)
        color_mask = np.zeros_like(img)
        color_mask[mask] = np.array(color)
        img[mask] = img[mask] * (1 - alpha) + color_mask[mask] * alpha
    if len(labels):
        raise UserWarning("Labels not implemented yet!")
    return img