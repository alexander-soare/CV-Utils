from typing import Union, Sequence, Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .bbox_utils import get_iou


def compute_tp_fp_fn(
        pred_bboxes: np.ndarray,
        pred_confs: np.ndarray,
        gt_bboxes: np.ndarray,
        conf_threshold: float,
        iou_thresholds: Union[Sequence[float], float]) -> Dict[float, Dict[str, int]]:
    """
    Computes the number of true positives (tp), false positives (fp), and
    (fn) false negatives at the given IoU thresholds for a single image and
    single category.
    """
    # Make sure iou_thresholds is a sequence
    if not isinstance(iou_thresholds, Sequence):
        iou_thresholds = [iou_thresholds]

    # Filter by confidence threshold
    filt = (pred_confs >= conf_threshold)
    pred_bboxes = pred_bboxes[filt]
    pred_confs = pred_confs[filt]

    # Check for trivial answer
    if not len(pred_bboxes) or not len(gt_bboxes):
        return {iou_thresh: {
            'tp': 0,
            'fp': len(pred_bboxes),
            'fn': len(gt_bboxes)
        } for iou_thresh in iou_thresholds}

    # Sort pred_bboxes by confidence
    pred_bboxes = pred_bboxes[pred_confs.argsort()[::-1]]

    # Compute iou_matrix (output is len(preds) x len(gts) matrix)
    iou_matrix = cdist(pred_bboxes, gt_bboxes, metric=get_iou)

    res = {}
    for iou_thresh in iou_thresholds:
        tp = 0
        fp = 0
        fn = 0
        # Keep track of which gt_bboxes are still in the running
        gt_mask = np.ones(len(gt_bboxes)).astype(bool)
        gt_indices = np.arange(len(gt_bboxes))
        for pred_ix in range(len(pred_bboxes)):
            if gt_mask.sum() == 0:  # no gt left to match
                # So whatever is left for the predictions counts as a FP
                fp += 1
                continue
            argmax = iou_matrix[pred_ix][gt_mask].argmax()
            best_match_gt_ix = gt_indices[gt_mask][argmax]
            best_match_iou = iou_matrix[pred_ix][gt_mask][argmax]
            if best_match_iou >= iou_thresh:
                tp += 1
                # Take the matched ground truth out of the running
                gt_mask[best_match_gt_ix] = False
            else:
                # FP: pred_bbox has no associate gt_bbox
                fp += 1
        # FN: indicates a gt box had no associated predicted box.
        fn = gt_mask.sum()
        # Report result for the given iou_thresh
        res[iou_thresh] = {'tp': tp, 'fp': fp, 'fn': fn}

    return res


def compute_precision_recall_f1(
        pred_bboxes: List[np.ndarray],
        pred_confs: List[np.ndarray],
        gt_bboxes: List[np.ndarray],
        conf_threshold: float,
        iou_thresholds: Union[Sequence[float], float]) -> Tuple[Dict[float, float],
                                                                Dict[float, float],
                                                                Dict[float, float]]:
    """
    Run `compute_tp_fp_fn` for multiple images and return the final precision
    and recall (aggregated over the lot)
    """
    assert len(pred_bboxes) == len(pred_confs) and len(pred_confs) == len(gt_bboxes), \
        "`pred_bboxes`, `pred_confs`, and `gt_bboxes` should all be lists fo the same length"

    # Make sure iou_thresholds is a sequence
    if not isinstance(iou_thresholds, Sequence):
        iou_thresholds = [iou_thresholds]

    # Keep tp, fp, and fn at various iou thresholds
    res = {th: {'tp': 0, 'fp': 0, 'fn': 0} for th in iou_thresholds}
    for pb, pc, gt in zip(pred_bboxes, pred_confs, gt_bboxes):
        r = compute_tp_fp_fn(pb, pc, gt, conf_threshold=conf_threshold,
                             iou_thresholds=iou_thresholds)
        for th in iou_thresholds:
            res[th]['tp'] += r[th]['tp']
            res[th]['fp'] += r[th]['fp']
            res[th]['fn'] += r[th]['fn']

    # Keep track of precision, recall and f1 at each iou threshold
    precision = {}
    recall = {}
    f1 = {}
    for th in iou_thresholds:
        tp, fp, fn = res[th]['tp'], res[th]['fp'], res[th]['fn']
        n_pred = tp + fp
        n_gt = tp + fn
        # Check for trivial answers
        if n_pred * n_gt == 0:
            precision[th] = 0
            recall[th] = 0
        else:
            # Non-trivial answer
            precision[th] = tp / n_pred
            recall[th] = tp / n_gt
        if precision[th] * recall[th] != 0:
            f1[th] = 2 / ((1 / precision[th]) + (1 / recall[th]))
        else:
            f1[th] = 0

    return precision, recall, f1
