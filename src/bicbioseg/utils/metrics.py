from .metrices import F1, F2, dice_score, jac_score, precision, recall

iou_score = jac_score
jaccard_score = jac_score

__all__ = [
    "precision",
    "recall",
    "F1",
    "F2",
    "dice_score",
    "jac_score",
    "iou_score",
    "jaccard_score",
]
