'''Confusion-count metrics with explicit empty and ignored-sample handling.'''

import numpy as np


def validate_metric_options(aggregation, empty_policy):
    if aggregation not in ("per_image", "dataset"):
        raise ValueError("metric_aggregation must be per_image or dataset.")
    if empty_policy not in ("exclude", "zero", "one"):
        raise ValueError("empty_policy must be exclude, zero, or one.")


class MetricAccumulator:
    def __init__(
        self,
        num_classes=1,
        metrics=("dice", "iou"),
        include_background=False,
        ignore_index=None,
        empty_policy="exclude",
        class_names=None,
        aggregation="per_image",
    ):
        validate_metric_options(aggregation, empty_policy)
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer.")
        if not isinstance(include_background, bool):
            raise ValueError("include_background must be boolean.")
        self.num_classes = num_classes
        self.metrics = list(metrics)
        if set(self.metrics) - {"dice", "iou", "jaccard", "precision", "recall"}:
            raise ValueError("Supported metrics are dice, iou, jaccard, precision, recall.")
        self.class_ids = [1] if num_classes == 1 else list(range(0 if include_background else 1, num_classes))
        self.names = {
            c: class_names[c] if class_names and c < len(class_names) else f"class_{c}"
            for c in self.class_ids
        }
        if len(set(self.names.values())) != len(self.names):
            raise ValueError("Class names must be unique.")
        self.ignore_index, self.empty_policy, self.aggregation = ignore_index, empty_policy, aggregation
        self.all_class_ids = [0, 1] if num_classes == 1 else list(range(num_classes))
        self.counts = {c: np.zeros(3, dtype=np.int64) for c in self.all_class_ids}
        self.support = {
            c: dict(pixels=0, predicted_pixels=0, samples=0, predicted_samples=0, empty_samples=0)
            for c in self.all_class_ids
        }
        self.rows = []
        self.valid_pixels = 0
        self.ignored_samples = 0
        self.empty_masks = 0

    def _scores(self, counts, valid=True):
        result = {}
        for c in self.class_ids:
            tp, fp, fn = counts[c]
            empty = tp + fp + fn == 0
            for metric in self.metrics:
                if not valid or (empty and self.empty_policy == "exclude"):
                    value = None
                elif empty:
                    value = 1.0 if self.empty_policy == "one" else 0.0
                else:
                    numerator, denominator = {
                        "dice": (2 * tp, 2 * tp + fp + fn),
                        "iou": (tp, tp + fp + fn),
                        "jaccard": (tp, tp + fp + fn),
                        "precision": (tp, tp + fp),
                        "recall": (tp, tp + fn),
                    }[metric]
                    value = float(numerator / denominator) if denominator else 0.0
                key = metric if self.num_classes == 1 else f"{metric}_{self.names[c]}"
                result[key] = value
        if self.num_classes > 1:
            for metric in self.metrics:
                values = [result[f"{metric}_{self.names[c]}"] for c in self.class_ids]
                values = [v for v in values if v is not None]
                result[f"macro_{metric}"] = float(np.mean(values)) if values else None
        return result

    def add(self, prediction, target):
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.shape != target.shape or target.ndim != 2:
            raise ValueError("Prediction and target must have matching HxW shapes.")
        valid = (
            target != self.ignore_index
            if self.ignore_index is not None
            else np.ones(target.shape, dtype=bool)
        )
        pred, truth = prediction[valid], target[valid]
        if not np.isfinite(pred).all() or not np.isfinite(truth).all():
            raise ValueError("Labels must be finite.")
        if self.num_classes > 1 and (
            np.any(truth < 0)
            or np.any(truth >= self.num_classes)
            or np.any(pred < 0)
            or np.any(pred >= self.num_classes)
            or np.any(truth != np.floor(truth))
            or np.any(pred != np.floor(pred))
        ):
            raise ValueError("Labels must be integer class IDs within num_classes.")
        if self.num_classes == 1:
            if np.any(truth < 0) or np.any(pred < 0):
                raise ValueError("Binary labels must be nonnegative, except ignore_index.")
            pred, truth = pred > 0, truth > 0
        self.valid_pixels += int(valid.sum())
        self.ignored_samples += int(not truth.size)
        self.empty_masks += int(bool(truth.size) and not np.any(truth > 0))
        counts = {}
        for c in self.all_class_ids:
            p, t = pred == c, truth == c
            counts[c] = np.asarray(
                [np.count_nonzero(p & t), np.count_nonzero(p & ~t), np.count_nonzero(~p & t)], dtype=np.int64
            )
            self.counts[c] += counts[c]
            support = self.support[c]
            support["pixels"] += int(t.sum())
            support["predicted_pixels"] += int(p.sum())
            support["samples"] += int(t.any())
            support["predicted_samples"] += int(p.any())
            support["empty_samples"] += int(bool(truth.size) and not (p.any() or t.any()))
        scores = self._scores(counts, valid=bool(truth.size))
        self.rows.append(scores)
        return scores

    def summary(self):
        keys = self._scores(self.counts).keys()
        mean = {}
        for key in keys:
            values = [r[key] for r in self.rows if r[key] is not None]
            mean[key] = float(np.mean(values)) if values else None
        dataset = self._scores(self.counts, valid=self.valid_pixels > 0)
        return dict(
            num_samples=len(self.rows),
            num_valid_samples=len(self.rows) - self.ignored_samples,
            ignored_only_samples=self.ignored_samples,
            empty_masks=self.empty_masks,
            valid_pixels=self.valid_pixels,
            aggregation=self.aggregation,
            empty_policy=self.empty_policy,
            mean=mean,
            dataset=dataset,
            scores=mean if self.aggregation == "per_image" else dataset,
            class_support={
                str(c): {
                    **self.support[c],
                    "true_positive": int(self.counts[c][0]),
                    "false_positive": int(self.counts[c][1]),
                    "false_negative": int(self.counts[c][2]),
                }
                for c in self.all_class_ids
            },
        )
