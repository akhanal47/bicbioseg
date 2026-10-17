import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch

from ..exceptions import InferenceError
from ..utils.evaluation import MetricAccumulator
from ..utils.progress import progress_iter


def _match_prediction_mask_pairs(predictions, masks):
    from ..trainer import Segmenter

    prediction_paths = [Path(path) for path in Segmenter._load_paths_static(predictions)]
    mask_paths = [Path(path) for path in Segmenter._load_paths_static(masks)]
    masks_by_stem = {}
    for path in mask_paths:
        if path.stem in masks_by_stem:
            raise InferenceError(f"Duplicate ground-truth mask stem: {path.stem}")
        masks_by_stem[path.stem] = path
    pairs = []
    for path in prediction_paths:
        # strip only the suffix written by inference, never internal '_mask'.
        stem = path.stem.removesuffix("_mask")
        mask = masks_by_stem.get(stem, masks_by_stem.get(path.stem))
        if mask is None:
            raise InferenceError(f"No ground-truth mask matches prediction: {path.name}")
        pairs.append((path, mask))
    if not pairs:
        raise InferenceError("No prediction/mask pairs to evaluate.")
    return pairs


def _score_prediction_arrays(
    pred_np,
    mask_np,
    metrics,
    num_classes=None,
    class_names=None,
    include_background=False,
    ignore_index=None,
    empty_policy="exclude",
):
    if pred_np.shape != mask_np.shape:
        pred_np = cv2.resize(
            pred_np, (mask_np.shape[1], mask_np.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    if num_classes is None:
        values = set(np.unique(pred_np)) | set(np.unique(mask_np))
        values.discard(ignore_index)
        num_classes = 1 if values.issubset({0, 1, 255}) else int(max(values, default=0)) + 1
    accumulator = MetricAccumulator(
        num_classes, metrics, include_background, ignore_index, empty_policy, class_names
    )
    return accumulator.add(pred_np, mask_np)


def evaluate_predictions(
    predictions,
    masks,
    metrics=("dice", "iou", "precision", "recall"),
    save_to=None,
    num_classes=None,
    class_names=None,
    include_background=False,
    ignore_index=None,
    empty_policy="exclude",
    aggregation="per_image",
):
    from ..trainer import Segmenter

    pairs = Segmenter._match_prediction_mask_pairs(predictions, masks)
    # Infer once for the complete dataset, so class absence cannot change task type per image.
    if num_classes is None:
        values = set()
        for pair in pairs:
            for path in pair:
                array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if array is None:
                    raise InferenceError(f"Could not read {path}")
                values.update(np.unique(array).tolist())
        values.discard(ignore_index)
        num_classes = 1 if values.issubset({0, 1, 255}) else int(max(values, default=0)) + 1
    accumulator = MetricAccumulator(
        num_classes, metrics, include_background, ignore_index, empty_policy, class_names, aggregation
    )
    rows = []
    for prediction_path, mask_path in pairs:
        prediction = cv2.imread(str(prediction_path), cv2.IMREAD_UNCHANGED)
        target = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if prediction is None or target is None:
            raise InferenceError(f"Could not read {prediction_path} or {mask_path}")
        if prediction.shape != target.shape:
            prediction = cv2.resize(
                prediction, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        scores = accumulator.add(prediction, target)
        valid = target != ignore_index if ignore_index is not None else np.ones(target.shape, dtype=bool)
        rows.append(
            {
                "prediction": Path(prediction_path).name,
                "mask": Path(mask_path).name,
                "valid_pixels": int(valid.sum()),
                "ignored_only": not bool(valid.any()),
                **scores,
            }
        )
    summary = {
        **accumulator.summary(),
        "rows": rows,
        "num_classes": num_classes,
        "include_background": include_background,
        "ignore_index": ignore_index,
        "class_names": class_names,
    }
    if save_to is not None:
        output_dir = Path(save_to)
        output_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            with (output_dir / "evaluation.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
                writer.writeheader()
                writer.writerows(rows)
        Segmenter._write_json(output_dir / "evaluation_summary.json", summary)
    return summary


def find_best_threshold(
    self,
    images,
    masks,
    thresholds: Optional[Sequence[float]] = None,
    metric: str = "dice",
    save_to: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, object]:
    if self.num_classes != 1:
        raise ValueError("Threshold tuning is only supported for binary segmentation.")

    thresholds = list(np.linspace(0.1, 0.9, 17) if thresholds is None else thresholds)
    if not thresholds:
        raise ValueError("Provide at least one threshold.")
    image_paths = self._load_inference_images(images)
    mask_pairs = self._match_prediction_mask_pairs(image_paths, masks)
    mask_by_stem = {Path(mask).stem.replace("_mask", ""): mask for _, mask in mask_pairs}
    rows = []

    self.model.eval()
    with torch.no_grad():
        probabilities = []
        ground_truths = []
        for image_path in progress_iter(image_paths, desc="Threshold tuning"):
            stem = Path(image_path).stem.replace("_mask", "")
            if stem not in mask_by_stem:
                continue
            _, _, probability, _ = self._predict_single_image(image_path, return_raw=True)
            mask = cv2.imread(str(mask_by_stem[stem]), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            if probability.shape != mask.shape:
                probability = cv2.resize(probability, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC)
            probabilities.append(probability)
            ground_truths.append(mask)

    if not probabilities:
        raise ValueError("No matched image/mask pairs found for threshold tuning.")

    best_threshold = None
    best_score = None
    metric_key = metric.lower()
    if metric_key not in {"dice", "iou", "jaccard"}:
        raise ValueError("metric must be dice, iou, or jaccard.")
    for threshold in thresholds:
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("thresholds must lie in [0, 1].")
        accumulator = MetricAccumulator(
            1,
            (metric_key,),
            ignore_index=self.ignore_index,
            empty_policy=self.empty_policy,
            aggregation=self.metric_aggregation,
        )
        for probability, target in zip(probabilities, ground_truths):
            accumulator.add((probability > threshold).astype(np.uint8), target)
        mean_score = accumulator.summary()["scores"][metric_key]
        rows.append({"threshold": float(threshold), metric_key: mean_score})
        if mean_score is not None and (best_score is None or mean_score > best_score):
            best_score = mean_score
            best_threshold = float(threshold)

    result = {
        "best_threshold": best_threshold,
        "best_score": best_score,
        "metric": metric_key,
        "rows": rows,
    }

    if save_to is not None:
        output_dir = Path(save_to)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "threshold_tuning.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["threshold", metric_key])
            writer.writeheader()
            writer.writerows(rows)
        self._write_json(output_dir / "threshold_tuning.json", result)

    return result


def evaluate(
    self,
    images,
    masks,
    save_to: Union[str, os.PathLike] = "evaluation",
    prediction_dir: Optional[Union[str, os.PathLike]] = None,
    metrics: Sequence[str] = ("dice", "iou", "precision", "recall"),
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    include_background: Optional[bool] = None,
    empty_policy: Optional[str] = None,
    aggregation: Optional[str] = None,
    split: Optional[str] = None,
    batch_size: int = 1,
) -> Dict[str, object]:
    prediction_dir = prediction_dir or Path(save_to) / "predictions"
    predictions = self.inference(
        images=images,
        save_to=prediction_dir,
        threshold=threshold,
        batch_size=batch_size,
        save_overlay=True,
    )
    result = self.evaluate_predictions(
        predictions,
        masks,
        metrics=metrics,
        save_to=save_to,
        num_classes=self.num_classes if num_classes is None else num_classes,
        class_names=class_names or [self.class_map[i] for i in sorted(self.class_map)],
        include_background=self.include_background if include_background is None else include_background,
        empty_policy=empty_policy or self.empty_policy,
        aggregation=aggregation or self.metric_aggregation,
        ignore_index=self.ignore_index,
    )

    result["provenance"] = {
        **self._prediction_metadata(threshold),
        "split": split,
        "images": str(images),
        "masks": str(masks),
        "prediction_dir": str(Path(prediction_dir).resolve()),
    }
    self._write_json(Path(save_to) / "evaluation_summary.json", result)
    return result


def find_worst_predictions(
    predictions,
    masks,
    metric: str = "dice",
    top_k: int = 10,
    save_to: Optional[Union[str, os.PathLike]] = None,
    images=None,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    include_background: bool = False,
    ignore_index: Optional[int] = None,
) -> List[Dict[str, object]]:
    from ..trainer import Segmenter

    evaluation = Segmenter.evaluate_predictions(
        predictions=predictions,
        masks=masks,
        metrics=(metric,),
        num_classes=num_classes,
        class_names=class_names,
        include_background=include_background,
        ignore_index=ignore_index,
    )
    rows = evaluation["rows"]
    metric_key = metric.lower()
    if rows and metric_key not in rows[0]:
        metric_candidates = [key for key in rows[0] if key == f"macro_{metric_key}" or key.startswith(f"{metric_key}_")]
        metric_key = f"macro_{metric_key}" if f"macro_{metric_key}" in metric_candidates else metric_candidates[0]

    worst = sorted(
        (row for row in rows if row.get(metric_key) is not None), key=lambda row: row[metric_key]
    )[:top_k]

    if save_to is not None:
        output_dir = Path(save_to)
        output_dir.mkdir(parents=True, exist_ok=True)
        if worst:
            with (output_dir / "worst_predictions.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({key for row in worst for key in row}))
                writer.writeheader()
                writer.writerows(worst)

        prediction_pairs = Segmenter._match_prediction_mask_pairs(predictions, masks)
        pred_by_name = {Path(pred).name: Path(pred) for pred, _ in prediction_pairs}
        mask_by_name = {Path(mask).name: Path(mask) for _, mask in prediction_pairs}
        image_by_stem = {}
        if images is not None:
            image_paths = Segmenter._load_paths_static(images)
            image_by_stem = {Path(path).stem: Path(path) for path in image_paths}

        for rank, row in enumerate(worst, start=1):
            case_dir = output_dir / f"rank_{rank}_{Path(row['prediction']).stem}"
            case_dir.mkdir(parents=True, exist_ok=True)
            if row["prediction"] in pred_by_name:
                shutil.copy2(pred_by_name[row["prediction"]], case_dir / row["prediction"])
            if row["mask"] in mask_by_name:
                shutil.copy2(mask_by_name[row["mask"]], case_dir / row["mask"])
            stem = Path(row["prediction"]).stem.replace("_mask", "")
            if stem in image_by_stem:
                shutil.copy2(image_by_stem[stem], case_dir / image_by_stem[stem].name)

    return worst
