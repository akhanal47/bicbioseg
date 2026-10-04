# Evaluate predictions

[Documentation](README.md) · [Main guide](../README.md)

Use validation data to select settings. Use the test split for the final evaluation.

Evaluate predictions against ground truth:

```python
results = model.evaluate(
    images="cell_dataset/test/images",
    masks="cell_dataset/test/masks",
    save_to="evaluation",
)
```

Multi-class evaluation:

```python
results = Segmenter.evaluate_predictions(
    predictions="predictions",
    masks="test/masks",
    metrics=["dice", "iou"],
    num_classes=3,
    class_names=["background", "nucleus", "cytoplasm"],
)
```

Find low-performing examples:

```python
worst = Segmenter.find_worst_predictions(
    predictions="predictions",
    masks="test/masks",
    images="test/images",
    metric="dice",
    top_k=10,
    save_to="failure_cases",
)
```

Tune a binary threshold:

```python
model.find_best_threshold(
    images="cell_dataset/validate/images",
    masks="cell_dataset/validate/masks",
    metric="dice",
    save_to="threshold_tuning",
)
```

## Evaluation settings

`evaluate` predicts masks and compares them with labels.
`evaluate_predictions` scores existing saved masks.
Both default to `metrics=("dice", "iou", "precision", "recall")` and `include_background=False`.
Set `num_classes` and `class_names` explicitly for multiclass masks. List names in class ID order.
Multiclass output includes per-class scores and macro averages.

`evaluate` accepts `threshold=0.5`, `save_to="evaluation"`, and `prediction_dir=None`.
With no prediction directory, it saves masks under the evaluation folder.
`evaluate_predictions` accepts `save_to=None` and `ignore_index=None`.
Instance `evaluate` uses the model's ignored label automatically.

`find_worst_predictions` accepts the same class and ignored-label settings.
Its defaults are `metric="dice"`, `top_k=10`, `save_to=None`, and `images=None`.
Provide source images to save visual examples.

`find_best_threshold` is binary-only. Supply `thresholds` to replace its 17 cutoffs from 0.1 to 0.9.
Pass the selected threshold explicitly to subsequent inference or evaluation calls.

## Empty masks, ignored pixels, and aggregation

Use `empty_policy="exclude"` (default), `"zero"`, or `"one"` for a class absent from both prediction and target.
False-positive predictions on an empty target receive zero overlap scores. They are not excluded.
Images containing only ignored labels always receive undefined scores and never contribute to averages.
Undefined scores appear as `None` in Python, `null` in JSON, and empty cells in CSV.

Use `aggregation="per_image"` to average defined image scores.
Use `aggregation="dataset"` to pool true positives, false positives, and false negatives before calculating scores.
The result always includes both `mean` and `dataset`. Its `scores` field contains the selected aggregation.

```python
from bicbioseg import Segmenter

result = Segmenter.evaluate_predictions(
    "predictions", "cell_dataset/test/masks",
    num_classes=3, ignore_index=255, include_background=False,
    aggregation="dataset", empty_policy="exclude",
)
print(result["scores"])
print(result["class_support"])
print(result["ignored_only_samples"])
```

`class_support` contains pixel counts, sample counts, and confusion counts for every class, including background.
`include_background` controls multiclass score averages. Binary scores describe foreground.
A class absent from the target but predicted by the model still contributes its false positives.
Training uses the same rules through `Segmenter(metric_aggregation=..., empty_policy=..., include_background=...)`.

Evaluation through a model saves provenance: checkpoint, epoch, split, threshold, class names, and prediction directory.
`evaluate` accepts `batch_size` for folder prediction and `split` for the report label.
`exp.evaluate(checkpoint="best", split="test")` selects a saved checkpoint without replacing the current model.
Threshold tuning and failure mining exclude undefined scores. If no threshold has a defined score, its selected threshold is `None`.
