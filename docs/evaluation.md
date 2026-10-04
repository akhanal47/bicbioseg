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
