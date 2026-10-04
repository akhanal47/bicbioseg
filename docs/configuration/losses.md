# Losses and metrics

[All configuration](README.md) · [Training guide](../training.md)

A loss drives training. A metric measures prediction quality.
Set the loss with `loss`, and pass its options through `loss_kwargs`.
All built-in losses consume logits. Do not apply sigmoid or softmax before them.

| Loss name | Aliases | Task | Options and defaults |
| --- | --- | --- | --- |
| `dice` | None | Binary or multiclass | `smooth=1e-6`, automatic `mode`. `weight=None` and `size_average=True` are accepted but unused. |
| `jaccard` | `iou` | Binary or multiclass | Same constructor options as Dice. |
| `bce` | `binary_cross_entropy` | Binary | No configurable loss options. |
| `dice_bce` | `combo` | Binary | `smooth=1e-6` |
| `focal` | None | Binary | `alpha=0.25`, `gamma=2` |
| `log_cosh_dice` | None | Binary | `smooth=1e-6` |
| `tversky` | None | Binary | `alpha=0.3`, `beta=0.7`, `smooth=1e-6` |
| `focal_tversky` | None | Binary | `alpha=0.3`, `beta=0.7`, `gamma=1.25`, `smooth=1e-6` |
| `sensitivity_specificity` | None | Binary | `alpha=0.3`, `smooth=1e-6` |
| `cross_entropy` | None | Multiclass only | PyTorch `CrossEntropyLoss` options. Set ignored labels on `Segmenter`. |

Tversky's `alpha` weights false positives. Its `beta` weights false negatives.
`Segmenter` selects `mode="binary"` or `mode="multiclass"` for Dice and Jaccard from `num_classes`.
An explicit conflicting mode causes an error.

```python
from bicbioseg import Segmenter

model = Segmenter(
    architecture="unet",
    loss="tversky",
    loss_kwargs={"alpha": 0.3, "beta": 0.7},
    metrics=["dice", "iou", "precision", "recall"],
)
print(Segmenter.available_losses())
```

For cross entropy, options include class `weight`, `reduction="mean"`, and `label_smoothing=0.0`.
PyTorch also accepts legacy `size_average=None` and `reduce=None` arguments.
Use a scalar loss for training. Tensor-valued class weights require Python setup and cannot be saved directly as JSON.
Set `ignore_index` on `Segmenter` so losses and metrics exclude the same pixels.

## Custom losses

Direct `Segmenter(loss=...)` accepts a module instance, a module class, or a callable.
`loss_kwargs` applies to classes and built-in names. Instances and callables manage their own settings.
The result must be a scalar suitable for backpropagation.
Unsupported placeholders are `UnifiedFocalLoss`, `ExponentialLogarithmicLoss`, and `ShapeAwareLoss`.
These are not registered loss choices.

## Metrics

Training supports `dice`, `iou`, `precision`, and `recall`. `jaccard` is an alias for `iou`.
Defaults are Dice and IoU. Pass an empty list to disable training metrics.
History keys use `train_` or `val_`, such as `val_dice`.

Evaluation also supports class names and `include_background=False` for multiclass scores.
Use [evaluation](../evaluation.md) to inspect per-image and per-class results.
