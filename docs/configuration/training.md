# Training settings

[All configuration](README.md) · [Main guide](../../README.md)

## TrainingConfig

Use this configuration with `model.train(data=..., config=config)` or `exp.train(config=config)`.

```python
from bicbioseg import TrainingConfig

config = TrainingConfig(
    epochs=50,
    batch_size=2,
    accumulation_steps=4,
    early_stopping=True,
    patience=10,
    experiment_dir="experiments",
    run_name="baseline",
)
```

## All fields

| Option | Default | Purpose and accepted values |
| --- | --- | --- |
| `epochs` | `1` | Number of epochs for this call. Must be positive. A resumed call adds this many epochs. |
| `batch_size` | `4` | Number of images per loader batch. |
| `lr` | `0.0001` | Optimizer learning rate. |
| `optimizer` | `'adam'` | Choose `adam`, `adamw`, or `sgd`. SGD uses momentum 0.9. |
| `num_workers` | `2` | Loader worker count. Use 0 for a simple notebook or troubleshooting setup. |
| `save_to` | `None` | Optional path for the final checkpoint. |
| `experiment_dir` | `None` | Parent directory for run artifacts. `None` disables the run directory. |
| `run_name` | `None` | Run subdirectory name. When omitted, the trainer creates a name. |
| `save_best` | `True` | Save `best_model.pt` when the monitored value improves and a run directory exists. |
| `monitor` | `'val_loss'` | History key to track, such as `val_loss`, `val_dice`, or `train_loss`. |
| `early_stopping` | `False` | Stop when the monitored value fails to improve for the configured patience. |
| `patience` | `10` | Number of epochs without improvement before early stopping. Must be nonnegative. |
| `min_delta` | `0.0` | Minimum improvement to count as progress. Must be nonnegative. |
| `monitor_mode` | `'auto'` | Choose `auto`, `min`, or `max`. Auto minimizes loss and maximizes other metrics. |
| `resume_from` | `None` | Checkpoint path. Restores model, optimizer, history, and saved training state. |
| `progress_bar` | `True` | Show progress bars. |
| `verbose` | `True` | Print epoch summaries. |
| `precision` | `'fp32'` | Choose `fp32`, `fp16`, or `bf16`, subject to device support below. |
| `accumulation_steps` | `1` | Positive number of microbatches per optimizer update. |
| `max_grad_norm` | `None` | Optional positive finite gradient clipping norm. |
| `scheduler` | `None` | Choose `None`, `step`, `cosine`, or `plateau`. |
| `scheduler_kwargs` | `{}` | Keyword arguments for the selected PyTorch scheduler. See examples below. |
| `aux_loss_weights` | `None` | Weights for auxiliary heads. Use two finite nonnegative values for the supported full models. |
| `crop_size` | `None` | Optional training crop as `(height, width)`. Must fit each source image. The loader then resizes it to `image_size`. |
| `foreground_probability` | `0.0` | Probability from 0 to 1 of centering a crop on a foreground pixel. Requires `crop_size` when positive. |
| `checkpoint_interval` | `None` | Positive epoch interval for atomic `last_model.pt` saves. Requires `experiment_dir`. |
| `dataset_id` | `None` | Optional explicit dataset identity. Otherwise fingerprint supported file, array, or tensor datasets. |

## Reduce memory use

Gradient accumulation combines several microbatches into one optimizer update.
For example, batch size 2 and four accumulation steps give an effective batch size of 8, except at a partial final group.
BatchNorm still sees each microbatch separately.

| Device | Supported precision |
| --- | --- |
| CPU | `fp32`, `bf16` |
| CUDA | `fp32`, `fp16`, and `bf16` on supported devices |
| MPS | `fp32` |

## Configure the learning rate

| Scheduler | Wrapper default | Example `scheduler_kwargs` |
| --- | --- | --- |
| `None` | Constant learning rate | `{}` |
| `step` | `step_size=10` | `{"step_size": 10, "gamma": 0.1}` |
| `cosine` | `T_max=epochs` | `{"T_max": 50, "eta_min": 1e-6}` |
| `plateau` | Mode follows `monitor_mode` | `{"patience": 5, "factor": 0.5}` |

Other scheduler arguments pass to `StepLR`, `CosineAnnealingLR`, or `ReduceLROnPlateau`, respectively.

## Configure auxiliary losses

Set `model_kwargs={"deep_supervision": True}` on a supported model.
Full Swin-Unet, PVTFormer, ResUNet++, and UNeXt each add two auxiliary heads.
Use `aux_loss_weights=(0.4, 0.2)`, for example. The total is the main loss plus each weighted auxiliary loss.
When weights are omitted, each auxiliary head receives weight 0.4. Inference uses the main head only.

## Resume a run

Load the checkpoint with `Segmenter.load`, then call `train(resume_from=...)`.
Keep the optimizer type, scheduler, scheduler arguments, normalization, and ignored label consistent with the saved run.
A saved gradient scaler also requires the saved precision.
Without `TrainingConfig`, omitted memory and schedule options inherit checkpoint values.
A supplied `TrainingConfig` provides explicit defaults, which can conflict with saved settings.

## Settings outside the dataclass

Pass `data`, `train_data`, `val_data`, and `transforms` directly to `train`.
Use a dataset root, separate loaders, or image/mask sources as described in [training](../training.md).
Direct `train(optimizer=...)` also accepts a PyTorch optimizer instance.

A supplied configuration replaces the matching training keywords.
The four optional path/name fields preserve direct keyword values when their config value is `None`.
These fields are `save_to`, `experiment_dir`, `run_name`, and `resume_from`.
Without validation data, the default `val_loss` monitor becomes `train_loss`.

## Checkpoint provenance

Checkpoints store a format version, package versions, completed epoch, dataset identity, and training settings.
Each write uses a temporary file followed by an atomic replacement. A failed write leaves the previous checkpoint intact.
`checkpoint_interval=1` saves `last_model.pt` after every completed epoch. Resume starts at the next epoch.

File, array, and TensorDataset inputs receive content fingerprints. Dataset fingerprints include train/validation membership.
Custom datasets without inspectable sources receive a descriptor marked `content_verified=False`.
Supply `dataset_id` for an externally managed dataset revision. Keep that ID unchanged when resuming.
A verified identity mismatch stops resume. Unknown future checkpoint formats produce an explicit error.
