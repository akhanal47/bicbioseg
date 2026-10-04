# Named experiment settings

[All configuration](README.md) · [Main guide](../../README.md)

## ExperimentRunConfig

Use one configuration for each run in a comparison.

```python
from bicbioseg import ExperimentRunConfig, SegmenterConfig, TrainingConfig, Segmenter

runs = {
    "unet_baseline": ExperimentRunConfig(
        segmenter=SegmenterConfig(architecture="unet"),
        training=TrainingConfig(epochs=30, batch_size=8),
        seed=42,
    ),
    "attention_baseline": ExperimentRunConfig(
        segmenter=SegmenterConfig(architecture="cattention_unet"),
        training=TrainingConfig(epochs=30, batch_size=8),
        seed=42,
    ),
}
results = Segmenter.run_experiment("cell_dataset", runs=runs, output_dir="comparisons")
runs["unet_baseline"].save("unet_run.json")
```

## All fields

| Option | Default | Purpose and accepted values |
| --- | --- | --- |
| `segmenter` | `SegmenterConfig()` | A `SegmenterConfig` for one model. |
| `training` | `TrainingConfig()` | A `TrainingConfig` for one training run. |
| `seed` | `42` | Seed for this run. `run_experiment` sets it before model construction. |

Use names without path separators. Each name identifies a run directory.
Each run can have its own model, loss, seed, and training settings.
Use the same seed and shared settings when a controlled architecture comparison is the goal.

`Segmenter.run_experiment` accepts config objects or serialized dictionaries in `runs`.
Do not combine `runs` with `architectures` or `losses`.
Extra training keywords passed to `run_experiment` override each run's training settings.
An explicit `training.experiment_dir` or `training.run_name` overrides the default output location or run name.

See [experiment comparison](../experiments.md) for the architecture/loss grid, reports, and comparison plots.
