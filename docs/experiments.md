# Compare experiments

[Documentation](README.md) · [Main guide](../README.md)

Keep dataset splits and evaluation settings consistent across runs.

```python
from bicbioseg import Segmenter
```

Run multiple architecture/loss combinations:

```python
results = Segmenter.run_experiment(
    dataset="cell_dataset",
    architectures=["unet", "cattention_unet"],
    losses=["dice", "jaccard"],
    metrics=["dice", "iou"],
    epochs=30,
    batch_size=8,
    output_dir="experiments",
)
```

The default U-Net and CAttention U-Net configurations share the same backbone.
For a controlled CBAM experiment, keep all model and training options matched.
See [CAttention U-Net](models/cattention_unet.md#compare-attention-with-u-net).

For separate model widths, losses, seeds and training settings, use named runs:

```python
from bicbioseg import Segmenter, ExperimentRunConfig, SegmenterConfig, TrainingConfig

runs = {
    "swin_tiny": ExperimentRunConfig(
        segmenter=SegmenterConfig(architecture="swin_unet_full", model_kwargs={"variant": "tiny"}),
        training=TrainingConfig(epochs=50, batch_size=2, accumulation_steps=4),
        seed=42,
    ),
    "pvt_b0": ExperimentRunConfig(
        segmenter=SegmenterConfig(architecture="pvtformer_full", model_kwargs={"variant": "b0", "decoder_channels": 32}),
        training=TrainingConfig(epochs=50, batch_size=4),
        seed=43,
    ),
}
results = Segmenter.run_experiment("cell_dataset", runs=runs, output_dir="comparisons")
runs["swin_tiny"].save("swin_run.json")
```

Each run saves its effective configuration, seed, environment, elapsed time and throughput in `experiment.json`, alongside normal checkpoints and history.

Compare experiment summaries:

```python
Segmenter.compare_experiments(
    "experiments",
    metric="val_dice",
    save_to="experiments/comparison.png",
)
```

Generate a run report:

```python
model.create_report("experiments/unet_dice")
```

The Swin and PVT example requires `bicbioseg[transformers]`.
Run-specific settings use [ExperimentRunConfig](configuration/runs.md).
For reproducible comparisons, retain `experiment.json`, the configuration files, and checkpoints.

`compare_experiments` defaults to `metric="val_loss"`, `save_to=None`, `show=True`, and `figsize=(8, 5)`.
`create_report(run_dir, save_to=None)` writes the report to the run directory by default.
