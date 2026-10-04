# Configuration reference

[Main guide](../../README.md) · [Documentation](../README.md)

Choose the configuration that matches your task.
Each dataclass page lists every field and its default.

| Configuration | Purpose | Used by |
| --- | --- | --- |
| [DatasetSplitConfig](dataset.md) | Split, resize, group, and patch data | `create_dataset_split`, `exp.prepare` |
| [SegmenterConfig](segmenter.md) | Choose the model, inputs, loss, and device | `Segmenter.from_config` |
| [TrainingConfig](training.md) | Train, stop, save, and resume | `model.train`, `exp.train` |
| [ExperimentConfig](experiment.md) | Connect data, model, and output folders | `SegmentationExperiment.from_config` |
| [ExperimentRunConfig](runs.md) | Compare named runs | `Segmenter.run_experiment` |
| [Normalization](normalization.md) | Scale uint8, uint16, and floating-point images | Model and experiment configuration |
| [Losses and metrics](losses.md) | Choose a training objective and scores | `loss`, `loss_kwargs`, `metrics` |
| [Model options](../models/README.md) | Select architecture variants and decoder settings | `model_kwargs` |

## Save and restore settings

All five dataclasses provide `to_dict()`, `from_dict(payload)`, `save(path)`, and `load(path)`.
`save` creates parent directories and writes JSON.

```python
from bicbioseg import TrainingConfig

config = TrainingConfig(epochs=50, batch_size=8)
config.save("training.json")
restored = TrainingConfig.load("training.json")
```

Configuration files contain settings. Checkpoints contain trained model state.
Use JSON-compatible values when you save settings. Custom functions, transforms, and optimizer objects require Python setup.
Model configurations validate architecture options and normalize JSON sequences automatically, including SegFormer stage settings.
`Segmenter.model_options(name)` lists option types, defaults, constraints, and accepted choices without constructing a model.

Dataset preparation and model settings use `(height, width)` for `resize`, `image_size`, `crop_size`, and `patch_size`.
The TIFF conversion helper also accepts `(height, width)`.
