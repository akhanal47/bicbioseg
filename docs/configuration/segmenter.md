# Model and input settings

[All configuration](README.md) · [Main guide](../../README.md)

## SegmenterConfig

Use this configuration to create a reusable model setup.

```python
from bicbioseg import Segmenter, SegmenterConfig

config = SegmenterConfig(
    architecture="unet",
    image_size=(256, 384),
    in_channels=1,
    num_classes=3,
    loss="cross_entropy",
    model_kwargs={"base_channels": 16},
)
config.save("model_config.json")
model = Segmenter.from_config(SegmenterConfig.load("model_config.json"))
```

## All fields

| Option | Default | Purpose and accepted values |
| --- | --- | --- |
| `architecture` | `'unet'` | Registered model name or alias. See [all models](../models/README.md). |
| `loss` | `'dice'` | Loss name. See [loss choices](losses.md). |
| `metrics` | `None` | Training metrics. `None` selects `dice` and `iou`. Also accepts `precision`, `recall`, and `jaccard` as an alias for `iou`. |
| `image_size` | `(224, 224)` | Model input size as `(height, width)`. The loader resizes each image and mask to this size. |
| `num_classes` | `1` | Use 1 for binary segmentation. Use at least 2 for multiclass segmentation. |
| `in_channels` | `3` | Input channel count. File workflows support 1 for grayscale or 3 for RGB. |
| `device` | `None` | `None` or `auto` selects CUDA, then MPS, then CPU. Explicit choices include `cpu`, `mps`, `cuda`, and `cuda:1`. |
| `model_kwargs` | `{}` | Architecture options. Each [model page](../models/README.md) lists the supported keys. |
| `loss_kwargs` | `{}` | Loss constructor options. See [loss choices](losses.md). |
| `normalization` | `{"mode": "standard"}` | Image scaling policy shared by training and inference. See [normalization](normalization.md). |
| `ignore_index` | `None` | Optional integer label to exclude from losses and metrics. Set it here to keep both consistent. |

## Match images and labels

Binary segmentation uses one output channel. Use background label 0 and positive foreground labels.
Multiclass segmentation uses integer class IDs from 0 through `num_classes - 1`.
Convert color masks to label masks before training.

The shared fields also exist on `Segmenter(...)`. Direct construction accepts an integer `image_size` for a square input.
Some direct models accept additional channel counts, but file workflows support grayscale and RGB images.
Do not repeat channel or class settings with conflicting values in `model_kwargs`.

Unavailable devices cause an error. Check `Segmenter.available_devices()` and call `model.validate_setup(backward=True)` before a long run.

Configuration JSON describes how to create a model. It does not contain trained weights.
Use a [checkpoint](../training.md) to restore trained weights.
