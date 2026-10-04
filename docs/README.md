# User guide

[Main README](../README.md)

Follow the workflow or open the reference for the setting you need.

| Task | Guide |
| --- | --- |
| Prepare pairs, splits, groups, and patches | [Dataset preparation](data.md) |
| Inspect data, convert formats, and measure objects | [Image operations](image-operations.md) |
| Choose an architecture and variant | [All 12 models](models/README.md) |
| Train, save, and resume | [Training](training.md) |
| Apply paired transforms | [Augmentation](augmentation.md) |
| Predict one image, a folder, or a large image | [Inference](inference.md) |
| Score predictions and inspect failures | [Evaluation](evaluation.md) |
| Compare model and loss combinations | [Experiments](experiments.md) |
| Find defaults and accepted values | [Configuration reference](configuration/README.md) |
| Find implementation references | [Acknowledgments](references.md) |

## Library components

```mermaid
flowchart TD
    A["SegmentationExperiment: complete workflow"] --> B["ImageOps: prepare and inspect"]
    A --> C["Segmenter: train, evaluate, predict"]
    D["Configuration: save reusable settings"] --> A
    D --> C
    E["DataLoader and augmentation: paired batches"] --> C
    F["Models: encoder and decoder"] --> C
    G["Losses and metrics: learn and score"] --> C
    C --> H["Checkpoints, masks, plots, and reports"]
```

Use `SegmentationExperiment` to connect the common steps.
Use `Segmenter` when you already have split data or custom loaders.
Use `ImageOps` without a model for preparation and inspection.
Use the model classes directly when you need a custom PyTorch loop.
