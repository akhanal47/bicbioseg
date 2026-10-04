# Train and save a model

[Documentation](README.md) · [Main guide](../README.md)

Use `Segmenter` when your dataset is already prepared. Use `SegmentationExperiment` for an end-to-end workflow.

```python
from bicbioseg import Segmenter

model = Segmenter(
    architecture="unet",
    loss="dice",
    metrics=["dice", "iou"],
    image_size=(224, 224),
    num_classes=1,
)

history = model.train(
    data="cell_dataset",
    epochs=50,
    batch_size=8,
    lr=1e-4,
    experiment_dir="experiments",
    run_name="unet_dice",
    early_stopping=True,
    patience=10,
)
```

Example multiclass setup:

```python
model = Segmenter(
    architecture="unet",
    num_classes=3,
    in_channels=1,
    image_size=(256, 384),
    loss="cross_entropy",
)
```

When `experiment_dir` is provided, training writes:

```text
experiments/unet_dice/
  config.json
  history.csv
  history.json
  summary.json
  best_model.pt
  final_model.pt
```

Plot training curves and samples:

```python
model.plot_history(save_to="experiments/unet_dice/history.png")

model.plot_training_samples(
    data="cell_dataset",
    save_to="experiments/unet_dice/training_samples.png",
)
```

Resume from a checkpoint:

```python
model = Segmenter.load("experiments/unet_dice/final_model.pt")
model.train(
    data="cell_dataset",
    epochs=20,
    resume_from="experiments/unet_dice/final_model.pt",
)
```

Use config objects:

```python
from bicbioseg import Segmenter, SegmenterConfig, TrainingConfig

model = Segmenter.from_config(
    SegmenterConfig(
        architecture="unet",
        loss="dice",
        image_size=(224, 224),
        device="auto",
    )
)

model.train(
    data="cell_dataset",
    config=TrainingConfig(
        epochs=50,
        batch_size=8,
        early_stopping=True,
        patience=10,
    ),
)
```



## Choose training data

`data="cell_dataset"` discovers `train` and `validate` subfolders.
Alternatively, pass `train_data` and `val_data` as loaders, image/mask pairs, or split directories.
A split directory must contain `images` and `masks` folders.
Validation is optional. Training crops and augmentations apply only to the training loader.

```python
from bicbioseg import DataLoader, Segmenter

train_loader = DataLoader.load_data("cell_dataset/train", batch_size=8, num_workers=0)
val_loader = DataLoader.load_data("cell_dataset/validate", batch_size=8, shuffle=False, num_workers=0)
model = Segmenter(architecture="unet")
model.train(train_data=train_loader, val_data=val_loader, epochs=50)
```

For prebuilt loaders, match the loader's size, channels, normalization, and ignored labels to the model.
The trainer uses these loaders as supplied. It does not replace their batch size or transforms.
On systems that spawn loader processes, place script execution under `if __name__ == "__main__":`.
Use `num_workers=0` when a notebook or small example needs no worker processes.

## Inspect the setup

```python
from bicbioseg import Segmenter, environment_info, set_seed

set_seed(42)
model = Segmenter(architecture="unet", device="auto")
print(environment_info())
print(model.validate_setup(backward=True))
```

Set the seed before model construction and training.
`set_seed(42, deterministic=True)` requests deterministic PyTorch behavior.

## Save and restore weights

```python
from bicbioseg import Segmenter

model = Segmenter(architecture="unet")
# Train the model before you save weights for prediction.
model.save("model.pt")
restored = Segmenter.load("model.pt", device="cpu")
```

Checkpoints retain the model, shared settings, and available training state.
`load_weights(path)` loads weights into an existing compatible model.
A configuration JSON alone does not restore trained weights.
Training keeps the final epoch in memory. Load `best_model.pt` explicitly to predict with the best saved epoch.

See [all training settings](configuration/training.md) for precision, accumulation, clipping, schedulers, crops, and resume constraints.
See [normalization](configuration/normalization.md) for uint16 microscopy and ignored labels.
See [augmentation](augmentation.md) for paired transforms.
