# bicbioseg

`bicbioseg` is a biomedical image segmentation toolkit focused on simple, experiment-friendly APIs.

## Prepare A Dataset

Create a train/validate/test split from image and mask folders:

```python
from bicbioseg import create_dataset_split

create_dataset_split(
    images="raw/images",
    masks="raw/masks",
    save_to="cell_dataset",
    split=(0.8, 0.1, 0.1),
    resize=(512, 512),
    overwrite=False,
)
```

Create patches after the split, so patches from the same source image do not leak across splits:

```python
create_dataset_split(
    images="raw/images",
    masks="raw/masks",
    save_to="cell_dataset_patches",
    split=(0.8, 0.1, 0.1),
    create_patches=True,
    patch_size=(224, 224),
    balance_empty_masks=True,
)
```

The legacy function still works:

```python
from bicbioseg import create_train_validate_test_split

create_train_validate_test_split(
    "raw/images",
    "raw/masks",
    output_dir="cell_dataset",
    create_patch=True,
    patch_dims=(224, 224),
)
```

## Image Operations

```python
from bicbioseg import ImageOps

summary = ImageOps.inspect_dataset("raw/images", "raw/masks")
unmatched = ImageOps.find_unmatched_masks("raw/images", "raw/masks")

ImageOps.resize_dataset(
    images_source="raw/images",
    masks_source="raw/masks",
    output_dir="resized",
    image_size=(512, 512),
)

frames, metadata = ImageOps.extract_tiff_frames(
    "stack.tif",
    resize=(512, 512),
    get_metadata=True,
)

ImageOps.convert_dicom(
    "scan.dcm",
    output_path="scan.png",
    method="clip",
    clip_percentiles=(1, 99),
)
```

## Train And Infer

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
    save_to="checkpoints/unet_dice.pt",
)

model.inference(
    images="new_images",
    save_to="predictions",
)
```

## Run Experiments

Compare model and loss choices without rewriting the training loop:

```python
from bicbioseg import Segmenter

results = Segmenter.run_experiment(
    dataset="cell_dataset",
    architectures=["unet", "attention_unet"],
    losses=["dice", "jaccard"],
    metrics=["dice", "iou"],
    epochs=30,
    batch_size=8,
)
```

Available model aliases:

```python
Segmenter.available_models()
```

Available loss aliases:

```python
Segmenter.available_losses()
```
