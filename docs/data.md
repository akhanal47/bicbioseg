# Prepare images and masks

[Documentation](README.md) · [Main guide](../README.md)

Start with paired images and masks. Split data before training or augmentation.

## Input files

```text
raw/
  images/
    sample_001.png
    sample_002.png
  masks/
    sample_001.png
    sample_002.png
```

Match image and mask filename stems. Extensions can differ.
Use grayscale or RGB image files. Masks must be two-dimensional integer label arrays with the same spatial size as their images.
Use 0 for background and positive values for binary foreground.
For multiclass masks, use class IDs from 0 through `num_classes - 1`.
Convert color masks to labels before training.

```mermaid
flowchart LR
    A["Match images and masks"] --> B["Group related samples"]
    B --> C["Split train / validate / test"]
    C --> D["Optional resize"]
    D --> E["Optional patches"]
    E --> F["Train on train split"]
```

## Create splits

Create a train/validate/test split:

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

The output structure is:

```text
cell_dataset/
  train/
    images/
    masks/
  validate/
    images/
    masks/
  test/
    images/
    masks/
```

Create K-fold train/validate splits:

```python
from bicbioseg import create_kfold_splits

folds = create_kfold_splits(
    images="raw/images",
    masks="raw/masks",
    save_to="cell_kfold",
    k=5,
)
```

Optional patch creation is available for advanced workflows. Patches are created only when explicitly requested, and splitting happens before patching to avoid leakage:

```python
create_dataset_split(
    images="raw/images",
    masks="raw/masks",
    save_to="cell_dataset_patches",
    create_patches=True,
    patch_size=(224, 224),
    balance_empty_masks=True,
)
```

Config objects are supported:

```python
from bicbioseg import DatasetSplitConfig, create_dataset_split

config = DatasetSplitConfig(
    split=(0.8, 0.1, 0.1),
    resize=(512, 512),
)

create_dataset_split("raw/images", "raw/masks", save_to="dataset", config=config)
```

See [all split settings](configuration/dataset.md) for grouping, seeds, overwrite behavior, and defaults.
`resize`, `image_size`, and `patch_size` use `(height, width)` in the dataset preparation API.
Training resizes loaded samples to the model's `image_size`, even if preparation saved a different size.
If both resize and patch creation are enabled, resize occurs first.
`balance_empty_masks=True` removes all empty patches.

K-fold preparation creates `fold_1`, `fold_2`, and subsequent folders. Each contains `train` and `validate` splits.
Train each fold separately. Reserve a separate test set if you need a final independent evaluation.

## Group by specimen

Use a grouping function when several images come from the same patient or specimen.
This example assumes filenames such as `patient17_field03.png`.

```python
from pathlib import Path
from bicbioseg import create_dataset_split

create_dataset_split(
    "raw/images", "raw/masks", save_to="patient_dataset",
    group_by=lambda path: Path(path).stem.split("_")[0],
)
```

The function must return the same key for all related images.
Use source lists or path-list files when your data spans multiple folders.
Directory discovery scans the selected folder. It does not recursively discover nested patient folders.

Continue with [quality checks and image operations](image-operations.md), [augmentation](augmentation.md), or [training](training.md).
