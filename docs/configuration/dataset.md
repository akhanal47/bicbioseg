# Dataset split settings

[All configuration](README.md) · [Main guide](../../README.md)

## DatasetSplitConfig

Use this configuration with `create_dataset_split(..., config=config)` or `exp.prepare(config=config)`.

```python
from bicbioseg import DatasetSplitConfig, create_dataset_split

config = DatasetSplitConfig(split=(0.8, 0.1, 0.1), resize=(512, 512))
create_dataset_split("raw/images", "raw/masks", save_to="dataset", config=config)
```

## All fields

| Option | Default | Purpose and accepted values |
| --- | --- | --- |
| `split` | `(0.8, 0.1, 0.1)` | Train, validation, and test fractions. Use three nonnegative values that sum to 1. |
| `resize` | `None` | Optional size for saved files. Use `(height, width)`. |
| `create_patches` | `False` | Set `True` to split images into patches after the dataset split. |
| `patch_size` | `(224, 224)` | Patch size as `(height, width)`. Used only when `create_patches=True`. |
| `balance_empty_masks` | `False` | When patching, discard every patch whose mask is entirely zero. |
| `group_by` | `'filename'` | Grouping rule: `filename`, `stem`, `parent`, `path`, or `None`. See grouping below. |
| `random_seed` | `42` | Seed for the dataset split. |
| `overwrite` | `False` | Set `True` to replace an existing output directory. Keep source data in a separate directory. |
| `progress` | `True` | Show preparation progress. |

## Group related images

`filename` and `stem` group files by stem after removal of a recognized patch suffix.
`path` and `None` use each full path as a group. `parent` uses the parent folder name.
For patient or specimen grouping, pass a callable directly to `create_dataset_split`.
A callable cannot be stored in JSON configuration.

Splitting occurs before patch creation. Keep related images in the same group to prevent leakage between splits.
`balance_empty_masks=True` can remove all patches from an entirely empty source mask.

A supplied configuration replaces the corresponding keyword settings in `create_dataset_split`.
With `exp.prepare`, explicit `group_by`, `random_seed`, and `progress` keywords take precedence over their config values.

`create_kfold_splits` accepts preparation options directly, plus `k=5`. It does not accept `DatasetSplitConfig`.
It creates train/validate folders for each fold, with no test folder. See [dataset preparation](../data.md).
