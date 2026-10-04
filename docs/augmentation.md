# Augment training images

[Documentation](README.md) · [Training guide](training.md)

Generate augmented image and mask files before training. Augmentation remains separate from normalization.
Split the dataset first. Augment only the training split.

```python
from functools import partial
from bicbioseg import AugmentImages, apply_and_save_augmentations

files = apply_and_save_augmentations(
    images_source="cell_dataset/train/images",
    masks_source="cell_dataset/train/masks",
    output_dir="augmented_train",
    augmentations=[
        partial(AugmentImages.flip, mode="horizontal", probability=0.5),
        partial(AugmentImages.rotate, angle_range=(5, 15), probability=0.5),
    ],
    num_augmented=3,
    random_seed=42,
    preview_to="qc/augmentation.png",
)
```

The output contains paired `images` and `masks` folders. Source stems identify each pair.
The function returns saved image paths. It preserves uint8 and uint16 images as PNG.
Normalized floating-point image files use uncompressed TIFF. Masks retain integer class IDs.
Original files remain in their source folders. Combine source lists explicitly if training should include originals and augmented copies.

## Transform contract

Each transform accepts `(image, mask)` and returns the transformed pair.
Images are grayscale `HxW` or RGB `HxWx3`, with dtype uint8, uint16, float32, or float64.
Floating-point images must already be finite and in `[0, 1]`. Augmentation does not apply a normalization policy.
Masks are integer `HxW` arrays with the same spatial dimensions as the image.

Geometric transforms use nearest-neighbor sampling for masks. Rotation can add background label 0 at image borders.
Brightness and histogram transforms leave the mask unchanged. Every built-in transform preserves image dtype and range.
The file writer checks that transforms do not introduce unexpected labels.

| Helper | Options and defaults |
| --- | --- |
| `rotate` | `angle_range=(5, 30)`, `resize_target=None`, `probability=1.0` |
| `flip` | `mode="random"`, `probability=1.0`. Modes: `horizontal`/`h`, `vertical`/`v`, `both`/`vh`, `random`. |
| `adjust_brightness` | `delta_range=(-30, 30)`, `probability=1.0` |
| `hist_equalize` | `clipLimit=3`, `tileGridSize=(8, 8)`, `probability=1.0` |
| `random_crop` | `crop_size=(224, 224)`, `probability=1.0` |

Resize and crop sizes use `(height, width)`. Brightness deltas use 8-bit units for all image dtypes.
For example, a delta of 25.5 represents 10% of the dtype's full range.
Histogram equalization applies CLAHE to RGB luminance. Float and uint16 luminance use a 16-bit working buffer.

## Preview label preservation

```python
import cv2
from bicbioseg import AugmentImages, AugmentationPipeline

image = cv2.cvtColor(cv2.imread("cell_dataset/train/images/sample_001.png"), cv2.COLOR_BGR2RGB)
mask = cv2.imread("cell_dataset/train/masks/sample_001.png", cv2.IMREAD_UNCHANGED)
pipeline = AugmentationPipeline([AugmentImages.rotate])
report = pipeline.preview(image, mask, save_to="qc/preview.png")
print(report["labels_preserved"], report["unexpected_labels"])
```

The preview shows original and augmented image/mask pairs. Its report lists labels before and after, dtype, and shape consistency.
`labels_preserved` means that the transform introduced no label IDs except background 0. Cropping can remove an object entirely.
Use `pipeline.add(transform)` to append a transform.

## Optional Albumentations adapter

Install `bicbioseg[albumentations]` to use `create_albumentations_pipeline` or `AlbumentationsTransform`.
The factory defaults are `horizontal_flip=True`, `vertical_flip=False`, `rotate_limit=30`,
`brightness_contrast=True`, `elastic=False`, and `probability=0.5`.
Pass `transforms=[...]` for a custom list, or wrap an existing composition with `AlbumentationsTransform(compose)`.
Check the selected external transforms against the dtype and mask-label contract.

The existing loader transform hook remains available for custom workflows. File generation does not require it.
