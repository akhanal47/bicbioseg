# Augment training images

[Documentation](README.md) · [Training guide](training.md)

Apply the same geometric transform to each image and mask. Keep validation and test data unchanged.
Split data before you save augmented copies.

```python
from functools import partial
from bicbioseg import AugmentImages, AugmentationPipeline, Segmenter

transforms = AugmentationPipeline([
    partial(AugmentImages.flip, mode="horizontal", probability=0.5),
    partial(AugmentImages.rotate, angle_range=(-15, 15), probability=0.5),
])
model = Segmenter(architecture="unet")
model.train(data="cell_dataset", transforms=transforms, epochs=50)
```

Each transform accepts `(image, mask)` and returns the transformed pair.
Use `AugmentationPipeline.add(transform)` to append a transform.

| Helper | Options and defaults |
| --- | --- |
| `rotate` | `angle_range=(5, 30)`, `resize_target=None`, `probability=1.0` |
| `flip` | `mode="random"` (`horizontal`/`h`, `vertical`/`v`, `both`/`vh`, or `random`), `probability=1.0` |
| `adjust_brightness` | `delta_range=(-30, 30)`, `probability=1.0` |
| `hist_equalize` | `clipLimit=3`, `tileGridSize=(8, 8)`, `probability=1.0` |
| `random_crop` | `crop_size=(224, 224)`, `probability=1.0` |

The built-in helpers require uint8 images and masks. Explicit intensity policies produce floats, so use compatible custom transforms for those policies.
`hist_equalize` expects BGR color arrays. The training loader supplies RGB, so convert color order if you use this helper there.
`rotate(resize_target=...)` passes its size directly to OpenCV as `(width, height)`.
For crop sampling before the loader resize, use [TrainingConfig.crop_size](configuration/training.md).

## Use Albumentations

Install `bicbioseg[albumentations]`.

```python
from bicbioseg import Segmenter, create_albumentations_pipeline

transforms = create_albumentations_pipeline(
    horizontal_flip=True,
    vertical_flip=False,
    rotate_limit=30,
    brightness_contrast=True,
    elastic=False,
    probability=0.5,
)
model = Segmenter(architecture="unet")
model.train(data="cell_dataset", transforms=transforms, epochs=50)
```

These are the factory defaults. Pass `transforms=[...]` to use your own Albumentations transform list.
Use `AlbumentationsTransform(compose)` to adapt an existing composition to the paired transform interface.

## Save augmented files

`apply_and_save_augmentations` accepts `images_source`, optional `masks_source`, and `output_dir="augmented"`.
Other options are `augmentations=None`, `num_augmented=1`, `random_seed=None`, and `valid_extensions=(".png", ".jpg", ".jpeg", ".bmp")`.
Use training sources only. See the [augmentation source](../src/bicbioseg/utils/augmentation.py) for dtype and shape requirements.
