# Inspect and process images

[Documentation](README.md) · [Main guide](../README.md)

Inspect the labels before training. Use these helpers independently or within an experiment.

Create a QC report:

```python
from bicbioseg import ImageOps

report = ImageOps.dataset_qc_report(
    images="raw/images",
    masks="raw/masks",
    save_to="qc/qc_report.json",
)

print(report["warnings"])
```

Preview image/mask/overlay samples:

```python
ImageOps.preview_dataset(
    images="raw/images",
    masks="raw/masks",
    num_samples=8,
    save_to="qc/preview.png",
)
```

Inspect and validate dataset files:

```python
summary = ImageOps.inspect_dataset("raw/images", "raw/masks")
mask_type = ImageOps.infer_mask_type("raw/masks")
unmatched = ImageOps.find_unmatched_masks("raw/images", "raw/masks")
```

Resize and normalize masks:

```python
ImageOps.resize_dataset(
    images_source="raw/images",
    masks_source="raw/masks",
    output_dir="resized",
    image_size=(512, 512),
)

ImageOps.normalize_masks(
    masks_source="raw/masks",
    output_dir="normalized_masks",
    mode="binary",
)
```

TIFF and DICOM helpers:

```python
frames, metadata = ImageOps.extract_tiff_frames("stack.tif", get_metadata=True)

ImageOps.convert_dicom(
    "scan.dcm",
    output_path="scan.png",
    method="clip",
    clip_percentiles=(1, 99),
)
```

Postprocess and measure masks:

```python
import cv2

mask = cv2.imread("predictions/cell_mask.png", cv2.IMREAD_GRAYSCALE)
mask = ImageOps.remove_small_objects(mask, min_size=64)
mask = ImageOps.fill_holes(mask)
mask = ImageOps.smooth_mask(mask)
instances = ImageOps.watershed_instances(mask)

measurements = ImageOps.measure_objects(
    mask="predictions/cell_mask.png",
    image="raw/images/cell.png",
    save_to="measurements.csv",
)
```

## Convert color masks

Folder conversion discovers colors across all masks before it assigns class IDs.
It saves the shared RGB map as `class_map.json` and writes `conversion.json` with source paths and unknown-color details.
Array conversion expects RGB. Convert OpenCV BGR input before calling it.

```python
import cv2
from bicbioseg import ImageOps

color_mask = cv2.cvtColor(
    cv2.imread("raw/color_masks/sample_001.png"), cv2.COLOR_BGR2RGB,
)
labels, color_map = ImageOps.convert_color_mask_to_labels(
    color_mask,
    color_map={(0, 0, 0): 0, (255, 0, 0): 1, (0, 255, 0): 2},
)
cv2.imwrite("sample_001_labels.png", labels)
```

`normalize_mask` accepts `mode="binary"`, `"labels"`, or `"color"`.
Binary defaults are `threshold=0`, `foreground_value=1`, and `dtype=np.uint8`.
Use `normalize_masks` for a folder. Preserve class IDs with `mode="labels"`.

## File conversion options

Install `bicbioseg[dicom]` for DICOM conversion.
`ImageOps.convert_dicom` and `dicom_to_uint8` accept `method="linear"`, `"clip"`, or `"log"`.
Other options are `output_path=None`, `clip_percentiles=(1, 99)`, `invert=False`, and `return_array=False`.
The conversion produces 8-bit images.

`ImageOps.extract_tiff_frames` and `tiff_extract_frames` share these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `resize` | `None` | Optional frame size as `(height, width)`. |
| `return_array` | `False` | Return arrays instead of Pillow images. |
| `get_metadata` | `False` | Return frame metadata with the images. |
| `save_image` | `False` | Save extracted images. |
| `save_image_location` | `None` | Destination for images. |
| `save_metadata` | `False` | Save metadata as JSON. |
| `save_metadata_location` | `None` | Destination for metadata. |

## Work with patches and masks

`ImageOps.standardize_images` pads to a patch multiple.
`ImageOps.create_patches` returns patches and grid coordinates.
`ImageOps.reconstruct_from_patches` combines patches into an image.
`ImageOps.overlay_mask` creates an image/mask overlay.

The postprocessing example expects a binary NumPy mask. Process each class separately for multiclass masks.
Object measurements include area, perimeter, centroid, bounding box, and optional mean image intensity.
See the [image operations source](../src/bicbioseg/imageops/preprocess.py) for helper signatures.

## Reuse a dataset color map

```python
from bicbioseg import ImageOps

ImageOps.normalize_masks("raw/color_masks", "labels", mode="color")
ImageOps.normalize_masks(
    "new/color_masks", "new_labels", mode="color",
    color_map="labels/class_map.json",
)
```

Unknown colors raise `DatasetError` by default. They cannot silently become background.
To allow them, choose `unknown_color="background"` or `unknown_color="ignore", ignore_index=255`.
Both explicit fallback policies emit a warning. Choose an ignored ID that is not a valid class.
Use `ImageOps.save_color_map(mapping, path)` and `ImageOps.load_color_map(path)` to manage maps directly.
Label mode preserves integer bit depth, including uint16 class IDs.

## Check label validity and split coverage

```python
from bicbioseg import ImageOps

report = ImageOps.dataset_qc_report(
    "raw/images", "raw/masks", num_classes=3, ignore_index=255,
)
coverage = ImageOps.dataset_split_qc_report(
    "cell_dataset", num_classes=3, ignore_index=255, save_to="qc/splits.json",
)
```

Reports include invalid labels, mismatched shapes, ignored-only masks, ignored pixel counts, and per-class sample/pixel counts.
Color masks are reported as non-label masks. Convert them before training.
Split reports list missing classes separately for `train`, `validate`, and `test`.
`exp.qc(splits=True)` checks the prepared experiment dataset with the model's class and ignored-label settings.
