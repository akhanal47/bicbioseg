# bicbioseg

`bicbioseg` is a biomedical image segmentation toolkit for preparing image datasets, training common segmentation models, running inference, and comparing experiments with a simple Python API.

## Status

This project is in early release & active developemnt. APIs may evolve or changes with each releases as more biomedical workflows are added.

## Stable Models
These are the stable models as of now (more to be added in future)
>Unet, DoubleUnet, Segformer

## Installation

```bash
pip install bicbioseg
```

## What The Package Provides

- Dataset splitting into `train`, `validate`, and `test`
- K-fold dataset splitting
- Image/mask filename matching
- Resize-based dataset preparation
- Optional patch creation after splitting
- TIFF frame extraction
- DICOM to 8-bit image conversion
- Mask normalization and color-mask-to-label conversion
- Dataset QC reports and preview plots
- Common mask postprocessing helpers
- Object measurements from masks
- PyTorch dataset/dataloader utilities
- High-level `Segmenter` wrapper for training/inference
- Experiment logging to JSON/CSV/checkpoints
- Training and inference visualizations
- Early stopping and checkpoint resume
- Binary and multi-class evaluation
- Worst-prediction/failure mining
- Binary threshold tuning
- Large-image tiled inference
- Ensemble inference from checkpoints
- Experiment comparison plots
- Config dataclasses for reproducible runs
- Reproducibility and environment helpers

## Expected Dataset Layout

Most high-level APIs expect image and mask folders with matching filename stems:

```text
raw/
  images/
    sample_001.png
    sample_002.png
  masks/
    sample_001.png
    sample_002.png
```

Image and mask extensions may differ (same extensions recommended), but stems should match.

## Quick Start

```python
from bicbioseg import SegmentationExperiment, set_seed

set_seed(42)

exp = SegmentationExperiment(
    images="raw/images",
    masks="raw/masks",
    model="unet",
    loss="dice",
    work_dir="cell_experiment",
    image_size=(224, 224),
)

exp.prepare(split=(0.8, 0.1, 0.1), resize=(224, 224))
exp.qc()
exp.preview(show=True)
exp.train(epochs=50, batch_size=8, early_stopping=True, patience=10)
exp.evaluate()
exp.predict("new_images", save_overlay=True)
exp.report()
```

## Dataset Preparation

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

## QC And Image Operations

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

## Training

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
model.train(
    data="cell_dataset",
    epochs=20,
    resume_from="experiments/unet_dice/final_model.pt",
)
```

Use config objects:

```python
from bicbioseg import SegmenterConfig, TrainingConfig

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

## Inference

Load a checkpoint:

```python
from bicbioseg import Segmenter

model = Segmenter.load("experiments/unet_dice/best_model.pt")
```

Run folder inference:

```python
predictions = model.inference(
    images="new_images",
    save_to="predictions",
    save_overlay=True,
    save_probability=True,
    save_logits=True,
    save_contours=True,
)
```

Single-image prediction:

```python
mask, overlay = model.predict_one(
    "new_images/cell.png",
    save_to="predictions/cell_mask.png",
    return_overlay=True,
)
```

Large-image tiled inference:

```python
model.inference_large_image(
    image="large_image.tif",
    patch_size=(512, 512),
    overlap=64,
    save_to="large_predictions",
)
```

Ensemble multiple checkpoints:

```python
Segmenter.ensemble_predict(
    checkpoints=[
        "experiments/unet_dice/best_model.pt",
        "experiments/attention_unet_dice/best_model.pt",
    ],
    images="test/images",
    save_to="ensemble_predictions",
)
```

Plot inference results:

```python
model.plot_inference_results(
    images="test/images",
    predictions=predictions,
    save_to="predictions/inference_grid.png",
)
```

## Evaluation And Failure Mining

Evaluate predictions against ground truth:

```python
results = model.evaluate(
    images="cell_dataset/test/images",
    masks="cell_dataset/test/masks",
    save_to="evaluation",
)
```

Multi-class evaluation:

```python
results = Segmenter.evaluate_predictions(
    predictions="predictions",
    masks="test/masks",
    metrics=["dice", "iou"],
    num_classes=3,
    class_names=["background", "nucleus", "cytoplasm"],
)
```

Find low-performing examples:

```python
worst = Segmenter.find_worst_predictions(
    predictions="predictions",
    masks="test/masks",
    images="test/images",
    metric="dice",
    top_k=10,
    save_to="failure_cases",
)
```

Tune a binary threshold:

```python
model.find_best_threshold(
    images="cell_dataset/validate/images",
    masks="cell_dataset/validate/masks",
    metric="dice",
    save_to="threshold_tuning",
)
```

## Experiment Comparison

Run multiple architecture/loss combinations:

```python
results = Segmenter.run_experiment(
    dataset="cell_dataset",
    architectures=["unet", "attention_unet"],
    losses=["dice", "jaccard"],
    metrics=["dice", "iou"],
    epochs=30,
    batch_size=8,
    output_dir="experiments",
)
```

Compare experiment summaries:

```python
Segmenter.compare_experiments(
    "experiments",
    metric="val_dice",
    save_to="experiments/comparison.png",
)
```

Generate a run report:

```python
model.create_report("experiments/unet_dice")
```

## Workflow Configs

Save and load a full experiment configuration:

```python
from bicbioseg import ExperimentConfig, SegmentationExperiment

config = ExperimentConfig(
    images="raw/images",
    masks="raw/masks",
    model="unet",
    loss="dice",
    work_dir="cell_experiment",
)

config.save("cell_experiment_config.json")
exp = SegmentationExperiment.from_config("cell_experiment_config.json")
```

## Reproducibility And Environment

```python
from bicbioseg import Segmenter, environment_info, set_seed

set_seed(42)

print(environment_info())
print(Segmenter.available_devices())
```

## Models, Losses, And Summary

```python
Segmenter.available_models()
Segmenter.available_losses()

model.summary(input_size=(224, 224))
```

## Notes

- Dataset patching is explicit. If `create_patches=True` is not passed, images are copied or resized as full images.
- Training resizes loaded images to `image_size` through the dataset loader.
- For very large images, use `inference_large_image(...)` for tiled prediction.
- DICOM support requires the `dicom` extra.
- Albumentations support requires the `albumentations` extra.
