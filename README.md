# bicbioseg

`bicbioseg` is a biomedical image segmentation toolkit focused on simple, experiment-friendly Python APIs for preparing data, training common segmentation models, running inference, and comparing results.

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

exp.prepare(split=(0.8, 0.1, 0.1), create_patches=True, patch_size=(224, 224))
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

Create patches after splitting, preventing patches from the same source image from leaking across splits:

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

Config objects are also supported:

```python
from bicbioseg import DatasetSplitConfig, create_dataset_split

config = DatasetSplitConfig(
    split=(0.8, 0.1, 0.1),
    create_patches=True,
    patch_size=(224, 224),
)

create_dataset_split("raw/images", "raw/masks", save_to="dataset", config=config)
```

## QC And Image Operations

```python
from bicbioseg import ImageOps

report = ImageOps.dataset_qc_report(
    images="raw/images",
    masks="raw/masks",
    save_to="qc/qc_report.json",
)

ImageOps.preview_dataset(
    images="raw/images",
    masks="raw/masks",
    num_samples=8,
    save_to="qc/preview.png",
)

mask_type = ImageOps.infer_mask_type("raw/masks")
unmatched = ImageOps.find_unmatched_masks("raw/images", "raw/masks")
```

Useful preprocessing helpers:

```python
ImageOps.resize_dataset("raw/images", "raw/masks", output_dir="resized", image_size=(512, 512))
ImageOps.normalize_masks("raw/masks", output_dir="normalized_masks", mode="binary")

frames, metadata = ImageOps.extract_tiff_frames("stack.tif", get_metadata=True)
ImageOps.convert_dicom("scan.dcm", output_path="scan.png", method="clip")
```

Postprocess and measure predictions:

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

Training writes run artifacts when `experiment_dir` is provided:

```text
experiments/unet_dice/
  config.json
  history.csv
  history.json
  summary.json
  best_model.pt
  final_model.pt
```

Plot curves and sample pairs:

```python
model.plot_history(save_to="experiments/unet_dice/history.png")
model.plot_training_samples(data="cell_dataset", save_to="experiments/unet_dice/training_samples.png")
```

Resume training:

```python
model.train(
    data="cell_dataset",
    epochs=20,
    resume_from="experiments/unet_dice/final_model.pt",
)
```

## Inference

```python
model = Segmenter.load("experiments/unet_dice/best_model.pt")

model.inference(
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
from bicbioseg import Segmenter

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
predictions = model.inference("test/images", save_to="predictions")

model.plot_inference_results(
    images="test/images",
    predictions=predictions,
    save_to="predictions/inference_grid.png",
)
```

## Evaluation And Failure Mining

Binary evaluation:

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

Find the worst predictions:

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

## Experiments

Compare models and losses without rewriting the training loop:

```python
from bicbioseg import Segmenter

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

## Configs, Devices, And Environment

```python
from bicbioseg import Segmenter, SegmenterConfig, TrainingConfig, set_seed, environment_info

set_seed(42)

print(environment_info())
print(Segmenter.available_devices())

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

Save/load workflow config:

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

## Available Models And Losses

```python
Segmenter.available_models()
Segmenter.available_losses()
model.summary(input_size=(224, 224))
```
