# bicbioseg

`bicbioseg` is a biomedical image segmentation toolkit for preparing image datasets, training common segmentation models, running inference, and comparing experiments with a simple Python API.

## Status

This project is in early release & active developemnt. APIs may evolve or changes with each releases as more biomedical workflows are added.

Architectures: U-Net, CAttention U-Net, DoubleUNet, SegFormer, TransUNet, ResUNet++, UNeXt, and optional DeiT, Swin, PVTv2, full Swin-Unet, and PVTFormer segmentation models.
DoubleUNet currently supports RGB binary segmentation only. TransUNet supports rectangular images with both dimensions divisible by 16, including an optional pretrained ResNet-50 CNN encoder.

## Installation

```bash
pip install bicbioseg
```

For DeiT, Swin, and PVTv2 encoders, install the optional maintained `timm` backend:

```bash
pip install 'bicbioseg[transformers]'
```

For full usages only, some of the TransUnet implementation e.g ResNet-50 TransUNet option, do not require `timm`.

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

## Model Options

All models plug into the same `Segmenter` and `SegmentationExperiment` training, evaluation, inference, and checkpoint APIs. Pass architecture options through `model_kwargs`. `Segmenter.available_models()` lists registered names, including models whose optional dependency may still need installation. New models return raw logits; the selected loss and inference API handle activations.

| Architecture | Encoder / variants | Initialization | Input size |
| --- | --- | --- | --- |
| `cattention_unet` | Configurable U-Net + decoder CBAM; `base_channels=16`, `num_decoder_blocks=4` | Random | Height and width >= `2**num_decoder_blocks`; odd sizes supported |
| `transunet` | Local CNN; `lightweight`, `standard`, `heavy` presets | Random | Height and width divisible by 16 |
| `transunet` with `encoder_name="resnet50"` | Torchvision ResNet-50 through layer3 + local transformer | Optional ImageNet CNN weights | Height and width divisible by 16 |
| `deit` | `tiny` (default), `small`, `base`; optional `distilled=True` | Optional ImageNet encoder weights | Rectangular and odd sizes; padded to 16 then cropped |
| `swin_unet` | Swin `tiny` (default), `small`, `base` | Optional ImageNet encoder weights | Rectangular and odd sizes; padded to 32 then cropped |
| `pvt_unet` | PVTv2 `b0` (default) through `b5` | Optional ImageNet encoder weights | Rectangular and odd sizes; padded to 32 then cropped |
| `swin_unet_full` | Swin `tiny` (default), `small`, `base`; Swin decoder, patch expansion and skip fusion | Optional ImageNet encoder weights; new decoder | Padded to 32, cropped to original size |
| `pvtformer_full` | PVTv2 `b3` (default), `b0`–`b5`; three-scale residual decoder and full-resolution fusion | Optional ImageNet encoder weights; new decoder | Padded to 32, cropped to original size |
| `resunetplusplus_full` | SE residual blocks, attention gates and bridge/output ASPP; `base_channels=16` | Random | Padded to 8, minimum 16; cropped to original size |
| `unext_full` | Convolutional and shifted token MLP stages; `base` (default) or `small` | Random | Padded to 32, cropped to original size |

Aliases include `trans_unet`, `deit_seg`, `swin` / `swinunet`, and `pvt` / `pvtv2`. The new pretrained encoder paths support RGB and grayscale inputs and binary or multiclass outputs. Grayscale is repeated to RGB internally. The custom TransUNet CNN also supports other channel counts when normalization is disabled.

### CAttention U-Net

Use `Segmenter(architecture="cattention_unet")` or import `CAttentionUNet` from
`bicbioseg.models.cattention_unet`. It shares U-Net's backbone, constructor options,
and defaults: `base_channels=16`, `num_decoder_blocks=4` (supported range 1–8),
`bilinear=False`, `n_channels=3`, and `n_classes=1`. `image_size` validates the
selected depth; it does not lock the model to one resolution.

```python
model = Segmenter(
    architecture="cattention_unet",
    image_size=(129, 193),
    in_channels=1,
    num_classes=3,
    model_kwargs={"base_channels": 8, "num_decoder_blocks": 5, "bilinear": True},
)
print(model.model.get_architecture_info())
```

Each decoder stage runs **upsample → concatenate skip → channel attention → spatial
attention → double convolution**. There is one CBAM per decoder, with reduction
ratio 8 and a 7×7 spatial kernel; the encoder and bottleneck have no CBAM. The
channel attention hidden width is at least one, including very narrow models.
`CAttentionUNet.calculate_max_decoder_blocks(image_size)` reports the depth limit.
Training BatchNorm needs multiple values per channel at the bottleneck.

With matching options, U-Net and CAttention U-Net differ only by CBAM in either
upsampling mode. See the [architecture review](docs/cattention_unet.md) for equations,
parameter counts, and the proposed controlled comparison. This configurable
version replaces the fixed-width architecture and its legacy names/checkpoints.

### Full architectures, discovery and preflight

```python
import torch
from bicbioseg import Segmenter

models = Segmenter.available_models(detailed=True)
print(models["swin_unet_full"])

model = Segmenter(
    architecture="swin_unet_full",
    image_size=(224, 320),
    in_channels=1,
    device="auto",
    model_kwargs={"variant": "tiny", "deep_supervision": True},
)

model.validate_setup((torch.rand(2, 1, 224, 320), torch.zeros(2, 224, 320)), backward=True)
```

`device="auto"` sets MPS on macOS when CUDA not available, then CPU. Explicit `"cpu"`, `"mps"`, `"cuda"`, and `"cuda:1"` are supported; unavailable backends and invalid CUDA fails. Please confirm that your device has the right CUDA and Torch Version bundled before running any experiments

### Configurable TransUNet

```python
from bicbioseg import Segmenter

model = Segmenter(
    architecture="transunet",
    image_size=(224, 320),
    in_channels=1,
    num_classes=3,
    loss="cross_entropy",
    model_kwargs={
        "preset": "lightweight",
        "out_channels": 32,                  # CNN width; also controls the bottleneck projection
        "embedding_dim": 256,                # transformer width, independent of CNN width
        "head_num": 8,                       # must divide embedding_dim
        "mlp_dim": 1024,
        "block_num": 4,
        "decoder_channels": [128, 64, 32, 16], # four stages, deepest to shallowest
        "dropout": 0.1,
        "n_skip": 3,                         # 0–3 skip connections, deepest first
    },
)
```

Explicit values override the preset. If omitted, `embedding_dim` is `8 * out_channels`, and decoder widths are derived from `out_channels`. `patch_dim=16` remains fixed because it describes the CNN stride; it is not a tunable raw-image patch size. Position embeddings are interpolated if the forward input uses a different valid spatial size. CNN and transformer widths, MLP size, depth, heads, decoder widths, dropout, and skip count are configurable. ResNet-50 has fixed CNN widths; there, `out_channels` controls the post-transformer bottleneck and default decoder widths.

To initialize the CNN encoder from maintained torchvision weights:

```python
model = Segmenter(
    architecture="transunet",
    image_size=(224, 224),
    model_kwargs={
        "encoder_name": "resnet50",
        "encoder_weights": "IMAGENET1K_V2",  # or IMAGENET1K_V1, DEFAULT, None
        "out_channels": 64,
        "embedding_dim": 512,
        "head_num": 8,
        "mlp_dim": 2048,
        "block_num": 6,
        "decoder_channels": [256, 128, 64, 32],
    },
)
```

`encoder_weights=None` (the default) uses random initialization. Weights are downloaded only when requested and cached by torchvision. Only the ResNet stem and layers 1–3 are retained, giving stride-16 features and three CNN skips. The local transformer and decoder are trained from scratch. Weight download or compatibility errors propagate; there is no silent fallback to random weights.

This is a **ResNet-50-backed TransUNet variant**, not the paper's exact R50–ViT-B/16 architecture. The [official TransUNet repository](https://github.com/Beckschen/TransUNet) notes that its original Google weight links expired and provides an alternative project-folder copy. Its hybrid checkpoint uses a different ResNetV2 and transformer layout, so those `.npz` weights cannot be loaded into this implementation. `pretrained_vit=True` remains explicitly unsupported. The supported CNN weights come from [torchvision ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).

The default custom encoder retains its parameter names and shapes for existing configurations. Attention scaling has been corrected to divide by the square root of head width, so existing TransUNet checkpoints remain loadable but their predictions can change; re-evaluate before reuse.

### DeiT, Swin, and PVTv2 Segmentation

```python
deit = Segmenter(
    architecture="deit",
    image_size=(224, 320),
    model_kwargs={"variant": "tiny", "pretrained": True,
                  "decoder_channels": [128, 64, 32, 16]},
)

swin = Segmenter(
    architecture="swin_unet",
    image_size=(256, 320),
    model_kwargs={"variant": "tiny", "pretrained": True,
                  "decoder_channels": 128, "freeze_encoder": False},
)

pvt = Segmenter(
    architecture="pvt_unet",
    image_size=(256, 320),
    model_kwargs={"variant": "b0", "pretrained": False,
                  "decoder_channels": 128},
)
```

These options use `timm`'s named ImageNet weights: `deit_{variant}[_distilled]_patch16_224.fb_in1k`, `swin_{variant}_patch4_window7_224.ms_in1k`, and `pvt_v2_{variant}.in1k`. All default to `pretrained=False`, so constructing them without pretrained weights requires no network. Set `freeze_encoder=True` to train just the segmentation decoder; the encoder stays in evaluation mode even during training. `dropout` configures decoder dropout; pretrained backbone dimensions and attention structure remain those of the selected variant.

DeiT decodes patch tokens through four convolutional upsampling stages. With `distilled=True`, both prefix tokens participate in encoder attention and are removed before spatial decoding; the wrapper does not implement teacher/student distillation training. Swin and PVTv2 use a local convolutional decoder that fuses four encoder scales. These are adaptations inspired by the research models, rather than reproductions of the pure-transformer Swin-Unet decoder or PVTFormer's exact decoder. No pretrained segmentation decoder is provided.

**Preprocessing and checkpoints:** ResNet-50 TransUNet and all `timm` adaptations normalize `[0, 1]` RGB inputs using ImageNet mean/std internally, regardless of whether initialization is pretrained. Custom TransUNet defaults to no normalization. The package's normal dataset/inference preprocessing already supplies `[0, 1]` tensors; do not normalize them a second time. Advanced users supplying already-normalized tensors directly to the model can set `normalize_input=False` (grayscale is still repeated for RGB encoders). Normalization and grayscale handling are identical during training and inference. `Segmenter.save()` stores the entire model; `Segmenter.load()` restores it without redownloading initialization weights while retaining the saved configuration. Restoring a `timm` model still requires the optional dependency.

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

### Memory controls and resumable schedules

```python
model.train(
    data="cell_dataset",
    epochs=50,
    batch_size=2,
    precision="fp32",             # CUDA: fp16 or bf16; CPU: bf16; MPS: fp32
    accumulation_steps=4,         # one optimizer update per four microbatches
    max_grad_norm=1.0,
    scheduler="cosine",           # also "step", "plateau", or None
    scheduler_kwargs={"T_max": 50, "eta_min": 1e-6},
    aux_loss_weights=(0.4, 0.2),   # two heads when deep_supervision=True
    experiment_dir="experiments",
    run_name="full_swin",
)
```

### Microscopy normalization and ignored annotations

```python
model = Segmenter(
    architecture="unext_full",
    in_channels=1,
    image_size=(256, 256),
    normalization={"mode": "dtype"},  # uint16 / 65535, uint8 / 255
    ignore_index=255,                  # reserve 255 for unannotated pixels
    model_kwargs={"variant": "small"},
)
model.train(
    data="microscopy_dataset",
    crop_size=(512, 512),
    foreground_probability=0.75,
    batch_size=2,
)
```

Normalization policies are saved in `SegmenterConfig`, `ExperimentConfig` and checkpoints and shared by training and inference:

| Mode | Behavior |
| --- | --- |
| `standard` (default) | uint8 divided by 255; finite floats already in [0, 1] |
| `dtype` | Unsigned integer images divided by their dtype maximum |
| `range` | Clip and scale a supplied `min`/`max`, e.g. `{"mode": "range", "min": 0, "max": 4095}` |
| `percentile` | Per-image clipping/scaling using `lower`/`upper` percentiles, defaults 1/99; constant images become zero |

Set `ignore_index` on `Segmenter`, not only inside loss arguments, to keep losses and metrics aligned.

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
    tile_batch_size=4,
    weighting="gaussian",
    save_to="large_predictions",
)
```

`tile_batch_size` controls how many tiles share a forward pass. `weighting="gaussian"` downweights tile edges (`gaussian_sigma=0.125`, as a fraction of tile size); `"uniform"` preserves the previous averaging behavior and remains the default. Partial edge tiles are padded, cropped and normalized by their accumulated weights.
Ensemble multiple checkpoints:

```python
Segmenter.ensemble_predict(
    checkpoints=[
        "experiments/unet_dice/best_model.pt",
        "experiments/cattention_unet_dice/best_model.pt",
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
    architectures=["unet", "cattention_unet"],
    losses=["dice", "jaccard"],
    metrics=["dice", "iou"],
    epochs=30,
    batch_size=8,
    output_dir="experiments",
)
```

The default U-Net and CAttention U-Net configurations share the same backbone.
For a controlled CBAM experiment, keep all model and training options matched; see the
[controlled comparison setup](docs/cattention_unet.md#scientific-interpretation-and-planned-comparison).

For separate model widths, losses, seeds and training settings, use named runs:

```python
from bicbioseg import ExperimentRunConfig, SegmenterConfig, TrainingConfig

runs = {
    "swin_tiny": ExperimentRunConfig(
        segmenter=SegmenterConfig(architecture="swin_unet_full", model_kwargs={"variant": "tiny"}),
        training=TrainingConfig(epochs=50, batch_size=2, accumulation_steps=4),
        seed=42,
    ),
    "pvt_b0": ExperimentRunConfig(
        segmenter=SegmenterConfig(architecture="pvtformer_full", model_kwargs={"variant": "b0", "decoder_channels": 32}),
        training=TrainingConfig(epochs=50, batch_size=4),
        seed=43,
    ),
}
results = Segmenter.run_experiment("cell_dataset", runs=runs, output_dir="comparisons")
runs["swin_tiny"].save("swin_run.json")
```

Each run saves its effective configuration, seed, environment, elapsed time and throughput in `experiment.json`, alongside normal checkpoints and history.

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

## Acknowledgments And Model References

Some of the model implementation references

- [JunZengz/dental-caries-segmentation](https://github.com/JunZengz/dental-caries-segmentation), especially its [model collection](https://github.com/JunZengz/dental-caries-segmentation/tree/main/models): Segmentation catalog & Swin and PVT decoder designs.

- [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet): the CNN/transformer hybrid architecture & R50–ViT pretrained checkpoints.

- [facebookresearch/deit](https://github.com/facebookresearch/deit): DeiT and distilled DeiT encoder architectures and weights.

- [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and [HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet): hierarchical shifted-window encoders and U-shaped segmentation inspiration.

- [whai362/PVT](https://github.com/whai362/PVT): pyramid vision transformers for dense prediction.
