# Choose a model

[Main guide](../../README.md) · [Documentation](../README.md)

All 15 architectures use the same training, evaluation, and prediction API.
An encoder extracts image features. A decoder converts those features into a mask.
A skip connection passes spatial detail from the encoder to the decoder.

Use `architecture` with `Segmenter`. Use `model` with `SegmentationExperiment`.
Put architecture-specific options in `model_kwargs`.

| Model | Encoder choices | Decoder | Optional installation |
| --- | --- | --- | --- |
| [`unet`](unet.md) | Configurable CNN | Convolutions and skips | None |
| [`cattention_unet`](cattention_unet.md) | Configurable CNN | Convolutions, skips, and CBAM | None |
| [`double_unet`](double_unet.md) | VGG19, ResNet-50, ResNet-101 | Two cascaded U-Nets | None |
| [`transunet`](transunet.md) | Custom CNN or ResNet-50, then local transformer | Convolutions and 0–3 skips | None |
| [`segformer`](segformer.md) | Configurable four-stage Mix Transformer | Multiscale projections | None |
| [`deit`](deit.md) | tiny, small, base, each with optional distilled encoder | Four convolutional upsampling stages | `transformers` |
| [`swin_unet`](swin_unet.md) | tiny, small, base | Convolutional pyramid | `transformers` |
| [`pvt_unet`](pvt_unet.md) | b0, b1, b2, b3, b4, b5 | Convolutional pyramid | `transformers` |
| [`swin_unet_full`](swin_unet_full.md) | tiny, small, base | Swin blocks and patch expansion | `transformers` |
| [`pvtformer_full`](pvtformer_full.md) | b0, b1, b2, b3, b4, b5 | Three-scale residual decoder and fusion | `transformers` |
| [`resunetplusplus_full`](resunetplusplus_full.md) | SE residual CNN | Attention and ASPP | None |
| [`unext_full`](unext_full.md) | base, small, or custom widths | Convolutions and shifted MLPs | None |
| [`dinov3_seg`](dinov3_seg.md) | small, base | Four-depth fusion and trainable convolutional upsampling | `transformers` |
| [`mednext_2d`](mednext_2d.md) | small, base, or custom widths/depth | MedNeXt blocks with learned upsampling and additive skips | None |
| [`efficientvit_seg`](efficientvit_seg.md) | b0, b1, b2, b3 | Additive multiscale fusion and residual MBConv head | `transformers` |

DeiT, Swin, and PVTv2 are separate model choices. They are not TransUNet encoder settings.
Each page lists all constructor options, defaults, aliases, and input constraints.

## Select shared settings

```python
from bicbioseg import Segmenter

model = Segmenter(
    architecture="pvt_unet",
    image_size=(256, 320),
    in_channels=1,
    num_classes=3,
    loss="cross_entropy",
    model_kwargs={"variant": "b0", "pretrained": False},
)
```

Install `bicbioseg[transformers]` before this example. Use `num_classes=1` for binary masks.
Use at least two classes for multiclass masks, including background when present.
See [shared settings](../configuration/segmenter.md) and [losses](../configuration/losses.md).

## Check a setup before training

```python
from bicbioseg import Segmenter

print(Segmenter.available_models())
print(Segmenter.model_options("transunet"))
print(Segmenter.available_models(detailed=True)["transunet"])
print(Segmenter.available_devices())

model = Segmenter(architecture="unet", device="auto")
print(model.validate_setup(backward=True))
print(model.summary())
```

Discovery lists optional models even if their dependencies are absent.
`device="auto"` selects CUDA, then MPS, then CPU, subject to availability.
`validate_setup` checks a forward pass and optionally a backward pass on the selected device.
You can pass `(images, masks)` as `sample_batch` and select `precision`.

## Weights and outputs

All models expose logits to `Segmenter`. The loss and prediction methods apply the required activations.
Pretrained options initialize encoders only. No pretrained segmentation decoder is supplied.
Requested pretrained weights can require network access.

Full Swin-Unet, PVTFormer, ResUNet++, and UNeXt support `deep_supervision=True`.
These models add two auxiliary heads during training. Inference uses only the main prediction.
MedNeXt in 2D also supports deep supervision, with four auxiliary heads during training.
See [auxiliary loss weights](../configuration/training.md).

Use [`Segmenter.save()` and `Segmenter.load()`](../training.md) to retain the model configuration.
Restoring an optional transformer model still requires `timm`, but does not download initialization weights again.

`model_options` returns a copy of each option schema with its default, type, description, and constraints.
Detailed model discovery includes the same schema under `options`. Unknown options fail before model construction or weight downloads.
