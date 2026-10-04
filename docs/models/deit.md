# DeiT segmentation

[All models](README.md) · [Main guide](../../README.md)

Use a transformer encoder with a segmentation decoder.

Architecture name: `deit`. Aliases: deit_seg.

```mermaid
flowchart LR
    s0["Image"]
    s1["DeiT patch tokens"]
    s2["Remove prefix tokens"]
    s3["Four convolutional upsampling stages"]
    s4["Class scores"]
    s0 --> s1 --> s2 --> s3 --> s4
```

The diagram shows the main flow. Skip connections carry encoder features into the decoder where described.

## Create the model

```python
from bicbioseg import Segmenter

model = Segmenter(
    architecture="deit",
    image_size=(224, 320),
    in_channels=3,
    num_classes=1,
    model_kwargs={"variant": "tiny", "pretrained": False},
)
```

## Configure the model

Pass these options through `model_kwargs`. Set `in_channels`, `num_classes`, and `image_size` on `Segmenter`.

| Option | Default | Purpose and accepted values |
| --- | --- | --- |
| `variant` | `'tiny'` | Encoder size. Supported values appear below. |
| `pretrained` | `False` | Set `True` to request ImageNet encoder weights. The decoder starts from random weights. |
| `distilled` | `False` | Select the distilled DeiT encoder architecture. |
| `decoder_channels` | `(128, 64, 32, 16)` | Four positive decoder widths, deepest to shallowest. |
| `dropout` | `0.1` | Dropout probability in the decoder or output head. Use `0 <= dropout < 1`. |
| `freeze_encoder` | `False` | Set `True` to disable encoder gradients and keep the encoder in evaluation mode. |
| `normalize_input` | `True` | Apply ImageNet mean and standard deviation inside the model. See [normalization](../configuration/normalization.md). |


## Inputs and behavior

Choose `tiny`, `small`, or `base`. Set `distilled=True` for the distilled encoder. Both prefix tokens participate in attention. The decoder removes them before it creates spatial maps. This option does not add teacher/student distillation training.

Inputs are padded to multiples of 16. Outputs are cropped to the original size.

Use one grayscale channel or three RGB channels. Grayscale repeats to RGB inside the model. Rectangular and odd sizes are supported.

Install `bicbioseg[transformers]`. The model uses `timm>=1.0.25,<2`. Pretrained weights apply only to the encoder. A weight download can occur when `pretrained=True`.

The encoder weight identifier is `deit_{variant}[_distilled]_patch16_224.fb_in1k`.

## Direct PyTorch use

Import `DeiTSegmenter` from `bicbioseg.models.transformer_segmentation`. The input/output constructor arguments are `in_channels`, `n_classes`.
`Segmenter` supplies these arguments from its shared settings. Do not pass conflicting values in `model_kwargs`.

For direct constructors, input channels default to 3. Output classes default to 1.


Inputs use `(batch, channels, height, width)`. Outputs contain raw class scores, called logits.


Continue with [training](../training.md), [shared configuration](../configuration/segmenter.md), or the [source](../../src/bicbioseg/models/transformer_segmentation.py).
