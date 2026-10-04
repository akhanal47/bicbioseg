# Swin with convolutional decoder

[All models](README.md) · [Main guide](../../README.md)

Use a transformer encoder with a segmentation decoder.

Architecture name: `swin_unet`. Aliases: swin, swinunet.

```mermaid
flowchart LR
    s0["Image"]
    s1["Four Swin encoder scales"]
    s2["Convolutional pyramid decoder"]
    s3["Class scores"]
    s0 --> s1 --> s2 --> s3
```

The diagram shows the main flow. Skip connections carry encoder features into the decoder where described.

## Create the model

```python
from bicbioseg import Segmenter

model = Segmenter(
    architecture="swin_unet",
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
| `decoder_channels` | `128` | One positive width for the convolutional pyramid decoder. |
| `dropout` | `0.1` | Dropout probability in the decoder or output head. Use `0 <= dropout < 1`. |
| `freeze_encoder` | `False` | Set `True` to disable encoder gradients and keep the encoder in evaluation mode. |
| `normalize_input` | `True` | Apply ImageNet mean and standard deviation inside the model. See [normalization](../configuration/normalization.md). |


## Inputs and behavior

Choose `tiny`, `small`, or `base`. This model uses a convolutional decoder. For Swin blocks in the decoder, choose [full Swin-Unet](swin_unet_full.md).

Inputs are padded to multiples of 32. Outputs are cropped to the original size.

Use one grayscale channel or three RGB channels. Grayscale repeats to RGB inside the model. Rectangular and odd sizes are supported.

Install `bicbioseg[transformers]`. The model uses `timm>=1.0.25,<2`. Pretrained weights apply only to the encoder. A weight download can occur when `pretrained=True`.

The encoder weight identifier is `swin_{variant}_patch4_window7_224.ms_in1k`.

## Direct PyTorch use

Import `SwinUNet` from `bicbioseg.models.transformer_segmentation`. The input/output constructor arguments are `in_channels`, `n_classes`.
`Segmenter` supplies these arguments from its shared settings. Do not pass conflicting values in `model_kwargs`.

For direct constructors, input channels default to 3. Output classes default to 1.


Inputs use `(batch, channels, height, width)`. Outputs contain raw class scores, called logits.


Continue with [training](../training.md), [shared configuration](../configuration/segmenter.md), or the [source](../../src/bicbioseg/models/transformer_segmentation.py).
