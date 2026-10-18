# IS-Net

[All models](README.md) · [Main guide](../../README.md)

IS-Net comes from *Highly Accurate Dichotomous Image Segmentation*
(Qin et al., ECCV 2022). This package implements the full `ISNetDIS` segmentation
network from the [authors' source](https://github.com/xuebinqin/DIS/blob/b6764e20381f6f42a70f83fa3324181529ed1403/IS-Net/models/isnet.py).

Architecture name: `isnet`. Aliases: `is-net`, `isnet_dis`.

The encoder has six stages with residual U-blocks. Each block contains a smaller
encoder and decoder. The two deepest stages use dilated convolutions.
Five decoder stages combine encoder skips with upsampled features.
The first side head produces the final prediction.

## Create the model

```python
from bicbioseg import Segmenter

model = Segmenter(
    architecture="isnet",
    image_size=(256, 320),
    in_channels=1,
    num_classes=1,
    loss="bce",
    model_kwargs={"deep_supervision": True},
)
```

## Configure the model

Pass these options through `model_kwargs`.
Set `in_channels`, `num_classes`, and `image_size` on `Segmenter`.

| Option | Default | Purpose |
| --- | --- | --- |
| `deep_supervision` | `False` | Apply losses to five auxiliary side predictions during training. |
| `normalize_input` | `False` | Subtract 0.5 from inputs in `[0, 1]`. Enable this for upstream pretrained weights. |

Encoder widths are `(64, 128, 256, 512, 512, 512)`.
The default RGB binary model has 44,046,790 parameters.
Widths and block depths match the upstream model and are fixed.
No optional dependency is required.

## Inputs and outputs

The model supports positive spatial dimensions, including odd and rectangular sizes.
Pooling rounds up. Interpolation restores each output to the input size.
No padding to a fixed multiple is added.
Training BatchNorm requires `batch * ceil(H / 64) * ceil(W / 64) > 1`.
For batch size 1, at least one spatial dimension must exceed 64.
Evaluation supports smaller inputs.
The BatchNorm requirement is a minimum. Very small training images can produce
large gradients. Use larger images and set `max_grad_norm` when needed.

Direct tensors can use any positive channel count.
The file loader supports grayscale and RGB images.
The paper uses one output channel for binary foreground masks.
This implementation also allows multiclass outputs through `num_classes`.
Upstream pretrained weights require three input channels and one output channel.

All predictions are logits. `Segmenter` applies the required activation.
With deep supervision enabled, training returns `logits` and five `aux_logits`.
The auxiliary heads are `side2` through `side6`, from fine to coarse.
Evaluation returns only the main logits from `side1`.

Custom `aux_loss_weights` must contain five entries.
The default is 0.4 per auxiliary head.
Use `loss="bce"` and `aux_loss_weights=(1, 1, 1, 1, 1)` to sum the six mask losses
with equal weights, as in the upstream output supervision loss.
See [training configuration](../configuration/training.md).

The paper also trains a separate `ISNetGTEncoder` on ground-truth masks.
It then uses that encoder's features to supervise the segmentation network.
This package does not include that two-stage training procedure or feature loss.
`deep_supervision=True` enables output mask losses only.
Training here does not reproduce the paper's complete training recipe.

## Load upstream weights

Initialization is random. There is no automatic weight download.
Download an `ISNetDIS` checkpoint from the
[authors' repository](https://github.com/xuebinqin/DIS#6-run-our-code).
Then load the plain state dictionary:

```python
from bicbioseg import Segmenter

model = Segmenter(
    architecture="isnet",
    image_size=(1024, 1024),
    in_channels=3,
    num_classes=1,
    model_kwargs={"normalize_input": True},
)
model.load_weights("isnet-general-use.pth")
model.save("isnet_bicbioseg.pt")
```

Parameter names and shapes match upstream `ISNetDIS`.
Use `load_weights` for plain upstream weights.
Use `Segmenter.load` for a saved bicbioseg checkpoint.
Ground-truth encoder checkpoints are not compatible.
This package does not reproduce the upstream inference script's uint8 resize
rounding or per-image min-max scaling of predictions.

## Direct PyTorch use

```python
import torch
from bicbioseg.models import ISNetDIS

model = ISNetDIS(in_channels=3, n_classes=1).eval()
with torch.no_grad():
    logits = model(torch.rand(1, 3, 65, 97))
    probabilities = logits.sigmoid()
```

Set `normalize_input=False` if you have already subtracted 0.5 from the input.
The source adaptation retains the upstream Apache-2.0 license.
See the [attribution](../../src/bicbioseg/models/licenses/NOTICE.txt).

Continue with [training](../training.md), [shared settings](../configuration/segmenter.md),
or the [source](../../src/bicbioseg/models/isnet.py).
