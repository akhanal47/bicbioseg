# CAttention U-Net architecture review

`cattention_unet` / `CAttentionUNet` shares the package's U-Net implementation and
configuration. It adds CBAM after skip concatenation in every decoder stage.
The earlier fixed-width model and its compatibility aliases have been replaced;
old checkpoints are not supported by this configurable version.

## Configuration

The constructor matches U-Net: `n_channels=3`, `n_classes=1`,
`num_decoder_blocks=4`, `base_channels=16`, `bilinear=False`, `image_size=None`.
Depth supports 1–8 decoder blocks; base width and input/output channels must be
positive integers. Both upsampling modes use the same channel progression as
U-Net, including the full-width bottleneck in bilinear mode.

Through `Segmenter`, use `in_channels`, `num_classes`, and `image_size` directly,
and pass depth, width, and upsampling via `model_kwargs`. Integer, rectangular,
and odd image dimensions are supported, subject to the depth minimum.
`image_size` validates construction without fixing the inference resolution;
the forward pass also validates actual tensor dimensions.

`calculate_max_decoder_blocks(image_size)` reports the spatial depth limit,
capped at 8. `get_architecture_info()` reports U-Net's architecture fields plus
`attention_parameters`. As with U-Net, increasing depth and width can grow
memory requirements substantially; spatial validity does not guarantee memory fit.

## Computation and placement

At each decoder level (four by default):

```text
decoder feature → upsample → align to skip size → concatenate with encoder skip
                → channel attention → spatial attention → DoubleConv
```

Let F have shape N×C×H×W after concatenation. This implementation computes:

```text
g(z) = W1 ReLU(W0 z)
Mc(F) = sigmoid(g(mean_HW(F)) + g(max_HW(F)))       # N×C×1×1
F' = Mc(F) * F
Ms(F') = sigmoid(Conv7×7([mean_C(F'), max_C(F')]))  # N×1×H×W
F'' = Ms(F') * F'
decoder output = DoubleConv(F'')
```

Multiplication broadcasts along the remaining axes. The two channel descriptors
share W0 and W1. The bias-free 1×1 convolutions on pooled descriptors implement
the MLP, with hidden width `max(1, floor(C/8))`. The spatial convolution maps two channels to one,
with padding 3. These operations follow the sequential attention equations in
[Woo et al., CBAM (2018)](https://arxiv.org/html/1807.06521).

With depth D and base width B, the encoder channels are B, 2B, ..., 2^D B.
At a decoder with skip width S, the concatenated width is 2S for transposed
convolution or 3S for bilinear upsampling. The attention hidden width is clamped
to one for small channel counts. Attention preserves shape; padding after
upsampling handles odd sizes. Both input dimensions must be at least 2^D.
Training BatchNorm requires more than one value per channel at the bottleneck
(batch size 2 suffices at the minimum image size). The final 1×1 convolution
returns raw logits for binary or multiclass losses.

This is a CBAM variant: reduction ratio 8, bias-free channel projections, and no
BatchNorm inside spatial attention. The
[authors' reference code](https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py)
defaults to ratio 16, biased linear layers, and spatial BatchNorm. These are
intentional architectural differences to disclose, not mathematical errors.
This model also differs from the skip attention gates of
[Oktay et al., Attention U-Net (2018)](https://arxiv.org/abs/1804.03999).

There is one CBAM per decoder block, each acting on the combined skip and decoder
features before DoubleConv. There is no separate encoder, bottleneck, or output
CBAM. The paper's diagram should show that placement and count explicitly.

## Parameter overhead

For each block, the number of trainable parameters is
`2*C*max(1, floor(C/8)) + 2*7*7`. Pooling, sigmoid, ReLU, and multiplication add no parameters.
The shared channel MLP is evaluated twice but its parameters are counted once.

Counts below use four decoder blocks, RGB input, and one output class,
including convolution biases and BatchNorm affine parameters, excluding
non-trainable buffers. The 16-channel configurations use the new defaults;
64 channels provide a wider experiment option.

| Base width | Upsampling | Matched U-Net | CAttention U-Net | CBAM parameters | Added |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | Transposed convolution (default) | 1,944,049 | 1,966,201 | 22,152 | 1.139% |
| 16 | Bilinear | 2,357,609 | 2,406,961 | 49,352 | 2.093% |
| 64 | Transposed convolution | 31,043,521 | 31,392,073 | 348,552 | 1.123% |
| 64 | Bilinear | 37,659,041 | 38,442,793 | 783,752 | 2.081% |

This supports a claim of modest **parameter** overhead. It does not establish
negligible runtime or memory overhead: pooling and gating traverse decoder
feature maps, including the full-resolution map. Measure latency and peak memory
on the intended image sizes and hardware.

## Scientific interpretation and planned comparison

The design is mathematically coherent. A plausible mechanism is that channel
gating selects useful feature types while spatial gating suppresses irrelevant
locations before skip/decoder fusion convolutions. Decoder features supply
context and skips supply spatial detail. This is a hypothesis about this model,
not an observed segmentation improvement.

Sigmoid gates only attenuate feature magnitudes directly. If both gate logits
are zero, their product scales a feature by 1/4; they are not identity gates at
zero. Learned suppression can help reject noise but can also discard subtle
foreground detail or impede gradients when gates saturate. There is no residual
bypass around CBAM. These are tradeoffs to test, not reasons to change the model
before obtaining evidence. The original CBAM results concern classification and
detection and do not establish gains for this package's segmentation datasets.

For an initial controlled experiment, use:

```python
from bicbioseg import Segmenter

options = {"base_channels": 16, "num_decoder_blocks": 4, "bilinear": False}
baseline = Segmenter(architecture="unet", model_kwargs=options)
candidate = Segmenter(architecture="cattention_unet", model_kwargs=options)
```

The same matched comparison also works with `bilinear=True`, arbitrary positive
base widths, and any supported depth. Regression tests copy matching weights and
confirm identical outputs when CBAM is replaced by identity, in both upsampling
modes. Train each model from scratch; disabling attention only at inference on a
trained model is a different experiment.

Keep train/validation/test splits, preprocessing, augmentation, losses, optimizer,
schedule, stopping rules, and tuning budget matched. Split at the patient/specimen
level where applicable. Use multiple paired seeds and, for a tightly controlled
ablation, copy corresponding initial backbone weights (the same seed alone does
not ensure this because CBAM initialization consumes random numbers). Report
held-out Dice/IoU and per-case paired differences with uncertainty, plus parameter
counts, latency, and peak memory. Optional later ablations can isolate channel-only,
spatial-only, and combined attention. No training comparison is claimed here.

Suggested wording before experiments: “CAttention U-Net applies sequential
channel and spatial attention to concatenated decoder features at four scales,
adding 22,152 parameters (1.14% over the matched 16-channel transposed-convolution
U-Net).
We hypothesize that this feature recalibration improves segmentation accuracy.”
Attribute CBAM to Woo et al.; naming this integration does not itself establish
architectural novelty.
