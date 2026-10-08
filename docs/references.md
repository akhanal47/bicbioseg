# Acknowledgments and implementation references

[Main guide](../README.md) · [Models](models/README.md)

The project acknowledges these model and implementation references:

- [JunZengz/dental-caries-segmentation](https://github.com/JunZengz/dental-caries-segmentation): model collection and decoder design references.
- [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet): CNN/transformer hybrid architecture.
- [facebookresearch/deit](https://github.com/facebookresearch/deit): DeiT and distilled DeiT encoders.
- [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer): hierarchical shifted-window encoders.
- [HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet): U-shaped transformer segmentation.
- [whai362/PVT](https://github.com/whai362/PVT): pyramid vision transformers.
- [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3): DINOv3 encoder features, paired here with a trainable bicbioseg decoder.
- [MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt): MedNeXt v1 blocks and encoder-decoder, adapted for 2D segmentation.
- [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit): EfficientViT B-series encoder and lightweight segmentation head.

MedNeXt and EfficientViT source adaptations retain their [Apache-2.0 license](../src/bicbioseg/models/licenses/Apache-2.0.txt)
and [attribution notices](../src/bicbioseg/models/licenses/NOTICE.txt).
DINOv3 pretrained weights retain the upstream [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).

The model pages describe the implementations and supported weights in this package.
A shared architecture name does not guarantee compatibility with a research repository's checkpoints.
