from .imageops.convert_images import ConvertImage, dicom_to_uint8, tiff_extract_frames
from .imageops.preprocess import ImageOps, create_dataset_split, create_train_validate_test_split


__all__ = [
    "create_dataset_split",
    "create_train_validate_test_split",
    "ImageOps",
    "ConvertImage",
    "RemoveNoise",
    "dicom_to_uint8",
    "tiff_extract_frames",
    "Segmenter",
    "AugmentImages",
    "AugmentationPipeline",
    "apply_and_save_augmentations",
    "DataLoader",
]


def __getattr__(name):
    if name == "Segmenter":
        from .trainer import Segmenter

        return Segmenter

    if name == "DataLoader":
        from .utils.load_data import DataLoader

        return DataLoader

    if name == "RemoveNoise":
        from .imageops.other_process import RemoveNoise

        return RemoveNoise

    if name in {"AugmentImages", "AugmentationPipeline", "apply_and_save_augmentations"}:
        from .utils.augmentation import AugmentImages, AugmentationPipeline, apply_and_save_augmentations

        lazy_exports = {
            "AugmentImages": AugmentImages,
            "AugmentationPipeline": AugmentationPipeline,
            "apply_and_save_augmentations": apply_and_save_augmentations,
        }
        return lazy_exports[name]

    raise AttributeError(f"module 'bicbioseg' has no attribute {name!r}")
