from .convert_images import ConvertImage, dicom_to_uint8, tiff_extract_frames
from .preprocess import ImageOps, create_dataset_split, create_kfold_splits, create_train_validate_test_split

__all__ = [
    "ImageOps",
    "ConvertImage",
    "RemoveNoise",
    "create_dataset_split",
    "create_kfold_splits",
    "create_train_validate_test_split",
    "dicom_to_uint8",
    "tiff_extract_frames",
]


def __getattr__(name):
    if name == "RemoveNoise":
        from .other_process import RemoveNoise

        return RemoveNoise

    raise AttributeError(f"module 'bicbioseg.imageops' has no attribute {name!r}")
