import random
from scipy import ndimage
import numpy as np
import cv2
import os
import glob
from pathlib import Path

class AugmentImages:
    @staticmethod
    def _check_type(img, mask):
        if img.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
            raise ValueError("Images must be uint8, uint16, or normalized float arrays.")
        if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[-1] != 3):
            raise ValueError("Images must be grayscale HxW or RGB HxWx3.")
        if not np.isfinite(img).all() or not img.size:
            raise ValueError("Images must be nonempty and finite.")
        if np.issubdtype(img.dtype, np.floating) and (img.min() < 0 or img.max() > 1):
            raise ValueError("Float images must already lie in [0, 1].")
        if mask is not None and (
            mask.ndim != 2 or mask.shape != img.shape[:2] or not np.issubdtype(mask.dtype, np.integer)
        ):
            raise ValueError("Masks must be integer HxW class IDs matching the image.")

    @staticmethod
    def _maximum(image):
        return float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else 1.0

    @staticmethod
    def _restore(array, image):
        return np.clip(array, 0, AugmentImages._maximum(image)).astype(image.dtype)

    @staticmethod
    def rotate(image, mask=None, angle_range=(5, 30), resize_target=None, probability=1.0):
        AugmentImages._check_type(image, mask)
        if random.random() > probability:
            return image, mask
        angle = random.uniform(*angle_range) * random.choice((-1, 1))
        result = AugmentImages._restore(
            ndimage.rotate(image, angle, axes=(0, 1), reshape=False, order=3), image
        )
        labels = (
            ndimage.rotate(mask, angle, reshape=False, order=0, prefilter=False) if mask is not None else None
        )
        if resize_target is not None:
            height, width = resize_target
            result = AugmentImages._restore(
                cv2.resize(result, (width, height), interpolation=cv2.INTER_CUBIC), image
            )
            if labels is not None:
                labels = cv2.resize(
                    labels.astype(np.float64), (width, height), interpolation=cv2.INTER_NEAREST
                ).astype(mask.dtype)
        return result, labels

    @staticmethod
    def flip(image, mask=None, mode="random", probability=1.0):
        AugmentImages._check_type(image, mask)
        modes = {"h": 1, "horizontal": 1, "v": 0, "vertical": 0, "vh": (0, 1), "both": (0, 1)}
        if mode != "random" and mode not in modes:
            raise ValueError("flip mode must be horizontal/h, vertical/v, both/vh, or random.")
        if random.random() > probability:
            return image, mask
        axis = random.choice((0, 1, (0, 1))) if mode == "random" else modes[mode]
        return np.flip(image, axis=axis).copy(), np.flip(mask, axis=axis).copy() if mask is not None else None

    @staticmethod
    def adjust_brightness(image, mask=None, delta_range=(-30, 30), probability=1.0):
        AugmentImages._check_type(image, mask)
        if random.random() > probability:
            return image, mask
        # Delta is in 8-bit units for all dtypes: 255 means the full intensity range.
        delta = random.uniform(*delta_range) / 255 * AugmentImages._maximum(image)
        return AugmentImages._restore(image.astype(np.float64) + delta, image), mask

    @staticmethod
    def hist_equalize(image, mask=None, clipLimit=3, tileGridSize=(8, 8), probability=1.0):
        AugmentImages._check_type(image, mask)
        if random.random() > probability:
            return image, mask
        maximum = AugmentImages._maximum(image)
        normalized = image.astype(np.float32) / maximum
        color = cv2.cvtColor(normalized, cv2.COLOR_RGB2YCrCb) if image.ndim == 3 else None
        luminance = color[..., 0] if color is not None else normalized
        bits = 255 if image.dtype == np.uint8 else 65535
        quantized = np.rint(np.clip(luminance, 0, 1) * bits).astype(np.uint8 if bits == 255 else np.uint16)
        equalized = (
            cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            .apply(quantized)
            .astype(np.float32)
            / bits
        )
        if color is not None:
            color[..., 0] = equalized
            equalized = cv2.cvtColor(color, cv2.COLOR_YCrCb2RGB)
        return AugmentImages._restore(equalized * maximum, image), mask

    @staticmethod
    def random_crop(image, mask=None, crop_size=(224, 224), probability=1.0):
        AugmentImages._check_type(image, mask)
        if len(crop_size) != 2 or any(not isinstance(v, int) or v < 1 for v in crop_size):
            raise ValueError("crop_size must contain two positive integers.")
        if random.random() > probability:
            return image, mask
        h, w = image.shape[:2]
        ch, cw = crop_size
        if h < ch or w < cw:
            result = AugmentImages._restore(cv2.resize(image, (cw, ch), interpolation=cv2.INTER_CUBIC), image)
            labels = (
                cv2.resize(mask.astype(np.float64), (cw, ch), interpolation=cv2.INTER_NEAREST).astype(
                    mask.dtype
                )
                if mask is not None
                else None
            )
            return result, labels
        y, x = random.randint(0, h - ch), random.randint(0, w - cw)
        return image[y : y + ch, x : x + cw].copy(), (
            mask[y : y + ch, x : x + cw].copy() if mask is not None else None
        )


class AugmentationPipeline:
    def __init__(self, augmentations=None):
        self.augmentations = augmentations if augmentations is not None else []

    def __call__(self, image, mask):
        for aug_func in self.augmentations:
            image, mask = aug_func(image, mask)
        return image, mask

    def preview(self, image, mask, save_to=None, show=False):
        """Preview a paired transform and report whether it introduced new labels."""
        import matplotlib.pyplot as plt

        AugmentImages._check_type(image, mask)
        output, labels = self(image.copy(), mask.copy())
        before = np.unique(mask).tolist()
        after = np.unique(labels).tolist()
        unexpected = sorted(set(after) - set(before) - {0})
        valid = not unexpected and np.issubdtype(labels.dtype, np.integer)
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for axis, array, title in zip(
            axes.flat,
            (image, mask, output, labels),
            ("Original RGB image", "Original labels", "Augmented RGB image", f"Labels preserved: {valid}"),
        ):
            display = array / AugmentImages._maximum(array) if array.ndim == 3 else array
            axis.imshow(display, cmap="gray" if array.ndim == 2 else None, interpolation="nearest")
            axis.set_title(title)
            axis.axis("off")
        fig.tight_layout()
        if save_to is not None:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_to)
        if show:
            plt.show()
        plt.close(fig)
        return {
            "labels_before": before,
            "labels_after": after,
            "unexpected_labels": unexpected,
            "labels_preserved": valid,
            "shape_matches": output.shape[:2] == labels.shape,
            "image_dtype": str(output.dtype),
            "mask_dtype": str(labels.dtype),
            "preview": str(Path(save_to).resolve()) if save_to else None,
        }

    def add(self, augmentation):
        self.augmentations.append(augmentation)
        return self


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, mask):
        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"]


def create_albumentations_pipeline(
    transforms=None,
    horizontal_flip=True,
    vertical_flip=False,
    rotate_limit=30,
    brightness_contrast=True,
    elastic=False,
    probability=0.5,
):
    try:
        import albumentations as A
    except ImportError as exc:
        raise ImportError(
            "albumentations is required for this pipeline. Install it with `pip install albumentations`."
        ) from exc

    if transforms is None:
        transforms = []
        if horizontal_flip:
            transforms.append(A.HorizontalFlip(p=probability))
        if vertical_flip:
            transforms.append(A.VerticalFlip(p=probability))
        if rotate_limit:
            transforms.append(A.Rotate(limit=rotate_limit, border_mode=cv2.BORDER_CONSTANT, p=probability))
        if brightness_contrast:
            transforms.append(A.RandomBrightnessContrast(p=probability))
        if elastic:
            transforms.append(A.ElasticTransform(p=probability))

    return AlbumentationsTransform(A.Compose(transforms))


def _load_file_paths(source, valid_extensions=('.png', '.jpg', '.jpeg', '.bmp')):
    if isinstance(source, list):
        return source
    elif isinstance(source, (str, Path)):
        if os.path.isdir(source):
            files = []
            for ext in valid_extensions:
                files.extend(glob.glob(os.path.join(source, f"*{ext}")))
            return sorted(files)
        else:
            raise FileNotFoundError(f"Directory not found: {source}")
    else:
        raise ValueError("Must be a directory or a list of file paths.")


def apply_and_save_augmentations(
    images_source,
    masks_source=None,
    output_dir="augmented",
    augmentations=None,
    num_augmented=1,
    valid_extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    random_seed=None,
    preview_to=None,
):
    """Save augmented copies. RGB arrays and original integer bit depth are preserved."""
    from ..imageops.preprocess import _match_image_mask_pairs

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    if augmentations is None:
        raise ValueError("At least one augmentation is required.")
    if not isinstance(num_augmented, int) or num_augmented < 1:
        raise ValueError("num_augmented must be a positive integer.")
    pairs = (
        _match_image_mask_pairs(images_source, masks_source, valid_extensions)
        if masks_source is not None
        else [(p, None) for p in _load_file_paths(images_source, valid_extensions)]
    )
    output = Path(output_dir)
    if any(Path(p).resolve().is_relative_to(output.resolve()) for pair in pairs for p in pair if p):
        raise ValueError("Augmentation output must not contain source files.")
    (output / "images").mkdir(parents=True, exist_ok=True)
    if masks_source is not None:
        (output / "masks").mkdir(exist_ok=True)
    pipeline = (
        augmentations
        if isinstance(augmentations, AugmentationPipeline)
        else AugmentationPipeline(augmentations)
    )
    saved = []
    for path, mask_path in pairs:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) if mask_path else None
        if mask_path and mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        AugmentImages._check_type(image, mask)
        if preview_to and mask is not None and not saved:
            state, numpy_state = random.getstate(), np.random.get_state()
            pipeline.preview(image, mask, save_to=preview_to)
            random.setstate(state)
            np.random.set_state(numpy_state)
        for index in range(1, num_augmented + 1):
            result, labels = pipeline(image.copy(), mask.copy() if mask is not None else None)
            if result.dtype != image.dtype:
                raise ValueError("File augmentation must preserve the image dtype.")
            AugmentImages._check_type(result, labels)
            if mask is not None and (
                labels is None or not set(np.unique(labels)).issubset(set(np.unique(mask)) | {0})
            ):
                raise ValueError("Augmentation introduced new mask labels or discarded the mask.")
            suffix = ".tiff" if np.issubdtype(result.dtype, np.floating) else ".png"
            filename = f"{Path(path).stem}_aug{index}{suffix}"
            dest = output / "images" / filename
            bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR) if result.ndim == 3 else result
            if not cv2.imwrite(
                str(dest), bgr, [cv2.IMWRITE_TIFF_COMPRESSION, 1] if suffix == ".tiff" else []
            ):
                raise OSError(f"Could not save {dest}")
            if labels is not None:
                if labels.min() < 0 or labels.max() > 65535:
                    raise ValueError("PNG mask labels must lie in [0,65535].")
                labels = labels.astype(np.uint8 if labels.max() <= 255 else np.uint16)
                cv2.imwrite(str(output / "masks" / filename), labels)
            saved.append(str(dest.resolve()))
    return saved
