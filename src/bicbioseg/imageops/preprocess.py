import csv
import glob
import os
import re
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


PathLike = Union[str, Path]
Source = Union[PathLike, Sequence[PathLike]]

VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _is_image_file(path: Path, valid_extensions=VALID_IMAGE_EXTENSIONS) -> bool:
    return path.suffix.lower() in valid_extensions


def _read_path_list(path: Path) -> List[str]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            rows = csv.reader(handle)
            return [row[0].strip() for row in rows if row and row[0].strip()]

    with path.open() as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_file_paths(source: Source, valid_extensions=VALID_IMAGE_EXTENSIONS) -> List[str]:
    if isinstance(source, (list, tuple)):
        return [str(Path(item)) for item in source]

    if isinstance(source, Path):
        source = str(source)

    if isinstance(source, str):
        path = Path(source)
        if path.is_dir():
            files: List[str] = []
            for ext in valid_extensions:
                files.extend(glob.glob(str(path / f"*{ext}")))
                files.extend(glob.glob(str(path / f"*{ext.upper()}")))
            return sorted(set(files))

        if path.is_file() and _is_image_file(path, valid_extensions):
            return [str(path)]

        if path.is_file() and path.suffix.lower() in {".txt", ".csv"}:
            return _read_path_list(path)

        raise FileNotFoundError(f"Could not find image source: {source}")

    raise ValueError("Source must be a directory, image path, text/csv list, or list of paths.")


def _stem_without_patch_suffix(path: PathLike) -> str:
    stem = Path(path).stem
    stem = re.sub(r"(_p|_patch)[0-9]+[_-][0-9]+$", "", stem)
    return stem


def _match_image_mask_pairs(
    images_source: Source,
    masks_source: Source,
    valid_extensions=VALID_IMAGE_EXTENSIONS,
) -> List[Tuple[str, str]]:
    images = _load_file_paths(images_source, valid_extensions)
    masks = _load_file_paths(masks_source, valid_extensions)

    mask_by_stem: Dict[str, List[str]] = defaultdict(list)
    for mask_path in masks:
        mask_by_stem[Path(mask_path).stem].append(mask_path)

    pairs: List[Tuple[str, str]] = []
    missing: List[str] = []
    duplicate_masks: List[str] = []

    for image_path in images:
        stem = Path(image_path).stem
        matches = mask_by_stem.get(stem, [])
        if len(matches) == 1:
            pairs.append((image_path, matches[0]))
        elif len(matches) > 1:
            duplicate_masks.append(stem)
        else:
            missing.append(image_path)

    if duplicate_masks:
        raise ValueError(f"Duplicate masks found for stems: {duplicate_masks[:5]}")

    if missing:
        raise FileNotFoundError(
            "Could not find corresponding masks for: "
            + ", ".join(Path(path).name for path in missing[:5])
        )

    extra_masks = sorted(set(mask_by_stem) - {Path(path).stem for path in images})
    if extra_masks:
        warnings.warn(f"Found {len(extra_masks)} mask(s) without matching images.")

    return sorted(pairs, key=lambda pair: Path(pair[0]).stem)


def _validate_split(split: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if len(split) != 3:
        raise ValueError("split must contain three ratios: train, validate, test.")

    train_ratio, val_ratio, test_ratio = split
    if min(split) < 0:
        raise ValueError("Split ratios must be non-negative.")

    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("At least one split ratio must be greater than zero.")

    return train_ratio / total, val_ratio / total, test_ratio / total


def _group_key(path: str, group_by: Optional[Union[str, Callable[[str], str]]]) -> str:
    if callable(group_by):
        return str(group_by(path))

    if group_by in (None, "path"):
        return str(path)

    if group_by in {"filename", "stem"}:
        return _stem_without_patch_suffix(path)

    if group_by == "parent":
        return Path(path).parent.name

    raise ValueError("group_by must be None, 'path', 'filename', 'stem', 'parent', or a callable.")


def _split_groups(
    pairs: List[Tuple[str, str]],
    split: Tuple[float, float, float],
    group_by: Optional[Union[str, Callable[[str], str]]],
    random_seed: int,
) -> Dict[str, List[Tuple[str, str]]]:
    train_ratio, val_ratio, test_ratio = _validate_split(split)

    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for image_path, mask_path in pairs:
        grouped[_group_key(image_path, group_by)].append((image_path, mask_path))

    group_items = list(grouped.items())
    if len(group_items) < 2:
        return {"train": pairs, "validate": [], "test": []}

    groups = [key for key, _ in group_items]
    rng = np.random.default_rng(random_seed)
    groups = list(rng.permutation(groups))

    num_groups = len(groups)
    test_count = int(round(num_groups * test_ratio))
    if test_ratio > 0:
        test_count = max(1, test_count)
    if test_count >= num_groups and (train_ratio > 0 or val_ratio > 0):
        test_count = num_groups - 1

    remaining_after_test = num_groups - test_count
    val_count = int(round(num_groups * val_ratio))
    if val_ratio > 0 and remaining_after_test > 1:
        val_count = max(1, val_count)
    if val_count >= remaining_after_test and train_ratio > 0:
        val_count = max(0, remaining_after_test - 1)

    test_groups = groups[:test_count]
    val_groups = groups[test_count : test_count + val_count]
    train_groups = groups[test_count + val_count :]

    def flatten(selected_groups: Iterable[str]) -> List[Tuple[str, str]]:
        return [pair for group in selected_groups for pair in grouped[group]]

    return {
        "train": flatten(train_groups),
        "validate": flatten(val_groups),
        "test": flatten(test_groups),
    }


def _prepare_output_dir(output_dir: PathLike, overwrite: bool) -> Path:
    output_path = Path(output_dir)
    if output_path.exists():
        if not overwrite and any(output_path.iterdir()):
            raise FileExistsError(
                f"{output_path} already exists and is not empty. "
                "Pass overwrite=True or choose another save location."
            )
        if overwrite:
            shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


class ImageOps:
    @staticmethod
    def _exact_divisor(num: int, patch_size: int = 224) -> int:
        return num if num % patch_size == 0 else num + (patch_size - (num % patch_size))

    @staticmethod
    def standardize_images(image: np.ndarray, patch_dims: Tuple[int, int] = (224, 224)) -> np.ndarray:
        img_height, img_width = image.shape[:2]
        canvas_height = ImageOps._exact_divisor(num=img_height, patch_size=patch_dims[0])
        canvas_width = ImageOps._exact_divisor(num=img_width, patch_size=patch_dims[1])

        if image.ndim == 3:
            canvas = np.zeros((canvas_height, canvas_width, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((canvas_height, canvas_width), dtype=image.dtype)

        canvas[:img_height, :img_width] = image
        return canvas

    @staticmethod
    def standarize_images(image: np.ndarray, patch_dims: Tuple[int, int] = (224, 224)) -> np.ndarray:
        warnings.warn(
            "standarize_images is deprecated; use standardize_images instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return ImageOps.standardize_images(image, patch_dims)

    @staticmethod
    def resize_image(
        image: np.ndarray,
        size: Tuple[int, int] = (224, 224),
        is_mask: bool = False,
    ) -> np.ndarray:
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
        return cv2.resize(image, size, interpolation=interpolation)

    @staticmethod
    def create_patches(
        img: np.ndarray,
        patch_dims: Tuple[int, int] = (224, 224),
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        img = ImageOps.standardize_images(img, patch_dims)
        patch_h, patch_w = patch_dims

        tiles = []
        coords = []
        for y in range(0, img.shape[0], patch_h):
            for x in range(0, img.shape[1], patch_w):
                tile = img[y : y + patch_h, x : x + patch_w]
                if tile.shape[0] == patch_h and tile.shape[1] == patch_w:
                    tiles.append(tile)
                    coords.append((y // patch_h, x // patch_w))

        return tiles, coords

    @staticmethod
    def reconstruct_from_patches(
        patches: Sequence[np.ndarray],
        coords: Sequence[Tuple[int, int]],
        original_shape: Optional[Tuple[int, ...]] = None,
        patch_dims: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        if not patches:
            raise ValueError("At least one patch is required for reconstruction.")

        patch_h, patch_w = patch_dims or patches[0].shape[:2]
        max_row = max(row for row, _ in coords)
        max_col = max(col for _, col in coords)

        canvas_shape = ((max_row + 1) * patch_h, (max_col + 1) * patch_w)
        if patches[0].ndim == 3:
            canvas_shape = canvas_shape + (patches[0].shape[2],)

        canvas = np.zeros(canvas_shape, dtype=patches[0].dtype)
        for patch, (row, col) in zip(patches, coords):
            y = row * patch_h
            x = col * patch_w
            canvas[y : y + patch_h, x : x + patch_w] = patch

        if original_shape is not None:
            slices = tuple(slice(0, dim) for dim in original_shape[: canvas.ndim])
            return canvas[slices]

        return canvas

    @staticmethod
    def overlay_mask(
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.4,
    ) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if mask.shape[:2] != image.shape[:2]:
            mask = ImageOps.resize_image(mask, (image.shape[1], image.shape[0]), is_mask=True)

        overlay = image.copy()
        color_arr = np.array(color, dtype=np.uint8)
        overlay[mask > 0] = (
            (1 - alpha) * overlay[mask > 0].astype(np.float32) + alpha * color_arr
        ).astype(np.uint8)
        return overlay

    @staticmethod
    def normalize_mask(
        mask: np.ndarray,
        mode: str = "binary",
        threshold: int = 0,
        foreground_value: int = 1,
        dtype=np.uint8,
    ) -> np.ndarray:
        """Normalize common biomedical mask formats to binary or label masks."""
        mode = mode.lower()

        if mask.ndim == 3 and mode != "color":
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if mode == "binary":
            return np.where(mask > threshold, foreground_value, 0).astype(dtype)

        if mode == "labels":
            return mask.astype(dtype)

        if mode == "color":
            labels, _ = ImageOps.convert_color_mask_to_labels(mask)
            return labels.astype(dtype)

        raise ValueError("mode must be 'binary', 'labels', or 'color'.")

    @staticmethod
    def convert_color_mask_to_labels(
        mask: np.ndarray,
        color_map: Optional[Mapping[Tuple[int, int, int], int]] = None,
        background_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], int]]:
        """Convert an RGB/BGR color-coded mask to integer label IDs."""
        if mask.ndim != 3 or mask.shape[2] < 3:
            raise ValueError("Color mask must have shape (height, width, channels).")

        rgb_mask = mask[:, :, :3]
        unique_colors = np.unique(rgb_mask.reshape(-1, 3), axis=0)

        if color_map is None:
            color_map = {}
            next_label = 1
            for color in unique_colors:
                color_tuple = tuple(int(value) for value in color)
                if color_tuple == background_color:
                    color_map[color_tuple] = 0
                else:
                    color_map[color_tuple] = next_label
                    next_label += 1
        else:
            color_map = {tuple(key): int(value) for key, value in color_map.items()}

        labels = np.zeros(mask.shape[:2], dtype=np.uint16)
        for color, label in color_map.items():
            labels[np.all(rgb_mask == np.array(color, dtype=rgb_mask.dtype), axis=-1)] = label

        return labels, dict(color_map)

    @staticmethod
    def normalize_masks(
        masks_source: Source,
        output_dir: PathLike = "normalized_masks",
        mode: str = "binary",
        threshold: int = 0,
        foreground_value: int = 1,
        color_map: Optional[Mapping[Tuple[int, int, int], int]] = None,
        overwrite: bool = False,
    ) -> str:
        output_path = _prepare_output_dir(output_dir, overwrite)
        mask_paths = _load_file_paths(masks_source)
        learned_color_map = None

        for mask_path in mask_paths:
            read_mode = cv2.IMREAD_COLOR if mode == "color" else cv2.IMREAD_UNCHANGED
            mask = cv2.imread(mask_path, read_mode)
            if mask is None:
                warnings.warn(f"Could not read mask {mask_path}. Skipping.")
                continue

            if mode == "color":
                normalized, learned_color_map = ImageOps.convert_color_mask_to_labels(
                    mask,
                    color_map=color_map or learned_color_map,
                )
            else:
                normalized = ImageOps.normalize_mask(
                    mask,
                    mode=mode,
                    threshold=threshold,
                    foreground_value=foreground_value,
                )

            cv2.imwrite(str(output_path / f"{Path(mask_path).stem}.png"), normalized)

        return str(output_path.resolve())

    @staticmethod
    def find_unmatched_masks(images_source: Source, masks_source: Source) -> Dict[str, List[str]]:
        images = _load_file_paths(images_source)
        masks = _load_file_paths(masks_source)

        image_stems = {Path(path).stem for path in images}
        mask_stems = {Path(path).stem for path in masks}

        return {
            "images_without_masks": sorted(image_stems - mask_stems),
            "masks_without_images": sorted(mask_stems - image_stems),
        }

    @staticmethod
    def inspect_dataset(images_source: Source, masks_source: Optional[Source] = None) -> Dict[str, object]:
        images = _load_file_paths(images_source)
        summary: Dict[str, object] = {
            "num_images": len(images),
            "image_extensions": sorted({Path(path).suffix.lower() for path in images}),
            "image_shapes": {},
        }

        for path in images:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                summary["image_shapes"][Path(path).name] = img.shape

        if masks_source is not None:
            masks = _load_file_paths(masks_source)
            summary["num_masks"] = len(masks)
            summary["unmatched"] = ImageOps.find_unmatched_masks(images, masks)

        return summary

    @staticmethod
    def resize_dataset(
        images_source: Source,
        masks_source: Optional[Source] = None,
        output_dir: PathLike = "resized",
        image_size: Tuple[int, int] = (224, 224),
        overwrite: bool = False,
    ) -> str:
        output_path = _prepare_output_dir(output_dir, overwrite)
        img_save_dir = output_path / "images"
        img_save_dir.mkdir(parents=True, exist_ok=True)

        if masks_source is None:
            image_paths = _load_file_paths(images_source)
            pairs = [(path, None) for path in image_paths]
            mask_save_dir = None
        else:
            pairs = _match_image_mask_pairs(images_source, masks_source)
            mask_save_dir = output_path / "masks"
            mask_save_dir.mkdir(parents=True, exist_ok=True)

        for image_path, mask_path in pairs:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                warnings.warn(f"Could not read image {image_path}. Skipping.")
                continue
            resized = ImageOps.resize_image(image, image_size, is_mask=False)
            cv2.imwrite(str(img_save_dir / f"{Path(image_path).stem}.png"), resized)

            if mask_path is not None and mask_save_dir is not None:
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask is None:
                    warnings.warn(f"Could not read mask {mask_path}. Skipping.")
                    continue
                mask = ImageOps.resize_image(mask, image_size, is_mask=True)
                cv2.imwrite(str(mask_save_dir / f"{Path(image_path).stem}.png"), mask)

        return str(output_path.resolve())

    @staticmethod
    def extract_tiff_frames(*args, **kwargs):
        from .convert_images import ConvertImage

        return ConvertImage.tiff_extract_frames(*args, **kwargs)

    @staticmethod
    def convert_dicom(*args, **kwargs):
        from .convert_images import ConvertImage

        return ConvertImage.dicom_to_uint8(*args, **kwargs)

    @staticmethod
    def split_dataset(*args, **kwargs) -> str:
        return create_dataset_split(*args, **kwargs)

    @staticmethod
    def prepare_dataset(*args, **kwargs) -> str:
        return create_dataset_split(*args, **kwargs)


def _save_pair(
    image_path: str,
    mask_path: str,
    img_save_dir: Path,
    mask_save_dir: Path,
    resize: Optional[Tuple[int, int]],
    create_patches: bool,
    patch_size: Tuple[int, int],
    balance_empty_masks: bool,
) -> int:
    base_name = Path(image_path).stem

    if not create_patches and resize is None:
        shutil.copy2(image_path, img_save_dir / Path(image_path).name)
        shutil.copy2(mask_path, mask_save_dir / Path(mask_path).name)
        return 1

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if image is None or mask is None:
        warnings.warn(f"Could not read {image_path} or {mask_path}. Skipping.")
        return 0

    if resize is not None:
        image = ImageOps.resize_image(image, resize, is_mask=False)
        mask = ImageOps.resize_image(mask, resize, is_mask=True)

    if not create_patches:
        filename = f"{base_name}.png"
        cv2.imwrite(str(img_save_dir / filename), image)
        cv2.imwrite(str(mask_save_dir / filename), mask)
        return 1

    image_patches, coords = ImageOps.create_patches(image, patch_size)
    mask_patches, _ = ImageOps.create_patches(mask, patch_size)
    saved = 0

    for image_patch, mask_patch, (row, col) in zip(image_patches, mask_patches, coords):
        if balance_empty_masks and np.max(mask_patch) == 0:
            continue

        filename = f"{base_name}_p{row}_{col}.png"
        cv2.imwrite(str(img_save_dir / filename), image_patch)
        cv2.imwrite(str(mask_save_dir / filename), mask_patch)
        saved += 1

    return saved


def create_dataset_split(
    images: Source,
    masks: Source,
    save_to: PathLike = "dataset_split",
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    resize: Optional[Tuple[int, int]] = None,
    create_patches: bool = False,
    patch_size: Tuple[int, int] = (224, 224),
    balance_empty_masks: bool = False,
    group_by: Optional[Union[str, Callable[[str], str]]] = "filename",
    random_seed: int = 42,
    overwrite: bool = False,
) -> str:
    """Create train/validate/test folders while keeping related images together."""
    pairs = _match_image_mask_pairs(images, masks)
    split_pairs = _split_groups(pairs, split, group_by, random_seed)
    output_path = _prepare_output_dir(save_to, overwrite)

    counts = {}
    for split_name, split_items in split_pairs.items():
        img_save_dir = output_path / split_name / "images"
        mask_save_dir = output_path / split_name / "masks"
        img_save_dir.mkdir(parents=True, exist_ok=True)
        mask_save_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for image_path, mask_path in split_items:
            saved_count += _save_pair(
                image_path=image_path,
                mask_path=mask_path,
                img_save_dir=img_save_dir,
                mask_save_dir=mask_save_dir,
                resize=resize,
                create_patches=create_patches,
                patch_size=patch_size,
                balance_empty_masks=balance_empty_masks,
            )
        counts[split_name] = saved_count

    print(f"Dataset created at: {output_path.resolve()}")
    print(f"Saved files: {counts}")
    return str(output_path.resolve())


def create_train_validate_test_split(
    images_source,
    masks_source,
    output_dir="dataset_split",
    val_ratio=0.1,
    test_ratio=0.1,
    create_patch=False,
    patch_dims=(224, 224),
    balance_dataset=False,
    random_seed=42,
    resize=None,
    overwrite=False,
    group_by="filename",
):
    """Backward-compatible wrapper around create_dataset_split."""
    train_ratio = max(0.0, 1.0 - val_ratio - test_ratio)
    return create_dataset_split(
        images=images_source,
        masks=masks_source,
        save_to=output_dir,
        split=(train_ratio, val_ratio, test_ratio),
        resize=resize,
        create_patches=create_patch,
        patch_size=patch_dims,
        balance_empty_masks=balance_dataset,
        group_by=group_by,
        random_seed=random_seed,
        overwrite=overwrite,
    )
