from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from bicbioseg import ImageOps, create_dataset_split


def _write_pair(image_dir: Path, mask_dir: Path, stem: str, image_ext: str = ".jpg"):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 1] = 120
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(image_dir / f"{stem}{image_ext}"), image)
    cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)


def test_create_dataset_split_matches_by_stem_and_preserves_masks(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    for idx in range(6):
        _write_pair(image_dir, mask_dir, f"sample_{idx}", image_ext=".jpg")

    output_dir = tmp_path / "split"
    create_dataset_split(
        images=image_dir,
        masks=mask_dir,
        save_to=output_dir,
        split=(0.5, 0.25, 0.25),
        resize=(16, 16),
        overwrite=True,
    )

    saved_masks = list(output_dir.glob("*/masks/*.png"))

    assert saved_masks
    for mask_path in saved_masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert set(np.unique(mask)).issubset({0, 255})


def test_patch_split_does_not_leak_source_images_across_splits(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    for idx in range(8):
        _write_pair(image_dir, mask_dir, f"field_{idx}", image_ext=".png")

    output_dir = tmp_path / "patch_split"
    create_dataset_split(
        images=image_dir,
        masks=mask_dir,
        save_to=output_dir,
        split=(0.5, 0.25, 0.25),
        create_patches=True,
        patch_size=(16, 16),
        overwrite=True,
    )

    source_to_split = {}
    for split_name in ("train", "validate", "test"):
        for patch_path in (output_dir / split_name / "images").glob("*.png"):
            source_stem = patch_path.stem.split("_p")[0]
            assert source_stem not in source_to_split or source_to_split[source_stem] == split_name
            source_to_split[source_stem] = split_name


def test_create_and_reconstruct_patches_round_trip_shape():
    image = np.arange(30 * 30, dtype=np.uint16).reshape(30, 30)
    patches, coords = ImageOps.create_patches(image, patch_dims=(16, 16))

    rebuilt = ImageOps.reconstruct_from_patches(patches, coords, original_shape=image.shape)

    assert len(patches) == 4
    assert rebuilt.shape == image.shape
    np.testing.assert_array_equal(rebuilt, image)


def test_normalize_binary_and_color_masks():
    binary = np.array([[0, 5], [255, 0]], dtype=np.uint8)
    normalized = ImageOps.normalize_mask(binary, mode="binary")

    np.testing.assert_array_equal(normalized, np.array([[0, 1], [1, 0]], dtype=np.uint8))

    color = np.zeros((2, 2, 3), dtype=np.uint8)
    color[0, 1] = (255, 0, 0)
    color[1, 0] = (0, 255, 0)

    labels, color_map = ImageOps.convert_color_mask_to_labels(color)

    assert labels[0, 0] == 0
    assert labels[0, 1] != labels[1, 0]
    assert color_map[(0, 0, 0)] == 0
