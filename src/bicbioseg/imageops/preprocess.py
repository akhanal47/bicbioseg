import cv2
import numpy as np
import os
import shutil
import glob
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings

class ImageOps:
    @staticmethod
    def _exact_divisor(num, patch_size=224):
        if (num % patch_size == 0):
            return num
        else:
            return num + (patch_size - (num % patch_size))
        
    @staticmethod
    def standarize_images(image: np.ndarray, patch_dims=(224, 224)):
        img_height, img_width = image.shape[:2]

        # target dims & canvas
        canvas_height = ImageOps._exact_divisor(num=img_height, patch_size=patch_dims[0])
        canvas_width = ImageOps._exact_divisor(num=img_width, patch_size=patch_dims[1])

        if image.ndim == 3:
            canvas = np.zeros((canvas_height, canvas_width, image.shape[2]), dtype=np.uint8)
        else:
            canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        
        canvas[:img_height, :img_width] = image
        return canvas

    @staticmethod
    def create_patches(img: np.ndarray, patch_dims=(224, 224)):
        """
        splits an image to patches
        returns:
            tiles (list): List of numpy array patches.
            coords (list): List of (x, y) tuples indicating patch indices.
        """
        img = ImageOps.standarize_images(img, patch_dims)
        
        M, N = patch_dims[0], patch_dims[1]
        
        # no of patch
        no_x_patch = img.shape[0] // M
        no_y_patch = img.shape[1] // N
        
        tiles = []
        coords = []

        for x in range(0, img.shape[0], M):
            for y in range(0, img.shape[1], N):
                if img.ndim == 3:
                    tile = img[x:x+M, y:y+N, :]
                else:
                    tile = img[x:x+M, y:y+N]
                
                # sanity check
                if tile.shape[0] == M and tile.shape[1] == N:
                    tiles.append(tile)
                    coords.append((x // M, y // N))

        return tiles, coords


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
        raise ValueError("Source must be a directory path or a list of file paths.")


def _process_and_save(image_paths, mask_paths, dest_root, split_name, create_patch, patch_dims, balance_dataset):    
    img_save_dir = os.path.join(dest_root, split_name, 'images')
    mask_save_dir = os.path.join(dest_root, split_name, 'masks')
    
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    print(f"Processing {split_name} split: {len(image_paths)} images...")

    for img_path, mask_path in zip(image_paths, mask_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        if create_patch:
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0) #mask assumed to be grayscale

            if img is None or mask is None:
                warnings.warn(f"Could not read {img_path} or {mask_path}. Skipping.")
                continue

            img_patches, img_coords = ImageOps.create_patches(img, patch_dims)
            mask_patches, _ = ImageOps.create_patches(mask, patch_dims)

            for patch, m_patch, (bx, by) in zip(img_patches, mask_patches, img_coords):
                
                # don't save blank mask & corresponding image
                if balance_dataset and np.max(m_patch) == 0:
                    continue

                fname = f"{base_name}_p{bx}_{by}.png"
                cv2.imwrite(os.path.join(img_save_dir, fname), patch)
                cv2.imwrite(os.path.join(mask_save_dir, fname), m_patch)
        
        else:
            shutil.copy(img_path, os.path.join(img_save_dir, os.path.basename(img_path)))
            shutil.copy(mask_path, os.path.join(mask_save_dir, os.path.basename(mask_path)))

def create_train_validate_test_split(
    images_source, 
    masks_source, 
    output_dir="dataset_split", 
    val_ratio=0.1, 
    test_ratio=0.1, 
    create_patch=False, 
    patch_dims=(224, 224),
    balance_dataset=False,
    random_seed=42
):
    """    
    Args:
        images_source: Path to folder of images OR list of file paths.
        masks_source: Path to folder of masks OR list of file paths.
        output_dir: Directory to save the output.
        val_ratio: Float (0.0 to 1.0) for validation set size.
        test_ratio: Float (0.0 to 1.0) for test set size.
        create_patch: Boolean, if True, images will be tiled.
        patch_dims: Tuple (H, W) for patch size.
        balance_dataset: Boolean, if True, patches with empty masks (0 pixel value) are discarded.
        random_seed: Int, seed for reproducibility.
    """
    
    all_images = _load_file_paths(images_source)
    all_masks = _load_file_paths(masks_source)
    
    if len(all_images) != len(all_masks):
        raise ValueError(f"Mismatch: Found {len(all_images)} images and {len(all_masks)} masks.")
    
    all_images.sort()
    all_masks.sort()

    train_images, test_images, train_masks, test_masks = train_test_split(
        all_images, all_masks, test_size=test_ratio, random_state=random_seed
    )
    
    relative_val_ratio = val_ratio / (1 - test_ratio)
    
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_images, train_masks, test_size=relative_val_ratio, random_state=random_seed
    )

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(f"Filter Empty Patches & Corresponding Image: {balance_dataset}")
    
    _process_and_save(train_images, train_masks, output_dir, 'train', create_patch, patch_dims, balance_dataset)
    _process_and_save(val_images, val_masks, output_dir, 'validate', create_patch, patch_dims, balance_dataset)
    _process_and_save(test_images, test_masks, output_dir, 'test', create_patch, patch_dims, balance_dataset)

    print(f"\nSuccess! Dataset created at: {os.path.abspath(output_dir)}")