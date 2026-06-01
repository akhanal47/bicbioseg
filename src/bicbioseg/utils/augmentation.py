import random
from scipy import ndimage
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import os
import glob
from pathlib import Path

class AugmentImages:
    @staticmethod
    def _check_type(img, mask):
        assert img.dtype == 'uint8', "Image must be uint8"
        if mask is not None:
             assert mask.dtype == 'uint8', "Mask must be uint8"

    @staticmethod
    def rotate(image: np.ndarray, mask: np.ndarray = None, angle_range=(5, 30), resize_target=None, probability=1.0):
        if random.random() > probability:
            return image, mask
            
        AugmentImages._check_type(image, mask)
        
        angle = random.randint(angle_range[0], angle_range[1])
        if random.random() > 0.5: 
            angle = -angle

        def _apply_rot(img, is_mask=False):
            if resize_target:
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
                interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
                return cv2.warpAffine(img, M, (w, h), flags=interp)
            else:
                return ndimage.rotate(img, angle, reshape=False, order=0 if is_mask else 3)

        aug_img = _apply_rot(image, is_mask=False)
        aug_mask = _apply_rot(mask, is_mask=True) if mask is not None else None
        
        if resize_target:
            aug_img = cv2.resize(aug_img, resize_target, interpolation=cv2.INTER_CUBIC)
            if aug_mask is not None:
                aug_mask = cv2.resize(aug_mask, resize_target, interpolation=cv2.INTER_NEAREST)

        return aug_img, aug_mask

    @staticmethod
    def flip(image: np.ndarray, mask: np.ndarray = None, mode='random', probability=1.0):
        if random.random() > probability:
            return image, mask
            
        AugmentImages._check_type(image, mask)
        
        if mode == 'random':
            flip_code = random.choice([-1, 0, 1])
        elif mode in ['v', 'vertical']:
            flip_code = 0
        elif mode in ['h', 'horizontal']:
            flip_code = 1
        elif mode in ['vh', 'both']:
            flip_code = -1
        else:
            raise ValueError(f"Unknown flip mode: {mode}. Use 'random', 'v', 'h', or 'vh'.")

        aug_img = cv2.flip(image, flip_code)
        aug_mask = cv2.flip(mask, flip_code) if mask is not None else None
        
        return aug_img, aug_mask

    @staticmethod
    def adjust_brightness(image: np.ndarray, mask: np.ndarray = None, delta_range=(-30, 30), probability=1.0):
        if random.random() > probability:
            return image, mask
            
        AugmentImages._check_type(image, mask)
        
        delta = random.randint(delta_range[0], delta_range[1])
        new_image = image.astype(np.int16) + delta
        new_image = np.clip(new_image, 0, 255).astype(np.uint8)
        
        return new_image, mask

    @staticmethod
    def hist_equalize(image: np.ndarray, mask: np.ndarray = None, clipLimit=3, tileGridSize=(8,8), probability=1.0):
        if random.random() > probability:
            return image, mask
            
        AugmentImages._check_type(image, mask)
        
        # grayscale -> direct apply, else YCrCb then apply
        if image.ndim == 2:
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            equalized = clahe.apply(image)
        else:
            converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            converted_img[:,:,0] = clahe.apply(converted_img[:,:,0])
            equalized = cv2.cvtColor(converted_img, cv2.COLOR_YCrCb2BGR)
            
        return equalized, mask

    @staticmethod
    def random_crop(image: np.ndarray, mask: np.ndarray = None, crop_size=(224, 224), probability=1.0):
        if random.random() > probability:
            return image, mask
            
        AugmentImages._check_type(image, mask)
        
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size
        
        if h < crop_h or w < crop_w:
            img_resized = cv2.resize(image, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
            mask_resized = None
            if mask is not None:
                mask_resized = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            return img_resized, mask_resized

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        
        img_crop = image[top:top+crop_h, left:left+crop_w]
        
        mask_crop = None
        if mask is not None:
            mask_crop = mask[top:top+crop_h, left:left+crop_w]
            
        return img_crop, mask_crop

class AugmentationPipeline:
    def __init__(self, augmentations=None):
        self.augmentations = augmentations if augmentations is not None else []
    
    def __call__(self, image, mask):
        for aug_func in self.augmentations:
            image, mask = aug_func(image, mask)
        return image, mask
    
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

# save the augmented image to a folder
def apply_and_save_augmentations(
    images_source,
    masks_source=None,
    output_dir="augmented",
    augmentations=None,
    num_augmented=1,
    valid_extensions=('.png', '.jpg', '.jpeg', '.bmp'),
    random_seed=None
    ):

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    if augmentations is None:
        raise ValueError("At least one augmentation required!")
    
    # Load file paths
    all_images = _load_file_paths(images_source, valid_extensions)
    all_masks = None
    
    if masks_source is not None:
        all_masks = _load_file_paths(masks_source, valid_extensions)
        if len(all_images) != len(all_masks):
            raise ValueError(f"No. Mismatch: {len(all_images)} images vs {len(all_masks)} masks")
        all_masks.sort()
    
    all_images.sort()
    
    # make dir
    img_save_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_save_dir, exist_ok=True)
    
    mask_save_dir = None
    if all_masks is not None:
        mask_save_dir = os.path.join(output_dir, 'masks')
        os.makedirs(mask_save_dir, exist_ok=True)
    
    print(f"Processing {len(all_images)} images with {num_augmented} augmented version(s) each...")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # aug pipeline
    pipeline = AugmentationPipeline(augmentations)
    
    # process and save
    for idx, img_path in enumerate(all_images):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue
        
        mask = None
        if all_masks is not None:
            mask = cv2.imread(all_masks[idx], 0)
            if mask is None:
                print(f"Warning: Could not read {all_masks[idx]}. Skipping.")
                continue
        
        for aug_num in range(1, num_augmented + 1):
            aug_img, aug_mask = pipeline(img.copy(), mask.copy() if mask is not None else None)
            
            aug_img_name = f"{base_name}_aug{aug_num}.png"
            cv2.imwrite(os.path.join(img_save_dir, aug_img_name), aug_img)
            
            if aug_mask is not None:
                cv2.imwrite(os.path.join(mask_save_dir, aug_img_name), aug_mask)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(all_images)} images...")
    
    print(f"\nAugmented images saved to: {os.path.abspath(output_dir)}")
    print(f"Generated Total Images: {len(all_images) * num_augmented}")
