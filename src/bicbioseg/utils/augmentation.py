import random
from scipy import ndimage
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

class AugmentImages:
    @staticmethod
    def _check_type(img, mask):
        assert img.dtype == 'uint8', "Image must be uint8"
        if mask is not None:
             assert mask.dtype == 'uint8', "Mask must be uint8"

    @staticmethod
    def rotate(image: np.ndarray, mask: np.ndarray = None, angle_range=(5, 30), resize_target=None):
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
    def flip(image: np.ndarray, mask: np.ndarray = None, mode='random'):
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
    def adjust_brightness(image: np.ndarray, mask: np.ndarray = None, delta_range=(-30, 30)):
        AugmentImages._check_type(image, mask)
        
        delta = random.randint(delta_range[0], delta_range[1])
        new_image = image.astype(np.int16) + delta
        new_image = np.clip(new_image, 0, 255).astype(np.uint8)
        
        return new_image, mask

    @staticmethod
    def hist_equalize(image: np.ndarray, mask: np.ndarray = None, clipLimit=3, tileGridSize=(8,8)):
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
    def random_crop(image: np.ndarray, mask: np.ndarray = None, crop_size=(224, 224)):
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