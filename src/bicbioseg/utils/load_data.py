import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader

from .preprocessing import image_tensor, normalize_image, validate_normalization


class BiosegDataset(TorchDataset):
    def __init__(self, images, masks, image_size=(224, 224), transform=None, in_channels=3, normalization=None, ignore_index=None,
                 crop_size=None, foreground_probability=0.0):
        if len(images) != len(masks):
            raise ValueError(f"Mismatched: {len(images)} images vs {len(masks)} masks")
        if not len(images):
            raise ValueError("Dataset contains no image/mask pairs.")
        self.images, self.masks = images, masks
        self.image_size, self.transform = image_size, transform
        self.in_channels = in_channels
        self.normalization = validate_normalization(normalization)
        self.ignore_index = ignore_index
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        if self.crop_size is not None and (len(self.crop_size) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in self.crop_size)):
            raise ValueError("crop_size must be a positive (height, width) pair.")
        if not 0 <= foreground_probability <= 1:
            raise ValueError("foreground_probability must lie in [0, 1].")
        if foreground_probability and self.crop_size is None:
            raise ValueError("foreground_probability requires crop_size.")
        self.foreground_probability = foreground_probability
        self.is_path_mode = isinstance(images[0], (str, os.PathLike, np.str_))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.is_path_mode:
            image = cv2.imread(str(self.images[idx]), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_UNCHANGED)
            if image is None or mask is None:
                raise ValueError(f"Could not read image/mask pair: {self.images[idx]}, {self.masks[idx]}")
            if image.ndim == 3 and image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image, mask = self.images[idx], self.masks[idx]
        if mask.ndim != 2:
            raise ValueError("Masks must be 2D class-ID arrays; convert color masks to labels first.")
        if image.shape[:2] != mask.shape:
            raise ValueError("Image and mask spatial dimensions must match before resizing.")
        if self.normalization["mode"] != "standard":
            image = normalize_image(image, self.normalization)
        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            h, w = mask.shape
            if crop_h > h or crop_w > w:
                raise ValueError("crop_size must fit within each source image.")
            foreground = (mask > 0) & (mask != self.ignore_index) if self.ignore_index is not None else mask > 0
            locations = np.argwhere(foreground)
            if len(locations) and torch.rand(()).item() < self.foreground_probability:
                cy, cx = locations[torch.randint(len(locations), ()).item()]
                y = int(np.clip(cy - crop_h // 2, 0, h - crop_h))
                x = int(np.clip(cx - crop_w // 2, 0, w - crop_w))
            else:
                y, x = torch.randint(h - crop_h + 1, ()).item(), torch.randint(w - crop_w + 1, ()).item()
            image, mask = image[y:y+crop_h, x:x+crop_w], mask[y:y+crop_h, x:x+crop_w]
        height, width = self.image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            image, mask = self.transform(image, mask)
        valid = mask != self.ignore_index if self.ignore_index is not None else np.ones(mask.shape, dtype=bool)
        if not np.isfinite(mask).all() or (mask[valid] < 0).any() or not np.equal(mask, np.floor(mask)).all():
            raise ValueError("Masks must contain nonnegative integer class IDs.")
        return image_tensor(image, self.image_size, self.in_channels), torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64))


class DataLoader:
    @staticmethod
    def load_data(source="train", masks_source=None, image_size=(224, 224), batch_size=4,
                  transforms=None, num_workers=2, shuffle=True, in_channels=3, normalization=None, ignore_index=None,
                  crop_size=None, foreground_probability=0.0, **kwargs):
        if isinstance(source, tuple) and len(source) == 2 and masks_source is None:
            source, masks_source = source
        if isinstance(source, (str, os.PathLike)):
            # local import avoids coupling package initialization to image operations.
            from ..imageops.preprocess import _match_image_mask_pairs
            root = Path(source)
            if masks_source is None:
                image_dir, mask_dir = root / "images", root / "masks"
                if not image_dir.is_dir() or not mask_dir.is_dir():
                    raise FileNotFoundError("Directory must contain 'images' and 'masks' subfolders.")
            else:
                image_dir, mask_dir = root, masks_source
            pairs = _match_image_mask_pairs(image_dir, mask_dir)
            images, masks = zip(*pairs) if pairs else ([], [])
        elif isinstance(source, (list, tuple, np.ndarray)):
            if masks_source is None:
                raise ValueError("Provide masks_source for image lists or arrays.")
            images, masks = source, masks_source
        else:
            raise TypeError("Source must be a directory, path list, array, or (images, masks) pair.")
        dataset = BiosegDataset(images, masks, image_size, transforms, in_channels, normalization,
                                ignore_index, crop_size, foreground_probability)
        return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

    @staticmethod
    def plot_samples(dataloader, num_samples=2, figsize=(12, 6)):
        import matplotlib.pyplot as plt
        
        batch = next(iter(dataloader))
        images, masks = batch
        
        num_samples = min(num_samples, len(images))
        
        fig, axes = plt.subplots(num_samples, 2, figsize=figsize)
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            img = images[i].numpy().transpose(1, 2, 0)
            mask = masks[i].numpy()
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Image {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title(f'Mask {i+1}')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
