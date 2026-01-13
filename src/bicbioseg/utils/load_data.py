import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from typing import Union, List, Tuple
import matplotlib.pyplot as plt

class BiosegDataset(TorchDataset):
    def __init__(self, 
                 images: Union[List[str], np.ndarray], 
                 masks: Union[List[str], np.ndarray], 
                 image_size=(224, 224), 
                 transform=None):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.transform = transform
        
        self.is_path_mode = False
        if isinstance(self.images, list) or (isinstance(self.images, np.ndarray) and self.images.ndim == 1):
             if len(self.images) > 0 and isinstance(self.images[0], (str, np.str_)):
                 self.is_path_mode = True

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.is_path_mode:
            img_path = str(self.images[idx])
            mask_path = str(self.masks[idx])
            
            image = cv2.imread(img_path, 1) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, 0) 
            
            if image is None: raise ValueError(f"Image not found: {img_path}")
            if mask is None: raise ValueError(f"Mask not found: {mask_path}")
        else:
            image = self.images[idx]
            mask = self.masks[idx]
            
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # optinal augmentation
        if self.transform:
            image, mask = self.transform(image, mask)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        mask = mask.astype(np.int64) 
        
        return torch.from_numpy(image), torch.from_numpy(mask)


class DataLoader:

    @staticmethod
    def load_data(
        source: Union[str, List, Tuple[np.ndarray, np.ndarray]] = 'train', 
        masks_source: Union[List, np.ndarray] = None,
        image_size=(224, 224),
        batch_size=4,
        transforms=None,
        num_workers=2,
        **kwargs
    ):
      
        images_data = None
        masks_data = None
        
        # from dirs
        if isinstance(source, str) and os.path.isdir(source):
            print(f"Loading from Directory: {source}")
            img_dir = os.path.join(source, 'images')
            mask_dir = os.path.join(source, 'masks')
            
            if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
                raise FileNotFoundError("Directory must contain 'images' and 'masks' subfolders.")
            
            valid_exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
            images_data = []
            for ext in valid_exts:
                images_data.extend(glob.glob(os.path.join(img_dir, ext)))
            
            images_data.sort()

            # for image in jpg but mask in png (quite often the case)
            masks_data = []
            for img_path in images_data:
                file_name = os.path.basename(img_path)
                name_no_ext = os.path.splitext(file_name)[0]
                
                # search priority: png > jpg > bmp
                found_mask = False
                
                for ext in [os.path.splitext(file_name)[1], '.png', '.jpg', '.jpeg', '.bmp']:
                    potential_mask = os.path.join(mask_dir, name_no_ext + ext)
                    if os.path.exists(potential_mask):
                        masks_data.append(potential_mask)
                        found_mask = True
                        break
                
                if not found_mask:
                    raise FileNotFoundError(f"Could not find a corresponding mask for {file_name} in {mask_dir}")
                
        # from list
        elif isinstance(source, list):
            print("Loading from List")
            if masks_source is None or not isinstance(masks_source, list):
                raise ValueError("For List mode, masks_source must be a provided as list of images.")
            if len(source) != len(masks_source):
                raise ValueError(f"Mismatched: {len(source)} images vs {len(masks_source)} masks")
            images_data = source
            masks_data = masks_source

        # from array
        elif isinstance(source, np.ndarray):
            print("Loading from Numpy Array")
            if masks_source is None or not isinstance(masks_source, np.ndarray):
                raise ValueError("For Array mode, masks_source must be a provided as numpy array.")
            if len(source) != len(masks_source):
                raise ValueError(f"Mismatched: {len(source)} images vs {len(masks_source)} masks")
            images_data = source
            masks_data = masks_source

        else:
            raise TypeError("Invalid source type provided.")

        dataset = BiosegDataset(
            images=images_data,
            masks=masks_data,
            image_size=image_size,
            transform=transforms
        )
        
        return TorchDataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            **kwargs
        )

    @staticmethod
    def plot_samples(dataloader, num_samples=2, figsize=(12, 6)):
        
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