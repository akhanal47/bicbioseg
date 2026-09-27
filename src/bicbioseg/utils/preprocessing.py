import cv2
import numpy as np
import torch


def image_tensor(image, image_size, in_channels=3):
    image = np.asarray(image)
    height, width = image_size
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif np.issubdtype(image.dtype, np.floating) and np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1:
        image = image.astype(np.float32)
    else:
        raise ValueError("Images must be uint8 or finite float arrays in [0, 1]. Normalize high-bit-depth images explicitly first.")
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError("Each image must have shape HxW or HxWxC.")
    if image.shape[-1] == 1 and in_channels == 3:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 3 and in_channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., None]
    if image.shape[-1] != in_channels:
        raise ValueError(f"Expected {in_channels} image channels, got {image.shape[-1]}.")
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[-1] != 3:
        raise ValueError("Image files must be grayscale or RGB; convert alpha/multichannel images explicitly.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
