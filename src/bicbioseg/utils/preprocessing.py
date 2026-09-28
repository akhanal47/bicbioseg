import cv2
import numpy as np
import torch


def validate_normalization(policy=None):
    policy = dict(policy or {"mode": "standard"})
    mode = policy.setdefault("mode", "standard")
    allowed = {"standard": {"mode"}, "dtype": {"mode"},
               "range": {"mode", "min", "max"}, "percentile": {"mode", "lower", "upper"}}
    if mode not in allowed or set(policy) - allowed[mode]:
        raise ValueError("normalization mode must be standard, dtype, range, or percentile, with matching parameters.")
    if mode == "range":
        low, high = policy.get("min"), policy.get("max")
        if low is None or high is None or not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("range normalization requires finite min < max.")
    if mode == "percentile":
        policy.setdefault("lower", 1.0)
        policy.setdefault("upper", 99.0)
        if not 0 <= policy["lower"] < policy["upper"] <= 100:
            raise ValueError("percentile normalization requires 0 <= lower < upper <= 100.")
    return policy


def normalize_image(image, normalization=None):
    policy = validate_normalization(normalization)
    image = np.asarray(image)
    if not image.size or not np.isfinite(image).all():
        raise ValueError("Images must be nonempty and finite.")
    mode = policy["mode"]
    if mode == "standard":
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        if np.issubdtype(image.dtype, np.floating) and image.min() >= 0 and image.max() <= 1:
            return image.astype(np.float32)
        raise ValueError("Images must be uint8 or finite float arrays in [0, 1]. Choose an explicit normalization policy for uint16.")
    if mode == "dtype":
        if not np.issubdtype(image.dtype, np.unsignedinteger):
            raise ValueError("dtype normalization requires unsigned integer images.")
        low, high = 0, np.iinfo(image.dtype).max
    elif mode == "range":
        low, high = policy["min"], policy["max"]
    else:
        low, high = np.percentile(image, [policy["lower"], policy["upper"]])
    image = image.astype(np.float32)
    return np.clip((image - low) / (high - low), 0, 1).astype(np.float32) if high > low else np.zeros_like(image)


def image_tensor(image, image_size, in_channels=3, normalization=None):
    policy = validate_normalization(normalization)
    # Preserve legacy uint8 resize rounding under the default policy. Explicit
    # high-bit-depth policies normalize first to avoid integer interpolation.
    if policy["mode"] != "standard":
        image = normalize_image(image, policy)
    height, width = image_size
    image = cv2.resize(np.asarray(image), (width, height), interpolation=cv2.INTER_LINEAR)
    image = normalize_image(image)
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
