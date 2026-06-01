import platform
import random
import sys

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> int:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def environment_info() -> dict:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
    }

    for package_name in ("numpy", "cv2", "torch", "torchvision"):
        try:
            module = __import__(package_name)
            info[package_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[package_name] = None

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        info["device_suggestion"] = "cuda" if info["cuda_available"] else "mps" if info["mps_available"] else "cpu"
    except ImportError:
        info["cuda_available"] = False
        info["mps_available"] = False
        info["device_suggestion"] = "cpu"

    return info
