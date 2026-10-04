"""Checkpoint provenance and atomic persistence."""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

CHECKPOINT_FORMAT_VERSION = 1


def package_versions():
    versions = {"torch": str(torch.__version__)}
    for name in ("bicbioseg", "torchvision", "numpy", "timm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain model state and configuration in a dictionary.")
    version = checkpoint.get("checkpoint_format_version", 0)
    if not isinstance(version, int) or version < 0 or version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {version!r}. This installation supports version {CHECKPOINT_FORMAT_VERSION}."
        )
    return checkpoint


def dataset_identity(train_loader, val_loader=None, dataset_id=None):
    if dataset_id is not None:
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("dataset_id must be a nonempty string.")
        return {"kind": "user", "id": dataset_id}
    digest = hashlib.sha256()
    samples = {}
    complete = True

    def update(value):
        nonlocal complete
        if isinstance(value, (str, os.PathLike)):
            path = Path(value)
            digest.update(path.name.encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        elif isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous()
            digest.update(str((tuple(array.shape), array.dtype)).encode())
            digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
        else:
            complete = False
        digest.update(b"\0")

    for split, loader in [("train", train_loader), ("validate", val_loader)]:
        if loader is None:
            continue
        dataset = loader.dataset
        samples[split] = len(dataset)
        digest.update(f"{split}:{len(dataset)}".encode())
        if hasattr(dataset, "images") and hasattr(dataset, "masks"):
            for pair in zip(dataset.images, dataset.masks):
                for value in pair:
                    update(value)
        elif hasattr(dataset, "tensors"):
            for value in dataset.tensors:
                update(value)
        else:
            complete = False
            digest.update(type(dataset).__qualname__.encode())
    return {
        "kind": "sha256" if complete else "descriptor",
        "id": digest.hexdigest(),
        "samples": samples,
        "content_verified": complete,
    }
