import os
import random
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch

from ..utils.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_torch_save,
    package_versions,
    read_checkpoint,
)


def _load_checkpoint_for_resume(self, path, optimizer=None, checkpoint=None):
    checkpoint = checkpoint if checkpoint is not None else read_checkpoint(path, map_location="cpu")
    self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    self.history = checkpoint.get("history", self.history)
    self.completed_epoch = checkpoint.get("completed_epoch", len(self.history.get("train_loss", [])))
    self.checkpoint_path = str(Path(path).resolve())
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for name, obj in (("scheduler", self._scheduler), ("scaler", self._scaler)):
        if obj is not None and checkpoint.get(f"{name}_state_dict") is not None:
            obj.load_state_dict(checkpoint[f"{name}_state_dict"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        state = checkpoint["numpy_rng_state"]
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    if self.device.type == "mps" and "mps_rng_state" in checkpoint:
        torch.mps.set_rng_state(checkpoint["mps_rng_state"].cpu())
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if self.device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return checkpoint


def save(self, path: Union[str, os.PathLike], optimizer=None) -> str:
    checkpoint = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "package_versions": package_versions(),
        "completed_epoch": self.completed_epoch,
        "dataset_identity": self.dataset_identity,
        "training_config": getattr(self, "last_training_config", {}),
        "architecture": self.architecture,
        "loss": self.loss_name,
        "metrics": self.metrics,
        "image_size": self.image_size,
        "num_classes": self.num_classes,
        "in_channels": self.in_channels,
        "model_kwargs": self.model_kwargs,
        "loss_kwargs": self.loss_kwargs,
        "normalization": self.normalization,
        "ignore_index": self.ignore_index,
        "include_background": self.include_background,
        "metric_aggregation": self.metric_aggregation,
        "empty_policy": self.empty_policy,
        "class_map": self.class_map,
        "model_state_dict": self.model.state_dict(),
        "history": self.history,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if optimizer is not None:
        checkpoint["optimizer_name"] = optimizer.__class__.__name__.lower()
        checkpoint["training_options"] = self._training_options
        checkpoint["scheduler_state_dict"] = self._scheduler.state_dict() if self._scheduler is not None else None
        checkpoint["scaler_state_dict"] = self._scaler.state_dict() if self._scaler is not None else None
        checkpoint["torch_rng_state"] = torch.get_rng_state()
        checkpoint["python_rng_state"] = random.getstate()
        numpy_rng = np.random.get_state()
        checkpoint["numpy_rng_state"] = (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:])
        if self.device.type == "mps":
            checkpoint["mps_rng_state"] = torch.mps.get_rng_state()
        if self.device.type == "cuda":
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(checkpoint, path)
    return str(path.resolve())


def load_weights(self, path: Union[str, os.PathLike]):
    checkpoint = read_checkpoint(path, map_location=self.device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    self.model.load_state_dict(state_dict)
    self.history = checkpoint.get("history", self.history)
    return self


def load(
    cls,
    path: Union[str, os.PathLike],
    device: Optional[str] = None,
    loss: Optional[Union[str, torch.nn.Module, Callable]] = None,
    **overrides,
):
    checkpoint = read_checkpoint(path, map_location=cls.resolve_device(device))
    saved_image_size = checkpoint.get("image_size", (224, 224))
    if isinstance(saved_image_size, list):
        saved_image_size = tuple(saved_image_size)

    config = {
        "architecture": checkpoint.get("architecture", "unet"),
        "loss": loss or checkpoint.get("loss", "dice"),
        "metrics": checkpoint.get("metrics", ["dice", "iou"]),
        "image_size": saved_image_size,
        "num_classes": checkpoint.get("num_classes", 1),
        "in_channels": checkpoint.get("in_channels", 3),
        "model_kwargs": checkpoint.get("model_kwargs", {}),
        "loss_kwargs": checkpoint.get("loss_kwargs", {}),
        "normalization": checkpoint.get("normalization"),
        "ignore_index": checkpoint.get("ignore_index"),
        "include_background": checkpoint.get("include_background", False),
        "metric_aggregation": checkpoint.get("metric_aggregation", "per_image"),
        "empty_policy": checkpoint.get("empty_policy", "exclude"),
        "class_map": checkpoint.get("class_map"),
        "device": device,
    }
    config.update(overrides)

    # Initialization weights are redundant when restoring the full state.
    # Preserve the saved configuration as provenance, but never download
    # encoder weights merely to overwrite them from this checkpoint.
    saved_model_kwargs = dict(config["model_kwargs"])
    restore_kwargs = dict(saved_model_kwargs)
    architecture = config["architecture"].lower()
    architecture = cls.MODEL_ALIASES.get(architecture, architecture)
    if architecture == "transunet":
        restore_kwargs["encoder_weights"] = None
    elif architecture in {"deit", "swin_unet", "swin_unet_full", "pvt_unet", "pvtformer_full", "double_unet", "dinov3_seg", "efficientvit_seg"}:
        restore_kwargs["pretrained"] = False
    config["model_kwargs"] = restore_kwargs
    segmenter = cls(**config)
    segmenter.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    segmenter.model_kwargs = saved_model_kwargs
    segmenter.history = checkpoint.get("history", {})
    segmenter.completed_epoch = checkpoint.get(
        "completed_epoch", len(segmenter.history.get("train_loss", []))
    )
    segmenter.dataset_identity = checkpoint.get("dataset_identity")
    segmenter.checkpoint_path = str(Path(path).resolve())
    segmenter.model.eval()
    return segmenter
