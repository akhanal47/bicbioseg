import csv
import json
import os
import shutil
import platform
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from .config import SegmenterConfig, TrainingConfig, ExperimentRunConfig
from .exceptions import InferenceError, ModelError
from .models.registry import MODEL_ALIASES, build_model, model_metadata
from .models.options import model_options, normalize_model_options
from .utils.preprocessing import image_tensor, read_rgb, validate_normalization, normalize_image
from .utils.training import autocast_context, validate_training_options, make_scheduler
from .utils import losses as loss_module
from .utils.load_data import DataLoader
from .utils.progress import progress_iter
from .utils.evaluation import MetricAccumulator, validate_metric_options
from .utils.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_torch_save,
    read_checkpoint,
    package_versions,
    dataset_identity,
)


class Segmenter:
    MODEL_ALIASES = MODEL_ALIASES

    LOSS_ALIASES = {
        "bce": loss_module.BCELoss,
        "binary_cross_entropy": loss_module.BCELoss,
        "dice": loss_module.DiceLoss,
        "dice_bce": loss_module.DiceBCELoss,
        "combo": loss_module.DiceBCELoss,
        "focal": loss_module.FocalLoss,
        "log_cosh_dice": loss_module.LogCoshDiceLoss,
        "jaccard": loss_module.JaccardLoss,
        "iou": loss_module.JaccardLoss,
        "tversky": loss_module.TverskyLoss,
        "focal_tversky": loss_module.FocalTverskyLoss,
        "cross_entropy": torch.nn.CrossEntropyLoss,
        "sensitivity_specificity": loss_module.SensitivitySpecificityLoss,
    }

    def __init__(
        self,
        architecture: str = "unet",
        loss: Union[str, torch.nn.Module, Callable] = "dice",
        metrics: Optional[Sequence[str]] = None,
        image_size=(224, 224),
        num_classes: int = 1,
        in_channels: int = 3,
        device: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        loss_kwargs: Optional[dict] = None,
        normalization: Optional[dict] = None,
        ignore_index: Optional[int] = None,
        include_background: bool = False,
        metric_aggregation: str = "per_image",
        empty_policy: str = "exclude",
        class_map: Optional[dict] = None,
    ):
        self.architecture = self.MODEL_ALIASES.get(architecture.lower(), architecture.lower())
        self.loss_name = loss.lower() if isinstance(loss, str) else loss.__class__.__name__
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        if len(self.image_size) != 2 or any(not isinstance(v, int) or v <= 0 for v in self.image_size):
            raise ModelError("image_size must be a positive (height, width) pair.")
        if num_classes < 1 or in_channels < 1:
            raise ModelError("num_classes and in_channels must be positive.")
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.model_kwargs = normalize_model_options(
            self.architecture,
            model_kwargs,
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=self.image_size,
        )
        self.loss_kwargs = dict(loss_kwargs or {})
        self.normalization = validate_normalization(normalization)
        if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, int)):
            raise ModelError("ignore_index must be an integer or None.")
        self.ignore_index = ignore_index
        validate_metric_options(metric_aggregation, empty_policy)
        self.include_background, self.metric_aggregation, self.empty_policy = (
            include_background,
            metric_aggregation,
            empty_policy,
        )
        self.class_map = {int(k): str(v) for k, v in (class_map or {}).items()}
        if not self.class_map:
            self.class_map = (
                {0: "background", 1: "foreground"}
                if num_classes == 1
                else {i: f"class_{i}" for i in range(num_classes)}
            )
        expected = set(range(2 if num_classes == 1 else num_classes))
        if set(self.class_map) != expected or len(set(self.class_map.values())) != len(self.class_map):
            raise ModelError("class_map must supply a unique name for every class ID, including background.")
        self._training_options = {}
        self._scheduler = self._scaler = None

        self.device = self.resolve_device(device or "auto")
        self.metrics = list(dict.fromkeys("iou" if m.lower() == "jaccard" else m.lower() for m in (metrics if metrics is not None else ["dice", "iou"])))
        unknown = set(self.metrics) - {"dice", "iou", "precision", "recall"}
        if unknown:
            raise ModelError(f"Unsupported metrics: {sorted(unknown)}")
        self.model = self._build_model(self.model_kwargs).to(self.device)
        self.loss_fn = self._build_loss(loss, self.loss_kwargs)
        if isinstance(self.loss_fn, torch.nn.Module):
            self.loss_fn.to(self.device)
        self.history: Dict[str, List[float]] = {}
        self.completed_epoch = 0
        self.dataset_identity = None
        self.checkpoint_path = None

    @classmethod
    def available_models(cls, detailed=False):
        names = sorted(set(cls.MODEL_ALIASES.values()))
        return {name: model_metadata(name) for name in names} if detailed else names

    @staticmethod
    def model_options(architecture):
        return model_options(architecture)

    @classmethod
    def available_losses(cls) -> List[str]:
        return sorted(cls.LOSS_ALIASES.keys())

    @classmethod
    def from_config(cls, config: SegmenterConfig):
        return cls(
            architecture=config.architecture,
            loss=config.loss,
            metrics=config.metrics,
            image_size=config.image_size,
            num_classes=config.num_classes,
            in_channels=config.in_channels,
            device=config.device,
            model_kwargs=config.model_kwargs,
            loss_kwargs=config.loss_kwargs,
            normalization=config.normalization,
            ignore_index=config.ignore_index,
            include_background=config.include_background,
            metric_aggregation=config.metric_aggregation,
            empty_policy=config.empty_policy,
            class_map=config.class_map,
        )

    @staticmethod
    def available_devices() -> List[str]:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
        return devices

    @staticmethod
    def resolve_device(device: Optional[str] = "auto") -> torch.device:
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if device is None or str(device) == "auto":
            if platform.system() == "Darwin" and mps_available:
                return torch.device("mps")
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            resolved = torch.device(device)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise ModelError(f"Invalid device: {device!r}.") from exc
        if resolved.type not in {"cpu", "cuda", "mps"}:
            raise ModelError("device must be 'auto', 'cpu', 'mps', or 'cuda[:index]'.")
        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise ModelError("CUDA was requested but is not available.")
            if resolved.index is not None and resolved.index >= torch.cuda.device_count():
                raise ModelError(f"CUDA device index {resolved.index} is unavailable.")
        if resolved.type == "mps" and not mps_available:
            raise ModelError("MPS was requested but is not available.")
        if resolved.type in {"cpu", "mps"} and resolved.index not in (None, 0):
            raise ModelError(f"{resolved.type} supports only device index 0.")
        return resolved

    def _build_model(self, model_kwargs: dict) -> torch.nn.Module:
        return build_model(self.architecture, self.in_channels, self.num_classes, self.image_size, model_kwargs)

    def _resize_logits(self, logits, images):
        if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
            raise ModelError("Model must return an NCHW logits tensor.")
        if logits.shape[:2] != (images.shape[0], self.num_classes):
            raise ModelError("Model output batch/classes do not match the Segmenter configuration.")
        if logits.shape[-2:] != images.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def _forward_predictions(self, images):
        output = self.model(images)
        if isinstance(output, dict):
            main, aux = output.get("logits"), output.get("aux_logits", [])
        elif isinstance(output, (tuple, list)) and output:
            main, aux = output[0], output[1:]
        else:
            main, aux = output, []
        if not isinstance(aux, (tuple, list)):
            raise ModelError("aux_logits must be a list or tuple of NCHW tensors.")
        return self._resize_logits(main, images), [self._resize_logits(item, images) for item in aux]

    def _forward_logits(self, images):
        return self._forward_predictions(images)[0]

    def _prediction_loss(self, main, auxiliary, targets):
        loss = self.loss_fn(main.float(), targets)
        weights = self._training_options.get("aux_loss_weights")
        if weights is None:
            weights = [0.4] * len(auxiliary)
        if (auxiliary or self.model.training) and len(weights) != len(auxiliary):
            raise ModelError("aux_loss_weights must have one weight per auxiliary prediction.")
        for weight, logits in zip(weights, auxiliary):
            loss = loss + weight * self.loss_fn(logits.float(), targets)
        return loss

    def validate_setup(self, sample_batch=None, *, backward=False, precision="fp32"):
        # check prepared NCHW tensors on the selected device without changing weights or gradients.
        options = dict(precision=precision, accumulation_steps=1, max_grad_norm=None, aux_loss_weights=None)
        validate_training_options(self.device, options)
        images, targets = (sample_batch if isinstance(sample_batch, (tuple, list)) else (sample_batch, None))
        if images is None:
            images = torch.zeros(1, self.in_channels, *self.image_size)
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != self.in_channels:
            raise ModelError("Preflight images must be an NCHW tensor with the configured channels.")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ModelError("Preflight images must be finite floating-point tensors after normalization.")
        modes = [(module, module.training) for module in self.model.modules()]
        buffers = [(buffer, buffer.clone()) for buffer in self.model.buffers()] if backward else []
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
        mps_rng = torch.mps.get_rng_state() if self.device.type == "mps" else None
        try:
            self.model.train(backward)
            images = images.to(self.device)
            with torch.set_grad_enabled(backward), autocast_context(self.device, precision):
                logits, auxiliary = self._forward_predictions(images)
                if not all(torch.isfinite(t).all() for t in [logits, *auxiliary]):
                    raise ModelError("Model produced nonfinite logits.")
                loss = None
                if targets is not None:
                    targets = self._prepare_targets(targets)
                    loss = self._prediction_loss(logits, auxiliary, targets)
                    if loss.ndim or not torch.isfinite(loss):
                        raise ModelError("Loss must be a finite scalar.")
                if backward:
                    objective = loss if loss is not None else logits.float().square().mean()
                    parameters = [p for p in self.model.parameters() if p.requires_grad]
                    grads = torch.autograd.grad(objective, parameters, allow_unused=True)
                    if not any(g is not None for g in grads) or any(g is not None and not torch.isfinite(g).all() for g in grads):
                        raise ModelError("Preflight gradients are missing or nonfinite.")
            return {"architecture": self.architecture, "device": str(self.device), "precision": precision,
                    "input_shape": tuple(images.shape), "output_shape": tuple(logits.shape),
                    "auxiliary_predictions": len(auxiliary), "backward_checked": backward,
                    "loss": None if loss is None else float(loss.detach().cpu()), "ok": True}
        except (RuntimeError, ValueError) as exc:
            raise ModelError(f"Preflight failed for {self.architecture} on {self.device}: {exc}") from exc
        finally:
            for module, mode in modes:
                module.training = mode
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            if mps_rng is not None:
                torch.mps.set_rng_state(mps_rng)

    def _build_loss(self, loss, loss_kwargs: dict):
        if isinstance(loss, torch.nn.Module):
            return loss_module.IgnoreLabelsLoss(loss, self.ignore_index) if self.ignore_index is not None else loss
        if isinstance(loss, type) and issubclass(loss, torch.nn.Module):
            result = loss(**loss_kwargs)
            return loss_module.IgnoreLabelsLoss(result, self.ignore_index) if self.ignore_index is not None else result
        if callable(loss) and not isinstance(loss, str):
            return loss_module.IgnoreLabelsLoss(loss, self.ignore_index) if self.ignore_index is not None else loss

        loss_key = loss.lower()
        if loss_key not in self.LOSS_ALIASES:
            raise ModelError(f"Unknown loss '{loss}'. Available: {self.available_losses()}")

        kwargs = dict(loss_kwargs)
        if self.num_classes > 1 and loss_key not in {"dice", "jaccard", "iou", "cross_entropy"}:
            raise ModelError(f"Loss '{loss_key}' is binary-only; use dice, jaccard, or cross_entropy for multiclass training.")
        if self.num_classes == 1 and loss_key == "cross_entropy":
            raise ModelError("cross_entropy requires num_classes >= 2; use bce for binary segmentation.")
        if loss_key in {"dice", "jaccard", "iou"}:
            mode = "binary" if self.num_classes == 1 else "multiclass"
            if "mode" in kwargs and kwargs["mode"] != mode:
                raise ModelError("loss_kwargs mode conflicts with num_classes.")
            kwargs["mode"] = mode
        if "ignore_index" in kwargs:
            if self.ignore_index != kwargs.pop("ignore_index"):
                raise ModelError("Set ignore_index on Segmenter so losses and metrics agree.")
        result = self.LOSS_ALIASES[loss_key](**kwargs)
        return loss_module.IgnoreLabelsLoss(result, self.ignore_index) if self.ignore_index is not None else result

    def _make_optimizer(self, optimizer: Union[str, torch.optim.Optimizer], lr: float):
        if isinstance(optimizer, torch.optim.Optimizer):
            return optimizer

        name = optimizer.lower()
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        if name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr)
        if name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        raise ModelError("optimizer must be 'adam', 'adamw', 'sgd', or a torch optimizer.")

    def _config(self) -> dict:
        return {
            "architecture": self.architecture,
            "loss": self.loss_name,
            "metrics": self.metrics,
            "image_size": list(self.image_size) if isinstance(self.image_size, tuple) else self.image_size,
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
            "device": str(self.device),
        }

    @staticmethod
    def _prepare_run_dir(experiment_dir, run_name: Optional[str], resume_from=None) -> Optional[Path]:
        if experiment_dir is None:
            return None

        if run_name is None:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")

        run_dir = Path(experiment_dir) / run_name
        if run_dir.exists() and any(run_dir.iterdir()):
            if resume_from is None or Path(resume_from).resolve().parent != run_dir.resolve():
                raise FileExistsError(f"Run directory already contains results: {run_dir}. Choose a new run_name or resume a checkpoint from this directory.")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _write_json(path: Path, payload: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(payload, handle, indent=2)

    @staticmethod
    def _write_history_csv(path: Path, history: Dict[str, List[float]]):
        if not history:
            return

        keys = sorted(history)
        max_len = max(len(values) for values in history.values())
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch"] + keys)
            writer.writeheader()
            for idx in range(max_len):
                row = {"epoch": idx + 1}
                for key in keys:
                    values = history.get(key, [])
                    row[key] = values[idx] if idx < len(values) else ""
                writer.writerow(row)

    def _update_run_artifacts(self, run_dir: Optional[Path]):
        if run_dir is None:
            return
        self._write_json(run_dir / "history.json", self.history)
        self._write_history_csv(run_dir / "history.csv", self.history)

    @staticmethod
    def _monitor_mode(monitor: str, mode: str = "auto") -> str:
        if mode != "auto":
            return mode
        if "loss" in monitor.lower():
            return "min"
        return "max"

    @staticmethod
    def _is_improved(current_value: float, best_value: Optional[float], mode: str, min_delta: float) -> bool:
        if best_value is None:
            return True
        if mode == "min":
            return current_value < best_value - min_delta
        if mode == "max":
            return current_value > best_value + min_delta
        raise ValueError("mode must be 'auto', 'min', or 'max'.")

    @staticmethod
    def _load_history(history_or_path) -> Dict[str, List[float]]:
        if isinstance(history_or_path, (str, os.PathLike)):
            path = Path(history_or_path)
            with path.open() as handle:
                return json.load(handle)
        return history_or_path

    @staticmethod
    def plot_history_file(
        history_path: Union[str, os.PathLike],
        metrics: Optional[Sequence[str]] = None,
        save_to: Optional[Union[str, os.PathLike]] = None,
        show: bool = True,
        figsize=(8, 5),
    ):
        history = Segmenter._load_history(history_path)
        return Segmenter._plot_history(
            history=history,
            metrics=metrics,
            save_to=save_to,
            show=show,
            figsize=figsize,
        )

    @staticmethod
    def _plot_history(
        history: Dict[str, List[float]],
        metrics: Optional[Sequence[str]] = None,
        save_to: Optional[Union[str, os.PathLike]] = None,
        show: bool = True,
        figsize=(8, 5),
    ):
        import matplotlib.pyplot as plt

        if not history:
            raise ValueError("No training history available to plot.")

        selected_metrics = list(metrics) if metrics is not None else sorted(history)
        missing = [metric for metric in selected_metrics if metric not in history]
        if missing:
            raise ValueError(f"Metrics not found in history: {missing}")

        fig, ax = plt.subplots(figsize=figsize)
        for metric in selected_metrics:
            values = history[metric]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, marker="o", label=metric)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric value")
        ax.set_title("Training History")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        if save_to is not None:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        return fig

    def plot_history(
        self,
        metrics: Optional[Sequence[str]] = None,
        save_to: Optional[Union[str, os.PathLike]] = None,
        show: bool = True,
        figsize=(8, 5),
    ):
        return self._plot_history(
            history=self.history,
            metrics=metrics,
            save_to=save_to,
            show=show,
            figsize=figsize,
        )

    def _loader_from_data(
        self,
        data,
        batch_size: int,
        transforms=None,
        num_workers: int = 2,
        shuffle: bool = True,
    ):
        if isinstance(data, TorchDataLoader):
            return data

        return DataLoader.load_data(
            source=data,
            in_channels=self.in_channels,
            image_size=self.image_size,
            batch_size=batch_size,
            transforms=transforms,
            num_workers=num_workers,
            shuffle=shuffle,
            normalization=self.normalization,
            ignore_index=self.ignore_index,
            **(getattr(self, "_crop_options", {}) if shuffle else {}),
        )

    def _resolve_loaders(
        self,
        data=None,
        train_data=None,
        val_data=None,
        batch_size: int = 4,
        transforms=None,
        num_workers: int = 2,
    ):
        if train_data is not None:
            train_loader = self._loader_from_data(train_data, batch_size, transforms, num_workers, shuffle=True)
            val_loader = (
                self._loader_from_data(val_data, batch_size, None, num_workers, shuffle=False)
                if val_data is not None
                else None
            )
            return train_loader, val_loader

        if data is None:
            raise ValueError("Provide either data or train_data.")

        data_path = Path(data) if isinstance(data, (str, os.PathLike)) else None
        if data_path is not None and (data_path / "train").is_dir():
            train_loader = self._loader_from_data(
                str(data_path / "train"), batch_size, transforms, num_workers, shuffle=True
            )
            if val_data is not None:
                return train_loader, self._loader_from_data(val_data, batch_size, None, num_workers, shuffle=False)
            validate_path = data_path / "validate"
            val_loader = None
            if validate_path.is_dir() and (not (validate_path / "images").is_dir() or self._load_paths_static(validate_path / "images")):
                val_loader = self._loader_from_data(
                    str(validate_path), batch_size, None, num_workers, shuffle=False
                )
            return train_loader, val_loader

        return (
            self._loader_from_data(data, batch_size, transforms, num_workers, shuffle=True),
            self._loader_from_data(val_data, batch_size, None, num_workers, shuffle=False) if val_data is not None else None,
        )

    def _prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(self.device)
        if self.num_classes == 1:
            if targets.ndim == 3:
                targets = targets.unsqueeze(1)
            prepared = (targets > 0).float()
            if self.ignore_index is not None:
                prepared = prepared.masked_fill(targets == self.ignore_index, self.ignore_index)
            return prepared
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        valid = targets != self.ignore_index if self.ignore_index is not None else torch.ones_like(targets, dtype=torch.bool)
        if targets.ndim != 3 or (targets[valid] < 0).any() or (targets[valid] >= self.num_classes).any() or (targets != targets.long()).any():
            raise ModelError(f"Multiclass targets must have shape NHW and integer IDs in [0, {self.num_classes - 1}].")
        return targets.long()

    def _metric_accumulator(self):
        return MetricAccumulator(
            self.num_classes,
            self.metrics,
            self.include_background,
            self.ignore_index,
            self.empty_policy,
            aggregation=self.metric_aggregation,
        )

    def _add_metric_batch(self, accumulator, outputs, targets):
        predictions = outputs.sigmoid()[:, 0] > 0.5 if self.num_classes == 1 else outputs.argmax(dim=1)
        targets = targets[:, 0] if targets.ndim == 4 else targets
        for prediction, target in zip(predictions.detach().cpu().numpy(), targets.detach().cpu().numpy()):
            accumulator.add(prediction, target)

    def _metric_values(self, outputs, targets):
        accumulator = self._metric_accumulator()
        self._add_metric_batch(accumulator, outputs, targets)
        scores = accumulator.summary()["scores"]
        return {name: scores[name if self.num_classes == 1 else f"macro_{name}"] for name in self.metrics}

    @staticmethod
    def _tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
        image_np = image.detach().cpu().numpy()
        if image_np.ndim == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
        return np.clip(image_np, 0, 1)

    @staticmethod
    def _tensor_mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
        mask_np = mask.detach().cpu().numpy()
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        return mask_np

    @staticmethod
    def _overlay_mask_on_image(
        image: np.ndarray,
        mask: np.ndarray,
        color=(1.0, 0.0, 0.0),
        alpha: float = 0.4,
    ) -> np.ndarray:
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if np.issubdtype(image.dtype, np.unsignedinteger):
            image = image.astype(np.float32) / np.iinfo(image.dtype).max
        elif image.max() > 1:
            image = normalize_image(image, {"mode": "percentile"})
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask_bool = mask > 0
        overlay = image.copy()
        color_arr = np.array(color, dtype=np.float32)
        overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * color_arr
        return np.clip(overlay, 0, 1)

    def plot_training_samples(
        self,
        data=None,
        train_data=None,
        batch_size: int = 4,
        num_samples: int = 4,
        transforms=None,
        num_workers: int = 0,
        overlay: bool = True,
        save_to: Optional[Union[str, os.PathLike]] = None,
        show: bool = True,
        figsize=None,
    ):
        import matplotlib.pyplot as plt

        train_loader, _ = self._resolve_loaders(
            data=data,
            train_data=train_data,
            batch_size=batch_size,
            transforms=transforms,
            num_workers=num_workers,
        )
        images, masks = next(iter(train_loader))
        num_samples = min(num_samples, len(images))
        num_cols = 3 if overlay else 2
        figsize = figsize or (4 * num_cols, 3 * num_samples)

        fig, axes = plt.subplots(num_samples, num_cols, figsize=figsize, squeeze=False)
        for idx in range(num_samples):
            image = self._tensor_image_to_numpy(images[idx])
            mask = self._tensor_mask_to_numpy(masks[idx])

            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title("Image")
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(mask, cmap="gray", interpolation="nearest")
            axes[idx, 1].set_title("Mask")
            axes[idx, 1].axis("off")

            if overlay:
                axes[idx, 2].imshow(self._overlay_mask_on_image(image, mask))
                axes[idx, 2].set_title("Overlay")
                axes[idx, 2].axis("off")

        fig.tight_layout()
        if save_to is not None:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        return fig

    def _run_epoch(self, loader, optimizer=None) -> Dict[str, float]:
        is_train = optimizer is not None
        self.model.train(is_train)
        totals: Dict[str, float] = {"loss": 0.0}
        samples = pending_batches = pending_samples = 0
        metric_accumulator = self._metric_accumulator()
        options = self._training_options
        precision = options.get("precision", "fp32")
        accumulation = options.get("accumulation_steps", 1)
        scaler = self._scaler
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        def step(sample_count):
            if scaler is not None:
                scaler.unscale_(optimizer)
            # Weight microbatches by sample count, including the final partial group.
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(sample_count)
            if options.get("max_grad_norm") is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), options["max_grad_norm"])
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for images, targets in loader:
            images = images.to(self.device)
            targets = self._prepare_targets(targets)
            batch_size = images.shape[0]
            with torch.set_grad_enabled(is_train), autocast_context(self.device, precision):
                outputs, auxiliary = self._forward_predictions(images)
                loss = self._prediction_loss(outputs, auxiliary, targets)
            if loss.ndim or not torch.isfinite(loss):
                raise ModelError("Training/validation loss must be a finite scalar.")
            if is_train:
                weighted_loss = loss * batch_size
                (scaler.scale(weighted_loss) if scaler is not None else weighted_loss).backward()
                pending_batches += 1
                pending_samples += batch_size
                if pending_batches == accumulation:
                    step(pending_samples)
                    pending_batches = pending_samples = 0
            self._add_metric_batch(metric_accumulator, outputs.detach().float(), targets.detach())
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            samples += batch_size
        if is_train and pending_batches:
            step(pending_samples)
        if samples == 0:
            raise ValueError("Cannot train or validate on an empty loader.")
        if is_train:
            self._last_train_samples = samples
        self.last_metric_summary = metric_accumulator.summary()
        scores = self.last_metric_summary["scores"]
        return {
            "loss": totals["loss"] / samples,
            **{name: scores[name if self.num_classes == 1 else f"macro_{name}"] for name in self.metrics},
        }

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

    def train(
        self,
        data=None,
        train_data=None,
        val_data=None,
        epochs: int = 1,
        batch_size: int = 4,
        lr: float = 1e-4,
        optimizer: Optional[Union[str, torch.optim.Optimizer]] = None,
        transforms=None,
        num_workers: int = 2,
        save_to: Optional[Union[str, os.PathLike]] = None,
        experiment_dir: Optional[Union[str, os.PathLike]] = None,
        run_name: Optional[str] = None,
        save_best: bool = True,
        monitor: str = "val_loss",
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.0,
        monitor_mode: str = "auto",
        resume_from: Optional[Union[str, os.PathLike]] = None,
        progress_bar: bool = True,
        verbose: bool = True,
        config: Optional[TrainingConfig] = None,
        precision: Optional[str] = None,
        accumulation_steps: Optional[int] = None,
        max_grad_norm: Optional[float] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[dict] = None,
        aux_loss_weights: Optional[Sequence[float]] = None,
        crop_size=None,
        foreground_probability: Optional[float] = None,
        checkpoint_interval: Optional[int] = None,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        if config is not None:
            epochs = config.epochs
            batch_size = config.batch_size
            lr = config.lr
            optimizer = config.optimizer
            num_workers = config.num_workers
            save_to = config.save_to if config.save_to is not None else save_to
            experiment_dir = config.experiment_dir if config.experiment_dir is not None else experiment_dir
            run_name = config.run_name if config.run_name is not None else run_name
            save_best = config.save_best
            monitor = config.monitor
            early_stopping = config.early_stopping
            patience = config.patience
            min_delta = config.min_delta
            monitor_mode = config.monitor_mode
            resume_from = config.resume_from if config.resume_from is not None else resume_from
            progress_bar = config.progress_bar
            verbose = config.verbose
            precision, accumulation_steps = config.precision, config.accumulation_steps
            max_grad_norm = config.max_grad_norm
            scheduler, scheduler_kwargs = config.scheduler, config.scheduler_kwargs
            aux_loss_weights = config.aux_loss_weights
            crop_size, foreground_probability = config.crop_size, config.foreground_probability
            checkpoint_interval, dataset_id = config.checkpoint_interval, config.dataset_id

        checkpoint = read_checkpoint(resume_from, map_location="cpu") if resume_from is not None else {}
        saved_options = checkpoint.get("training_options", {})
        if optimizer is None:
            optimizer = checkpoint.get("optimizer_name", "adam")
        if checkpoint.get("optimizer_name") and isinstance(optimizer, str) and optimizer.lower() != checkpoint["optimizer_name"]:
            raise ValueError("Resume must use the saved optimizer type.")
        if checkpoint:
            for key, current in (("normalization", self.normalization), ("ignore_index", self.ignore_index)):
                if checkpoint.get(key, {"mode": "standard"} if key == "normalization" else None) != current:
                    raise ValueError(f"Resume {key} differs from the checkpoint; load the checkpoint with Segmenter.load first.")
        options = dict(precision=precision, accumulation_steps=accumulation_steps,
                       max_grad_norm=max_grad_norm, scheduler=scheduler,
                       scheduler_kwargs=scheduler_kwargs, aux_loss_weights=aux_loss_weights,
                       crop_size=crop_size, foreground_probability=foreground_probability)
        defaults = dict(precision="fp32", accumulation_steps=1, max_grad_norm=None,
                        scheduler=None, scheduler_kwargs={}, aux_loss_weights=None,
                        crop_size=None, foreground_probability=0.0)
        for key in options:
            if options[key] is None:
                options[key] = saved_options.get(key, defaults[key])
        if options["scheduler"] == "cosine":
            options["scheduler_kwargs"] = dict(options["scheduler_kwargs"])
            options["scheduler_kwargs"].setdefault("T_max", epochs)
        if saved_options and (options["scheduler"] != saved_options.get("scheduler") or
                              options["scheduler_kwargs"] != saved_options.get("scheduler_kwargs")):
            raise ValueError("Resume must use the saved scheduler and scheduler_kwargs.")
        if checkpoint.get("scaler_state_dict") and options["precision"] != saved_options.get("precision"):
            raise ValueError("Resume must use the saved precision when a gradient scaler is present.")
        validate_training_options(self.device, options)
        self._training_options = options
        self._crop_options = {key: options[key] for key in ("crop_size", "foreground_probability")}
        train_loader, val_loader = self._resolve_loaders(
            data=data,
            train_data=train_data,
            val_data=val_data,
            batch_size=batch_size,
            transforms=transforms,
            num_workers=num_workers,
        )
        if checkpoint_interval is not None and (
            isinstance(checkpoint_interval, bool)
            or not isinstance(checkpoint_interval, int)
            or checkpoint_interval < 1
        ):
            raise ValueError("checkpoint_interval must be a positive integer or None.")
        identity = dataset_identity(train_loader, val_loader, dataset_id)
        saved_identity = checkpoint.get("dataset_identity")
        if saved_identity and saved_identity.get("kind") != "descriptor" and saved_identity != identity:
            raise ValueError(
                "Resume dataset identity differs from the checkpoint. Use the same data and dataset_id."
            )
        self.dataset_identity = identity
        if epochs < 1 or patience < 0 or min_delta < 0:
            raise ValueError("epochs must be positive; patience and min_delta must be nonnegative.")
        if monitor_mode not in {"auto", "min", "max"}:
            raise ValueError("monitor_mode must be 'auto', 'min', or 'max'.")
        if val_loader is None and monitor == "val_loss":
            monitor = "train_loss"
        available_monitors = {f"{prefix}_{metric}" for prefix in (["train", "val"] if val_loader is not None else ["train"]) for metric in ["loss", *self.metrics]}
        if monitor not in available_monitors:
            raise ValueError(f"Monitor '{monitor}' is unavailable. Choose from {sorted(available_monitors)}.")
        optimizer_obj = self._make_optimizer(optimizer, lr)
        self._scheduler = make_scheduler(optimizer_obj, options["scheduler"], options["scheduler_kwargs"],
                                         epochs, self._monitor_mode(monitor, monitor_mode))
        self._scaler = torch.amp.GradScaler("cuda") if options["precision"] == "fp16" else None
        if resume_from is not None:
            self._load_checkpoint_for_resume(resume_from, optimizer=optimizer_obj, checkpoint=checkpoint)

        if resume_from is None:
            self.completed_epoch = 0
            self.checkpoint_path = None
        if resume_from is None or not self.history:
            self.history = {"train_loss": []}
        else:
            self.history.setdefault("train_loss", [])
        run_dir = self._prepare_run_dir(experiment_dir, run_name, resume_from)
        self.last_run_dir = run_dir
        if checkpoint_interval is not None and run_dir is None:
            raise ValueError("checkpoint_interval requires experiment_dir.")
        best_value = None
        stale_epochs = 0
        resolved_monitor_mode = self._monitor_mode(monitor, monitor_mode)
        # reconstruct the best value and patience from completed epochs on resume.
        for value in self.history.get(monitor, []):
            if value is None:
                continue
            if self._is_improved(value, best_value, resolved_monitor_mode, min_delta):
                best_value, stale_epochs = value, 0
            else:
                stale_epochs += 1

        self.last_training_config = {
            **options,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "lr": optimizer_obj.param_groups[0]["lr"],
            "optimizer": optimizer_obj.__class__.__name__,
            "monitor": monitor,
            "early_stopping": early_stopping,
            "patience": patience,
            "min_delta": min_delta,
            "monitor_mode": resolved_monitor_mode,
            "checkpoint_interval": checkpoint_interval,
            "dataset_id": dataset_id,
        }
        if run_dir is not None:
            config = self._config()
            config["training"] = self.last_training_config
            self._write_json(run_dir / "config.json", config)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        processed_samples = 0
        epoch_iter = progress_iter(range(1, epochs + 1), enabled=progress_bar, desc="Training")
        for epoch in epoch_iter:
            self.history.setdefault("lr", []).append(optimizer_obj.param_groups[0]["lr"])
            train_stats = self._run_epoch(train_loader, optimizer=optimizer_obj)
            train_metric_summary = getattr(self, "last_metric_summary", {})
            processed_samples += getattr(self, "_last_train_samples", 0)
            self.history["train_loss"].append(train_stats["loss"])
            for name, value in train_stats.items():
                if name != "loss":
                    self.history.setdefault(f"train_{name}", []).append(value)

            message = f"Epoch {epoch}/{epochs} - train_loss: {train_stats['loss']:.4f}"

            if val_loader is not None:
                with torch.no_grad():
                    val_stats = self._run_epoch(val_loader, optimizer=None)
                self.history.setdefault("val_loss", []).append(val_stats["loss"])
                for name, value in val_stats.items():
                    if name != "loss":
                        self.history.setdefault(f"val_{name}", []).append(value)
                message += f" - val_loss: {val_stats['loss']:.4f}"

            if verbose:
                print(message)

            if self._scheduler is not None:
                if options["scheduler"] == "plateau":
                    if self.history[monitor][-1] is not None:
                        self._scheduler.step(self.history[monitor][-1])
                else:
                    self._scheduler.step()
            self.completed_epoch += 1
            self.checkpoint_path = None
            self.training_metric_summary = {
                "train": train_metric_summary,
                "validate": getattr(self, "last_metric_summary", {}) if val_loader is not None else None,
            }
            if (
                run_dir is not None
                and checkpoint_interval
                and self.completed_epoch % checkpoint_interval == 0
            ):
                self.save(run_dir / "last_model.pt", optimizer=optimizer_obj)
            self._update_run_artifacts(run_dir)
            monitor_values = self.history.get(monitor)
            if monitor_values and monitor_values[-1] is not None:
                current_value = monitor_values[-1]
                improved = self._is_improved(current_value, best_value, resolved_monitor_mode, min_delta)
                if improved:
                    best_value = current_value
                    stale_epochs = 0
                    if run_dir is not None and save_best:
                        best_value = current_value
                        self.save(run_dir / "best_model.pt", optimizer=optimizer_obj)
                else:
                    stale_epochs += 1

                if early_stopping and not improved and stale_epochs >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}; {monitor} did not improve for {patience} epoch(s).")
                    break

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        self.performance = {"elapsed_seconds": elapsed, "training_samples": processed_samples,
                            "training_samples_per_second": processed_samples / elapsed,
                            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
                            "mps_allocated_memory_bytes": torch.mps.current_allocated_memory() if self.device.type == "mps" else None}
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.performance["process_peak_rss_bytes"] = rss if platform.system() == "Darwin" else rss * 1024
        except ImportError:
            self.performance["process_peak_rss_bytes"] = None
        if save_to is not None:
            self.save(save_to, optimizer=optimizer_obj)

        if run_dir is not None:
            self.save(run_dir / "final_model.pt", optimizer=optimizer_obj)
            self._write_json(
                run_dir / "summary.json",
                {
                    "config": self._config(),
                    "final_metrics": {key: values[-1] for key, values in self.history.items() if values},
                    "best_monitor": monitor,
                    "best_value": best_value,
                    "performance": self.performance,
                    "metric_summary": self.training_metric_summary,
                },
            )

        return self.history

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

    def summary(self, input_size=None) -> Dict[str, object]:
        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_params = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        info = {
            "architecture": self.architecture,
            "loss": self.loss_name,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

        if hasattr(self.model, "get_architecture_info"):
            info["architecture_info"] = self.model.get_architecture_info()

        if input_size is not None:
            if len(input_size) == 2:
                shape = (1, self.in_channels, input_size[0], input_size[1])
            elif len(input_size) == 3:
                shape = (1, input_size[0], input_size[1], input_size[2])
            else:
                shape = tuple(input_size)
            self.model.eval()
            with torch.no_grad():
                dummy = torch.zeros(shape, device=self.device)
                output = self.model(dummy)
            info["input_shape"] = tuple(shape)
            info["output_shape"] = tuple(output.shape)

        return info

    def load_weights(self, path: Union[str, os.PathLike]):
        checkpoint = read_checkpoint(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.history = checkpoint.get("history", self.history)
        return self

    def create_report(self, run_dir, save_to=None, evaluation_dirs=None, prediction_dirs=None):
        """Write a report using explicit artifact locations and their saved provenance."""
        run_path = Path(run_dir)
        report_path = Path(save_to) if save_to is not None else run_path / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        history_plot = run_path / "history.png"
        if self.history and not history_plot.exists():
            self.plot_history(save_to=history_plot, show=False)
        lines = ["# Segmentation run report", ""]

        def link(path):
            return Path(os.path.relpath(path, report_path.parent)).as_posix().replace(" ", "%20")

        for title, filename in [
            ("Model and training settings", "config.json"),
            ("Training summary", "summary.json"),
        ]:
            path = run_path / filename
            if path.exists():
                lines.extend([f"## {title}", "", "```json", path.read_text().strip(), "```", ""])
        if history_plot.exists():
            lines.extend([f"![Training history]({link(history_plot)})", ""])
        example_dirs = list(prediction_dirs or [run_path / "predictions"])
        for directory in evaluation_dirs or [run_path / "evaluation"]:
            path = Path(directory) / "evaluation_summary.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            provenance = result.get("provenance", {})
            lines.extend(
                [
                    "## Evaluation",
                    "",
                    f"Source: [{path.parent.name}]({link(path)})",
                    "",
                    f"Checkpoint: {provenance.get('checkpoint') or 'current in-memory model'}",
                    f"Completed epoch: {provenance.get('completed_epoch', 'unknown')}",
                    f"Dataset split: {provenance.get('split') or 'custom'}",
                    f"Threshold: {provenance.get('threshold')}",
                    f"Aggregation: {result.get('aggregation', 'per_image')}",
                    f"Empty policy: {result.get('empty_policy', 'unknown')}",
                    f"Valid samples: {result.get('num_valid_samples', result.get('num_samples'))}",
                    f"Ignored-only samples: {result.get('ignored_only_samples', 0)}",
                    "",
                    "| Metric | Score |",
                    "| --- | --- |",
                ]
            )
            for metric, value in result.get("scores", result.get("mean", {})).items():
                lines.append(f"| {metric} | {value if value is not None else 'undefined (excluded)'} |")
            lines.extend(
                [
                    "",
                    "Class support:",
                    "",
                    "```json",
                    json.dumps(result.get("class_support", {}), indent=2),
                    "```",
                    "",
                ]
            )
            if provenance.get("prediction_dir"):
                example_dirs.append(provenance["prediction_dir"])
        for directory in dict.fromkeys(map(str, example_dirs)):
            root = Path(directory)
            manifest = root / "predictions.json"
            if manifest.exists():
                records = json.loads(manifest.read_text()).get("predictions", [])
                if records:
                    lines.extend(
                        [
                            "## Prediction provenance",
                            "",
                            "```json",
                            json.dumps(records[0]["metadata"], indent=2),
                            "```",
                            "",
                        ]
                    )
            overlays = sorted((root / "overlays").glob("*.png"))[:6]
            if overlays:
                lines.extend(["## Example predictions", ""])
                for overlay in overlays:
                    lines.extend([f"![{overlay.stem}]({link(overlay)})", ""])
        report_path.write_text("\n".join(lines))
        return str(report_path.resolve())

    @classmethod
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
        elif architecture in {"deit", "swin_unet", "swin_unet_full", "pvt_unet", "pvtformer_full", "double_unet"}:
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

    def _load_inference_images(self, images) -> List[str]:
        return self._load_paths_static(images)

    @staticmethod
    def _load_paths_static(paths_or_dir) -> List[str]:
        if isinstance(paths_or_dir, (list, tuple)):
            return [str(path) for path in paths_or_dir]

        image_path = Path(paths_or_dir)
        if image_path.is_dir():
            extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            return [str(path) for path in sorted(image_path.iterdir()) if path.is_file() and path.suffix.lower() in extensions]

        if image_path.is_file():
            return [str(image_path)]

        raise InferenceError(f"Could not find image source: {paths_or_dir}")

    def _predict_array(self, rgb: np.ndarray, threshold: float = 0.5, return_raw: bool = False):
        self._validate_prediction_request([], 1, threshold)
        prediction, probability, logits = self._predict_batch([rgb], threshold)[0]
        return (prediction, probability, logits) if return_raw else prediction

    def _predict_single_image(self, image_path: Union[str, os.PathLike], threshold: float = 0.5, return_raw=False):
        rgb = read_rgb(image_path)
        prediction = self._predict_array(rgb, threshold=threshold, return_raw=return_raw)
        if return_raw:
            pred, probability, logits = prediction
            return rgb, pred, probability, logits
        return rgb, prediction

    @staticmethod
    def _save_contours(prediction: np.ndarray, path: Path):
        mask = prediction
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.max() <= 1:
            mask = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        canvas = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(canvas, contours, -1, 255, 1)
        cv2.imwrite(str(path), canvas)

    def _prediction_metadata(self, threshold):
        return {
            "architecture": self.architecture,
            "checkpoint": self.checkpoint_path,
            "completed_epoch": self.completed_epoch,
            "dataset_identity": self.dataset_identity,
            "threshold": threshold if self.num_classes == 1 else None,
            "num_classes": self.num_classes,
            "class_map": self.class_map,
            "normalization": self.normalization,
            "image_size": list(self.image_size),
        }

    def _prediction_record(
        self, source, paths, threshold, prediction, probability=None, logits=None, arrays=False
    ):
        return {
            "source": str(Path(source).resolve()),
            "shape": list(prediction.shape),
            "paths": paths,
            "metadata": self._prediction_metadata(threshold),
            "mask": prediction if arrays else None,
            "probability": probability if arrays else None,
            "logits": logits if arrays else None,
        }

    @staticmethod
    def _save_prediction_manifest(path, records):
        Segmenter._write_json(
            path,
            {
                "format_version": 1,
                "predictions": [
                    {k: v for k, v in r.items() if k not in ("mask", "probability", "logits")}
                    for r in records
                ],
            },
        )

    @staticmethod
    def _validate_prediction_request(paths, batch_size, threshold):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must lie in [0,1].")
        stems = [Path(path).stem for path in paths]
        if len(stems) != len(set(stems)):
            raise InferenceError("Duplicate image stems would overwrite predictions.")

    def _predict_batch(self, images, threshold):
        tensors = torch.stack(
            [image_tensor(rgb, self.image_size, self.in_channels, self.normalization) for rgb in images]
        ).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self._forward_logits(tensors)
            results = []
            for index, rgb in enumerate(images):
                logits = torch.nn.functional.interpolate(
                    output[index : index + 1], size=rgb.shape[:2], mode="bilinear", align_corners=False
                )
                if self.num_classes == 1:
                    probability = logits.sigmoid()[0, 0].cpu().numpy()
                    prediction = (probability > threshold).astype(np.uint8) * 255
                    scores = logits[0, 0].cpu().numpy()
                else:
                    probability = logits.softmax(1)[0].cpu().numpy()
                    prediction = probability.argmax(0).astype(
                        np.uint8 if self.num_classes <= 256 else np.uint16
                    )
                    scores = logits[0].cpu().numpy()
                results.append((prediction, probability, scores))
            return results

    def _save_prediction_files(
        self,
        output_dir,
        source,
        rgb,
        prediction,
        probability=None,
        logits=None,
        save_overlay=False,
        save_probability=False,
        save_logits=False,
        save_contours=False,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(source).stem
        mask_path = output_dir / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), prediction)
        paths = {"mask": str(mask_path.resolve())}
        if save_overlay:
            directory = output_dir / "overlays"
            directory.mkdir(exist_ok=True)
            path = directory / f"{stem}_overlay.png"
            overlay = self._overlay_mask_on_image(rgb, prediction)
            cv2.imwrite(str(path), cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            paths["overlay"] = str(path.resolve())
        for enabled, kind, array in [
            (save_probability, "probability", probability),
            (save_logits, "logits", logits),
        ]:
            if enabled:
                directory = output_dir / ("probabilities" if kind == "probability" else "logits")
                directory.mkdir(exist_ok=True)
                path = directory / f"{stem}_{kind}.npy"
                np.save(path, array)
                paths[kind] = str(path.resolve())
        if save_contours:
            directory = output_dir / "contours"
            directory.mkdir(exist_ok=True)
            path = directory / f"{stem}_contours.png"
            self._save_contours(prediction, path)
            paths["contours"] = str(path.resolve())
        return paths

    def inference(
        self,
        images,
        save_to="predictions",
        threshold=0.5,
        return_arrays=False,
        save_overlay=False,
        save_probability=False,
        save_logits=False,
        save_contours=False,
        batch_size=1,
        structured=False,
    ):
        image_paths = self._load_inference_images(images)
        self._validate_prediction_request(image_paths, batch_size, threshold)
        records = []
        for start in progress_iter(range(0, len(image_paths), batch_size), desc="Inference"):
            sources = image_paths[start : start + batch_size]
            rgbs = [read_rgb(path) for path in sources]
            for source, rgb, (prediction, probability, logits) in zip(
                sources, rgbs, self._predict_batch(rgbs, threshold)
            ):
                paths = self._save_prediction_files(
                    save_to,
                    source,
                    rgb,
                    prediction,
                    probability,
                    logits,
                    save_overlay,
                    save_probability,
                    save_logits,
                    save_contours,
                )
                records.append(
                    self._prediction_record(
                        source, paths, threshold, prediction, probability, logits, return_arrays
                    )
                )
        self._save_prediction_manifest(Path(save_to) / "predictions.json", records)
        self.last_predictions = records
        return (
            records if structured else [r["mask"] if return_arrays else r["paths"]["mask"] for r in records]
        )

    def predict_one(
        self,
        image: Union[str, os.PathLike],
        threshold: float = 0.5,
        save_to: Optional[Union[str, os.PathLike]] = None,
        return_overlay: bool = False,
        show: bool = False,
        structured: bool = False,
    ):
        self.model.eval()
        with torch.no_grad():
            rgb, pred, probability, logits = self._predict_single_image(
                image, threshold=threshold, return_raw=True
            )

        paths = {}
        overlay = self._overlay_mask_on_image(rgb, pred)
        if save_to is not None:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), pred)
            overlay_path = save_path.with_name(f"{save_path.stem}_overlay.png")
            cv2.imwrite(str(overlay_path), cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            paths = {"mask": str(save_path.resolve()), "overlay": str(overlay_path.resolve())}
        record = self._prediction_record(image, paths, threshold, pred, probability, logits, True)
        if save_to is not None:
            self._save_prediction_manifest(save_path.with_suffix(".json"), [record])
        self.last_predictions = [record]

        if show:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(rgb)
            axes[0].set_title("Image")
            axes[1].imshow(pred, cmap="gray", interpolation="nearest")
            axes[1].set_title("Prediction")
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            plt.show()

        if structured:
            return record
        if return_overlay:
            return pred, overlay
        return pred

    @staticmethod
    def ensemble_predict(
        checkpoints,
        images,
        save_to="ensemble_predictions",
        threshold=0.5,
        device=None,
        save_overlay=True,
        batch_size=1,
        structured=False,
        return_arrays=False,
        save_probability=False,
    ):
        if not checkpoints:
            raise InferenceError("At least one checkpoint is required for ensemble prediction.")
        models = [Segmenter.load(path, device=device) for path in checkpoints]
        reference = models[0]
        if any(m.num_classes != reference.num_classes or m.class_map != reference.class_map for m in models):
            raise InferenceError("Ensemble checkpoints must have the same class count and class map.")
        image_paths = reference._load_inference_images(images)
        reference._validate_prediction_request(image_paths, batch_size, threshold)
        records = []
        for start in progress_iter(range(0, len(image_paths), batch_size), desc="Ensemble inference"):
            sources = image_paths[start : start + batch_size]
            rgbs = [read_rgb(path) for path in sources]
            batches = [model._predict_batch(rgbs, threshold) for model in models]
            for index, (source, rgb) in enumerate(zip(sources, rgbs)):
                probability = np.mean([batch[index][1] for batch in batches], axis=0)
                prediction = (
                    (probability > threshold).astype(np.uint8) * 255
                    if reference.num_classes == 1
                    else probability.argmax(0).astype(np.uint8 if reference.num_classes <= 256 else np.uint16)
                )
                paths = reference._save_prediction_files(
                    save_to,
                    source,
                    rgb,
                    prediction,
                    probability,
                    save_overlay=save_overlay,
                    save_probability=save_probability,
                )
                record = reference._prediction_record(
                    source, paths, threshold, prediction, probability, arrays=return_arrays
                )
                record["metadata"]["checkpoint"] = None
                record["metadata"]["architecture"] = "ensemble"
                record["metadata"]["members"] = [model._prediction_metadata(threshold) for model in models]
                record["metadata"]["checkpoints"] = [str(Path(p).resolve()) for p in checkpoints]
                records.append(record)
        reference._save_prediction_manifest(Path(save_to) / "predictions.json", records)
        return (
            records if structured else [r["mask"] if return_arrays else r["paths"]["mask"] for r in records]
        )

    def inference_large_image(
        self,
        image: Union[str, os.PathLike],
        save_to: Union[str, os.PathLike] = "predictions",
        patch_size=(512, 512),
        overlap: int = 64,
        threshold: float = 0.5,
        save_overlay: bool = True,
        save_probability: bool = False,
        tile_batch_size: int = 1,
        weighting: str = "uniform",
        gaussian_sigma: float = 0.125,
        structured: bool = False,
        return_arrays: bool = False,
    ) -> Dict[str, str]:
        if isinstance(tile_batch_size, bool) or not isinstance(tile_batch_size, int) or tile_batch_size < 1:
            raise ValueError("tile_batch_size must be a positive integer.")
        if weighting not in {"uniform", "gaussian"}:
            raise ValueError("weighting must be 'uniform' or 'gaussian'.")
        if not np.isfinite(gaussian_sigma) or gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be finite and positive.")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must lie in [0, 1].")
        self.model.eval()
        image_path = Path(image)
        rgb = read_rgb(image_path)
        height, width = rgb.shape[:2]
        if len(patch_size) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in patch_size):
            raise ValueError("patch_size must contain two positive integers.")
        if isinstance(overlap, bool) or not isinstance(overlap, int):
            raise ValueError("overlap must be an integer.")
        patch_h, patch_w = patch_size
        stride_h = patch_h - overlap
        stride_w = patch_w - overlap
        if overlap < 0 or patch_h <= 0 or patch_w <= 0 or stride_h <= 0 or stride_w <= 0:
            raise ValueError("overlap must be smaller than both patch dimensions.")

        if self.num_classes == 1:
            probability_acc = np.zeros((height, width), dtype=np.float32)
        else:
            probability_acc = np.zeros((self.num_classes, height, width), dtype=np.float32)
        count_acc = np.zeros((height, width), dtype=np.float32)

        y_starts = list(range(0, max(1, height - patch_h + 1), stride_h))
        x_starts = list(range(0, max(1, width - patch_w + 1), stride_w))
        if y_starts[-1] != max(0, height - patch_h):
            y_starts.append(max(0, height - patch_h))
        if x_starts[-1] != max(0, width - patch_w):
            x_starts.append(max(0, width - patch_w))

        # Normalize once over the source image, including percentile policies, so
        # overlap pixels have identical intensities in every tile.
        normalized = rgb if self.normalization["mode"] == "standard" else normalize_image(rgb, self.normalization)
        weights = np.ones((patch_h, patch_w), dtype=np.float32)
        if weighting == "gaussian":
            yy = (np.arange(patch_h) - (patch_h - 1) / 2) / (patch_h * gaussian_sigma)
            xx = (np.arange(patch_w) - (patch_w - 1) / 2) / (patch_w * gaussian_sigma)
            weights = np.maximum(np.exp(-0.5 * (yy[:, None] ** 2 + xx[None, :] ** 2)), 1e-6).astype(np.float32)
        with torch.no_grad():
            tiles = [(y, x) for y in y_starts for x in x_starts]
            for start in progress_iter(range(0, len(tiles), tile_batch_size), desc="Tiled inference"):
                positions = tiles[start:start + tile_batch_size]
                tensors = []
                for y, x in positions:
                    patch = normalized[y:y+patch_h, x:x+patch_w]
                    pad_h, pad_w = patch_h - patch.shape[0], patch_w - patch.shape[1]
                    patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                    tensors.append(image_tensor(patch, self.image_size, self.in_channels))
                logits = self._forward_logits(torch.stack(tensors).to(self.device))
                logits = torch.nn.functional.interpolate(logits, size=(patch_h, patch_w), mode="bilinear", align_corners=False)
                probabilities = logits.sigmoid()[:, 0] if self.num_classes == 1 else logits.softmax(1)
                for (y, x), probability in zip(positions, probabilities.float().cpu().numpy()):
                    h, w = min(patch_h, height-y), min(patch_w, width-x)
                    probability_acc[..., y:y+h, x:x+w] += probability[..., :h, :w] * weights[:h, :w]
                    count_acc[y:y+h, x:x+w] += weights[:h, :w]

        if self.num_classes == 1:
            probability = probability_acc / count_acc
            prediction = (probability > threshold).astype(np.uint8) * 255
        else:
            probability = probability_acc / count_acc[None, :, :]
            prediction = np.argmax(probability, axis=0).astype(np.uint8 if self.num_classes <= 256 else np.uint16)

        output_dir = Path(save_to)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), prediction)

        outputs = {"mask": str(mask_path.resolve())}
        if save_probability:
            probability_path = output_dir / f"{image_path.stem}_probability.npy"
            np.save(probability_path, probability)
            outputs["probability"] = str(probability_path.resolve())
        if save_overlay:
            overlay = self._overlay_mask_on_image(rgb, prediction)
            overlay_path = output_dir / f"{image_path.stem}_overlay.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            outputs["overlay"] = str(overlay_path.resolve())

        record = self._prediction_record(
            image, outputs, threshold, prediction, probability, arrays=return_arrays
        )
        record["metadata"]["tiling"] = {
            "patch_size": list(patch_size),
            "overlap": overlap,
            "weighting": weighting,
            "gaussian_sigma": gaussian_sigma,
            "tile_batch_size": tile_batch_size,
        }
        self._save_prediction_manifest(output_dir / f"{image_path.stem}_prediction.json", [record])
        self.last_predictions = [record]
        return record if structured else outputs

    @staticmethod
    def _match_prediction_mask_pairs(predictions, masks):
        prediction_paths = [Path(path) for path in Segmenter._load_paths_static(predictions)]
        mask_paths = [Path(path) for path in Segmenter._load_paths_static(masks)]
        masks_by_stem = {}
        for path in mask_paths:
            if path.stem in masks_by_stem:
                raise InferenceError(f"Duplicate ground-truth mask stem: {path.stem}")
            masks_by_stem[path.stem] = path
        pairs = []
        for path in prediction_paths:
            # strip only the suffix written by inference, never internal '_mask'.
            stem = path.stem.removesuffix("_mask")
            mask = masks_by_stem.get(stem, masks_by_stem.get(path.stem))
            if mask is None:
                raise InferenceError(f"No ground-truth mask matches prediction: {path.name}")
            pairs.append((path, mask))
        if not pairs:
            raise InferenceError("No prediction/mask pairs to evaluate.")
        return pairs

    @staticmethod
    def _score_prediction_arrays(
        pred_np,
        mask_np,
        metrics,
        num_classes=None,
        class_names=None,
        include_background=False,
        ignore_index=None,
        empty_policy="exclude",
    ):
        if pred_np.shape != mask_np.shape:
            pred_np = cv2.resize(
                pred_np, (mask_np.shape[1], mask_np.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        if num_classes is None:
            values = set(np.unique(pred_np)) | set(np.unique(mask_np))
            values.discard(ignore_index)
            num_classes = 1 if values.issubset({0, 1, 255}) else int(max(values, default=0)) + 1
        accumulator = MetricAccumulator(
            num_classes, metrics, include_background, ignore_index, empty_policy, class_names
        )
        return accumulator.add(pred_np, mask_np)

    @staticmethod
    def evaluate_predictions(
        predictions,
        masks,
        metrics=("dice", "iou", "precision", "recall"),
        save_to=None,
        num_classes=None,
        class_names=None,
        include_background=False,
        ignore_index=None,
        empty_policy="exclude",
        aggregation="per_image",
    ):
        pairs = Segmenter._match_prediction_mask_pairs(predictions, masks)
        # Infer once for the complete dataset, so class absence cannot change task type per image.
        if num_classes is None:
            values = set()
            for pair in pairs:
                for path in pair:
                    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if array is None:
                        raise InferenceError(f"Could not read {path}")
                    values.update(np.unique(array).tolist())
            values.discard(ignore_index)
            num_classes = 1 if values.issubset({0, 1, 255}) else int(max(values, default=0)) + 1
        accumulator = MetricAccumulator(
            num_classes, metrics, include_background, ignore_index, empty_policy, class_names, aggregation
        )
        rows = []
        for prediction_path, mask_path in pairs:
            prediction = cv2.imread(str(prediction_path), cv2.IMREAD_UNCHANGED)
            target = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if prediction is None or target is None:
                raise InferenceError(f"Could not read {prediction_path} or {mask_path}")
            if prediction.shape != target.shape:
                prediction = cv2.resize(
                    prediction, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            scores = accumulator.add(prediction, target)
            valid = target != ignore_index if ignore_index is not None else np.ones(target.shape, dtype=bool)
            rows.append(
                {
                    "prediction": Path(prediction_path).name,
                    "mask": Path(mask_path).name,
                    "valid_pixels": int(valid.sum()),
                    "ignored_only": not bool(valid.any()),
                    **scores,
                }
            )
        summary = {
            **accumulator.summary(),
            "rows": rows,
            "num_classes": num_classes,
            "include_background": include_background,
            "ignore_index": ignore_index,
            "class_names": class_names,
        }
        if save_to is not None:
            output_dir = Path(save_to)
            output_dir.mkdir(parents=True, exist_ok=True)
            if rows:
                with (output_dir / "evaluation.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
                    writer.writeheader()
                    writer.writerows(rows)
            Segmenter._write_json(output_dir / "evaluation_summary.json", summary)
        return summary

    def find_best_threshold(
        self,
        images,
        masks,
        thresholds: Optional[Sequence[float]] = None,
        metric: str = "dice",
        save_to: Optional[Union[str, os.PathLike]] = None,
    ) -> Dict[str, object]:
        if self.num_classes != 1:
            raise ValueError("Threshold tuning is only supported for binary segmentation.")

        thresholds = list(np.linspace(0.1, 0.9, 17) if thresholds is None else thresholds)
        if not thresholds:
            raise ValueError("Provide at least one threshold.")
        image_paths = self._load_inference_images(images)
        mask_pairs = self._match_prediction_mask_pairs(image_paths, masks)
        mask_by_stem = {Path(mask).stem.replace("_mask", ""): mask for _, mask in mask_pairs}
        rows = []

        self.model.eval()
        with torch.no_grad():
            probabilities = []
            ground_truths = []
            for image_path in progress_iter(image_paths, desc="Threshold tuning"):
                stem = Path(image_path).stem.replace("_mask", "")
                if stem not in mask_by_stem:
                    continue
                _, _, probability, _ = self._predict_single_image(image_path, return_raw=True)
                mask = cv2.imread(str(mask_by_stem[stem]), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    continue
                if probability.shape != mask.shape:
                    probability = cv2.resize(probability, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC)
                probabilities.append(probability)
                ground_truths.append(mask)

        if not probabilities:
            raise ValueError("No matched image/mask pairs found for threshold tuning.")

        best_threshold = None
        best_score = None
        metric_key = metric.lower()
        if metric_key not in {"dice", "iou", "jaccard"}:
            raise ValueError("metric must be dice, iou, or jaccard.")
        for threshold in thresholds:
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("thresholds must lie in [0, 1].")
            accumulator = MetricAccumulator(
                1,
                (metric_key,),
                ignore_index=self.ignore_index,
                empty_policy=self.empty_policy,
                aggregation=self.metric_aggregation,
            )
            for probability, target in zip(probabilities, ground_truths):
                accumulator.add((probability > threshold).astype(np.uint8), target)
            mean_score = accumulator.summary()["scores"][metric_key]
            rows.append({"threshold": float(threshold), metric_key: mean_score})
            if mean_score is not None and (best_score is None or mean_score > best_score):
                best_score = mean_score
                best_threshold = float(threshold)

        result = {
            "best_threshold": best_threshold,
            "best_score": best_score,
            "metric": metric_key,
            "rows": rows,
        }

        if save_to is not None:
            output_dir = Path(save_to)
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "threshold_tuning.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["threshold", metric_key])
                writer.writeheader()
                writer.writerows(rows)
            self._write_json(output_dir / "threshold_tuning.json", result)

        return result

    def evaluate(
        self,
        images,
        masks,
        save_to: Union[str, os.PathLike] = "evaluation",
        prediction_dir: Optional[Union[str, os.PathLike]] = None,
        metrics: Sequence[str] = ("dice", "iou", "precision", "recall"),
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        include_background: Optional[bool] = None,
        empty_policy: Optional[str] = None,
        aggregation: Optional[str] = None,
        split: Optional[str] = None,
        batch_size: int = 1,
    ) -> Dict[str, object]:
        prediction_dir = prediction_dir or Path(save_to) / "predictions"
        predictions = self.inference(
            images=images,
            save_to=prediction_dir,
            threshold=threshold,
            batch_size=batch_size,
            save_overlay=True,
        )
        result = self.evaluate_predictions(
            predictions,
            masks,
            metrics=metrics,
            save_to=save_to,
            num_classes=self.num_classes if num_classes is None else num_classes,
            class_names=class_names or [self.class_map[i] for i in sorted(self.class_map)],
            include_background=self.include_background if include_background is None else include_background,
            empty_policy=empty_policy or self.empty_policy,
            aggregation=aggregation or self.metric_aggregation,
            ignore_index=self.ignore_index,
        )

        result["provenance"] = {
            **self._prediction_metadata(threshold),
            "split": split,
            "images": str(images),
            "masks": str(masks),
            "prediction_dir": str(Path(prediction_dir).resolve()),
        }
        self._write_json(Path(save_to) / "evaluation_summary.json", result)
        return result

    @staticmethod
    def find_worst_predictions(
        predictions,
        masks,
        metric: str = "dice",
        top_k: int = 10,
        save_to: Optional[Union[str, os.PathLike]] = None,
        images=None,
        num_classes: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        include_background: bool = False,
        ignore_index: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        evaluation = Segmenter.evaluate_predictions(
            predictions=predictions,
            masks=masks,
            metrics=(metric,),
            num_classes=num_classes,
            class_names=class_names,
            include_background=include_background,
            ignore_index=ignore_index,
        )
        rows = evaluation["rows"]
        metric_key = metric.lower()
        if rows and metric_key not in rows[0]:
            metric_candidates = [key for key in rows[0] if key == f"macro_{metric_key}" or key.startswith(f"{metric_key}_")]
            metric_key = f"macro_{metric_key}" if f"macro_{metric_key}" in metric_candidates else metric_candidates[0]

        worst = sorted(
            (row for row in rows if row.get(metric_key) is not None), key=lambda row: row[metric_key]
        )[:top_k]

        if save_to is not None:
            output_dir = Path(save_to)
            output_dir.mkdir(parents=True, exist_ok=True)
            if worst:
                with (output_dir / "worst_predictions.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=sorted({key for row in worst for key in row}))
                    writer.writeheader()
                    writer.writerows(worst)

            prediction_pairs = Segmenter._match_prediction_mask_pairs(predictions, masks)
            pred_by_name = {Path(pred).name: Path(pred) for pred, _ in prediction_pairs}
            mask_by_name = {Path(mask).name: Path(mask) for _, mask in prediction_pairs}
            image_by_stem = {}
            if images is not None:
                image_paths = Segmenter._load_paths_static(images)
                image_by_stem = {Path(path).stem: Path(path) for path in image_paths}

            for rank, row in enumerate(worst, start=1):
                case_dir = output_dir / f"rank_{rank}_{Path(row['prediction']).stem}"
                case_dir.mkdir(parents=True, exist_ok=True)
                if row["prediction"] in pred_by_name:
                    shutil.copy2(pred_by_name[row["prediction"]], case_dir / row["prediction"])
                if row["mask"] in mask_by_name:
                    shutil.copy2(mask_by_name[row["mask"]], case_dir / row["mask"])
                stem = Path(row["prediction"]).stem.replace("_mask", "")
                if stem in image_by_stem:
                    shutil.copy2(image_by_stem[stem], case_dir / image_by_stem[stem].name)

        return worst

    def plot_inference_results(
        self,
        images,
        predictions=None,
        num_samples: int = 4,
        threshold: float = 0.5,
        overlay: bool = True,
        save_to: Optional[Union[str, os.PathLike]] = None,
        show: bool = True,
        figsize=None,
    ):
        import matplotlib.pyplot as plt

        image_paths = self._load_inference_images(images)
        image_paths = image_paths[:num_samples]
        if not image_paths:
            raise ValueError("No inference images found to plot.")

        if predictions is not None:
            if isinstance(predictions, (str, os.PathLike)):
                prediction_paths = self._load_inference_images(predictions)
                prediction_items = prediction_paths[: len(image_paths)]
            else:
                prediction_items = list(predictions)[: len(image_paths)]
        else:
            prediction_items = [None] * len(image_paths)

        self.model.eval()
        num_cols = 3 if overlay else 2
        figsize = figsize or (4 * num_cols, 3 * max(1, len(image_paths)))
        fig, axes = plt.subplots(len(image_paths), num_cols, figsize=figsize, squeeze=False)

        with torch.no_grad():
            for idx, (image_path, prediction_item) in enumerate(zip(image_paths, prediction_items)):
                if prediction_item is None:
                    rgb, pred = self._predict_single_image(image_path, threshold=threshold)
                else:
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"Could not read image: {image_path}")
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if isinstance(prediction_item, np.ndarray):
                        pred = prediction_item
                    else:
                        pred = cv2.imread(str(prediction_item), cv2.IMREAD_UNCHANGED)
                        if pred is None:
                            raise ValueError(f"Could not read prediction: {prediction_item}")
                    if pred.shape[:2] != rgb.shape[:2]:
                        pred = cv2.resize(pred, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

                axes[idx, 0].imshow(rgb)
                axes[idx, 0].set_title("Image")
                axes[idx, 0].axis("off")

                axes[idx, 1].imshow(pred, cmap="gray", interpolation="nearest")
                axes[idx, 1].set_title("Prediction")
                axes[idx, 1].axis("off")

                if overlay:
                    axes[idx, 2].imshow(self._overlay_mask_on_image(rgb, pred))
                    axes[idx, 2].set_title("Overlay")
                    axes[idx, 2].axis("off")

        fig.tight_layout()
        if save_to is not None:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        return fig

    @staticmethod
    def run_experiment(
        dataset,
        architectures: Optional[Sequence[str]] = None,
        losses: Optional[Sequence[str]] = None,
        epochs: int = 1,
        batch_size: int = 4,
        image_size=(224, 224),
        metrics: Optional[Sequence[str]] = None,
        num_classes: int = 1,
        in_channels: int = 3,
        model_kwargs: Optional[dict] = None,
        loss_kwargs: Optional[dict] = None,
        output_dir: Optional[Union[str, os.PathLike]] = "experiments",
        runs: Optional[Dict[str, ExperimentRunConfig]] = None,
        seed: int = 42,
        device: Optional[str] = None,
        **train_kwargs,
    ) -> Dict[str, Dict[str, List[float]]]:
        from .utils.environment import set_seed, environment_info

        if runs is not None and (architectures is not None or losses is not None):
            raise ValueError("Use named runs or architectures/losses, not both.")
        if runs is None:
            if not architectures or not losses:
                raise ValueError("Provide named runs or nonempty architectures and losses.")
            runs = {}
            for architecture in architectures:
                for loss in losses:
                    name = f"{architecture}_{loss}"
                    if name in runs:
                        raise ValueError(f"Duplicate experiment name: {name}.")
                    runs[name] = ExperimentRunConfig(
                        segmenter=SegmenterConfig(architecture=architecture, loss=loss, metrics=metrics,
                            image_size=image_size, num_classes=num_classes, in_channels=in_channels,
                            model_kwargs=model_kwargs or {}, loss_kwargs=loss_kwargs or {}, device=device),
                        training=TrainingConfig(epochs=epochs, batch_size=batch_size), seed=seed)
        if not runs:
            raise ValueError("runs cannot be empty.")
        prepared = {}
        for name, run in runs.items():
            if not isinstance(name, str) or not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Run names must be nonempty directory names without path separators.")
            prepared[name] = ExperimentRunConfig.from_dict(run) if isinstance(run, dict) else run
            if not isinstance(prepared[name], ExperimentRunConfig):
                raise TypeError("Each run must be an ExperimentRunConfig or its serialized dictionary.")
        results, summary_rows = {}, []
        for name, run in prepared.items():
            set_seed(run.seed)
            segmenter = Segmenter.from_config(run.segmenter)
            options = run.training.to_dict()
            options.update(train_kwargs)
            options.setdefault("experiment_dir", output_dir)
            if options["experiment_dir"] is None:
                options["experiment_dir"] = output_dir
            if options.get("run_name") is None:
                options["run_name"] = name
            results[name] = segmenter.train(data=dataset, **options)
            final_metrics = {key: values[-1] for key, values in results[name].items() if values}
            row = {"run": name, "architecture": segmenter.architecture, "loss": segmenter.loss_name,
                   "seed": run.seed, "device": str(segmenter.device), **segmenter.performance, **final_metrics}
            summary_rows.append(row)
            if segmenter.last_run_dir is not None:
                Segmenter._write_json(segmenter.last_run_dir / "experiment.json", {
                    "seed": run.seed, "segmenter": segmenter._config(), "training": segmenter.last_training_config,
                    "environment": environment_info(), "performance": segmenter.performance})
            del segmenter
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with (output_path / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summary_rows for key in row}))
                writer.writeheader()
                writer.writerows(summary_rows)
            Segmenter._write_json(output_path / "summary.json", {"runs": summary_rows})
        return results

    @staticmethod
    def compare_experiments(
        experiments_dir: Union[str, os.PathLike],
        metric: str = "val_loss",
        save_to: Optional[Union[str, os.PathLike]] = None,
        show: bool = True,
        figsize=(8, 5),
    ):
        import matplotlib.pyplot as plt

        experiments_path = Path(experiments_dir)
        rows = []
        summary_path = experiments_path / "summary.csv"
        if summary_path.exists():
            with summary_path.open() as handle:
                rows = list(csv.DictReader(handle))
        else:
            for history_path in experiments_path.glob("*/history.json"):
                history = Segmenter._load_history(history_path)
                if metric in history and history[metric]:
                    rows.append({"run": history_path.parent.name, metric: history[metric][-1]})

        if not rows:
            raise ValueError(f"No experiment data found in {experiments_dir}.")

        run_names = [row.get("run", f"run_{idx}") for idx, row in enumerate(rows)]
        values = [float(row[metric]) for row in rows if metric in row and row[metric] != ""]
        if len(values) != len(run_names):
            raise ValueError(f"Metric '{metric}' was not found for every experiment.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(run_names, values)
        ax.set_ylabel(metric)
        ax.set_title(f"Experiment Comparison: {metric}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        if save_to is not None:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        return fig
