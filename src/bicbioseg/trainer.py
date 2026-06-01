import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from .config import SegmenterConfig, TrainingConfig
from .exceptions import InferenceError, ModelError
from .models.attention_unet import AttUNet
from .models.doubleunet import DoubleUNet
from .models.segformer import Segformer
from .models.transunet import TransUNet
from .models.unet import UNet
from .utils import losses as loss_module
from .utils.load_data import DataLoader
from .utils.metrics import dice_score, jac_score, precision, recall
from .utils.progress import progress_iter


class Segmenter:
    MODEL_ALIASES = {
        "unet": "unet",
        "u-net": "unet",
        "attention_unet": "attention_unet",
        "attunet": "attention_unet",
        "attention-u-net": "attention_unet",
        "double_unet": "double_unet",
        "doubleunet": "double_unet",
        "transunet": "transunet",
        "trans_unet": "transunet",
        "segformer": "segformer",
    }

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
        "unified_focal": loss_module.UnifiedFocalLoss,
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
    ):
        self.architecture = self.MODEL_ALIASES.get(architecture.lower(), architecture.lower())
        self.loss_name = loss.lower() if isinstance(loss, str) else loss.__class__.__name__
        self.image_size = image_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.model_kwargs = dict(model_kwargs or {})
        self.loss_kwargs = dict(loss_kwargs or {})
        self.device = self.resolve_device(device or "auto")
        self.metrics = list(metrics or ["dice", "iou"])
        self.model = self._build_model(self.model_kwargs).to(self.device)
        self.loss_fn = self._build_loss(loss, self.loss_kwargs)
        self.history: Dict[str, List[float]] = {}

    @classmethod
    def available_models(cls) -> List[str]:
        return sorted(set(cls.MODEL_ALIASES.values()))

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
        if device in (None, "auto"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        if device == "cuda" and not torch.cuda.is_available():
            raise ModelError("CUDA was requested but is not available.")
        if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise ModelError("MPS was requested but is not available.")
        return torch.device(device)

    def _build_model(self, model_kwargs: dict) -> torch.nn.Module:
        if self.architecture == "unet":
            kwargs = {
                "n_channels": self.in_channels,
                "n_classes": self.num_classes,
                "image_size": self.image_size,
            }
            kwargs.update(model_kwargs)
            return UNet(**kwargs)

        if self.architecture == "attention_unet":
            kwargs = {"n_channels": self.in_channels, "n_classes": self.num_classes}
            kwargs.update(model_kwargs)
            return AttUNet(**kwargs)

        if self.architecture == "double_unet":
            kwargs = {"n_classes": self.num_classes, "pretrained": False}
            kwargs.update(model_kwargs)
            return DoubleUNet(**kwargs)

        if self.architecture == "transunet":
            img_dim = self.image_size[0] if isinstance(self.image_size, tuple) else self.image_size
            kwargs = {"img_dim": img_dim, "in_channels": self.in_channels, "n_classes": self.num_classes}
            kwargs.update(model_kwargs)
            return TransUNet(**kwargs)

        if self.architecture == "segformer":
            kwargs = {"channels": self.in_channels, "num_classes": self.num_classes}
            kwargs.update(model_kwargs)
            return Segformer(**kwargs)

        raise ModelError(f"Unknown architecture '{self.architecture}'. Available: {self.available_models()}")

    def _build_loss(self, loss, loss_kwargs: dict):
        if isinstance(loss, torch.nn.Module):
            return loss
        if isinstance(loss, type) and issubclass(loss, torch.nn.Module):
            return loss(**loss_kwargs)
        if callable(loss) and not isinstance(loss, str):
            return loss

        loss_key = loss.lower()
        if loss_key not in self.LOSS_ALIASES:
            raise ModelError(f"Unknown loss '{loss}'. Available: {self.available_losses()}")

        return self.LOSS_ALIASES[loss_key](**loss_kwargs)

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
            "device": str(self.device),
        }

    @staticmethod
    def _prepare_run_dir(experiment_dir, run_name: Optional[str]) -> Optional[Path]:
        if experiment_dir is None:
            return None

        if run_name is None:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        run_dir = Path(experiment_dir) / run_name
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
            image_size=self.image_size,
            batch_size=batch_size,
            transforms=transforms,
            num_workers=num_workers,
            shuffle=shuffle,
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
            validate_path = data_path / "validate"
            val_loader = None
            if validate_path.is_dir():
                val_loader = self._loader_from_data(
                    str(validate_path), batch_size, None, num_workers, shuffle=False
                )
            return train_loader, val_loader

        return self._loader_from_data(data, batch_size, transforms, num_workers, shuffle=True), None

    def _prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(self.device)
        if self.num_classes == 1:
            if targets.ndim == 3:
                targets = targets.unsqueeze(1)
            return (targets > 0).float()
        return targets.long()

    def _metric_values(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        values = {}
        if self.num_classes == 1:
            preds = (torch.sigmoid(outputs) > 0.5).float()
            target_values = (targets > 0).float()
        else:
            preds = torch.argmax(outputs, dim=1)
            target_values = targets

        for metric in self.metrics:
            metric_key = metric.lower()
            if metric_key == "dice":
                values["dice"] = float(dice_score(target_values, preds).detach().cpu())
            elif metric_key in {"iou", "jaccard"}:
                values["iou"] = float(jac_score(target_values, preds).detach().cpu())
        return values

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
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
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
        batches = 0

        for images, targets in loader:
            images = images.to(self.device)
            targets = self._prepare_targets(targets)

            if is_train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                if is_train:
                    loss.backward()
                    optimizer.step()

            metric_values = self._metric_values(outputs.detach(), targets.detach())
            totals["loss"] += float(loss.detach().cpu())
            for name, value in metric_values.items():
                totals[name] = totals.get(name, 0.0) + value
            batches += 1

        return {name: value / max(1, batches) for name, value in totals.items()}

    def _load_checkpoint_for_resume(self, path: Union[str, os.PathLike], optimizer=None):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.history = checkpoint.get("history", self.history)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def train(
        self,
        data=None,
        train_data=None,
        val_data=None,
        epochs: int = 1,
        batch_size: int = 4,
        lr: float = 1e-4,
        optimizer: Union[str, torch.optim.Optimizer] = "adam",
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

        train_loader, val_loader = self._resolve_loaders(
            data=data,
            train_data=train_data,
            val_data=val_data,
            batch_size=batch_size,
            transforms=transforms,
            num_workers=num_workers,
        )
        optimizer_obj = self._make_optimizer(optimizer, lr)
        if resume_from is not None:
            self._load_checkpoint_for_resume(resume_from, optimizer=optimizer_obj)

        if resume_from is None or not self.history:
            self.history = {"train_loss": []}
        else:
            self.history.setdefault("train_loss", [])
        run_dir = self._prepare_run_dir(experiment_dir, run_name)
        best_value = None
        stale_epochs = 0
        resolved_monitor_mode = self._monitor_mode(monitor, monitor_mode)

        if run_dir is not None:
            config = self._config()
            config["training"] = {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "optimizer": optimizer if isinstance(optimizer, str) else optimizer.__class__.__name__,
                "monitor": monitor,
                "early_stopping": early_stopping,
                "patience": patience,
                "min_delta": min_delta,
                "monitor_mode": resolved_monitor_mode,
            }
            self._write_json(run_dir / "config.json", config)

        epoch_iter = progress_iter(range(1, epochs + 1), enabled=progress_bar, desc="Training")
        for epoch in epoch_iter:
            train_stats = self._run_epoch(train_loader, optimizer=optimizer_obj)
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

            self._update_run_artifacts(run_dir)
            monitor_values = self.history.get(monitor)
            if monitor_values:
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

                if early_stopping and stale_epochs >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}; {monitor} did not improve for {patience} epoch(s).")
                    break

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
                },
            )

        return self.history

    def save(self, path: Union[str, os.PathLike], optimizer=None) -> str:
        checkpoint = {
            "architecture": self.architecture,
            "loss": self.loss_name,
            "metrics": self.metrics,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "model_kwargs": self.model_kwargs,
            "loss_kwargs": self.loss_kwargs,
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
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
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.history = checkpoint.get("history", self.history)
        return self

    def create_report(
        self,
        run_dir: Union[str, os.PathLike],
        save_to: Optional[Union[str, os.PathLike]] = None,
    ) -> str:
        run_path = Path(run_dir)
        report_path = Path(save_to) if save_to is not None else run_path / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        config_path = run_path / "config.json"
        summary_path = run_path / "summary.json"
        history_plot = run_path / "history.png"
        if self.history and not history_plot.exists():
            self.plot_history(save_to=history_plot, show=False)

        lines = ["# Segmentation Run Report", ""]
        if config_path.exists():
            lines.extend(["## Config", "", "```json", config_path.read_text().strip(), "```", ""])
        if summary_path.exists():
            lines.extend(["## Summary", "", "```json", summary_path.read_text().strip(), "```", ""])
        if history_plot.exists():
            lines.extend(["## Training Curves", "", f"![Training history]({history_plot.name})", ""])

        prediction_overlays = sorted((run_path / "predictions" / "overlays").glob("*.png"))
        if prediction_overlays:
            lines.extend(["## Prediction Overlays", ""])
            for overlay_path in prediction_overlays[:12]:
                lines.append(f"![{overlay_path.name}]({overlay_path.relative_to(run_path)})")
            lines.append("")

        evaluation_summary = run_path / "evaluation" / "evaluation_summary.json"
        if evaluation_summary.exists():
            lines.extend(["## Evaluation", "", "```json", evaluation_summary.read_text().strip(), "```", ""])

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
        checkpoint = torch.load(path, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
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
            "device": device,
        }
        config.update(overrides)

        segmenter = cls(**config)
        segmenter.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        segmenter.history = checkpoint.get("history", {})
        return segmenter

    def _load_inference_images(self, images) -> List[str]:
        return self._load_paths_static(images)

    @staticmethod
    def _load_paths_static(paths_or_dir) -> List[str]:
        if isinstance(paths_or_dir, (list, tuple)):
            return [str(path) for path in paths_or_dir]

        image_path = Path(paths_or_dir)
        if image_path.is_dir():
            paths = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
                paths.extend(image_path.glob(ext))
            return [str(path) for path in sorted(paths)]

        if image_path.is_file():
            return [str(image_path)]

        raise InferenceError(f"Could not find image source: {paths_or_dir}")

    def _predict_array(self, rgb: np.ndarray, threshold: float = 0.5, return_raw: bool = False):
        original_size = (rgb.shape[1], rgb.shape[0])
        resized = cv2.resize(rgb, self.image_size, interpolation=cv2.INTER_CUBIC)
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

        output = self.model(tensor)
        if self.num_classes == 1:
            logits = output[0, 0].detach().cpu().numpy()
            probability = torch.sigmoid(output)[0, 0].detach().cpu().numpy()
            pred = (probability > threshold).astype(np.uint8) * 255
        else:
            logits = output[0].detach().cpu().numpy()
            probability = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
            pred = np.argmax(probability, axis=0).astype(np.uint8)

        pred = cv2.resize(pred, original_size, interpolation=cv2.INTER_NEAREST)
        if self.num_classes == 1:
            probability = cv2.resize(probability, original_size, interpolation=cv2.INTER_CUBIC)
            logits = cv2.resize(logits, original_size, interpolation=cv2.INTER_CUBIC)
        else:
            probability = np.stack(
                [cv2.resize(channel, original_size, interpolation=cv2.INTER_CUBIC) for channel in probability],
                axis=0,
            )
            logits = np.stack(
                [cv2.resize(channel, original_size, interpolation=cv2.INTER_CUBIC) for channel in logits],
                axis=0,
            )

        if return_raw:
            return pred, probability, logits
        return pred

    def _predict_single_image(self, image_path: Union[str, os.PathLike], threshold: float = 0.5, return_raw=False):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise InferenceError(f"Could not read image: {image_path}")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

    def inference(
        self,
        images,
        save_to: Union[str, os.PathLike] = "predictions",
        threshold: float = 0.5,
        return_arrays: bool = False,
        save_overlay: bool = False,
        save_probability: bool = False,
        save_logits: bool = False,
        save_contours: bool = False,
    ):
        self.model.eval()
        output_dir = Path(save_to)
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir = output_dir / "overlays"
        if save_overlay:
            overlay_dir.mkdir(parents=True, exist_ok=True)
        if save_probability:
            (output_dir / "probabilities").mkdir(parents=True, exist_ok=True)
        if save_logits:
            (output_dir / "logits").mkdir(parents=True, exist_ok=True)
        if save_contours:
            (output_dir / "contours").mkdir(parents=True, exist_ok=True)
        image_paths = self._load_inference_images(images)
        predictions = []

        with torch.no_grad():
            for image_path in progress_iter(image_paths, desc="Inference"):
                rgb, pred, probability, logits = self._predict_single_image(
                    image_path,
                    threshold=threshold,
                    return_raw=True,
                )
                out_path = output_dir / f"{Path(image_path).stem}_mask.png"
                cv2.imwrite(str(out_path), pred)
                if save_overlay:
                    overlay = self._overlay_mask_on_image(rgb, pred)
                    overlay_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(overlay_dir / f"{Path(image_path).stem}_overlay.png"), overlay_bgr)
                if save_probability:
                    np.save(output_dir / "probabilities" / f"{Path(image_path).stem}_probability.npy", probability)
                if save_logits:
                    np.save(output_dir / "logits" / f"{Path(image_path).stem}_logits.npy", logits)
                if save_contours:
                    self._save_contours(pred, output_dir / "contours" / f"{Path(image_path).stem}_contours.png")
                predictions.append(pred if return_arrays else str(out_path.resolve()))

        return predictions

    def predict_one(
        self,
        image: Union[str, os.PathLike],
        threshold: float = 0.5,
        save_to: Optional[Union[str, os.PathLike]] = None,
        return_overlay: bool = False,
        show: bool = False,
    ):
        self.model.eval()
        with torch.no_grad():
            rgb, pred = self._predict_single_image(image, threshold=threshold)

        overlay = self._overlay_mask_on_image(rgb, pred)
        if save_to is not None:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), pred)
            overlay_path = save_path.with_name(f"{save_path.stem}_overlay.png")
            cv2.imwrite(str(overlay_path), cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

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

        if return_overlay:
            return pred, overlay
        return pred

    @staticmethod
    def ensemble_predict(
        checkpoints: Sequence[Union[str, os.PathLike]],
        images,
        save_to: Union[str, os.PathLike] = "ensemble_predictions",
        threshold: float = 0.5,
        device: Optional[str] = None,
        save_overlay: bool = True,
    ) -> List[str]:
        if not checkpoints:
            raise InferenceError("At least one checkpoint is required for ensemble prediction.")

        models = [Segmenter.load(checkpoint, device=device) for checkpoint in checkpoints]
        reference = models[0]
        image_paths = reference._load_inference_images(images)
        output_dir = Path(save_to)
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir = output_dir / "overlays"
        if save_overlay:
            overlay_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        with torch.no_grad():
            for image_path in progress_iter(image_paths, desc="Ensemble inference"):
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                probabilities = []
                for model in models:
                    _, probability, _ = model._predict_array(rgb, threshold=threshold, return_raw=True)
                    probabilities.append(probability)
                mean_probability = np.mean(probabilities, axis=0)

                if reference.num_classes == 1:
                    prediction = (mean_probability > threshold).astype(np.uint8) * 255
                else:
                    prediction = np.argmax(mean_probability, axis=0).astype(np.uint8)

                out_path = output_dir / f"{Path(image_path).stem}_mask.png"
                cv2.imwrite(str(out_path), prediction)
                if save_overlay:
                    overlay = reference._overlay_mask_on_image(rgb, prediction)
                    cv2.imwrite(
                        str(overlay_dir / f"{Path(image_path).stem}_overlay.png"),
                        cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    )
                saved_paths.append(str(out_path.resolve()))

        return saved_paths

    def inference_large_image(
        self,
        image: Union[str, os.PathLike],
        save_to: Union[str, os.PathLike] = "predictions",
        patch_size=(512, 512),
        overlap: int = 64,
        threshold: float = 0.5,
        save_overlay: bool = True,
        save_probability: bool = False,
    ) -> Dict[str, str]:
        self.model.eval()
        image_path = Path(image)
        raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f"Could not read image: {image}")

        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        patch_h, patch_w = patch_size
        stride_h = patch_h - overlap
        stride_w = patch_w - overlap
        if stride_h <= 0 or stride_w <= 0:
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

        with torch.no_grad():
            tiles = [(y, x) for y in y_starts for x in x_starts]
            for y, x in progress_iter(tiles, desc="Tiled inference"):
                patch = rgb[y : y + patch_h, x : x + patch_w]
                pad_h = patch_h - patch.shape[0]
                pad_w = patch_w - patch.shape[1]
                if pad_h or pad_w:
                    patch = cv2.copyMakeBorder(
                        patch,
                        0,
                        pad_h,
                        0,
                        pad_w,
                        borderType=cv2.BORDER_CONSTANT,
                        value=0,
                    )

                _, probability, _ = self._predict_array(patch, threshold=threshold, return_raw=True)
                if probability.ndim == 3:
                    probability = probability[..., : patch_h - pad_h, : patch_w - pad_w]
                else:
                    probability = probability[: patch_h - pad_h, : patch_w - pad_w]
                valid_h, valid_w = probability.shape[-2:]

                if self.num_classes == 1:
                    probability_acc[y : y + valid_h, x : x + valid_w] += probability
                else:
                    probability_acc[:, y : y + valid_h, x : x + valid_w] += probability
                count_acc[y : y + valid_h, x : x + valid_w] += 1

        count_acc = np.maximum(count_acc, 1)
        if self.num_classes == 1:
            probability = probability_acc / count_acc
            prediction = (probability > threshold).astype(np.uint8) * 255
        else:
            probability = probability_acc / count_acc[None, :, :]
            prediction = np.argmax(probability, axis=0).astype(np.uint8)

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

        return outputs

    @staticmethod
    def _match_prediction_mask_pairs(predictions, masks):
        prediction_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            if isinstance(predictions, (str, os.PathLike)) and Path(predictions).is_dir():
                prediction_paths.extend(Path(predictions).glob(ext))
        if not prediction_paths:
            prediction_paths = [Path(path) for path in (predictions if isinstance(predictions, (list, tuple)) else [predictions])]

        mask_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            if isinstance(masks, (str, os.PathLike)) and Path(masks).is_dir():
                mask_paths.extend(Path(masks).glob(ext))
        if not mask_paths:
            mask_paths = [Path(path) for path in (masks if isinstance(masks, (list, tuple)) else [masks])]

        masks_by_stem = {path.stem.replace("_mask", ""): path for path in mask_paths}
        pairs = []
        for pred_path in prediction_paths:
            stem = pred_path.stem.replace("_mask", "")
            if stem in masks_by_stem:
                pairs.append((pred_path, masks_by_stem[stem]))
        return pairs

    @staticmethod
    def _binary_metric_value(pred_binary: np.ndarray, mask_binary: np.ndarray, metric: str) -> float:
        pred_tensor = torch.from_numpy(pred_binary.astype(np.float32))
        mask_tensor = torch.from_numpy(mask_binary.astype(np.float32))
        if metric == "dice":
            return float(dice_score(mask_tensor, pred_tensor))
        if metric in {"iou", "jaccard"}:
            return float(jac_score(mask_tensor, pred_tensor))
        if metric == "precision":
            return float(precision(mask_tensor, pred_tensor))
        if metric == "recall":
            return float(recall(mask_tensor, pred_tensor))
        raise ValueError(f"Unsupported metric: {metric}")

    @staticmethod
    def _score_prediction_arrays(
        pred_np: np.ndarray,
        mask_np: np.ndarray,
        metrics: Sequence[str],
        num_classes: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        include_background: bool = False,
    ) -> Dict[str, float]:
        pred_np = pred_np.squeeze()
        mask_np = mask_np.squeeze()
        if pred_np.shape != mask_np.shape:
            pred_np = cv2.resize(pred_np, (mask_np.shape[1], mask_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        unique_values = set(np.unique(pred_np).tolist()) | set(np.unique(mask_np).tolist())
        is_binary = unique_values.issubset({0, 1, 255})
        metric_keys = [metric.lower() for metric in metrics]
        scores: Dict[str, float] = {}

        if is_binary and (num_classes is None or num_classes <= 2):
            pred_binary = pred_np > 0
            mask_binary = mask_np > 0
            for metric in metric_keys:
                scores[metric] = Segmenter._binary_metric_value(pred_binary, mask_binary, metric)
            return scores

        if num_classes is None:
            num_classes = int(max(np.max(pred_np), np.max(mask_np))) + 1

        class_ids = list(range(num_classes))
        if not include_background and 0 in class_ids:
            class_ids.remove(0)

        for class_id in class_ids:
            class_label = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
            pred_binary = pred_np == class_id
            mask_binary = mask_np == class_id
            for metric in metric_keys:
                key = f"{metric}_{class_label}"
                scores[key] = Segmenter._binary_metric_value(pred_binary, mask_binary, metric)

        for metric in metric_keys:
            values = [value for key, value in scores.items() if key.startswith(f"{metric}_")]
            if values:
                scores[f"macro_{metric}"] = float(np.mean(values))

        return scores

    @staticmethod
    def evaluate_predictions(
        predictions,
        masks,
        metrics: Sequence[str] = ("dice", "iou", "precision", "recall"),
        save_to: Optional[Union[str, os.PathLike]] = None,
        num_classes: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        include_background: bool = False,
    ) -> Dict[str, object]:
        pairs = Segmenter._match_prediction_mask_pairs(predictions, masks)
        rows = []
        totals: Dict[str, List[float]] = {}

        for prediction_path, mask_path in pairs:
            pred_np = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
            mask_np = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if pred_np is None or mask_np is None:
                continue

            row = {"prediction": Path(prediction_path).name, "mask": Path(mask_path).name}
            scores = Segmenter._score_prediction_arrays(
                pred_np=pred_np,
                mask_np=mask_np,
                metrics=metrics,
                num_classes=num_classes,
                class_names=class_names,
                include_background=include_background,
            )
            row.update(scores)
            for key, value in scores.items():
                totals.setdefault(key, []).append(value)

            rows.append(row)

        summary = {
            "num_samples": len(rows),
            "mean": {metric: float(np.mean(values)) for metric, values in totals.items() if values},
            "rows": rows,
        }

        if save_to is not None:
            output_dir = Path(save_to)
            output_dir.mkdir(parents=True, exist_ok=True)
            if rows:
                fields = sorted({key for row in rows for key in row})
                with (output_dir / "evaluation.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
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

        thresholds = list(thresholds or np.linspace(0.1, 0.9, 17))
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
                mask = cv2.imread(str(mask_by_stem[stem]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                if probability.shape != mask.shape:
                    probability = cv2.resize(probability, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC)
                probabilities.append(probability)
                ground_truths.append((mask > 0).astype(np.float32))

        if not probabilities:
            raise ValueError("No matched image/mask pairs found for threshold tuning.")

        best_threshold = None
        best_score = None
        metric_key = metric.lower()
        for threshold in thresholds:
            scores = []
            for probability, target in zip(probabilities, ground_truths):
                pred = (probability > threshold).astype(np.float32)
                pred_tensor = torch.from_numpy(pred)
                target_tensor = torch.from_numpy(target)
                if metric_key == "dice":
                    score = float(dice_score(target_tensor, pred_tensor))
                elif metric_key in {"iou", "jaccard"}:
                    score = float(jac_score(target_tensor, pred_tensor))
                else:
                    raise ValueError("metric must be 'dice', 'iou', or 'jaccard'.")
                scores.append(score)
            mean_score = float(np.mean(scores))
            rows.append({"threshold": float(threshold), metric_key: mean_score})
            if best_score is None or mean_score > best_score:
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
        include_background: bool = False,
    ) -> Dict[str, object]:
        prediction_dir = prediction_dir or Path(save_to) / "predictions"
        predictions = self.inference(images=images, save_to=prediction_dir, threshold=threshold)
        return self.evaluate_predictions(
            predictions,
            masks,
            metrics=metrics,
            save_to=save_to,
            num_classes=num_classes,
            class_names=class_names,
            include_background=include_background,
        )

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
    ) -> List[Dict[str, object]]:
        evaluation = Segmenter.evaluate_predictions(
            predictions=predictions,
            masks=masks,
            metrics=(metric,),
            num_classes=num_classes,
            class_names=class_names,
            include_background=include_background,
        )
        rows = evaluation["rows"]
        metric_key = metric.lower()
        if rows and metric_key not in rows[0]:
            metric_candidates = [key for key in rows[0] if key == f"macro_{metric_key}" or key.startswith(f"{metric_key}_")]
            metric_key = f"macro_{metric_key}" if f"macro_{metric_key}" in metric_candidates else metric_candidates[0]

        worst = sorted(rows, key=lambda row: row.get(metric_key, float("inf")))[:top_k]

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
        architectures: Sequence[str],
        losses: Sequence[str],
        epochs: int = 1,
        batch_size: int = 4,
        image_size=(224, 224),
        metrics: Optional[Sequence[str]] = None,
        num_classes: int = 1,
        in_channels: int = 3,
        model_kwargs: Optional[dict] = None,
        loss_kwargs: Optional[dict] = None,
        output_dir: Optional[Union[str, os.PathLike]] = "experiments",
        **train_kwargs,
    ) -> Dict[str, Dict[str, List[float]]]:
        results = {}
        summary_rows = []
        for architecture in architectures:
            for loss in losses:
                run_name = f"{architecture}_{loss}"
                print(f"Running experiment: {run_name}")
                segmenter = Segmenter(
                    architecture=architecture,
                    loss=loss,
                    metrics=metrics,
                    image_size=image_size,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    model_kwargs=model_kwargs,
                    loss_kwargs=loss_kwargs,
                )
                per_run_train_kwargs = dict(train_kwargs)
                per_run_train_kwargs.setdefault("experiment_dir", output_dir)
                per_run_train_kwargs.setdefault("run_name", run_name)
                results[run_name] = segmenter.train(
                    data=dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    **per_run_train_kwargs,
                )
                final_metrics = {
                    key: values[-1] for key, values in results[run_name].items() if values
                }
                summary_rows.append({"run": run_name, "architecture": architecture, "loss": loss, **final_metrics})

        if output_dir is not None and summary_rows:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fields = sorted({key for row in summary_rows for key in row})
            with (output_path / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
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
