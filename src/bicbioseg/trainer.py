import os
import platform
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from .config import SegmenterConfig, TrainingConfig, ExperimentRunConfig
from .exceptions import ModelError
from .models.registry import MODEL_ALIASES, build_model, model_metadata
from .models.options import model_options, normalize_model_options
from .utils.preprocessing import validate_normalization
from .utils.training import autocast_context, validate_training_options
from .utils import losses as loss_module
from .utils.evaluation import MetricAccumulator, validate_metric_options

from ._segmenter import (
    training as _training,
    checkpoints as _checkpoints,
    inference as _inference,
    evaluation as _evaluation,
    plotting as _plotting,
    artifacts as _artifacts,
    experiments as _experiments,
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

    def _make_optimizer(self, optimizer: Union[str, torch.optim.Optimizer], lr: float):
        return _training._make_optimizer(self, optimizer=optimizer, lr=lr)

    @staticmethod
    def _monitor_mode(monitor: str, mode: str = "auto") -> str:
        return _training._monitor_mode(monitor=monitor, mode=mode)

    @staticmethod
    def _is_improved(current_value: float, best_value: Optional[float], mode: str, min_delta: float) -> bool:
        return _training._is_improved(
            current_value=current_value, best_value=best_value, mode=mode, min_delta=min_delta,
        )

    def _loader_from_data(
        self,
        data,
        batch_size: int,
        transforms=None,
        num_workers: int = 2,
        shuffle: bool = True,
    ):
        return _training._loader_from_data(
            self, data=data, batch_size=batch_size, transforms=transforms, num_workers=num_workers,
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
        return _training._resolve_loaders(
            self, data=data, train_data=train_data, val_data=val_data, batch_size=batch_size,
            transforms=transforms, num_workers=num_workers,
        )

    def _run_epoch(self, loader, optimizer=None) -> Dict[str, float]:
        return _training._run_epoch(self, loader=loader, optimizer=optimizer)

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
        return _training.train(
            self, data=data, train_data=train_data, val_data=val_data, epochs=epochs,
            batch_size=batch_size, lr=lr, optimizer=optimizer, transforms=transforms,
            num_workers=num_workers, save_to=save_to, experiment_dir=experiment_dir,
            run_name=run_name, save_best=save_best, monitor=monitor, early_stopping=early_stopping,
            patience=patience, min_delta=min_delta, monitor_mode=monitor_mode,
            resume_from=resume_from, progress_bar=progress_bar, verbose=verbose, config=config,
            precision=precision, accumulation_steps=accumulation_steps, max_grad_norm=max_grad_norm,
            scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
            aux_loss_weights=aux_loss_weights, crop_size=crop_size,
            foreground_probability=foreground_probability, checkpoint_interval=checkpoint_interval,
            dataset_id=dataset_id,
        )

    def _load_checkpoint_for_resume(self, path, optimizer=None, checkpoint=None):
        return _checkpoints._load_checkpoint_for_resume(
            self, path=path, optimizer=optimizer, checkpoint=checkpoint,
        )

    def save(self, path: Union[str, os.PathLike], optimizer=None) -> str:
        return _checkpoints.save(self, path=path, optimizer=optimizer)

    def load_weights(self, path: Union[str, os.PathLike]):
        return _checkpoints.load_weights(self, path=path)

    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike],
        device: Optional[str] = None,
        loss: Optional[Union[str, torch.nn.Module, Callable]] = None,
        **overrides,
    ):
        return _checkpoints.load(cls, path=path, device=device, loss=loss, **overrides)

    def _load_inference_images(self, images) -> List[str]:
        return _inference._load_inference_images(self, images=images)

    def _predict_array(self, rgb: np.ndarray, threshold: float = 0.5, return_raw: bool = False):
        return _inference._predict_array(self, rgb=rgb, threshold=threshold, return_raw=return_raw)

    def _predict_single_image(self, image_path: Union[str, os.PathLike], threshold: float = 0.5, return_raw=False):
        return _inference._predict_single_image(
            self, image_path=image_path, threshold=threshold, return_raw=return_raw,
        )

    def _prediction_metadata(self, threshold):
        return _inference._prediction_metadata(self, threshold=threshold)

    def _prediction_record(
        self, source, paths, threshold, prediction, probability=None, logits=None, arrays=False
    ):
        return _inference._prediction_record(
            self, source=source, paths=paths, threshold=threshold, prediction=prediction,
            probability=probability, logits=logits, arrays=arrays,
        )

    @staticmethod
    def _validate_prediction_request(paths, batch_size, threshold):
        return _inference._validate_prediction_request(
            paths=paths, batch_size=batch_size, threshold=threshold,
        )

    def _predict_batch(self, images, threshold):
        return _inference._predict_batch(self, images=images, threshold=threshold)

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
        return _inference.inference(
            self, images=images, save_to=save_to, threshold=threshold, return_arrays=return_arrays,
            save_overlay=save_overlay, save_probability=save_probability, save_logits=save_logits,
            save_contours=save_contours, batch_size=batch_size, structured=structured,
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
        return _inference.predict_one(
            self, image=image, threshold=threshold, save_to=save_to, return_overlay=return_overlay,
            show=show, structured=structured,
        )

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
        return _inference.ensemble_predict(
            checkpoints=checkpoints, images=images, save_to=save_to, threshold=threshold,
            device=device, save_overlay=save_overlay, batch_size=batch_size, structured=structured,
            return_arrays=return_arrays, save_probability=save_probability,
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
        return _inference.inference_large_image(
            self, image=image, save_to=save_to, patch_size=patch_size, overlap=overlap,
            threshold=threshold, save_overlay=save_overlay, save_probability=save_probability,
            tile_batch_size=tile_batch_size, weighting=weighting, gaussian_sigma=gaussian_sigma,
            structured=structured, return_arrays=return_arrays,
        )

    @staticmethod
    def _match_prediction_mask_pairs(predictions, masks):
        return _evaluation._match_prediction_mask_pairs(predictions=predictions, masks=masks)

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
        return _evaluation._score_prediction_arrays(
            pred_np=pred_np, mask_np=mask_np, metrics=metrics, num_classes=num_classes,
            class_names=class_names, include_background=include_background,
            ignore_index=ignore_index, empty_policy=empty_policy,
        )

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
        return _evaluation.evaluate_predictions(
            predictions=predictions, masks=masks, metrics=metrics, save_to=save_to,
            num_classes=num_classes, class_names=class_names, include_background=include_background,
            ignore_index=ignore_index, empty_policy=empty_policy, aggregation=aggregation,
        )

    def find_best_threshold(
        self,
        images,
        masks,
        thresholds: Optional[Sequence[float]] = None,
        metric: str = "dice",
        save_to: Optional[Union[str, os.PathLike]] = None,
    ) -> Dict[str, object]:
        return _evaluation.find_best_threshold(
            self, images=images, masks=masks, thresholds=thresholds, metric=metric, save_to=save_to,
        )

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
        return _evaluation.evaluate(
            self, images=images, masks=masks, save_to=save_to, prediction_dir=prediction_dir,
            metrics=metrics, threshold=threshold, num_classes=num_classes, class_names=class_names,
            include_background=include_background, empty_policy=empty_policy,
            aggregation=aggregation, split=split, batch_size=batch_size,
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
        ignore_index: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        return _evaluation.find_worst_predictions(
            predictions=predictions, masks=masks, metric=metric, top_k=top_k, save_to=save_to,
            images=images, num_classes=num_classes, class_names=class_names,
            include_background=include_background, ignore_index=ignore_index,
        )

    @staticmethod
    def plot_history_file(
        history_path: Union[str, os.PathLike],
        metrics: Optional[Sequence[str]] = None,
        save_to: Optional[Union[str, os.PathLike]] = None,
        show: bool = True,
        figsize=(8, 5),
    ):
        return _plotting.plot_history_file(
            history_path=history_path, metrics=metrics, save_to=save_to, show=show, figsize=figsize,
        )

    @staticmethod
    def _plot_history(
        history: Dict[str, List[float]],
        metrics: Optional[Sequence[str]] = None,
        save_to: Optional[Union[str, os.PathLike]] = None,
        show: bool = True,
        figsize=(8, 5),
    ):
        return _plotting._plot_history(
            history=history, metrics=metrics, save_to=save_to, show=show, figsize=figsize,
        )

    def plot_history(
        self,
        metrics: Optional[Sequence[str]] = None,
        save_to: Optional[Union[str, os.PathLike]] = None,
        show: bool = True,
        figsize=(8, 5),
    ):
        return _plotting.plot_history(
            self, metrics=metrics, save_to=save_to, show=show, figsize=figsize,
        )

    @staticmethod
    def _tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
        return _plotting._tensor_image_to_numpy(image=image)

    @staticmethod
    def _tensor_mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
        return _plotting._tensor_mask_to_numpy(mask=mask)

    @staticmethod
    def _overlay_mask_on_image(
        image: np.ndarray,
        mask: np.ndarray,
        color=(1.0, 0.0, 0.0),
        alpha: float = 0.4,
    ) -> np.ndarray:
        return _plotting._overlay_mask_on_image(image=image, mask=mask, color=color, alpha=alpha)

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
        return _plotting.plot_training_samples(
            self, data=data, train_data=train_data, batch_size=batch_size, num_samples=num_samples,
            transforms=transforms, num_workers=num_workers, overlay=overlay, save_to=save_to,
            show=show, figsize=figsize,
        )

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
        return _plotting.plot_inference_results(
            self, images=images, predictions=predictions, num_samples=num_samples,
            threshold=threshold, overlay=overlay, save_to=save_to, show=show, figsize=figsize,
        )

    @staticmethod
    def compare_experiments(
        experiments_dir: Union[str, os.PathLike],
        metric: str = "val_loss",
        save_to: Optional[Union[str, os.PathLike]] = None,
        show: bool = True,
        figsize=(8, 5),
    ):
        return _plotting.compare_experiments(
            experiments_dir=experiments_dir, metric=metric, save_to=save_to, show=show,
            figsize=figsize,
        )

    @staticmethod
    def _prepare_run_dir(experiment_dir, run_name: Optional[str], resume_from=None) -> Optional[Path]:
        return _artifacts._prepare_run_dir(
            experiment_dir=experiment_dir, run_name=run_name, resume_from=resume_from,
        )

    @staticmethod
    def _write_json(path: Path, payload: dict):
        return _artifacts._write_json(path=path, payload=payload)

    @staticmethod
    def _write_history_csv(path: Path, history: Dict[str, List[float]]):
        return _artifacts._write_history_csv(path=path, history=history)

    def _update_run_artifacts(self, run_dir: Optional[Path]):
        return _artifacts._update_run_artifacts(self, run_dir=run_dir)

    @staticmethod
    def _load_history(history_or_path) -> Dict[str, List[float]]:
        return _artifacts._load_history(history_or_path=history_or_path)

    @staticmethod
    def _load_paths_static(paths_or_dir) -> List[str]:
        return _artifacts._load_paths_static(paths_or_dir=paths_or_dir)

    @staticmethod
    def _save_contours(prediction: np.ndarray, path: Path):
        return _artifacts._save_contours(prediction=prediction, path=path)

    @staticmethod
    def _save_prediction_manifest(path, records):
        return _artifacts._save_prediction_manifest(path=path, records=records)

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
        return _artifacts._save_prediction_files(
            self, output_dir=output_dir, source=source, rgb=rgb, prediction=prediction,
            probability=probability, logits=logits, save_overlay=save_overlay,
            save_probability=save_probability, save_logits=save_logits, save_contours=save_contours,
        )

    def create_report(self, run_dir, save_to=None, evaluation_dirs=None, prediction_dirs=None):
        """Write a report using explicit artifact locations and their saved provenance."""
        return _artifacts.create_report(
            self, run_dir=run_dir, save_to=save_to, evaluation_dirs=evaluation_dirs,
            prediction_dirs=prediction_dirs,
        )

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
        return _experiments.run_experiment(
            dataset=dataset, architectures=architectures, losses=losses, epochs=epochs,
            batch_size=batch_size, image_size=image_size, metrics=metrics, num_classes=num_classes,
            in_channels=in_channels, model_kwargs=model_kwargs, loss_kwargs=loss_kwargs,
            output_dir=output_dir, runs=runs, seed=seed, device=device, **train_kwargs,
        )
