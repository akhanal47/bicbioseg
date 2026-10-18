import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from ..config import TrainingConfig
from ..exceptions import ModelError
from ..utils.checkpoints import dataset_identity, read_checkpoint
from ..utils.load_data import DataLoader
from ..utils.progress import progress_iter
from ..utils.training import autocast_context, make_scheduler, validate_training_options


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


def _monitor_mode(monitor: str, mode: str = "auto") -> str:
    if mode != "auto":
        return mode
    if "loss" in monitor.lower():
        return "min"
    return "max"


def _is_improved(current_value: float, best_value: Optional[float], mode: str, min_delta: float) -> bool:
    if best_value is None:
        return True
    if mode == "min":
        return current_value < best_value - min_delta
    if mode == "max":
        return current_value > best_value + min_delta
    raise ValueError("mode must be 'auto', 'min', or 'max'.")


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
