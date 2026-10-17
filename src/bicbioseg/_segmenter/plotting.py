import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch

from ..utils.preprocessing import normalize_image


def plot_history_file(
    history_path: Union[str, os.PathLike],
    metrics: Optional[Sequence[str]] = None,
    save_to: Optional[Union[str, os.PathLike]] = None,
    show: bool = True,
    figsize=(8, 5),
):
    from ..trainer import Segmenter

    history = Segmenter._load_history(history_path)
    return Segmenter._plot_history(
        history=history,
        metrics=metrics,
        save_to=save_to,
        show=show,
        figsize=figsize,
    )


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


def _tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().numpy()
    if image_np.ndim == 3:
        image_np = np.transpose(image_np, (1, 2, 0))
    return np.clip(image_np, 0, 1)


def _tensor_mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy()
    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]
    return mask_np


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


def compare_experiments(
    experiments_dir: Union[str, os.PathLike],
    metric: str = "val_loss",
    save_to: Optional[Union[str, os.PathLike]] = None,
    show: bool = True,
    figsize=(8, 5),
):
    from ..trainer import Segmenter

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
