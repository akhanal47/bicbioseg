import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from ..exceptions import InferenceError
from ..utils.preprocessing import image_tensor, normalize_image, read_rgb
from ..utils.progress import progress_iter
from .artifacts import _write_prediction_image


def _load_inference_images(self, images) -> List[str]:
    paths = self._load_paths_static(images)
    if not paths:
        raise InferenceError("No images found. Provide image files or a folder that contains images.")
    return paths


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
    rgb, pred, probability, logits = self._predict_single_image(
        image, threshold=threshold, return_raw=True
    )

    paths = {}
    overlay = self._overlay_mask_on_image(rgb, pred) if save_to is not None or return_overlay or show else None
    if save_to is not None:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _write_prediction_image(save_path, pred)
        overlay_path = save_path.with_name(f"{save_path.stem}_overlay.png")
        _write_prediction_image(overlay_path, cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

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
    from ..trainer import Segmenter

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

    # Normalize the full image once to keep shared pixels equal in all tiles.
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
    _write_prediction_image(mask_path, prediction)

    outputs = {"mask": str(mask_path.resolve())}
    if save_probability:
        probability_path = output_dir / f"{image_path.stem}_probability.npy"
        np.save(probability_path, probability)
        outputs["probability"] = str(probability_path.resolve())
    if save_overlay:
        overlay = self._overlay_mask_on_image(rgb, prediction)
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        _write_prediction_image(overlay_path, cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
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
