import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..exceptions import InferenceError


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


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


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


def _load_history(history_or_path) -> Dict[str, List[float]]:
    if isinstance(history_or_path, (str, os.PathLike)):
        path = Path(history_or_path)
        with path.open() as handle:
            return json.load(handle)
    return history_or_path


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


def _save_prediction_manifest(path, records):
    from ..trainer import Segmenter

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
