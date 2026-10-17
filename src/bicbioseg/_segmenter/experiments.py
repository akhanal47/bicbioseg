import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from ..config import ExperimentRunConfig, SegmenterConfig, TrainingConfig


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
    from ..trainer import Segmenter

    from ..utils.environment import set_seed, environment_info

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
