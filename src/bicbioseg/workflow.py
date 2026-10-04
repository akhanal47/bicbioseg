import json
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from .config import DatasetSplitConfig, ExperimentConfig, TrainingConfig
from .imageops.preprocess import ImageOps, create_dataset_split
from .trainer import Segmenter


class SegmentationExperiment:

    def __init__(
        self,
        images,
        masks,
        model: str = "unet",
        loss: str = "dice",
        work_dir: Union[str, Path] = "bicbioseg_experiment",
        image_size=(224, 224),
        num_classes: int = 1,
        in_channels: int = 3,
        metrics: Optional[Sequence[str]] = None,
        model_kwargs: Optional[dict] = None,
        loss_kwargs: Optional[dict] = None,
        device: Optional[str] = None,
        config: Optional[ExperimentConfig] = None,
        normalization: Optional[dict] = None,
        ignore_index: Optional[int] = None,
        include_background: bool = False,
        metric_aggregation: str = "per_image",
        empty_policy: str = "exclude",
        class_map: Optional[dict] = None,
    ):
        if config is not None:
            images = config.images
            masks = config.masks
            model = config.model
            loss = config.loss
            work_dir = config.work_dir
            image_size = config.image_size
            num_classes = config.num_classes
            in_channels = config.in_channels
            metrics = config.metrics
            model_kwargs = config.model_kwargs
            loss_kwargs = config.loss_kwargs
            device = config.device
            normalization, ignore_index = config.normalization, config.ignore_index
            include_background, metric_aggregation = config.include_background, config.metric_aggregation
            empty_policy, class_map = config.empty_policy, config.class_map

        self.images = images
        self.masks = masks
        self.experiment_config = ExperimentConfig(
            images=str(images) if isinstance(images, Path) else images,
            masks=str(masks) if isinstance(masks, Path) else masks,
            model=model,
            loss=loss,
            work_dir=str(work_dir),
            image_size=image_size,
            num_classes=num_classes,
            in_channels=in_channels,
            metrics=metrics,
            model_kwargs=model_kwargs or {},
            loss_kwargs=loss_kwargs or {},
            normalization=normalization or {"mode": "standard"},
            ignore_index=ignore_index,
            include_background=include_background,
            metric_aggregation=metric_aggregation,
            empty_policy=empty_policy,
            class_map=class_map,
            device=device,
        )
        self.work_dir = Path(work_dir)
        self.dataset_dir = self.work_dir / "dataset"
        self.qc_dir = self.work_dir / "qc"
        self.run_dir = self.work_dir / "runs" / f"{model}_{loss}"
        self.prediction_dir = self.work_dir / "predictions"
        self.evaluation_dir = self.work_dir / "evaluation"
        self.segmenter = Segmenter(
            architecture=model,
            loss=loss,
            metrics=metrics,
            image_size=image_size,
            num_classes=num_classes,
            in_channels=in_channels,
            model_kwargs=model_kwargs,
            loss_kwargs=loss_kwargs,
            normalization=normalization,
            ignore_index=ignore_index,
            include_background=include_background,
            metric_aggregation=metric_aggregation,
            empty_policy=empty_policy,
            class_map=class_map,
            device=device,
        )

    @classmethod
    def from_config(cls, config: Union[ExperimentConfig, str, Path]):
        if isinstance(config, (str, Path)):
            config = ExperimentConfig.load(config)
        return cls(None, None, config=config)

    def save_config(self, path: Union[str, Path]) -> str:
        return self.experiment_config.save(path)

    def prepare(
        self,
        split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        resize=None,
        create_patches: bool = False,
        patch_size=(224, 224),
        balance_empty_masks: bool = False,
        overwrite: bool = False,
        config: Optional[DatasetSplitConfig] = None,
        **kwargs,
    ) -> str:
        if config is not None:
            split = config.split
            resize = config.resize
            create_patches = config.create_patches
            patch_size = config.patch_size
            balance_empty_masks = config.balance_empty_masks
            overwrite = config.overwrite
            kwargs.setdefault("group_by", config.group_by)
            kwargs.setdefault("random_seed", config.random_seed)
            kwargs.setdefault("progress", config.progress)

        self.dataset_dir = Path(
            create_dataset_split(
                images=self.images,
                masks=self.masks,
                save_to=self.dataset_dir,
                split=split,
                resize=resize,
                create_patches=create_patches,
                patch_size=patch_size,
                balance_empty_masks=balance_empty_masks,
                overwrite=overwrite,
                **kwargs,
            )
        )
        return str(self.dataset_dir)

    def qc(self, save: bool = True, splits: bool = False):
        self.qc_dir.mkdir(parents=True, exist_ok=True)
        save_to = self.qc_dir / "qc_report.json" if save else None
        options = dict(
            num_classes=self.segmenter.num_classes, ignore_index=self.segmenter.ignore_index, save_to=save_to
        )
        if splits:
            return ImageOps.dataset_split_qc_report(self.dataset_dir, **options)
        return ImageOps.dataset_qc_report(self.images, self.masks, **options)

    def preview(self, num_samples: int = 6, show: bool = True):
        self.qc_dir.mkdir(parents=True, exist_ok=True)
        return ImageOps.preview_dataset(
            self.images,
            self.masks,
            num_samples=num_samples,
            save_to=self.qc_dir / "dataset_preview.png",
            show=show,
        )

    def train(self, epochs: int = 1, batch_size: int = 4, config: Optional[TrainingConfig] = None, **kwargs):
        data = kwargs.pop("data", self.dataset_dir)
        kwargs.setdefault("experiment_dir", self.run_dir.parent)
        kwargs.setdefault("run_name", self.run_dir.name)
        result = self.segmenter.train(data=data, epochs=epochs, batch_size=batch_size, config=config, **kwargs)
        directory = config.experiment_dir if config and config.experiment_dir is not None else kwargs["experiment_dir"]
        name = config.run_name if config and config.run_name is not None else kwargs["run_name"]
        if directory is not None and name is not None:
            self.run_dir = Path(directory) / name
        return result

    def _select_checkpoint(self, checkpoint):
        if checkpoint == "current":
            return self.segmenter
        path = (
            self.run_dir / f"{checkpoint}_model.pt"
            if checkpoint in ("best", "final", "last")
            else Path(checkpoint)
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {path}. Train and save this checkpoint first."
            )
        return Segmenter.load(path, device=str(self.segmenter.device))

    def _record_artifact(self, kind, directory):
        path = self.work_dir / "artifacts.json"
        payload = json.loads(path.read_text()) if path.exists() else {}
        entries = payload.setdefault(kind, [])
        location = str(Path(directory).resolve())
        if location not in entries:
            entries.append(location)
        self.segmenter._write_json(path, payload)

    def evaluate(self, split: str = "test", checkpoint="current", **kwargs):
        custom_data = "images" in kwargs or "masks" in kwargs
        images = kwargs.pop("images", self.dataset_dir / split / "images")
        masks = kwargs.pop("masks", self.dataset_dir / split / "masks")
        kwargs.setdefault("save_to", self.evaluation_dir)
        kwargs.setdefault("split", "custom" if custom_data else split)
        model = self._select_checkpoint(checkpoint)
        result = model.evaluate(images=images, masks=masks, **kwargs)
        self._record_artifact("evaluation", kwargs["save_to"])
        return result

    def predict(self, images, checkpoint="current", **kwargs):
        kwargs.setdefault("save_to", self.prediction_dir)
        result = self._select_checkpoint(checkpoint).inference(images=images, **kwargs)
        self._record_artifact("predictions", kwargs["save_to"])
        return result

    def report(self, save_to=None):
        path = self.work_dir / "artifacts.json"
        artifacts = json.loads(path.read_text()) if path.exists() else {}
        return self.segmenter.create_report(
            self.run_dir,
            save_to=save_to,
            evaluation_dirs=artifacts.get("evaluation", [self.evaluation_dir]),
            prediction_dirs=artifacts.get("predictions", [self.prediction_dir]),
        )
