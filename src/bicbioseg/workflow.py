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

    def qc(self, save: bool = True):
        self.qc_dir.mkdir(parents=True, exist_ok=True)
        save_to = self.qc_dir / "qc_report.json" if save else None
        return ImageOps.dataset_qc_report(self.images, self.masks, save_to=save_to)

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
        if config is not None:
            epochs = config.epochs
            batch_size = config.batch_size
        return self.segmenter.train(
            data=data,
            epochs=epochs,
            batch_size=batch_size,
            config=config,
            experiment_dir=self.run_dir.parent,
            run_name=self.run_dir.name,
            **kwargs,
        )

    def evaluate(self, split: str = "test", **kwargs):
        images = kwargs.pop("images", self.dataset_dir / split / "images")
        masks = kwargs.pop("masks", self.dataset_dir / split / "masks")
        return self.segmenter.evaluate(
            images=images,
            masks=masks,
            save_to=self.evaluation_dir,
            **kwargs,
        )

    def predict(self, images, **kwargs):
        return self.segmenter.inference(
            images=images,
            save_to=self.prediction_dir,
            **kwargs,
        )

    def report(self):
        return self.segmenter.create_report(self.run_dir)
