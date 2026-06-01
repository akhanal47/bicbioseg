import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union


class SerializableConfig:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Union[str, Path]) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return str(path.resolve())

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]):
        return cls(**payload)

    @classmethod
    def load(cls, path: Union[str, Path]):
        with Path(path).open() as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class DatasetSplitConfig(SerializableConfig):
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    resize: Optional[Tuple[int, int]] = None
    create_patches: bool = False
    patch_size: Tuple[int, int] = (224, 224)
    balance_empty_masks: bool = False
    group_by: str = "filename"
    random_seed: int = 42
    overwrite: bool = False
    progress: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]):
        payload = dict(payload)
        if payload.get("split") is not None:
            payload["split"] = tuple(payload["split"])
        if payload.get("resize") is not None:
            payload["resize"] = tuple(payload["resize"])
        if payload.get("patch_size") is not None:
            payload["patch_size"] = tuple(payload["patch_size"])
        return cls(**payload)


@dataclass
class TrainingConfig(SerializableConfig):
    epochs: int = 1
    batch_size: int = 4
    lr: float = 1e-4
    optimizer: str = "adam"
    num_workers: int = 2
    save_to: Optional[str] = None
    experiment_dir: Optional[str] = None
    run_name: Optional[str] = None
    save_best: bool = True
    monitor: str = "val_loss"
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 0.0
    monitor_mode: str = "auto"
    resume_from: Optional[str] = None
    progress_bar: bool = True
    verbose: bool = True


@dataclass
class SegmenterConfig(SerializableConfig):
    architecture: str = "unet"
    loss: str = "dice"
    metrics: Optional[Sequence[str]] = None
    image_size: Tuple[int, int] = (224, 224)
    num_classes: int = 1
    in_channels: int = 3
    device: Optional[str] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]):
        payload = dict(payload)
        if payload.get("image_size") is not None:
            payload["image_size"] = tuple(payload["image_size"])
        return cls(**payload)


@dataclass
class ExperimentConfig(SerializableConfig):
    images: Union[str, Sequence[str]]
    masks: Union[str, Sequence[str]]
    model: str = "unet"
    loss: str = "dice"
    work_dir: str = "bicbioseg_experiment"
    image_size: Tuple[int, int] = (224, 224)
    num_classes: int = 1
    in_channels: int = 3
    metrics: Optional[Sequence[str]] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    device: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]):
        payload = dict(payload)
        if payload.get("image_size") is not None:
            payload["image_size"] = tuple(payload["image_size"])
        return cls(**payload)
