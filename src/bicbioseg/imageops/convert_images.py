import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image


PathLike = Union[str, Path]


class ConvertImage:
    @staticmethod
    def _resize_image(pil_image: Image.Image, resize: Tuple[int, int] = (224, 224)) -> Image.Image:
        if not (isinstance(resize[0], int) and isinstance(resize[1], int)):
            raise ValueError(f"The x, y size in {resize} should be int.")
        return pil_image.resize(resize)

    @staticmethod
    def _read_tiff_image(
        input_image: PathLike,
        resize: Optional[Tuple[int, int]] = None,
        return_array: bool = False,
    ):
        image = Image.open(input_image)
        frame_list: List[Union[Image.Image, np.ndarray]] = []

        for frame_idx in range(getattr(image, "n_frames", 1)):
            image.seek(frame_idx)
            current_frame = image.convert("RGB")
            if resize is not None:
                current_frame = ConvertImage._resize_image(current_frame, resize=resize)
            frame_list.append(np.array(current_frame) if return_array else current_frame.copy())

        if len(frame_list) == 1:
            return frame_list[0]
        return np.array(frame_list) if return_array else frame_list

    @staticmethod
    def _read_tiff_metadata(input_image: PathLike) -> dict:
        image = Image.open(input_image)
        return {
            "filename": str(input_image),
            "size": image.size,
            "height": image.height,
            "width": image.width,
            "format": image.format,
            "mode": image.mode,
            "is_animated": getattr(image, "is_animated", False),
            "frames": getattr(image, "n_frames", 1),
        }

    @staticmethod
    def tiff_extract_frames(
        input_image: PathLike,
        resize: Optional[Tuple[int, int]] = None,
        return_array: bool = False,
        get_metadata: bool = False,
        save_image: bool = False,
        save_image_location: Optional[PathLike] = None,
        save_metadata: bool = False,
        save_metadata_location: Optional[PathLike] = None,
    ):
        input_path = Path(input_image)
        if input_path.suffix.lower() not in (".tif", ".tiff"):
            raise TypeError(f"{input_image} is not a tiff/tif file.")

        converted_image = ConvertImage._read_tiff_image(
            input_image=input_path,
            resize=resize,
            return_array=return_array,
        )
        metadata = ConvertImage._read_tiff_metadata(input_path) if get_metadata or save_metadata else None

        if save_image:
            save_dir = Path(save_image_location or f"{input_path.stem}_frames")
            save_dir.mkdir(parents=True, exist_ok=True)
            frames = converted_image if isinstance(converted_image, list) else [converted_image]
            for idx, frame in enumerate(frames):
                frame_img = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
                frame_img.save(save_dir / f"{input_path.stem}_frame_{idx:04d}.png")

        if save_metadata:
            metadata_path = Path(save_metadata_location or f"{input_path.stem}_metadata.json")
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_path.open("w") as handle:
                json.dump(metadata, handle, indent=2)

        return (converted_image, metadata) if get_metadata else (converted_image, None)

    @staticmethod
    def dicom_to_uint8(
        input_image: PathLike,
        output_path: Optional[PathLike] = None,
        method: str = "linear",
        clip_percentiles: Tuple[float, float] = (1.0, 99.0),
        invert: bool = False,
        return_array: bool = False,
    ):
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError("pydicom is required for DICOM conversion. Install it with `pip install pydicom`.") from exc

        dataset = pydicom.dcmread(str(input_image))
        image = dataset.pixel_array.astype(np.float32)

        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        image = image * slope + intercept

        method = method.lower()
        if method == "clip":
            low, high = np.percentile(image, clip_percentiles)
            image = np.clip(image, low, high)
        elif method == "log":
            image = image - np.min(image)
            image = np.log1p(image)
        elif method != "linear":
            raise ValueError("method must be 'linear', 'clip', or 'log'.")

        image = image - np.min(image)
        max_value = np.max(image)
        if max_value == 0:
            warnings.warn("DICOM image has no dynamic range after scaling.")
            image_uint8 = np.zeros_like(image, dtype=np.uint8)
        else:
            image_uint8 = np.clip((image / max_value) * 255, 0, 255).astype(np.uint8)

        if invert:
            image_uint8 = 255 - image_uint8

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image_uint8).save(output_path)

        if return_array or output_path is None:
            return image_uint8
        return str(output_path.resolve())


tiff_extract_frames = ConvertImage.tiff_extract_frames
dicom_to_uint8 = ConvertImage.dicom_to_uint8
