from PIL import Image
import os
import pandas as pd
import glob
import numpy as np
import warnings

class ConvertImage:
    def __init__(self) -> None:
        pass

    def _resize_image(self, pil_image, resize=(224,224)):
        if isinstance(resize[0], int) and isinstance(resize[1],int):
            resized_image = pil_image.resize(resize)
        else:
            raise ValueError (f'the x, y size in {resize} should be int')
        return resized_image 

    def _read_tiff_image(self, input_image, resize=None, return_array=False):
        image = Image.open(input_image)
        frame_list = list()
        # to seek if there are more than one frames
        if image.n_frames > 1:
            for frame_idx in range(image.n_frames):
                image.seek(frame_idx)
                current_frame = image.tell()
                current_frame_jpg = current_frame.convert('RGB')
                if resize is not None:
                    current_frame_jpg = self._resize_image(current_frame_jpg, resize=resize)
                frame_list.append(current_frame_jpg)
            if return_array:
                frame_list = np.array(frame_list)
                return frame_list
            else:
                return frame_list
        elif image.n_frames == 1:
            current_frame_jpg = image.convert('RGB')
            if resize is not None:
                current_frame_jpg = self._resize_image(current_frame_jpg, resize=resize)
            if return_array:
                current_frame_jpg = np.array(current_frame_jpg)
            return current_frame_jpg
        else:
            raise ValueError (f"error reading file {input_image}")


    def _read_tiff_metadata(self, input_image):
        image = Image.open(input_image)
        metadata = {
                    "filename": str(input_image),
                    "size": image.size,
                    "height": image.height,
                    "width": image.width,
                    "format": image.format,
                    "mode": image.mode,
                    "is_animated": getattr(image, "is_animated", False),
                    "frames": getattr(image, "n_frames", 1)
                    }
        return metadata
    
    def tiff_extract_frames(self,input_image:str, resize=None, return_array=False, get_metadata=False, save_image=False, save_image_location=None, save_metadata=False, save_metadata_location=None):
        _, input_image_ext = os.path.splitext((os.path.basename(input_image)))
        if input_image_ext.lower() in ('.tif', '.tiff'):
            if get_metadata == False:
                converted_image = self._read_tiff_image(input_image=input_image, resize=resize, return_array=return_array)
                return converted_image, None
            elif get_metadata == True:
                converted_image = self._read_tiff_image(input_image=input_image, resize=resize, return_array=return_array)
                meta_data = self._read_tiff_metadata(input_image=input_image)
                return converted_image, meta_data
        else:
            raise TypeError (f"{input_image} is not a tiff/tif file")
    