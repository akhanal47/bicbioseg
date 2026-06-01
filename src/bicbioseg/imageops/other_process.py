import random
from typing import List, Tuple

import cv2
import numpy as np


class RemoveNoise:
    @staticmethod
    def apply_bilateral_blur(img: np.ndarray, kernal_size: int = 10) -> np.ndarray:
        mean = np.mean(img)
        variance = np.mean(np.square(img - mean))
        sig_color = int(int(variance // 100) * 1.5)
        return cv2.bilateralFilter(img, kernal_size, sig_color, kernal_size)

    @staticmethod
    def plot_tiles(tiles: List[np.ndarray], num_rows: int, num_cols: int):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
        axes = np.atleast_2d(axes)

        for i, tile in enumerate(tiles[: num_rows * num_cols]):
            ax = axes[i // num_cols, i % num_cols]
            ax.imshow(tile, cmap="gray" if tile.ndim == 2 else None)
            ax.axis("off")

        fig.tight_layout()
        plt.show()

    @staticmethod
    def _read_mask(masks: List[str]) -> Tuple[List[str], List[str]]:
        blank_masks = []
        non_blank_masks = []

        for mask_path in masks:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if np.max(mask) == 0:
                blank_masks.append(str(mask_path))
            else:
                non_blank_masks.append(str(mask_path))

        return blank_masks, non_blank_masks

    @staticmethod
    def balance_data(masks: List[str], ratio: float = 1) -> List[str]:
        blank_masks, non_blank_masks = RemoveNoise._read_mask(masks=masks)
        sample_size = int(ratio * len(non_blank_masks))

        if sample_size < len(blank_masks):
            sampled_blank_masks = random.sample(blank_masks, sample_size)
        else:
            sampled_blank_masks = blank_masks

        print(f"No. of Sampled Blank Mask Images: {len(sampled_blank_masks)}")

        all_masks = sampled_blank_masks + non_blank_masks
        random.shuffle(all_masks)
        if len(all_masks) == 0:
            raise ValueError("No masks read.")

        return all_masks


remove_noise = RemoveNoise
