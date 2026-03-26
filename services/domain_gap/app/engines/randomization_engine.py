"""
Domain Randomization Engine
============================
Domain-agnostic image augmentation for increasing synthetic dataset diversity.
Operates entirely on CPU using OpenCV and NumPy.

Generates multiple variants of each synthetic image with controlled
randomization of color, brightness, contrast, noise, and blur.
Optionally matches histogram statistics from real reference images.
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class RandomizationEngine:
    """Applies domain randomization to synthetic images."""

    def __init__(self):
        logger.info("RandomizationEngine initialized (CPU-only)")

    def apply_single(
        self,
        image_path: str,
        output_dir: str,
        num_variants: int = 3,
        intensity: float = 0.5,
        color_jitter: float = 0.3,
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.7, 1.3),
        noise_intensity: float = 0.02,
        blur_range: Tuple[float, float] = (0.0, 1.5),
        reference_stats: Optional[dict] = None,
        histogram_match_strength: float = 0.5,
        annotations_path: Optional[str] = None,
    ) -> List[str]:
        """Apply domain randomization to a single image, producing N variants.

        Args:
            image_path: Path to source image
            output_dir: Directory for output variants
            num_variants: Number of variants to generate
            intensity: Global intensity multiplier (0-1)
            color_jitter: Color variation strength
            brightness_range: Min/max brightness multiplier
            contrast_range: Min/max contrast multiplier
            saturation_range: Min/max saturation multiplier
            noise_intensity: Gaussian noise sigma
            blur_range: Min/max Gaussian blur sigma
            reference_stats: Pre-computed reference set stats for histogram matching
            histogram_match_strength: How strongly to match reference histogram (0-1)
            annotations_path: Path to annotations file to copy alongside

        Returns:
            List of paths to generated variant images
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return []

        os.makedirs(output_dir, exist_ok=True)
        stem = Path(image_path).stem
        ext = Path(image_path).suffix or ".jpg"

        output_paths = []
        for i in range(num_variants):
            variant = self._generate_variant(
                image,
                intensity=intensity,
                color_jitter=color_jitter,
                brightness_range=brightness_range,
                contrast_range=contrast_range,
                saturation_range=saturation_range,
                noise_intensity=noise_intensity,
                blur_range=blur_range,
                reference_stats=reference_stats,
                histogram_match_strength=histogram_match_strength,
            )

            variant_name = f"{stem}_dr{i:02d}{ext}"
            variant_path = os.path.join(output_dir, variant_name)
            cv2.imwrite(variant_path, variant)
            output_paths.append(variant_path)

            # Copy annotations if provided
            if annotations_path and os.path.exists(annotations_path):
                ann_ext = Path(annotations_path).suffix
                ann_name = f"{stem}_dr{i:02d}{ann_ext}"
                ann_out = os.path.join(output_dir, ann_name)
                shutil.copy2(annotations_path, ann_out)

        return output_paths

    def apply_batch(
        self,
        images_dir: str,
        output_dir: str,
        num_variants: int = 3,
        intensity: float = 0.5,
        color_jitter: float = 0.3,
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.7, 1.3),
        noise_intensity: float = 0.02,
        blur_range: Tuple[float, float] = (0.0, 1.5),
        reference_stats: Optional[dict] = None,
        histogram_match_strength: float = 0.5,
        annotations_dir: Optional[str] = None,
        progress_callback=None,
    ) -> dict:
        """Apply domain randomization to all images in a directory.

        Args:
            images_dir: Directory with source images
            output_dir: Directory for output variants
            progress_callback: Optional callable(processed, total) for progress updates

        Returns:
            Dict with total_images, total_variants, failed count
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = sorted([
            f for f in Path(images_dir).iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ])

        if not image_files:
            logger.warning(f"No images found in {images_dir}")
            return {"total_images": 0, "total_variants": 0, "failed": 0}

        os.makedirs(output_dir, exist_ok=True)
        total_variants = 0
        failed = 0

        for idx, img_path in enumerate(image_files):
            try:
                # Find matching annotation file if annotations_dir provided
                ann_path = None
                if annotations_dir:
                    for ann_ext in [".json", ".txt", ".xml"]:
                        candidate = Path(annotations_dir) / f"{img_path.stem}{ann_ext}"
                        if candidate.exists():
                            ann_path = str(candidate)
                            break

                variants = self.apply_single(
                    image_path=str(img_path),
                    output_dir=output_dir,
                    num_variants=num_variants,
                    intensity=intensity,
                    color_jitter=color_jitter,
                    brightness_range=brightness_range,
                    contrast_range=contrast_range,
                    saturation_range=saturation_range,
                    noise_intensity=noise_intensity,
                    blur_range=blur_range,
                    reference_stats=reference_stats,
                    histogram_match_strength=histogram_match_strength,
                    annotations_path=ann_path,
                )
                total_variants += len(variants)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                failed += 1

            if progress_callback:
                progress_callback(idx + 1, len(image_files))

        logger.info(
            f"Domain randomization complete: {len(image_files)} images → "
            f"{total_variants} variants ({failed} failed)"
        )
        return {
            "total_images": len(image_files),
            "total_variants": total_variants,
            "failed": failed,
        }

    def _generate_variant(
        self,
        image: np.ndarray,
        intensity: float,
        color_jitter: float,
        brightness_range: Tuple[float, float],
        contrast_range: Tuple[float, float],
        saturation_range: Tuple[float, float],
        noise_intensity: float,
        blur_range: Tuple[float, float],
        reference_stats: Optional[dict],
        histogram_match_strength: float,
    ) -> np.ndarray:
        """Generate a single randomized variant of the image."""
        result = image.copy()

        # Scale all randomization by intensity
        eff_color = color_jitter * intensity
        eff_noise = noise_intensity * intensity
        eff_blur_max = blur_range[1] * intensity

        # 1. Color jitter in LAB space
        if eff_color > 0:
            result = self._apply_color_jitter(result, eff_color)

        # 2. Brightness adjustment
        result = self._apply_brightness(result, brightness_range, intensity)

        # 3. Contrast adjustment
        result = self._apply_contrast(result, contrast_range, intensity)

        # 4. Saturation adjustment
        result = self._apply_saturation(result, saturation_range, intensity)

        # 5. Gaussian noise
        if eff_noise > 0:
            result = self._apply_noise(result, eff_noise)

        # 6. Gaussian blur
        if eff_blur_max > 0:
            result = self._apply_blur(result, blur_range[0], eff_blur_max)

        # 7. Histogram matching to reference (if provided)
        if reference_stats and histogram_match_strength > 0:
            result = self._apply_histogram_match(
                result, reference_stats, histogram_match_strength * intensity
            )

        return result

    def _apply_color_jitter(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply random color jitter in LAB space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Perturb A and B channels
        a_shift = random.gauss(0, strength * 10)
        b_shift = random.gauss(0, strength * 10)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + a_shift, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + b_shift, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _apply_brightness(
        self, image: np.ndarray, brange: Tuple[float, float], intensity: float
    ) -> np.ndarray:
        """Adjust brightness by scaling the L channel."""
        # Interpolate range toward 1.0 based on inverse intensity
        low = 1.0 + (brange[0] - 1.0) * intensity
        high = 1.0 + (brange[1] - 1.0) * intensity
        factor = random.uniform(low, high)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * factor, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _apply_contrast(
        self, image: np.ndarray, crange: Tuple[float, float], intensity: float
    ) -> np.ndarray:
        """Adjust contrast by scaling L channel variance around its mean."""
        low = 1.0 + (crange[0] - 1.0) * intensity
        high = 1.0 + (crange[1] - 1.0) * intensity
        factor = random.uniform(low, high)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]
        mean_l = l_channel.mean()
        lab[:, :, 0] = np.clip((l_channel - mean_l) * factor + mean_l, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _apply_saturation(
        self, image: np.ndarray, srange: Tuple[float, float], intensity: float
    ) -> np.ndarray:
        """Adjust saturation by scaling A/B channels magnitude."""
        low = 1.0 + (srange[0] - 1.0) * intensity
        high = 1.0 + (srange[1] - 1.0) * intensity
        factor = random.uniform(low, high)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        # A and B are centered at 128 in uint8 LAB
        for ch in [1, 2]:
            lab[:, :, ch] = np.clip((lab[:, :, ch] - 128) * factor + 128, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _apply_noise(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, sigma * 255, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)

    def _apply_blur(self, image: np.ndarray, sigma_min: float, sigma_max: float) -> np.ndarray:
        """Apply random Gaussian blur."""
        sigma = random.uniform(sigma_min, sigma_max)
        if sigma < 0.3:
            return image
        ksize = int(sigma * 6) | 1  # Ensure odd kernel size
        ksize = max(ksize, 3)
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    def _apply_histogram_match(
        self, image: np.ndarray, ref_stats: dict, strength: float
    ) -> np.ndarray:
        """Partially match image histogram to reference statistics.

        Shifts mean and scales std toward reference values, blended by strength.
        """
        if "channel_means_lab" not in ref_stats or "channel_stds_lab" not in ref_stats:
            return image

        ref_means = ref_stats["channel_means_lab"]
        ref_stds = ref_stats["channel_stds_lab"]

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        for ch in range(3):
            ch_data = lab[:, :, ch]
            src_mean = ch_data.mean()
            src_std = max(ch_data.std(), 1e-6)

            # Target mean/std: blend between source and reference
            target_mean = src_mean + (ref_means[ch] - src_mean) * strength
            target_std = src_std + (ref_stds[ch] - src_std) * strength

            # Normalize and re-scale
            lab[:, :, ch] = np.clip(
                (ch_data - src_mean) * (target_std / src_std) + target_mean, 0, 255
            )

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
