"""
Domain Gap Metrics Engine
=========================
Computes domain gap metrics (FID, KID, color distribution) between
synthetic and real image sets using Inception v3 features and LAB
color space analysis.
"""

import os
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from scipy.stats import wasserstein_distance
from torchvision import models, transforms
from loguru import logger

from app.models.schemas import (
    MetricsResult,
    ColorDistribution,
    ChannelStats,
    GapLevel,
)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Inception input size
INCEPTION_SIZE = 299

# Batch size for GPU processing
BATCH_SIZE = 32


def _list_images(directory: str, max_images: int) -> List[Path]:
    """List image files in a directory, up to max_images."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    images = []
    for entry in sorted(dir_path.rglob("*")):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(entry)
            if len(images) >= max_images:
                break

    if not images:
        raise ValueError(f"No valid images found in: {directory}")

    return images


class MetricsEngine:
    """
    Computes domain gap metrics between synthetic and real image sets.

    Supports Frechet Inception Distance (FID), Kernel Inception Distance (KID),
    and LAB color distribution comparison with Earth Mover's Distance.
    """

    def __init__(self, use_gpu: bool = True) -> None:
        """
        Initialize the metrics engine with lazy model loading.

        Args:
            use_gpu: Whether to use GPU acceleration if available.
        """
        self._model: Optional[torch.nn.Module] = None
        self._device: torch.device = torch.device("cpu")
        self._use_gpu = use_gpu

        if use_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
            logger.info(
                "MetricsEngine initialized with GPU: {}",
                torch.cuda.get_device_name(0),
            )
        else:
            logger.info("MetricsEngine initialized on CPU")

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INCEPTION_SIZE, INCEPTION_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        synthetic_dir: str,
        real_dir: str,
        max_images: int = 100,
        compute_fid: bool = True,
        compute_kid: bool = True,
        compute_color: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> MetricsResult:
        """
        Compute domain gap metrics between synthetic and real image sets.

        Args:
            synthetic_dir: Path to directory containing synthetic images.
            real_dir: Path to directory containing real images.
            max_images: Maximum number of images to sample from each set.
            compute_fid: Whether to compute Frechet Inception Distance.
            compute_kid: Whether to compute Kernel Inception Distance.
            compute_color: Whether to compute color distribution metrics.

        Returns:
            MetricsResult with all computed metrics and an overall gap score.
        """
        start_time = time.time()
        logger.info(
            "Computing metrics: synthetic={}, real={}, max_images={}",
            synthetic_dir,
            real_dir,
            max_images,
        )

        fid_score: Optional[float] = None
        kid_score: Optional[float] = None
        kid_std: Optional[float] = None
        color_dist: Optional[ColorDistribution] = None

        # Count images actually used
        synthetic_images = _list_images(synthetic_dir, max_images)
        real_images = _list_images(real_dir, max_images)
        synthetic_count = len(synthetic_images)
        real_count = len(real_images)

        # Shorthand for progress reporting
        def _cb(phase: str, fraction: float) -> None:
            if progress_callback:
                progress_callback(phase, fraction)

        # Extract Inception features if needed for FID or KID
        synthetic_features: Optional[np.ndarray] = None
        real_features: Optional[np.ndarray] = None

        if compute_fid or compute_kid:
            logger.info("Extracting Inception features...")
            _cb("extracting_features_synthetic", 0.0)
            synthetic_features = self._extract_inception_features(
                synthetic_dir, max_images,
                progress_callback=lambda p: _cb("extracting_features_synthetic", p),
            )
            _cb("extracting_features_real", 0.0)
            real_features = self._extract_inception_features(
                real_dir, max_images,
                progress_callback=lambda p: _cb("extracting_features_real", p),
            )

            if compute_fid:
                logger.info("Computing FID...")
                _cb("computing_fid", 0.0)
                fid_score = self.compute_fid(synthetic_features, real_features)
                _cb("computing_fid", 1.0)
                logger.info("FID score: {:.4f}", fid_score)

            if compute_kid:
                logger.info("Computing KID...")
                _cb("computing_kid", 0.0)
                kid_score, kid_std = self.compute_kid(
                    synthetic_features, real_features
                )
                _cb("computing_kid", 1.0)
                logger.info("KID score: {:.6f} +/- {:.6f}", kid_score, kid_std)

        if compute_color:
            logger.info("Computing color distribution metrics...")
            _cb("computing_color", 0.0)
            color_dist = self.compute_color_distribution(
                synthetic_dir, real_dir, max_images
            )
            _cb("computing_color", 1.0)
            logger.info("Color EMD total: {:.4f}", color_dist.emd_total)

        # Compute overall gap score
        color_emd = color_dist.emd_total if color_dist is not None else None
        overall_score, gap_level = self._compute_gap_score(
            fid_score, kid_score, color_emd
        )

        processing_time = (time.time() - start_time) * 1000
        logger.info(
            "Metrics complete in {:.0f}ms - gap_score={:.1f} ({})",
            processing_time,
            overall_score,
            gap_level.value,
        )

        return MetricsResult(
            fid_score=fid_score,
            kid_score=kid_score,
            kid_std=kid_std,
            color_distribution=color_dist,
            overall_gap_score=overall_score,
            gap_level=gap_level,
            synthetic_count=synthetic_count,
            real_count=real_count,
            processing_time_ms=processing_time,
        )

    def compute_fid(
        self,
        synthetic_features: np.ndarray,
        real_features: np.ndarray,
    ) -> float:
        """
        Compute the Frechet Inception Distance between two feature sets.

        FID = ||mu_s - mu_r||^2 + Tr(Sigma_s + Sigma_r - 2 * sqrt(Sigma_s @ Sigma_r))

        Args:
            synthetic_features: Inception pool3 features for synthetic images (N, 2048).
            real_features: Inception pool3 features for real images (N, 2048).

        Returns:
            FID score (lower is better, 0 means identical distributions).
        """
        mu_s = np.mean(synthetic_features, axis=0)
        mu_r = np.mean(real_features, axis=0)
        sigma_s = np.cov(synthetic_features, rowvar=False)
        sigma_r = np.cov(real_features, rowvar=False)

        diff = mu_s - mu_r
        mean_diff_sq = np.dot(diff, diff)

        # Matrix square root of product of covariances
        covmean, _ = linalg.sqrtm(sigma_s @ sigma_r, disp=False)

        # Handle numerical instabilities: discard imaginary components
        if np.iscomplexobj(covmean):
            if not np.allclose(np.imag(covmean), 0, atol=1e-3):
                logger.warning(
                    "Imaginary component in sqrtm result (max={:.4e}), discarding",
                    np.max(np.abs(np.imag(covmean))),
                )
            covmean = np.real(covmean)

        trace_term = np.trace(sigma_s + sigma_r - 2.0 * covmean)
        fid = float(mean_diff_sq + trace_term)

        # Clamp to non-negative (numerical precision can produce small negatives)
        return max(0.0, fid)

    def compute_kid(
        self,
        synthetic_features: np.ndarray,
        real_features: np.ndarray,
        subset_size: int = 100,
        num_subsets: int = 100,
    ) -> Tuple[float, float]:
        """
        Compute the Kernel Inception Distance using polynomial kernel MMD.

        Uses an unbiased MMD^2 estimator averaged over random subsets.
        Kernel: k(x, y) = ((x . y) / d + 1)^3  where d = feature_dim.

        Args:
            synthetic_features: Inception pool3 features for synthetic images (N, 2048).
            real_features: Inception pool3 features for real images (N, 2048).
            subset_size: Number of samples per subset.
            num_subsets: Number of subsets to average over.

        Returns:
            Tuple of (mean KID score, standard deviation).
        """
        n_s = len(synthetic_features)
        n_r = len(real_features)
        d = synthetic_features.shape[1]  # feature dimensionality (2048)

        # Clamp subset size to available samples
        effective_subset = min(subset_size, n_s, n_r)
        if effective_subset < subset_size:
            logger.warning(
                "Subset size reduced from {} to {} due to sample count",
                subset_size,
                effective_subset,
            )

        mmd_values: List[float] = []
        rng = np.random.RandomState(42)

        for _ in range(num_subsets):
            idx_s = rng.choice(n_s, size=effective_subset, replace=False)
            idx_r = rng.choice(n_r, size=effective_subset, replace=False)
            s = synthetic_features[idx_s]
            r = real_features[idx_r]

            # Polynomial kernel: k(x, y) = ((x . y) / d + 1)^3
            k_ss = ((s @ s.T) / d + 1.0) ** 3
            k_rr = ((r @ r.T) / d + 1.0) ** 3
            k_sr = ((s @ r.T) / d + 1.0) ** 3

            m = effective_subset

            # Unbiased MMD^2 estimator
            # sum of off-diagonal elements divided by m*(m-1)
            diag_ss = np.diag(k_ss).copy()
            diag_rr = np.diag(k_rr).copy()
            np.fill_diagonal(k_ss, 0.0)
            np.fill_diagonal(k_rr, 0.0)

            mmd2 = (
                np.sum(k_ss) / (m * (m - 1))
                + np.sum(k_rr) / (m * (m - 1))
                - 2.0 * np.sum(k_sr) / (m * m)
            )
            mmd_values.append(float(mmd2))

        kid_mean = float(np.mean(mmd_values))
        kid_std = float(np.std(mmd_values))

        return kid_mean, kid_std

    def compute_color_distribution(
        self,
        synthetic_dir: str,
        real_dir: str,
        max_images: int = 100,
    ) -> ColorDistribution:
        """
        Compare LAB color distributions between synthetic and real image sets.

        Computes per-channel histograms and Earth Mover's Distance for
        the L, A, and B channels.

        Args:
            synthetic_dir: Directory containing synthetic images.
            real_dir: Directory containing real images.
            max_images: Maximum images to sample per set.

        Returns:
            ColorDistribution with per-channel EMD and statistics.
        """
        synthetic_images = _list_images(synthetic_dir, max_images)
        real_images = _list_images(real_dir, max_images)

        # Collect LAB values
        syn_l, syn_a, syn_b = self._collect_lab_histograms(synthetic_images)
        real_l, real_a, real_b = self._collect_lab_histograms(real_images)

        # Compute Earth Mover's Distance per channel
        bin_centers = np.arange(256)
        emd_l = float(wasserstein_distance(bin_centers, bin_centers, syn_l, real_l))
        emd_a = float(wasserstein_distance(bin_centers, bin_centers, syn_a, real_a))
        emd_b = float(wasserstein_distance(bin_centers, bin_centers, syn_b, real_b))
        emd_total = emd_l + emd_a + emd_b

        # Compute per-channel stats
        syn_stats = self._compute_channel_stats(synthetic_images)
        real_stats = self._compute_channel_stats(real_images)

        return ColorDistribution(
            emd_l=emd_l,
            emd_a=emd_a,
            emd_b=emd_b,
            emd_total=emd_total,
            synthetic_stats=syn_stats,
            real_stats=real_stats,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_inception(self) -> None:
        """
        Load Inception v3 model with pool3 feature extraction.

        The model is modified to output the 2048-dimensional features
        from the average pooling layer (pool3) instead of classification logits.
        """
        if self._model is not None:
            return

        logger.info("Loading Inception v3 model...")
        start_time = time.time()

        try:
            # Load pretrained Inception v3
            inception = models.inception_v3(
                weights=models.Inception_V3_Weights.DEFAULT
            )
            inception.eval()

            # Replace the fully connected layer with identity to extract
            # the 2048-dim pool3 features directly.
            inception.fc = torch.nn.Identity()
            # Disable auxiliary logits (not needed for feature extraction)
            inception.aux_logits = False
            inception.AuxLogits = None

            inception.to(self._device)
            self._model = inception

            load_time = time.time() - start_time
            logger.info("Inception v3 loaded in {:.2f}s on {}", load_time, self._device)

        except Exception as e:
            logger.error("Failed to load Inception v3: {}", e)
            raise

    @torch.no_grad()
    def _extract_inception_features(
        self,
        image_dir: str,
        max_images: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """
        Extract 2048-dimensional pool3 features from Inception v3.

        Images are resized to 299x299 and normalized with ImageNet statistics.
        Processing is done in batches for GPU efficiency.

        Args:
            image_dir: Directory containing images.
            max_images: Maximum number of images to process.

        Returns:
            Feature array of shape (N, 2048).
        """
        self._load_inception()
        image_paths = _list_images(image_dir, max_images)

        all_features: List[np.ndarray] = []
        batch: List[torch.Tensor] = []
        skipped = 0

        for i, img_path in enumerate(image_paths):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning("Failed to read image, skipping: {}", img_path)
                    skipped += 1
                    continue

                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = self._transform(img_rgb)
                batch.append(tensor)

            except Exception as e:
                logger.warning("Error processing {}: {}", img_path, e)
                skipped += 1
                continue

            # Process batch when full or at the end
            if len(batch) >= BATCH_SIZE or i == len(image_paths) - 1:
                if batch:
                    batch_tensor = torch.stack(batch).to(self._device)
                    features = self._model(batch_tensor)
                    # Handle case where model returns InceptionOutputs namedtuple
                    if isinstance(features, tuple):
                        features = features[0]
                    all_features.append(features.cpu().numpy())
                    batch = []
                    if progress_callback:
                        progress_callback((i + 1) / len(image_paths))

        if skipped > 0:
            logger.warning(
                "Skipped {}/{} corrupted/unreadable images in {}",
                skipped,
                len(image_paths),
                image_dir,
            )

        if not all_features:
            raise ValueError(
                f"No valid images could be processed from: {image_dir}"
            )

        # Free GPU memory
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        return np.concatenate(all_features, axis=0)

    def _compute_gap_score(
        self,
        fid: Optional[float],
        kid: Optional[float],
        color_emd: Optional[float],
    ) -> Tuple[float, GapLevel]:
        """
        Compute a normalized 0-100 gap score from individual metrics.

        Scoring breakdown:
            - FID contributes 0-50 points (clipped at FID=300 for maximum)
            - KID contributes 0-30 points (clipped at KID=0.1 for maximum)
            - Color EMD contributes 0-20 points

        Gap levels:
            - < 20: LOW
            - 20-45: MEDIUM
            - 45-70: HIGH
            - > 70: CRITICAL

        Args:
            fid: FID score (None if not computed).
            kid: KID score (None if not computed).
            color_emd: Total color EMD (None if not computed).

        Returns:
            Tuple of (overall score 0-100, GapLevel).
        """
        total_score = 0.0
        max_possible = 0.0

        if fid is not None:
            fid_clamped = min(fid, 300.0)
            fid_contribution = (fid_clamped / 300.0) * 50.0
            total_score += fid_contribution
            max_possible += 50.0

        if kid is not None:
            kid_clamped = min(kid, 0.1)
            kid_contribution = (kid_clamped / 0.1) * 30.0
            total_score += kid_contribution
            max_possible += 30.0

        if color_emd is not None:
            # Normalize color EMD: a total EMD of ~60 across 3 channels is
            # a reasonable maximum for highly divergent distributions.
            emd_max = 60.0
            emd_clamped = min(color_emd, emd_max)
            emd_contribution = (emd_clamped / emd_max) * 20.0
            total_score += emd_contribution
            max_possible += 20.0

        # Scale to 0-100 if not all metrics were computed
        if max_possible > 0:
            total_score = (total_score / max_possible) * 100.0
        else:
            total_score = 0.0

        # Determine gap level
        if total_score < 20.0:
            gap_level = GapLevel.LOW
        elif total_score < 45.0:
            gap_level = GapLevel.MEDIUM
        elif total_score < 70.0:
            gap_level = GapLevel.HIGH
        else:
            gap_level = GapLevel.CRITICAL

        return round(total_score, 2), gap_level

    def _collect_lab_histograms(
        self,
        image_paths: List[Path],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect aggregated 256-bin LAB histograms across a set of images.

        Args:
            image_paths: List of image file paths.

        Returns:
            Tuple of normalized histograms (L, A, B), each shape (256,).
        """
        hist_l = np.zeros(256, dtype=np.float64)
        hist_a = np.zeros(256, dtype=np.float64)
        hist_b = np.zeros(256, dtype=np.float64)
        processed = 0

        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)

                hist_l += cv2.calcHist([l_ch], [0], None, [256], [0, 256]).flatten()
                hist_a += cv2.calcHist([a_ch], [0], None, [256], [0, 256]).flatten()
                hist_b += cv2.calcHist([b_ch], [0], None, [256], [0, 256]).flatten()
                processed += 1

            except Exception as e:
                logger.warning("Error computing LAB histogram for {}: {}", img_path, e)
                continue

        if processed == 0:
            raise ValueError("No images could be processed for LAB histograms")

        # Normalize to probability distributions
        hist_l /= hist_l.sum()
        hist_a /= hist_a.sum()
        hist_b /= hist_b.sum()

        return hist_l, hist_a, hist_b

    def _compute_channel_stats(
        self,
        image_paths: List[Path],
    ) -> dict:
        """
        Compute per-channel LAB statistics across a set of images.

        Args:
            image_paths: List of image file paths.

        Returns:
            Dictionary mapping channel name to ChannelStats.
        """
        l_values: List[float] = []
        a_values: List[float] = []
        b_values: List[float] = []

        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)

                l_values.append(float(l_ch.mean()))
                a_values.append(float(a_ch.mean()))
                b_values.append(float(b_ch.mean()))

            except Exception as e:
                logger.warning("Error computing stats for {}: {}", img_path, e)
                continue

        def _make_stats(values: List[float]) -> ChannelStats:
            arr = np.array(values)
            return ChannelStats(
                mean=float(np.mean(arr)),
                std=float(np.std(arr)),
                min=float(np.min(arr)),
                max=float(np.max(arr)),
            )

        return {
            "L": _make_stats(l_values),
            "A": _make_stats(a_values),
            "B": _make_stats(b_values),
        }
