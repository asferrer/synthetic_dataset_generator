"""
Domain Gap Metrics Engine
=========================
Computes domain gap metrics (RADIO-MMD, FD-RADIO, FID, KID, CMMD, PRDC,
color distribution) between synthetic and real image sets using C-RADIOv4
embeddings (primary), Inception v3 features (legacy), CLIP embeddings,
and LAB color space analysis.
"""

import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
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

# C-RADIOv4 backbone (NVIDIA)
RADIO_MODEL_NAME = "nvidia/C-RADIOv4-H"
RADIO_EMBED_DIM = 1280
RADIO_INPUT_SIZE = 512


def _get_radio_batch_size() -> int:
    """Choose RADIO batch size based on FREE GPU VRAM (not total).

    Other services (SAM3, depth models) may hold VRAM while RADIO runs,
    so we check actual availability to avoid CUDA OOM / thrashing.
    """
    if not torch.cuda.is_available():
        return 2
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    free_gb = free_mem / (1024 ** 3)
    logger.debug("GPU VRAM: {:.1f} GB free / {:.1f} GB total", free_gb, total_mem / (1024 ** 3))
    if free_gb >= 6:
        return 8
    if free_gb >= 4:
        return 4
    # < 4 GB free — other models (SAM3, depth) likely resident
    return 2


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
        self._clip_model = None
        self._clip_preprocess = None
        self._radio_model = None
        self._radio_processor = None
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
        compute_radio_mmd: bool = True,
        compute_fd_radio: bool = True,
        compute_fid: bool = False,
        compute_kid: bool = False,
        compute_color: bool = True,
        compute_cmmd: bool = False,
        compute_prdc: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> MetricsResult:
        """
        Compute domain gap metrics between synthetic and real image sets.

        Processing order (sequential VRAM management):
            1. RADIO (3-5GB) → extract features → compute RADIO-MMD + FD-RADIO → unload
            2. CLIP (2GB, if CMMD enabled) → extract features → compute CMMD → unload
            3. Inception (0.5GB, if FID/KID enabled) → extract features → compute FID/KID

        PRDC uses RADIO features when available (better than Inception), falling
        back to Inception features if RADIO is not computed.

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

        radio_mmd_score: Optional[float] = None
        fd_radio_score: Optional[float] = None
        fid_score: Optional[float] = None
        kid_score: Optional[float] = None
        kid_std: Optional[float] = None
        cmmd_score: Optional[float] = None
        prdc_result: Optional[Dict[str, float]] = None
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

        # Features that may be shared across metrics (raw for FD, normalized for MMD/PRDC)
        radio_syn_raw: Optional[np.ndarray] = None
        radio_real_raw: Optional[np.ndarray] = None
        radio_syn_norm: Optional[np.ndarray] = None
        radio_real_norm: Optional[np.ndarray] = None

        # --- RADIO: primary backbone (C-RADIOv4, 1280-dim) ---
        need_radio = compute_radio_mmd or compute_fd_radio
        if need_radio:
            logger.info("Extracting C-RADIOv4 features...")
            _cb("extracting_radio_synthetic", 0.0)
            radio_syn_raw, radio_syn_norm = self._extract_radio_features(
                synthetic_dir, max_images,
                progress_callback=lambda p: _cb("extracting_radio_synthetic", p),
            )
            # Free intermediate CUDA tensors between extraction passes
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            _cb("extracting_radio_real", 0.0)
            radio_real_raw, radio_real_norm = self._extract_radio_features(
                real_dir, max_images,
                progress_callback=lambda p: _cb("extracting_radio_real", p),
            )
            # Unload RADIO to free VRAM before loading other models
            self._unload_radio()

            if compute_radio_mmd:
                logger.info("Computing RADIO-MMD...")
                _cb("computing_radio_mmd", 0.0)
                radio_mmd_score = self.compute_cmmd(radio_syn_norm, radio_real_norm)
                _cb("computing_radio_mmd", 1.0)
                logger.info("RADIO-MMD score: {:.6f}", radio_mmd_score)

            if compute_fd_radio:
                logger.info("Computing FD-RADIO (raw features)...")
                _cb("computing_fd_radio", 0.0)
                # Use RAW (unnormalized) features for Frechet Distance —
                # L2-norm compresses magnitudes and produces artificially low FD values
                fd_syn, fd_real = self._reduce_dimensionality(
                    radio_syn_raw, radio_real_raw, target_dims=256
                )
                fd_radio_score = self.compute_fid(fd_syn, fd_real)
                _cb("computing_fd_radio", 1.0)
                logger.info("FD-RADIO score: {:.4f}", fd_radio_score)

        # --- PRDC: prefer RADIO normalized features, fallback to Inception ---
        if compute_prdc and radio_syn_norm is not None and radio_real_norm is not None:
            if len(radio_syn_norm) >= 10 and len(radio_real_norm) >= 10:
                logger.info("Computing PRDC with RADIO features...")
                _cb("computing_prdc", 0.0)
                prdc_result = self.compute_prdc(radio_syn_norm, radio_real_norm)
                _cb("computing_prdc", 1.0)
                logger.info(
                    "PRDC - P={:.3f} R={:.3f} D={:.3f} C={:.3f}",
                    prdc_result["precision"],
                    prdc_result["recall"],
                    prdc_result["density"],
                    prdc_result["coverage"],
                )

        # --- CMMD: uses CLIP features (separate model, load/unload) ---
        if compute_cmmd:
            logger.info("Computing CMMD with CLIP features...")
            _cb("extracting_clip_synthetic", 0.0)
            syn_clip = self._extract_clip_features(
                synthetic_dir, max_images,
                progress_callback=lambda p: _cb("extracting_clip_synthetic", p),
            )
            _cb("extracting_clip_real", 0.0)
            real_clip = self._extract_clip_features(
                real_dir, max_images,
                progress_callback=lambda p: _cb("extracting_clip_real", p),
            )
            # Unload CLIP to free VRAM before loading Inception
            self._unload_clip()

            _cb("computing_cmmd", 0.0)
            cmmd_score = self.compute_cmmd(syn_clip, real_clip)
            _cb("computing_cmmd", 1.0)
            logger.info("CMMD score: {:.6f}", cmmd_score)

        # --- Inception features: legacy FID, KID, and fallback PRDC ---
        synthetic_features: Optional[np.ndarray] = None
        real_features: Optional[np.ndarray] = None
        need_inception = compute_fid or compute_kid or (compute_prdc and prdc_result is None)

        if need_inception:
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

            # Fallback PRDC with Inception features if RADIO was not used
            if compute_prdc and prdc_result is None:
                if len(synthetic_features) >= 10 and len(real_features) >= 10:
                    logger.info("Computing PRDC with Inception features (fallback)...")
                    _cb("computing_prdc", 0.0)
                    prdc_result = self.compute_prdc(synthetic_features, real_features)
                    _cb("computing_prdc", 1.0)
                    logger.info(
                        "PRDC - P={:.3f} R={:.3f} D={:.3f} C={:.3f}",
                        prdc_result["precision"],
                        prdc_result["recall"],
                        prdc_result["density"],
                        prdc_result["coverage"],
                    )

        if compute_color:
            logger.info("Computing color distribution metrics...")
            _cb("computing_color", 0.0)
            color_dist = self.compute_color_distribution(
                synthetic_dir, real_dir, max_images
            )
            _cb("computing_color", 1.0)
            logger.info("Color EMD total: {:.4f}", color_dist.emd_total)

        # Sharpness metrics (CPU-only, no VRAM impact)
        sharpness_metrics: Optional[Dict[str, float]] = None
        if compute_color:  # Piggyback on the same gate — always compute alongside color
            logger.info("Computing sharpness metrics...")
            _cb("computing_sharpness", 0.0)
            sharpness_metrics = self._compute_sharpness_metrics(
                synthetic_dir, real_dir, max_images
            )
            _cb("computing_sharpness", 1.0)

        sharpness_ratio_val = (
            sharpness_metrics["sharpness_ratio"] if sharpness_metrics else None
        )

        # Compute overall gap score (v2.0: recalibrated thresholds + sqrt + sharpness)
        color_emd = color_dist.emd_total if color_dist is not None else None
        overall_score, gap_level = self._compute_gap_score(
            fid=fid_score,
            kid=kid_score,
            color_emd=color_emd,
            cmmd=cmmd_score,
            radio_mmd=radio_mmd_score,
            fd_radio=fd_radio_score,
            sharpness_ratio=sharpness_ratio_val,
        )

        # Diagnostics: sample size warning and PCA info
        sample_warning: Optional[str] = None
        pca_was_applied = False
        pca_dims_used: Optional[int] = None
        n_min = min(synthetic_count, real_count)
        if n_min < 50 and (compute_fd_radio or compute_fid):
            sample_warning = (
                f"Only {n_min} images per set — Frechet Distance estimates have high "
                f"variance. Use at least 100 images for reliable gap scoring."
            )
        if radio_syn_raw is not None and n_min < radio_syn_raw.shape[1] / 4:
            pca_was_applied = True
            pca_dims_used = min(256, n_min - 1) if n_min > 2 else None

        processing_time = (time.time() - start_time) * 1000
        logger.info(
            "Metrics complete in {:.0f}ms - gap_score={:.1f} ({})",
            processing_time,
            overall_score,
            gap_level.value,
        )

        return MetricsResult(
            radio_mmd_score=radio_mmd_score,
            fd_radio_score=fd_radio_score,
            fid_score=fid_score,
            kid_score=kid_score,
            kid_std=kid_std,
            cmmd_score=cmmd_score,
            precision=prdc_result["precision"] if prdc_result else None,
            recall=prdc_result["recall"] if prdc_result else None,
            density=prdc_result["density"] if prdc_result else None,
            coverage=prdc_result["coverage"] if prdc_result else None,
            color_distribution=color_dist,
            overall_gap_score=overall_score,
            gap_level=gap_level,
            synthetic_count=synthetic_count,
            real_count=real_count,
            processing_time_ms=processing_time,
            sample_size_warning=sample_warning,
            pca_applied=pca_was_applied,
            pca_dims=pca_dims_used,
            sharpness_ratio=sharpness_ratio_val,
            synthetic_sharpness=(
                sharpness_metrics["synthetic_sharpness"] if sharpness_metrics else None
            ),
            real_sharpness=(
                sharpness_metrics["real_sharpness"] if sharpness_metrics else None
            ),
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

    def compute_cmmd(
        self,
        synthetic_features: np.ndarray,
        real_features: np.ndarray,
        sigma: Optional[float] = None,
    ) -> float:
        """
        Compute CLIP Maximum Mean Discrepancy (CMMD).

        Uses squared MMD with Gaussian RBF kernel on CLIP embeddings.
        More robust than FID: no normality assumption, unbiased estimator.

        Args:
            synthetic_features: CLIP embeddings for synthetic images (N, D).
            real_features: CLIP embeddings for real images (N, D).
            sigma: RBF bandwidth. If None, uses the median heuristic.

        Returns:
            CMMD score (lower is better, 0 means identical distributions).
        """
        x = synthetic_features.astype(np.float64)
        y = real_features.astype(np.float64)

        # Median heuristic for bandwidth selection
        if sigma is None:
            # Compute pairwise distances on a subsample for efficiency
            n_sub = min(500, len(x), len(y))
            rng = np.random.RandomState(42)
            x_sub = x[rng.choice(len(x), n_sub, replace=len(x) < n_sub)]
            y_sub = y[rng.choice(len(y), n_sub, replace=len(y) < n_sub)]
            combined = np.concatenate([x_sub, y_sub], axis=0)
            # Squared pairwise distances
            sq_norms = np.sum(combined ** 2, axis=1)
            dists_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * combined @ combined.T
            # Median of positive distances
            median_dist = np.median(dists_sq[dists_sq > 0])
            sigma = np.sqrt(median_dist / 2.0)

        gamma = 1.0 / (2.0 * sigma ** 2)

        # Compute kernel matrices
        def _rbf_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            sq_a = np.sum(a ** 2, axis=1)
            sq_b = np.sum(b ** 2, axis=1)
            dists_sq = sq_a[:, None] + sq_b[None, :] - 2 * a @ b.T
            return np.exp(-gamma * np.maximum(dists_sq, 0))

        k_xx = _rbf_kernel(x, x)
        k_yy = _rbf_kernel(y, y)
        k_xy = _rbf_kernel(x, y)

        m = len(x)
        n = len(y)

        # Unbiased MMD^2 estimator
        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)

        mmd2 = (
            np.sum(k_xx) / (m * (m - 1))
            + np.sum(k_yy) / (n * (n - 1))
            - 2.0 * np.sum(k_xy) / (m * n)
        )

        return max(0.0, float(mmd2))

    def compute_prdc(
        self,
        synthetic_features: np.ndarray,
        real_features: np.ndarray,
        nearest_k: int = 5,
    ) -> Dict[str, float]:
        """
        Compute Precision, Recall, Density, and Coverage (PRDC).

        Decomposes generation quality into four independent axes:
        - Precision: fraction of synthetic samples within real manifold (fidelity)
        - Recall: fraction of real manifold covered by synthetic (diversity)
        - Density: robust fidelity via real-neighborhood counting
        - Coverage: robust diversity via real-neighbor coverage

        Based on Naeem et al. (ICML 2020).

        Args:
            synthetic_features: Inception features for synthetic images (N, 2048).
            real_features: Inception features for real images (N, 2048).
            nearest_k: Number of nearest neighbors for manifold estimation.

        Returns:
            Dictionary with precision, recall, density, coverage (all 0-1 range).
        """
        # Ensure enough samples for nearest neighbor computation
        effective_k = min(nearest_k, len(real_features) - 1, len(synthetic_features) - 1)
        if effective_k < 1:
            logger.warning("Not enough samples for PRDC computation")
            return {"precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0}

        # Fit nearest neighbors on real and synthetic features
        nn_real = NearestNeighbors(n_neighbors=effective_k + 1, algorithm="auto")
        nn_real.fit(real_features)

        nn_synth = NearestNeighbors(n_neighbors=effective_k + 1, algorithm="auto")
        nn_synth.fit(synthetic_features)

        # Get k-th nearest neighbor distances (radius of manifold balls)
        # Index 0 is the point itself (distance=0), so k-th neighbor is at index k
        real_nn_dists, _ = nn_real.kneighbors(real_features)
        real_radii = real_nn_dists[:, effective_k]  # (N_real,)

        synth_nn_dists, _ = nn_synth.kneighbors(synthetic_features)
        synth_radii = synth_nn_dists[:, effective_k]  # (N_synth,)

        # Distance from each synthetic sample to its nearest real neighbor
        dist_synth_to_real, idx_synth_to_real = nn_real.kneighbors(
            synthetic_features, n_neighbors=1
        )
        dist_synth_to_real = dist_synth_to_real[:, 0]  # (N_synth,)
        idx_synth_to_real = idx_synth_to_real[:, 0]

        # Distance from each real sample to its nearest synthetic neighbor
        dist_real_to_synth, _ = nn_synth.kneighbors(real_features, n_neighbors=1)
        dist_real_to_synth = dist_real_to_synth[:, 0]  # (N_real,)

        # Precision: fraction of synthetic in real manifold
        precision = float(np.mean(
            dist_synth_to_real <= real_radii[idx_synth_to_real]
        ))

        # Recall: fraction of real covered by synthetic manifold
        dist_real_to_synth_nn, idx_real_to_synth = nn_synth.kneighbors(
            real_features, n_neighbors=1
        )
        idx_real_to_synth = idx_real_to_synth[:, 0]
        recall = float(np.mean(
            dist_real_to_synth <= synth_radii[idx_real_to_synth]
        ))

        # Density: average number of real-neighbor balls containing each synthetic point
        # For each synthetic sample, count how many real samples have it within their radius
        dist_synth_to_real_all, idx_synth_to_real_all = nn_real.kneighbors(
            synthetic_features, n_neighbors=effective_k
        )
        density_counts = np.sum(
            dist_synth_to_real_all <= real_radii[idx_synth_to_real_all],
            axis=1,
        )
        density = float(np.mean(density_counts) / effective_k)

        # Coverage: fraction of real samples with at least one synthetic neighbor within radius
        dist_real_to_synth_all, _ = nn_synth.kneighbors(
            real_features, n_neighbors=1
        )
        coverage = float(np.mean(
            dist_real_to_synth_all[:, 0] <= real_radii
        ))

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "density": round(density, 4),
            "coverage": round(coverage, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers — dimensionality reduction
    # ------------------------------------------------------------------

    def _reduce_dimensionality(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        target_dims: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply PCA dimensionality reduction for stable Frechet Distance computation.

        When the number of samples is much smaller than the feature dimensionality,
        the covariance matrix is rank-deficient and sqrtm() becomes numerically
        unstable.  PCA projects both sets into a lower-dimensional subspace where
        the covariance estimate is well-conditioned.

        Only reduces if n_samples < n_features / 4.  Otherwise returns inputs
        unchanged.
        """
        from sklearn.decomposition import PCA

        n_samples = min(len(features_a), len(features_b))
        n_features = features_a.shape[1]

        if n_samples >= n_features / 4:
            return features_a, features_b

        effective_dims = min(target_dims, n_samples - 1)
        if effective_dims < 2:
            logger.warning(
                "Too few samples ({}) for PCA reduction, skipping", n_samples
            )
            return features_a, features_b

        combined = np.concatenate([features_a, features_b], axis=0)
        pca = PCA(n_components=effective_dims, random_state=42)
        pca.fit(combined)

        reduced_a = pca.transform(features_a)
        reduced_b = pca.transform(features_b)

        logger.info(
            "PCA reduction: {} -> {} dims (explained variance: {:.1%}, samples: {})",
            n_features,
            effective_dims,
            sum(pca.explained_variance_ratio_),
            n_samples,
        )

        return reduced_a, reduced_b

    # ------------------------------------------------------------------
    # Internal helpers — RADIO
    # ------------------------------------------------------------------

    def _load_radio(self) -> None:
        """Load C-RADIOv4-H model for feature extraction. Lazy loading."""
        if self._radio_model is not None:
            return

        logger.info("Loading C-RADIOv4-H model...")
        start_time = time.time()
        try:
            from transformers import AutoModel, CLIPImageProcessor

            model = AutoModel.from_pretrained(
                RADIO_MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            model.eval()
            model.to(self._device)
            self._radio_model = model

            self._radio_processor = CLIPImageProcessor.from_pretrained(
                RADIO_MODEL_NAME,
            )

            load_time = time.time() - start_time
            logger.info("C-RADIOv4-H loaded in {:.2f}s on {}", load_time, self._device)
        except Exception as e:
            logger.error("Failed to load C-RADIOv4-H: {}", e)
            raise

    def _unload_radio(self) -> None:
        """Unload RADIO model to free VRAM."""
        if self._radio_model is not None:
            del self._radio_model
            del self._radio_processor
            self._radio_model = None
            self._radio_processor = None
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info("C-RADIOv4-H model unloaded")

    @torch.no_grad()
    def _extract_radio_features(
        self,
        image_dir: str,
        max_images: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 1280-dimensional summary features from C-RADIOv4-H.

        Returns two versions of the features:
            - raw (unnormalized): for Frechet Distance (needs real magnitudes)
            - L2-normalized: for MMD/PRDC (works best with cosine/RBF kernels)

        Args:
            image_dir: Directory containing images.
            max_images: Maximum number of images to process.

        Returns:
            Tuple of (raw_features, normalized_features), each shape (N, 1280).
        """
        from PIL import Image

        self._load_radio()
        image_paths = _list_images(image_dir, max_images)
        all_raw_features: List[np.ndarray] = []
        all_norm_features: List[np.ndarray] = []
        batch_pixels: List[torch.Tensor] = []
        skipped = 0
        radio_batch_size = _get_radio_batch_size()
        logger.debug("RADIO batch size: {} (VRAM: {:.1f} GB)", radio_batch_size,
                      torch.cuda.get_device_properties(0).total_memory / (1024**3)
                      if torch.cuda.is_available() else 0)

        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(str(img_path)).convert("RGB")
                inputs = self._radio_processor(
                    images=img,
                    return_tensors="pt",
                    do_resize=True,
                    size={"height": RADIO_INPUT_SIZE, "width": RADIO_INPUT_SIZE},
                )
                batch_pixels.append(inputs["pixel_values"].squeeze(0))
            except Exception as e:
                logger.warning("Error processing {} for RADIO: {}", img_path, e)
                skipped += 1
                continue

            if len(batch_pixels) >= radio_batch_size or i == len(image_paths) - 1:
                if batch_pixels:
                    batch_tensor = torch.stack(batch_pixels).to(
                        self._device, dtype=torch.float16
                    )
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self._device.type == "cuda"):
                        output = self._radio_model(batch_tensor)

                    # C-RADIOv4 returns (summary, spatial_features)
                    if isinstance(output, tuple):
                        summary = output[0]
                    else:
                        summary = output.last_hidden_state[:, 0]

                    summary_float = summary.float()
                    # Raw features preserve magnitude (for Frechet Distance)
                    all_raw_features.append(summary_float.cpu().numpy())
                    # L2-normalized features (for MMD with RBF kernel, PRDC)
                    all_norm_features.append(
                        F.normalize(summary_float, dim=-1).cpu().numpy()
                    )
                    batch_pixels = []
                    if progress_callback:
                        progress_callback((i + 1) / len(image_paths))

        if skipped > 0:
            logger.warning(
                "Skipped {}/{} images for RADIO in {}",
                skipped, len(image_paths), image_dir,
            )

        if not all_raw_features:
            raise ValueError(f"No valid images for RADIO from: {image_dir}")

        return (
            np.concatenate(all_raw_features, axis=0),
            np.concatenate(all_norm_features, axis=0),
        )

    # ------------------------------------------------------------------
    # Internal helpers — CLIP
    # ------------------------------------------------------------------

    def _load_clip(self) -> None:
        """Load CLIP ViT-L/14 model for CMMD computation. Lazy loading."""
        if self._clip_model is not None:
            return

        logger.info("Loading CLIP ViT-L/14 for CMMD...")
        start_time = time.time()
        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device=self._device,
            )
            model.eval()
            self._clip_model = model
            self._clip_preprocess = preprocess

            load_time = time.time() - start_time
            logger.info("CLIP loaded in {:.2f}s on {}", load_time, self._device)
        except Exception as e:
            logger.error("Failed to load CLIP: {}", e)
            raise

    def _unload_clip(self) -> None:
        """Unload CLIP model to free VRAM."""
        if self._clip_model is not None:
            del self._clip_model
            del self._clip_preprocess
            self._clip_model = None
            self._clip_preprocess = None
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info("CLIP model unloaded")

    @torch.no_grad()
    def _extract_clip_features(
        self,
        image_dir: str,
        max_images: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """
        Extract CLIP embeddings for images in a directory.

        Args:
            image_dir: Directory containing images.
            max_images: Maximum number of images to process.

        Returns:
            Feature array of shape (N, embed_dim).
        """
        from PIL import Image

        self._load_clip()
        image_paths = _list_images(image_dir, max_images)
        all_features: List[np.ndarray] = []
        batch: List[torch.Tensor] = []
        skipped = 0

        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(str(img_path)).convert("RGB")
                tensor = self._clip_preprocess(img)
                batch.append(tensor)
            except Exception as e:
                logger.warning("Error processing {} for CLIP: {}", img_path, e)
                skipped += 1
                continue

            if len(batch) >= BATCH_SIZE or i == len(image_paths) - 1:
                if batch:
                    batch_tensor = torch.stack(batch).to(self._device)
                    features = self._clip_model.encode_image(batch_tensor)
                    # L2 normalize embeddings
                    features = F.normalize(features, dim=-1)
                    all_features.append(features.cpu().numpy())
                    batch = []
                    if progress_callback:
                        progress_callback((i + 1) / len(image_paths))

        if skipped > 0:
            logger.warning(
                "Skipped {}/{} images for CLIP in {}",
                skipped, len(image_paths), image_dir,
            )

        if not all_features:
            raise ValueError(f"No valid images for CLIP from: {image_dir}")

        return np.concatenate(all_features, axis=0)

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

    # ------------------------------------------------------------------
    # Internal helpers — perceptual quality
    # ------------------------------------------------------------------

    def _compute_sharpness_metrics(
        self,
        synthetic_dir: str,
        real_dir: str,
        max_images: int,
    ) -> Dict[str, float]:
        """
        Compute Laplacian variance (sharpness proxy) for both image sets.

        Laplacian variance measures edge density/sharpness.  Blurry images
        (common in bad compositing) will have significantly lower variance
        than real photographs.

        Returns:
            Dict with synthetic_sharpness, real_sharpness, sharpness_ratio.
        """
        def _laplacian_variances(image_paths: List[Path]) -> List[float]:
            variances = []
            for p in image_paths:
                try:
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        lap = cv2.Laplacian(img, cv2.CV_64F)
                        variances.append(float(lap.var()))
                except Exception:
                    continue
            return variances

        syn_images = _list_images(synthetic_dir, max_images)
        real_images = _list_images(real_dir, max_images)

        syn_vars = _laplacian_variances(syn_images)
        real_vars = _laplacian_variances(real_images)

        syn_sharp = float(np.mean(syn_vars)) if syn_vars else 0.0
        real_sharp = float(np.mean(real_vars)) if real_vars else 0.0
        ratio = syn_sharp / real_sharp if real_sharp > 0 else 0.0

        logger.info(
            "Sharpness — synthetic: {:.1f}, real: {:.1f}, ratio: {:.3f}",
            syn_sharp, real_sharp, ratio,
        )

        return {
            "synthetic_sharpness": round(syn_sharp, 2),
            "real_sharpness": round(real_sharp, 2),
            "sharpness_ratio": round(ratio, 4),
        }

    def _compute_gap_score(
        self,
        fid: Optional[float] = None,
        kid: Optional[float] = None,
        color_emd: Optional[float] = None,
        cmmd: Optional[float] = None,
        radio_mmd: Optional[float] = None,
        fd_radio: Optional[float] = None,
        sharpness_ratio: Optional[float] = None,
    ) -> Tuple[float, GapLevel]:
        """
        Compute a normalized 0-100 gap score from individual metrics.

        Uses sqrt scaling for more uniform sensitivity across the value range
        (linear scaling under-weights moderate gaps and over-weights extreme ones).

        Metrics version 2.0 — calibrated for raw (unnormalized) RADIO features
        for FD-RADIO and L2-normalized features for RADIO-MMD.

        When RADIO metrics are present (recommended):
            - RADIO-MMD: 0-30pts, sqrt scaling, clip at 0.5
            - FD-RADIO:  0-25pts, sqrt scaling, clip at 200 (raw features)
            - Color EMD: 0-15pts, linear, clip at 60
            - Sharpness: 0-10pts, linear, clip at 1.0 deviation from 1.0
            - Legacy FID: 0-10pts (if present)
            - Legacy KID: 0-10pts (if present)

        Legacy mode (no RADIO, has CMMD):
            - CMMD: 0-30pts, FID: 0-30pts, KID: 0-20pts, Color: 0-20pts

        Legacy mode (no RADIO, no CMMD):
            - FID: 0-50pts, KID: 0-30pts, Color: 0-20pts

        Gap levels:
            - < 20: LOW
            - 20-45: MEDIUM
            - 45-70: HIGH
            - > 70: CRITICAL
        """
        total_score = 0.0
        max_possible = 0.0

        has_radio = radio_mmd is not None or fd_radio is not None

        if has_radio:
            # --- RADIO-primary scoring (v2.0: recalibrated thresholds + sqrt) ---

            # RADIO-MMD on L2-normalized features: typical good=0.001-0.01, bad=0.05-0.5+
            if radio_mmd is not None:
                radio_mmd_clamped = min(radio_mmd, 0.5)
                total_score += math.sqrt(radio_mmd_clamped / 0.5) * 30.0
                max_possible += 30.0

            # FD-RADIO on raw features: typical good=5-30, bad=100-500+
            if fd_radio is not None:
                fd_radio_clamped = min(fd_radio, 200.0)
                total_score += math.sqrt(fd_radio_clamped / 200.0) * 25.0
                max_possible += 25.0

            # Color EMD: already well-calibrated, keep linear
            if color_emd is not None:
                emd_clamped = min(color_emd, 60.0)
                total_score += (emd_clamped / 60.0) * 15.0
                max_possible += 15.0

            # Sharpness ratio penalty: deviation from 1.0 (perfectly matched)
            if sharpness_ratio is not None:
                deviation = abs(1.0 - sharpness_ratio)
                sharp_clamped = min(deviation, 1.0)
                total_score += sharp_clamped * 10.0
                max_possible += 10.0

            # Legacy metrics get reduced weight when RADIO is primary
            if fid is not None:
                fid_clamped = min(fid, 300.0)
                total_score += math.sqrt(fid_clamped / 300.0) * 10.0
                max_possible += 10.0

            if kid is not None:
                kid_clamped = min(kid, 0.1)
                total_score += math.sqrt(kid_clamped / 0.1) * 10.0
                max_possible += 10.0
        else:
            # --- Legacy scoring (no RADIO) ---
            if cmmd is not None:
                cmmd_clamped = min(cmmd, 2.0)
                total_score += (cmmd_clamped / 2.0) * 30.0
                max_possible += 30.0

            if fid is not None:
                fid_clamped = min(fid, 300.0)
                fid_max_pts = 30.0 if cmmd is not None else 50.0
                total_score += (fid_clamped / 300.0) * fid_max_pts
                max_possible += fid_max_pts

            if kid is not None:
                kid_clamped = min(kid, 0.1)
                kid_max_pts = 20.0 if cmmd is not None else 30.0
                total_score += (kid_clamped / 0.1) * kid_max_pts
                max_possible += kid_max_pts

            if color_emd is not None:
                emd_clamped = min(color_emd, 60.0)
                total_score += (emd_clamped / 60.0) * 20.0
                max_possible += 20.0

            # Sharpness applies in legacy mode too
            if sharpness_ratio is not None:
                deviation = abs(1.0 - sharpness_ratio)
                sharp_clamped = min(deviation, 1.0)
                total_score += sharp_clamped * 10.0
                max_possible += 10.0

        # Scale to 0-100
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
