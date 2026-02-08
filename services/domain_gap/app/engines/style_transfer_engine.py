"""
Neural Style Transfer Engine (DA-WCT)
======================================
Depth-Aware Whitening and Coloring Transform for domain gap reduction.

Transfers the visual style (colors, textures) from real reference images
to synthetic images while preserving structural content. Uses a pretrained
VGG-19 encoder for multi-scale feature extraction and applies the WCT
algorithm at each scale (relu1_1 through relu4_1) from coarse to fine.

When a depth map is provided and ``depth_guided`` is enabled, the style
transfer strength is modulated spatially: foreground objects (closer to
the camera) receive less style transfer so that fine details are preserved,
while background regions receive stronger transfer.

GPU usage: ~4 GB VRAM at 512x512 resolution.  Falls back to CPU
automatically when no CUDA device is detected.
"""

import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ImageNet normalization constants used by VGG
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Supported image extensions (consistent with other engines)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# VGG-19 layer indices (0-indexed into ``vgg19.features``)
# These correspond to the output of each relu*_1 block.
VGG19_LAYER_INDICES: Dict[str, int] = {
    "relu1_1": 1,   # conv1_1 -> relu
    "relu2_1": 6,   # conv2_1 -> relu
    "relu3_1": 11,  # conv3_1 -> relu
    "relu4_1": 20,  # conv4_1 -> relu
}

# Processing order: coarse (deep) to fine (shallow)
SCALE_ORDER = ["relu4_1", "relu3_1", "relu2_1", "relu1_1"]

# Small epsilon to prevent division by zero in eigenvalue decomposition
EPS = 1e-5


# ---------------------------------------------------------------------------
# VGG-19 Multi-Scale Feature Encoder
# ---------------------------------------------------------------------------

class _VGG19Encoder(nn.Module):
    """Thin wrapper around ``torchvision.models.vgg19`` that exposes
    intermediate feature maps at the four relu*_1 layers used by WCT.

    The encoder is frozen (no gradient computation) and used purely for
    feature extraction during inference.
    """

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        features = list(vgg.features.children())

        # Build sub-networks so we can extract features at each scale
        max_idx = max(VGG19_LAYER_INDICES.values())
        self.slices = nn.ModuleList()
        prev = 0
        for name in SCALE_ORDER[::-1]:  # build in forward order (shallow->deep)
            idx = VGG19_LAYER_INDICES[name] + 1  # +1 because slicing is exclusive
            self.slices.append(nn.Sequential(*features[prev:idx]))
            prev = idx

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale feature maps.

        Args:
            x: Input tensor of shape (1, 3, H, W), ImageNet-normalised.

        Returns:
            Dictionary mapping layer name to feature tensor.
        """
        features: Dict[str, torch.Tensor] = {}
        h = x
        for slice_module, name in zip(self.slices, SCALE_ORDER[::-1]):
            h = slice_module(h)
            features[name] = h
        return features


# ---------------------------------------------------------------------------
# StyleTransferEngine
# ---------------------------------------------------------------------------

class StyleTransferEngine:
    """Applies depth-aware neural style transfer using the WCT algorithm.

    The engine lazily loads a VGG-19 encoder on first use and supports both
    full neural WCT transfer and a lightweight colour-only mode that operates
    in LAB colour space without neural features.
    """

    def __init__(self, use_gpu: bool = True) -> None:
        """Initialise the style transfer engine.

        Args:
            use_gpu: Attempt to use CUDA if available.  Falls back to CPU
                when no GPU is detected.
        """
        self._encoder: Optional[_VGG19Encoder] = None
        self._device = torch.device("cpu")
        self._use_gpu = use_gpu

        if use_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
            logger.info(
                "StyleTransferEngine initialized with GPU: {}",
                torch.cuda.get_device_name(0),
            )
        else:
            logger.info("StyleTransferEngine initialized on CPU")

        # Preprocessing transform: resize is handled per-image, so we only
        # define normalisation here.
        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_single(
        self,
        content_path: str,
        style_dir: str,
        output_path: str,
        style_weight: float = 0.6,
        content_weight: float = 1.0,
        preserve_structure: float = 0.8,
        color_only: bool = False,
        depth_map_path: Optional[str] = None,
        depth_guided: bool = True,
        annotations_path: Optional[str] = None,
    ) -> str:
        """Apply style transfer to a single content image.

        A random style image is sampled from *style_dir* for each call.
        If ``color_only`` is True, only LAB colour statistics are
        transferred (much faster, no GPU required).

        Args:
            content_path: Path to the synthetic content image.
            style_dir: Directory containing real reference style images.
            output_path: Destination path for the styled result.
            style_weight: Blending strength toward the style (0-1).
            content_weight: Content preservation multiplier.
            preserve_structure: How much structural detail to keep (0-1).
            color_only: If True, only match colour statistics in LAB space.
            depth_map_path: Optional path to a depth map (single-channel or
                greyscale image with the same spatial dimensions as the
                content image).
            depth_guided: When True and a depth map is available, modulate
                ``style_weight`` spatially so foreground objects receive
                less transfer.
            annotations_path: If provided, the annotation file is copied
                alongside the output image (matching the output stem).

        Returns:
            The ``output_path`` that was written to.
        """
        start_time = time.time()

        # --- Load content image -----------------------------------------------
        content_bgr = cv2.imread(content_path)
        if content_bgr is None:
            raise FileNotFoundError(f"Could not read content image: {content_path}")

        # --- Pick a random style image ----------------------------------------
        style_path = self._pick_random_style(style_dir)
        style_bgr = cv2.imread(style_path)
        if style_bgr is None:
            raise FileNotFoundError(f"Could not read style image: {style_path}")

        logger.debug(
            "Style transfer: content={} style={} weight={:.2f}",
            content_path, style_path, style_weight,
        )

        # --- Load depth map (optional) ----------------------------------------
        depth_map: Optional[np.ndarray] = None
        if depth_guided and depth_map_path and os.path.isfile(depth_map_path):
            depth_map = self._load_depth_map(depth_map_path, content_bgr.shape[:2])

        # --- Apply transfer ---------------------------------------------------
        if color_only:
            result_bgr = self._transfer_color_only(
                content_bgr, style_bgr, style_weight, depth_map,
            )
        else:
            result_bgr = self._transfer_wct(
                content_bgr, style_bgr,
                style_weight=style_weight,
                content_weight=content_weight,
                preserve_structure=preserve_structure,
                depth_map=depth_map,
            )

        # --- Write output -----------------------------------------------------
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, result_bgr)

        # Copy annotations alongside if provided
        if annotations_path and os.path.exists(annotations_path):
            ann_ext = Path(annotations_path).suffix
            ann_out = str(Path(output_path).with_suffix(ann_ext))
            shutil.copy2(annotations_path, ann_out)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "Style transfer complete: {} -> {} ({:.0f}ms)",
            content_path, output_path, elapsed_ms,
        )

        # Free GPU memory
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        return output_path

    def apply_batch(
        self,
        images_dir: str,
        style_dir: str,
        output_dir: str,
        style_weight: float = 0.6,
        content_weight: float = 1.0,
        preserve_structure: float = 0.8,
        color_only: bool = False,
        depth_guided: bool = True,
        annotations_dir: Optional[str] = None,
        progress_callback=None,
    ) -> dict:
        """Apply style transfer to all images in a directory.

        Args:
            images_dir: Directory containing synthetic content images.
            style_dir: Directory containing real reference style images.
            output_dir: Directory for styled output images.
            style_weight: Blending strength toward the style (0-1).
            content_weight: Content preservation multiplier.
            preserve_structure: How much structural detail to keep (0-1).
            color_only: If True, only match colour statistics in LAB space.
            depth_guided: Enable depth-guided transfer when depth maps are
                available alongside the content images.
            annotations_dir: Optional directory containing annotation files
                whose stems match the image file stems.
            progress_callback: Optional ``callable(processed, total)`` for
                progress updates.

        Returns:
            Dictionary with keys ``total_images``, ``processed``, ``failed``.
        """
        image_files = sorted([
            f for f in Path(images_dir).iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()
        ])

        if not image_files:
            logger.warning("No images found in {}", images_dir)
            return {"total_images": 0, "processed": 0, "failed": 0}

        os.makedirs(output_dir, exist_ok=True)
        processed = 0
        failed = 0

        for idx, img_path in enumerate(image_files):
            try:
                out_path = os.path.join(output_dir, img_path.name)

                # Look for a depth map with the same stem
                depth_map_path: Optional[str] = None
                if depth_guided:
                    depth_map_path = self._find_depth_map(
                        images_dir, img_path.stem,
                    )

                # Find matching annotation file
                ann_path: Optional[str] = None
                if annotations_dir:
                    for ann_ext in [".json", ".txt", ".xml"]:
                        candidate = Path(annotations_dir) / f"{img_path.stem}{ann_ext}"
                        if candidate.exists():
                            ann_path = str(candidate)
                            break

                self.apply_single(
                    content_path=str(img_path),
                    style_dir=style_dir,
                    output_path=out_path,
                    style_weight=style_weight,
                    content_weight=content_weight,
                    preserve_structure=preserve_structure,
                    color_only=color_only,
                    depth_map_path=depth_map_path,
                    depth_guided=depth_guided,
                    annotations_path=ann_path,
                )
                processed += 1

            except Exception as e:
                logger.error("Failed to process {}: {}", img_path, e)
                failed += 1

            if progress_callback:
                progress_callback(idx + 1, len(image_files))

        logger.info(
            "Style transfer batch complete: {} images -> {} processed ({} failed)",
            len(image_files), processed, failed,
        )
        return {
            "total_images": len(image_files),
            "processed": processed,
            "failed": failed,
        }

    # ------------------------------------------------------------------
    # VGG-19 lazy loading
    # ------------------------------------------------------------------

    def _load_encoder(self) -> None:
        """Lazily load the VGG-19 encoder on first use."""
        if self._encoder is not None:
            return

        logger.info("Loading VGG-19 encoder for style transfer...")
        start = time.time()

        try:
            self._encoder = _VGG19Encoder()
            self._encoder.eval()
            self._encoder.to(self._device)

            elapsed = time.time() - start
            logger.info(
                "VGG-19 encoder loaded in {:.2f}s on {}", elapsed, self._device,
            )
        except Exception as e:
            logger.error("Failed to load VGG-19 encoder: {}", e)
            raise

    # ------------------------------------------------------------------
    # WCT (Whitening and Coloring Transform) core
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _transfer_wct(
        self,
        content_bgr: np.ndarray,
        style_bgr: np.ndarray,
        style_weight: float,
        content_weight: float,
        preserve_structure: float,
        depth_map: Optional[np.ndarray],
    ) -> np.ndarray:
        """Run the full multi-scale WCT pipeline.

        Processing order is coarse-to-fine (relu4_1 -> relu1_1).
        At each scale the content features are whitened, coloured with style
        statistics, and blended back with the original content features.

        The ``preserve_structure`` parameter further damps the transfer by
        interpolating between the styled image and the original content in
        pixel space after each scale.

        Args:
            content_bgr: Content image in BGR uint8 format.
            style_bgr: Style image in BGR uint8 format.
            style_weight: Alpha blending factor for WCT (0 = no style).
            content_weight: Multiplier applied to content features before
                blending (values > 1 strengthen content retention).
            preserve_structure: Pixel-space blend toward the original content
                after each WCT scale (0 = no preservation, 1 = full).
            depth_map: Optional HxW float32 array in [0, 1] where 0 is near
                and 1 is far.

        Returns:
            Styled image as BGR uint8.
        """
        self._load_encoder()

        # Convert to RGB float for tensor conversion
        content_rgb = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2RGB)
        style_rgb = cv2.cvtColor(style_bgr, cv2.COLOR_BGR2RGB)

        # Resize style to match content spatial dimensions
        h, w = content_rgb.shape[:2]
        style_rgb = cv2.resize(style_rgb, (w, h), interpolation=cv2.INTER_AREA)

        # Build tensors: (1, 3, H, W) normalised
        content_tensor = self._image_to_tensor(content_rgb)
        style_tensor = self._image_to_tensor(style_rgb)

        # Extract multi-scale features
        content_features = self._encoder(content_tensor)
        style_features = self._encoder(style_tensor)

        # Iteratively apply WCT from coarse to fine
        result_tensor = content_tensor.clone()

        for layer_name in SCALE_ORDER:
            cf = content_features[layer_name]
            sf = style_features[layer_name]

            # Apply WCT at this scale
            styled_f = self._wct_transform(cf, sf, alpha=style_weight)

            # Blend with the original content features (content_weight)
            blended_f = (
                content_weight * cf + (styled_f - cf)
            )
            # This simplifies to: styled_f when content_weight==1
            # and emphasises content when content_weight > 1.
            blended_f = cf + (styled_f - cf) / max(content_weight, EPS)

            # Replace the feature map for the next (finer) scale
            content_features[layer_name] = blended_f

        # Reconstruct the image from the finest-scale features.
        # Since we don't have a trained decoder, we use a simple approach:
        # propagate the WCT-blended features back through the encoder to
        # get a pixel-space delta, then apply it to the original image.
        result_rgb = self._reconstruct_from_features(
            content_tensor, content_features, style_weight,
        )

        # Pixel-space blending with original for structure preservation
        if preserve_structure > 0:
            original_rgb = content_rgb.astype(np.float32)
            result_rgb = (
                (1.0 - preserve_structure) * result_rgb
                + preserve_structure * original_rgb
            )

        # Apply depth-guided spatial modulation
        if depth_map is not None:
            result_rgb = self._apply_depth_modulation(
                content_rgb.astype(np.float32),
                result_rgb,
                depth_map,
            )

        # Clamp and convert back to BGR uint8
        result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        return result_bgr

    def _wct_transform(
        self,
        content_feat: torch.Tensor,
        style_feat: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Whitening and Coloring Transform on a single feature map.

        Steps:
        1. Flatten spatial dims: (1, C, H, W) -> (C, HW)
        2. Centre and compute covariance of content features.
        3. Whiten content via eigendecomposition of its covariance.
        4. Colour with style statistics via eigendecomposition of the
           style covariance.
        5. Blend the coloured result with the original content features
           using ``alpha``.

        Args:
            content_feat: Content feature tensor (1, C, H, W).
            style_feat: Style feature tensor (1, C, H', W').
            alpha: Blending factor (0 = pure content, 1 = pure style).

        Returns:
            Transformed feature tensor (1, C, H, W).
        """
        # Unpack spatial dimensions
        _, c, h, w = content_feat.shape

        # Flatten: (C, N) where N = H*W
        cf = content_feat.squeeze(0).view(c, -1)  # (C, N_c)
        sf = style_feat.squeeze(0).view(c, -1)    # (C, N_s)

        # --- Whiten content features ---
        c_mean = cf.mean(dim=1, keepdim=True)
        cf_centered = cf - c_mean

        c_cov = (cf_centered @ cf_centered.t()) / max(cf_centered.shape[1] - 1, 1) + EPS * torch.eye(c, device=cf.device)
        c_eigvals, c_eigvecs = torch.linalg.eigh(c_cov)

        # Clamp eigenvalues to positive for numerical stability
        c_eigvals = c_eigvals.clamp(min=EPS)

        # Whitening matrix: D^{-1/2} @ E^T
        c_d_inv_sqrt = torch.diag(c_eigvals.pow(-0.5))
        whitening_matrix = c_eigvecs @ c_d_inv_sqrt @ c_eigvecs.t()

        whitened = whitening_matrix @ cf_centered

        # --- Colour with style features ---
        s_mean = sf.mean(dim=1, keepdim=True)
        sf_centered = sf - s_mean

        s_cov = (sf_centered @ sf_centered.t()) / max(sf_centered.shape[1] - 1, 1) + EPS * torch.eye(c, device=sf.device)
        s_eigvals, s_eigvecs = torch.linalg.eigh(s_cov)
        s_eigvals = s_eigvals.clamp(min=EPS)

        # Coloring matrix: E @ D^{1/2} @ E^T
        s_d_sqrt = torch.diag(s_eigvals.pow(0.5))
        coloring_matrix = s_eigvecs @ s_d_sqrt @ s_eigvecs.t()

        colored = coloring_matrix @ whitened + s_mean

        # --- Blend ---
        result = alpha * colored + (1.0 - alpha) * cf

        return result.view(1, c, h, w)

    def _reconstruct_from_features(
        self,
        content_tensor: torch.Tensor,
        blended_features: Dict[str, torch.Tensor],
        style_weight: float,
    ) -> np.ndarray:
        """Reconstruct an RGB image from WCT-blended feature maps.

        Since we do not train a decoder network, we use a feature-space
        matching heuristic: for each scale the difference between the
        blended and original features is back-projected into pixel space
        via the pseudo-inverse of the encoder's Jacobian (approximated by
        a single optimisation step with Adam).

        For simplicity and speed this implementation uses an iterative
        optimisation approach: start from the content image and optimise
        the pixel values to match the blended features across all scales.

        Args:
            content_tensor: Original content tensor (1, 3, H, W).
            blended_features: WCT-blended features per layer.
            style_weight: Used to scale the number of optimisation steps.

        Returns:
            Reconstructed RGB image as float32 (H, W, 3) in [0, 255].
        """
        # Clone content as starting point; enable gradient
        result = content_tensor.clone().detach().requires_grad_(True)

        # Number of optimisation iterations scales with style_weight
        num_iters = max(10, int(50 * style_weight))
        optimizer = torch.optim.Adam([result], lr=0.02)

        for _ in range(num_iters):
            optimizer.zero_grad()
            current_features = self._encoder(result)

            loss = torch.tensor(0.0, device=self._device)
            for layer_name in SCALE_ORDER:
                target = blended_features[layer_name].detach()
                current = current_features[layer_name]
                loss = loss + nn.functional.mse_loss(current, target)

            loss.backward()
            optimizer.step()

        # Convert back to numpy RGB
        result_np = self._tensor_to_image(result.detach())
        return result_np

    # ------------------------------------------------------------------
    # Colour-only transfer (LAB space, no neural network)
    # ------------------------------------------------------------------

    def _transfer_color_only(
        self,
        content_bgr: np.ndarray,
        style_bgr: np.ndarray,
        strength: float,
        depth_map: Optional[np.ndarray],
    ) -> np.ndarray:
        """Transfer colour statistics from style to content in LAB space.

        For each LAB channel the content distribution is shifted and scaled
        to match the style distribution, blended by ``strength``.

        Args:
            content_bgr: Content image (BGR uint8).
            style_bgr: Style image (BGR uint8).
            strength: Transfer strength (0-1).
            depth_map: Optional depth map for spatial modulation.

        Returns:
            Colour-transferred image (BGR uint8).
        """
        # Resize style to match content
        h, w = content_bgr.shape[:2]
        style_resized = cv2.resize(style_bgr, (w, h), interpolation=cv2.INTER_AREA)

        content_lab = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        style_lab = cv2.cvtColor(style_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

        result_lab = content_lab.copy()

        for ch in range(3):
            c_ch = content_lab[:, :, ch]
            s_ch = style_lab[:, :, ch]

            c_mean, c_std = c_ch.mean(), max(c_ch.std(), 1e-6)
            s_mean, s_std = s_ch.mean(), max(s_ch.std(), 1e-6)

            # Normalise content, apply style statistics, blend
            normalised = (c_ch - c_mean) * (s_std / c_std) + s_mean
            result_lab[:, :, ch] = c_ch + strength * (normalised - c_ch)

        # Apply depth-guided spatial modulation
        if depth_map is not None:
            original_lab = content_lab.copy()
            result_lab = self._apply_depth_modulation_lab(
                original_lab, result_lab, depth_map,
            )

        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Depth-guided modulation
    # ------------------------------------------------------------------

    @staticmethod
    def _load_depth_map(
        depth_path: str,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Load and normalise a depth map to float32 in [0, 1].

        Convention: 0 = near (foreground), 1 = far (background).

        Args:
            depth_path: Path to the depth map image.
            target_size: (H, W) to resize the depth map to.

        Returns:
            Normalised depth map of shape (H, W) as float32.
        """
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth map: {depth_path}")

        h, w = target_size
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        depth_f = depth.astype(np.float32)
        d_min, d_max = depth_f.min(), depth_f.max()
        if d_max - d_min > 1e-6:
            depth_f = (depth_f - d_min) / (d_max - d_min)
        else:
            depth_f = np.zeros_like(depth_f)

        return depth_f

    @staticmethod
    def _apply_depth_modulation(
        original_rgb: np.ndarray,
        styled_rgb: np.ndarray,
        depth_map: np.ndarray,
    ) -> np.ndarray:
        """Modulate style transfer strength based on depth.

        Foreground (depth ~ 0) retains more of the original content.
        Background (depth ~ 1) retains more of the styled result.

        Args:
            original_rgb: Original content image (float32, H, W, 3).
            styled_rgb: Styled image (float32, H, W, 3).
            depth_map: Depth map (float32, H, W) in [0, 1].

        Returns:
            Depth-modulated result (float32, H, W, 3).
        """
        # Expand depth to 3 channels for broadcasting
        alpha = depth_map[:, :, np.newaxis]  # (H, W, 1)

        # Smoothly blend: foreground -> original, background -> styled
        result = (1.0 - alpha) * original_rgb + alpha * styled_rgb
        return result

    @staticmethod
    def _apply_depth_modulation_lab(
        original_lab: np.ndarray,
        styled_lab: np.ndarray,
        depth_map: np.ndarray,
    ) -> np.ndarray:
        """Depth modulation in LAB space (used by colour-only transfer).

        Same logic as ``_apply_depth_modulation`` but operates on LAB
        float32 arrays.
        """
        alpha = depth_map[:, :, np.newaxis]
        return (1.0 - alpha) * original_lab + alpha * styled_lab

    # ------------------------------------------------------------------
    # Image <-> Tensor conversion helpers
    # ------------------------------------------------------------------

    def _image_to_tensor(self, rgb_image: np.ndarray) -> torch.Tensor:
        """Convert an RGB uint8 image to a normalised (1, 3, H, W) tensor.

        Uses ImageNet mean/std normalisation as expected by VGG-19.
        """
        from PIL import Image
        pil_img = Image.fromarray(rgb_image)
        tensor = self._to_tensor(pil_img)  # (3, H, W)
        return tensor.unsqueeze(0).to(self._device)

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a normalised (1, 3, H, W) tensor back to RGB float32
        in [0, 255].

        Applies ImageNet denormalisation.
        """
        img = tensor.squeeze(0).cpu().clone()  # (3, H, W)

        # Denormalise
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img = img * std + mean

        # Convert to (H, W, 3) float32 [0, 255]
        img = img.clamp(0.0, 1.0) * 255.0
        img_np = img.permute(1, 2, 0).numpy().astype(np.float32)
        return img_np

    # ------------------------------------------------------------------
    # Style image selection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_random_style(style_dir: str) -> str:
        """Select a random image from the style directory.

        Raises:
            FileNotFoundError: If the directory does not exist.
            ValueError: If no valid images are found.
        """
        dir_path = Path(style_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Style directory not found: {style_dir}")

        candidates = [
            p for p in sorted(dir_path.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not candidates:
            raise ValueError(f"No valid style images found in: {style_dir}")

        return str(random.choice(candidates))

    @staticmethod
    def _find_depth_map(images_dir: str, stem: str) -> Optional[str]:
        """Look for a depth map matching the given image stem.

        Searches for files named ``{stem}_depth.*`` or in a ``depth/``
        sub-directory alongside the images.

        Returns:
            Path to the depth map if found, else ``None``.
        """
        base = Path(images_dir)

        # Pattern 1: {stem}_depth.{ext} in the same directory
        for ext in [".png", ".jpg", ".exr", ".tiff"]:
            candidate = base / f"{stem}_depth{ext}"
            if candidate.is_file():
                return str(candidate)

        # Pattern 2: depth/{stem}.{ext} sub-directory
        depth_dir = base / "depth"
        if depth_dir.is_dir():
            for ext in [".png", ".jpg", ".exr", ".tiff"]:
                candidate = depth_dir / f"{stem}{ext}"
                if candidate.is_file():
                    return str(candidate)

        return None
