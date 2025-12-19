# src/augmentation/depth_engine.py
"""
Depth Estimation Engine using Depth Anything V3 (November 2025)

This module provides depth map estimation from single images using the
state-of-the-art Depth Anything V3 model for photorealistic object compositing.

Depth Anything V3 improvements over V2:
- +44.3% camera pose accuracy
- +25.1% geometric accuracy
- Unified architecture for depth + pose + multi-view geometry
- Plain transformer (DINO encoder) vs convolutional V2

Backward compatibility: Supports both V2 and V3 models via model_version parameter.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    Sistema de estimación de profundidad con Depth Anything V3/V2

    Uses monocular depth estimation to create depth maps from background images,
    enabling depth-aware object scaling, atmospheric perspective, and depth-of-field effects.

    Args:
        model_size: 'small' (Apache 2.0, commercial use), 'base', or 'large' (CC-BY-NC-4.0)
        model_version: 'v3' (default, SOTA Nov 2025) or 'v2' (backward compatibility)
        device: 'cuda' or 'cpu' - GPU highly recommended for performance
        cache_dir: Directory to store/load model checkpoints
        enable_pose_estimation: Enable camera pose estimation (V3 only, default: False)
    """

    def __init__(self, model_size='small', model_version='v3', device='cuda',
                 cache_dir='checkpoints', enable_pose_estimation=False):
        """Initialize depth estimator with specified model size, version, and device"""
        self.device = device if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")

        self.model_size = model_size
        self.model_version = model_version.lower()
        self.enable_pose_estimation = enable_pose_estimation
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

        # Validate model version
        if self.model_version not in ['v2', 'v3']:
            raise ValueError(f"model_version must be 'v2' or 'v3', got '{self.model_version}'")

        # Pose estimation only available in V3
        if self.enable_pose_estimation and self.model_version == 'v2':
            logger.warning("Pose estimation only available in V3, disabling for V2")
            self.enable_pose_estimation = False

        # Load model lazily
        self._load_model()

    def _load_model(self):
        """Load Depth Anything V2 or V3 model with lazy loading"""
        try:
            if self.model_version == 'v3':
                # Depth Anything V3 (November 2025 - SOTA)
                from depth_anything_3.api import DepthAnything3

                # Map model sizes to HuggingFace model IDs
                model_ids = {
                    'small': 'depth-anything/da3-small',
                    'base': 'depth-anything/da3-base',
                    'large': 'depth-anything/da3-large',
                    'giant': 'depth-anything/da3-giant'  # Non-commercial
                }

                if self.model_size not in model_ids:
                    raise ValueError(f"Model size must be 'small', 'base', 'large', or 'giant', got '{self.model_size}'")

                model_id = model_ids[self.model_size]

                logger.info(f"Loading Depth Anything V3 ({self.model_size}) from {model_id}...")

                # Load model directly from HuggingFace
                self.model = DepthAnything3.from_pretrained(model_id)
                self.model = self.model.to(device=self.device)
                self.model.eval()

                logger.info(f"Depth Anything V3 ({self.model_size}) loaded successfully on {self.device}")
                logger.info("V3 improvements: +44.3% pose accuracy, +25.1% geometric accuracy vs V2")

            else:  # v2 (backward compatibility)
                from depth_anything_v2.dpt import DepthAnythingV2

                # Model configurations for different sizes
                model_configs = {
                    'small': {
                        'encoder': 'vits',
                        'features': 64,
                        'out_channels': [48, 96, 192, 384]
                    },
                    'base': {
                        'encoder': 'vitb',
                        'features': 128,
                        'out_channels': [96, 192, 384, 768]
                    },
                    'large': {
                        'encoder': 'vitl',
                        'features': 256,
                        'out_channels': [256, 512, 1024, 1024]
                    }
                }

                if self.model_size not in model_configs:
                    raise ValueError(f"Model size must be 'small', 'base', or 'large', got '{self.model_size}'")

                config = model_configs[self.model_size]

                # Initialize model
                self.model = DepthAnythingV2(**config)

                # Load pre-trained weights
                checkpoint_path = self.cache_dir / f"depth_anything_v2_{self.model_size}.pth"

                if checkpoint_path.exists():
                    logger.info(f"Loading V2 checkpoint from {checkpoint_path}")
                    state_dict = torch.load(checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                else:
                    logger.warning(f"Checkpoint not found: {checkpoint_path}")
                    logger.info("Attempting auto-download from HuggingFace...")
                    self._download_checkpoint()

                # Move model to device and set to evaluation mode
                self.model.to(self.device)
                self.model.eval()

                logger.info(f"Depth Anything V2 ({self.model_size}) loaded successfully on {self.device}")

        except ImportError as e:
            if self.model_version == 'v3':
                error_msg = (
                    "depth_anything_3 not installed. "
                    "Install with: pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
                )
            else:
                error_msg = (
                    "depth_anything_v2 not installed. "
                    "Install with: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
                )
            logger.error(error_msg)
            raise ImportError(
                f"Depth Anything {self.model_version.upper()} not found. Please ensure it's installed."
            ) from e
        except Exception as e:
            logger.error(f"Error loading Depth Anything {self.model_version.upper()} model: {e}")
            raise

    def _download_checkpoint(self):
        """Download model checkpoint automatically from HuggingFace"""
        try:
            from huggingface_hub import hf_hub_download

            # Map model sizes to HuggingFace repository IDs
            repo_ids = {
                'small': 'depth-anything/Depth-Anything-V2-Small',
                'base': 'depth-anything/Depth-Anything-V2-Base',
                'large': 'depth-anything/Depth-Anything-V2-Large'
            }

            repo_id = repo_ids.get(self.model_size)
            if not repo_id:
                raise ValueError(f"Unknown model size: {self.model_size}")

            filename = f"depth_anything_v2_{self.model_size}.pth"

            logger.info(f"Downloading checkpoint from {repo_id}...")
            checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir)
            )

            # Copy to local cache directory
            import shutil
            dest = self.cache_dir / filename
            shutil.copy(checkpoint_path, dest)

            logger.info(f"Checkpoint downloaded successfully to {dest}")

        except Exception as e:
            logger.error(f"Failed to download checkpoint: {e}")
            raise RuntimeError(
                f"Could not download Depth Anything V2 checkpoint. "
                f"Please download manually from HuggingFace and place in {self.cache_dir}"
            ) from e

    @torch.no_grad()
    def estimate_depth(self, image, normalize=True):
        """
        Estimate depth map from a single image

        Args:
            image: numpy array (H, W, 3) in BGR format (OpenCV standard)
            normalize: If True, normalize depth values to [0, 1] range

        Returns:
            depth_map: numpy array (H, W) with depth values
                      Higher values = closer to camera (normalized convention)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        h, w = rgb.shape[:2]

        # Run inference based on model version
        try:
            if self.model_version == 'v3':
                # Depth Anything V3 API
                # V3 expects list of images and returns prediction object
                prediction = self.model.inference([rgb])

                # Extract depth map from prediction
                # prediction.depth is [N, H, W] where N=1 in our case
                depth_map = prediction.depth[0]  # Get first (and only) depth map

                # Convert to numpy if it's a tensor
                if torch.is_tensor(depth_map):
                    depth_map = depth_map.cpu().numpy()

            else:  # v2
                # Depth Anything V2 API (original)
                # Prepare input tensor (accepts any resolution)
                # Note: Model is faster with dimensions that are multiples of 14
                input_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
                input_tensor = input_tensor.unsqueeze(0).to(self.device)

                # Direct model call
                depth = self.model(input_tensor)

                # Convert to numpy
                depth_map = depth.squeeze().cpu().numpy()

        except Exception as e:
            logger.error(f"Error during depth inference: {e}")
            raise RuntimeError(f"Depth estimation failed: {e}") from e

        # Normalize if requested
        if normalize:
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max - depth_min > 1e-8:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                # If depth map is uniform, return zeros
                depth_map = np.zeros_like(depth_map)

        return depth_map

    def get_depth_at_position(self, depth_map, x, y, radius=5):
        """
        Get average depth value at a specific position

        Args:
            depth_map: Depth map array (H, W)
            x, y: Coordinates of center point
            radius: Radius of region to average over (default: 5 pixels)

        Returns:
            depth_value: Average depth value in [0, 1] range
        """
        h, w = depth_map.shape

        # Safe clamping to image boundaries
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)

        # Extract region and compute mean
        region = depth_map[y_min:y_max, x_min:x_max]

        if region.size == 0:
            logger.warning(f"Empty region at position ({x}, {y})")
            return 0.5  # Return middle depth as fallback

        return np.mean(region)

    def classify_depth_zones(self, depth_map, num_zones=3):
        """
        Classify image into depth zones (near/mid/far)

        Uses quantile-based division to ensure balanced zones

        Args:
            depth_map: Depth map array (H, W)
            num_zones: Number of depth zones (default: 3 for near/mid/far)

        Returns:
            zones_mask: numpy array (H, W) with zone IDs (0=far, 1=mid, 2=near for 3 zones)
            depth_ranges: List of tuples (min_depth, max_depth) for each zone
        """
        # Divide into quantiles for balanced zones
        depth_flat = depth_map.flatten()
        quantiles = np.linspace(0, 1, num_zones + 1)
        thresholds = np.quantile(depth_flat, quantiles)

        # Create zone mask
        zones_mask = np.zeros_like(depth_map, dtype=np.uint8)
        depth_ranges = []

        for i in range(num_zones):
            # Create mask for this zone
            if i < num_zones - 1:
                mask = (depth_map >= thresholds[i]) & (depth_map < thresholds[i + 1])
            else:
                # Last zone includes maximum value
                mask = (depth_map >= thresholds[i]) & (depth_map <= thresholds[i + 1])

            zones_mask[mask] = i
            depth_ranges.append((thresholds[i], thresholds[i + 1]))

        return zones_mask, depth_ranges

    def visualize_depth(self, depth_map, colormap=cv2.COLORMAP_INFERNO):
        """
        Create a colorized visualization of the depth map

        Args:
            depth_map: Depth map array (H, W) in [0, 1] range
            colormap: OpenCV colormap to use (default: COLORMAP_INFERNO)

        Returns:
            colored_depth: BGR image for visualization
        """
        # Convert to 8-bit
        depth_uint8 = (depth_map * 255).astype(np.uint8)

        # Apply colormap
        colored_depth = cv2.applyColorMap(depth_uint8, colormap)

        return colored_depth

    @torch.no_grad()
    def estimate_pose(self, images):
        """
        Estimate camera poses from multiple images (V3 only)

        This is a new capability in Depth Anything V3, providing unified architecture
        for depth + pose + multi-view geometry.

        Args:
            images: List of numpy arrays (H, W, 3) in BGR format

        Returns:
            poses: Camera pose estimation results
                  None if model_version is V2 (not supported)

        Raises:
            RuntimeError: If called with V2 model or pose estimation disabled
        """
        if self.model_version == 'v2':
            raise RuntimeError("Pose estimation only available in Depth Anything V3")

        if not self.enable_pose_estimation:
            raise RuntimeError(
                "Pose estimation is disabled. Initialize with enable_pose_estimation=True"
            )

        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Convert BGR to RGB
        rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

        try:
            # V3 inference returns prediction object with pose information
            prediction = self.model.inference(rgb_images)

            # Extract pose information from prediction
            # NOTE: Actual pose extraction depends on V3 API structure
            # This might need adjustment based on actual API
            logger.info(f"Pose estimation completed for {len(images)} images")

            return prediction  # Return full prediction object

        except Exception as e:
            logger.error(f"Error during pose estimation: {e}")
            raise RuntimeError(f"Pose estimation failed: {e}") from e


# Global singleton instance
_depth_estimator = None


def get_depth_estimator(model_size='small', model_version='v3', device='cuda',
                       cache_dir='checkpoints', enable_pose_estimation=False):
    """
    Get or create singleton instance of DepthEstimator

    This ensures only one model is loaded in memory, even if called multiple times.

    Args:
        model_size: 'small', 'base', 'large', or 'giant' (V3 only)
        model_version: 'v3' (default, SOTA Nov 2025) or 'v2' (backward compatibility)
        device: 'cuda' or 'cpu'
        cache_dir: Directory for model checkpoints
        enable_pose_estimation: Enable camera pose estimation (V3 only, default: False)

    Returns:
        DepthEstimator instance
    """
    global _depth_estimator

    if _depth_estimator is None:
        logger.info(
            f"Initializing Depth Estimator "
            f"(model={model_size}, version={model_version}, device={device})"
        )
        _depth_estimator = DepthEstimator(
            model_size=model_size,
            model_version=model_version,
            device=device,
            cache_dir=cache_dir,
            enable_pose_estimation=enable_pose_estimation
        )

    return _depth_estimator
