"""
VRAM Monitor
============
Shared utility for monitoring GPU VRAM usage across services.
Triggers cleanup when usage exceeds threshold to prevent OOM errors.
"""

import gc
import logging
from typing import Dict, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class VRAMMonitor:
    """Monitors VRAM usage and triggers cleanup when needed."""

    def __init__(self, threshold: float = 0.7, device: int = 0, check_interval: int = 2):
        """
        Initialize VRAM monitor.

        Args:
            threshold: VRAM usage fraction to trigger cleanup (default 0.7 = 70%)
            device: CUDA device ID to monitor (default 0)
            check_interval: Check every N iterations (default 2)
        """
        self.threshold = threshold
        self.device = device
        self.last_check = 0
        self.check_interval = check_interval

    def get_vram_usage(self) -> float:
        """
        Returns VRAM usage as fraction (0.0-1.0).

        Returns:
            VRAM usage fraction, or 0.0 if CUDA not available
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0

        try:
            allocated = torch.cuda.memory_allocated(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            return allocated / total
        except Exception as e:
            logger.warning(f"Failed to get VRAM usage: {e}")
            return 0.0

    def get_vram_stats(self) -> Dict[str, float]:
        """
        Get detailed VRAM statistics.

        Returns:
            Dict with allocated_gb, reserved_gb, total_gb, usage_fraction
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}

        try:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            usage = allocated / (total or 1)

            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'usage_fraction': usage,
            }
        except Exception as e:
            logger.warning(f"Failed to get VRAM stats: {e}")
            return {}

    def should_cleanup(self) -> bool:
        """
        Check if cleanup is needed (VRAM > threshold).

        Returns:
            True if VRAM usage exceeds threshold
        """
        self.last_check += 1
        if self.last_check % self.check_interval == 0:
            usage = self.get_vram_usage()
            if usage > self.threshold:
                stats = self.get_vram_stats()
                logger.warning(
                    f"VRAM at {usage*100:.1f}% "
                    f"({stats.get('allocated_gb', 0):.2f}GB / {stats.get('total_gb', 32):.0f}GB), "
                    f"triggering cleanup"
                )
                return True
        return False

    def cleanup(self) -> None:
        """
        Perform VRAM cleanup: garbage collection and cache clearing.
        """
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            stats = self.get_vram_stats()
            logger.info(
                f"VRAM cleanup complete: {stats.get('allocated_gb', 0):.2f}GB / "
                f"{stats.get('total_gb', 32):.0f}GB"
            )

    def cleanup_if_needed(self) -> bool:
        """
        Check if cleanup is needed and perform it if so.

        Returns:
            True if cleanup was performed
        """
        if self.should_cleanup():
            self.cleanup()
            return True
        return False
