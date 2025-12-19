"""
Caustics Generation Module

Provides efficient caustics map generation using a caching system.
Templates are pre-generated and then transformed for variation (500-1000x faster).
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class CausticsCache:
    """Cache system for pre-generated caustics maps with random transformations.

    Generates templates once and transforms them (rotation, flip, scale) for
    variation without expensive Perlin noise calculation.
    """

    def __init__(self, cache_dir: str = "/app/cache/caustics", num_templates: int = 15):
        """Initialize caustics cache.

        Args:
            cache_dir: Directory for storing templates on disk
            num_templates: Number of base templates to generate
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_templates = num_templates
        self.templates: Dict[str, np.ndarray] = {}
        self._initialized = False

    def generate_templates(self, sizes: List[Tuple[int, int]] = None) -> None:
        """Pre-generate caustics templates at various sizes.

        Args:
            sizes: List of (width, height) tuples
        """
        if sizes is None:
            sizes = [(1024, 1024), (2048, 2048)]

        try:
            import noise
        except ImportError:
            logger.warning("noise library not available. Using fallback pattern.")
            self._generate_fallback_templates(sizes)
            return

        logger.info(f"Generating {self.num_templates} caustics templates...")

        for size in sizes:
            width, height = size

            for template_id in range(self.num_templates):
                # Vary parameters for diversity
                complexity = np.random.uniform(0.008, 0.015)
                octaves = np.random.randint(3, 6)

                caustics_map = np.zeros((height, width), dtype=np.float32)

                for y in range(height):
                    for x in range(width):
                        value = noise.pnoise2(
                            x * complexity,
                            y * complexity,
                            octaves=octaves
                        )
                        caustics_map[y, x] = value

                # Normalize
                caustics_map = (caustics_map - np.min(caustics_map)) / \
                              (np.max(caustics_map) - np.min(caustics_map))

                # Save to cache
                template_key = f"{width}x{height}_{template_id}"
                cache_file = self.cache_dir / f"{template_key}.npy"
                np.save(cache_file, caustics_map)
                self.templates[template_key] = caustics_map

        self._initialized = True
        logger.info(f"Caustics cache generated: {len(self.templates)} templates")

    def _generate_fallback_templates(self, sizes: List[Tuple[int, int]]) -> None:
        """Generate templates using fast sinusoidal patterns.

        Args:
            sizes: List of (width, height) tuples
        """
        logger.info("Generating fallback sinusoidal templates...")

        for size in sizes:
            width, height = size

            for template_id in range(self.num_templates):
                # Generate sinusoidal pattern with variation
                freq_x = np.random.uniform(3, 6)
                freq_y = np.random.uniform(3, 6)
                phase = np.random.uniform(0, 2 * np.pi)

                x = np.linspace(0, freq_x * np.pi, width)
                y = np.linspace(0, freq_y * np.pi, height)
                X, Y = np.meshgrid(x, y)

                caustics_map = (np.sin(X + phase) * np.cos(Y) +
                               np.sin(2 * X + Y + phase) * 0.5 +
                               np.cos(X - 2 * Y) * 0.3)

                # Normalize
                caustics_map = (caustics_map - caustics_map.min()) / \
                              (caustics_map.max() - caustics_map.min())

                template_key = f"{width}x{height}_{template_id}"
                cache_file = self.cache_dir / f"{template_key}.npy"
                np.save(cache_file, caustics_map.astype(np.float32))
                self.templates[template_key] = caustics_map.astype(np.float32)

        self._initialized = True
        logger.info(f"Fallback caustics templates generated: {len(self.templates)}")

    def load_from_disk(self) -> bool:
        """Load templates from disk if they exist.

        Returns:
            True if templates were loaded, False otherwise
        """
        cache_files = list(self.cache_dir.glob("*.npy"))

        if not cache_files:
            return False

        for cache_file in cache_files:
            template_key = cache_file.stem
            self.templates[template_key] = np.load(cache_file)

        self._initialized = True
        logger.info(f"Loaded {len(cache_files)} caustics templates from disk")
        return True

    def ensure_initialized(self) -> None:
        """Ensure cache is initialized, loading or generating as needed."""
        if self._initialized:
            return

        if not self.load_from_disk():
            self.generate_templates()

    def get_caustics(self, width: int, height: int, brightness: float = 0.4) -> np.ndarray:
        """Get caustics map adapted to size with random transformations.

        Args:
            width: Desired width
            height: Desired height
            brightness: Brightness factor (0.0-1.0)

        Returns:
            BGR caustics map ready to apply
        """
        self.ensure_initialized()

        # Find nearest template
        target_size = width * height
        best_template_key = None
        best_diff = float('inf')

        for template_key in self.templates.keys():
            size_str = template_key.rsplit('_', 1)[0]
            w, h = map(int, size_str.split('x'))
            diff = abs(w * h - target_size)

            if diff < best_diff:
                best_diff = diff
                best_template_key = template_key

        if best_template_key is None:
            return self._generate_instant_fallback(width, height, brightness)

        # Get and transform template
        caustics = self.templates[best_template_key].copy()
        caustics = self._random_transform(caustics)

        # Resize if needed
        if caustics.shape[1] != width or caustics.shape[0] != height:
            caustics = cv2.resize(caustics, (width, height), interpolation=cv2.INTER_LINEAR)

        # Apply brightness
        caustics = np.clip(caustics * brightness + (1 - brightness), 0, 2)

        # Convert to BGR
        return cv2.cvtColor(caustics.astype(np.float32), cv2.COLOR_GRAY2BGR)

    def _random_transform(self, caustics: np.ndarray) -> np.ndarray:
        """Apply random transformations for visual variation.

        Args:
            caustics: Base template (grayscale float32)

        Returns:
            Transformed template
        """
        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            caustics = np.rot90(caustics, k)

        # Random flips
        if np.random.random() > 0.5:
            caustics = np.fliplr(caustics)
        if np.random.random() > 0.5:
            caustics = np.flipud(caustics)

        # Random contrast adjustment
        contrast = np.random.uniform(0.8, 1.2)
        mean = np.mean(caustics)
        caustics = (caustics - mean) * contrast + mean
        caustics = np.clip(caustics, 0, 1)

        return caustics

    def _generate_instant_fallback(self, width: int, height: int,
                                    brightness: float = 0.4) -> np.ndarray:
        """Generate instant fallback pattern (faster than Perlin).

        Args:
            width: Output width
            height: Output height
            brightness: Brightness factor

        Returns:
            BGR caustics map
        """
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 4 * np.pi, height)
        X, Y = np.meshgrid(x, y)

        caustics = (np.sin(X) * np.cos(Y) +
                   np.sin(2 * X + Y) * 0.5 +
                   np.cos(X - 2 * Y) * 0.3)

        caustics = (caustics - caustics.min()) / (caustics.max() - caustics.min())
        caustics = np.clip(caustics * brightness + (1 - brightness), 0, 2)

        return cv2.cvtColor(caustics.astype(np.float32), cv2.COLOR_GRAY2BGR)

    @property
    def is_ready(self) -> bool:
        """Check if cache is ready."""
        return self._initialized

    @property
    def template_count(self) -> int:
        """Get number of loaded templates."""
        return len(self.templates)


# Singleton instance
_caustics_cache: Optional[CausticsCache] = None


def get_caustics_cache() -> CausticsCache:
    """Get singleton caustics cache instance."""
    global _caustics_cache
    if _caustics_cache is None:
        _caustics_cache = CausticsCache()
        _caustics_cache.ensure_initialized()
    return _caustics_cache


def generate_caustics_map(width: int, height: int,
                          brightness: float = 0.4) -> np.ndarray:
    """Generate caustics map using cache system.

    Args:
        width: Map width
        height: Map height
        brightness: Brightness factor (0.0-1.0)

    Returns:
        BGR caustics map
    """
    cache = get_caustics_cache()
    return cache.get_caustics(width, height, brightness)


def apply_caustics(image: np.ndarray, caustics_map: np.ndarray) -> np.ndarray:
    """Apply caustics map to an image.

    Args:
        image: Input image (BGR)
        caustics_map: Caustics map (BGR float)

    Returns:
        Image with caustics applied
    """
    # Convert to HLS to modify luminosity
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float32)
    hls[:, :, 1] *= caustics_map[:, :, 1]
    hls[:, :, 1] = np.clip(hls[:, :, 1], 0, 255)

    return cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR)
