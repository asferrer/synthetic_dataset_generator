"""
Reference Image Manager
=======================
Manages sets of real reference images for domain gap analysis.
Handles storage, metadata persistence, and pre-computation of
image statistics (LAB/RGB distributions, edge sharpness, brightness).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.job_database import JobDatabase, get_job_db

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np
from loguru import logger

from app.models.schemas import ReferenceImageSet, ReferenceImageStats


# Supported image extensions for reference sets
_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class ReferenceManager:
    """Manages real reference image sets for domain gap analysis.

    Each reference set is stored on disk under ``base_dir/{set_id}/`` and its
    metadata (including pre-computed statistics) is persisted in the shared
    SQLite database via ``JobDatabase``.
    """

    def __init__(self, db: JobDatabase, base_dir: str = "/shared/references"):
        self.db = db
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._init_table()
        logger.info("ReferenceManager initialized (base_dir={})", self.base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_set(
        self,
        name: str,
        description: str,
        domain_id: str,
        image_paths: List[str],
    ) -> ReferenceImageSet:
        """Create a new reference set by copying images and computing stats.

        Args:
            name: Human-readable name for the set.
            description: Optional description.
            domain_id: The domain this set is associated with.
            image_paths: List of source image file paths to include.

        Returns:
            The created ``ReferenceImageSet`` with pre-computed statistics.

        Raises:
            ValueError: If no valid images are provided.
        """
        set_id = str(uuid4())
        image_dir = self.base_dir / set_id
        image_dir.mkdir(parents=True, exist_ok=True)

        # Copy images into the set directory
        copied = 0
        for src_path in image_paths:
            src = Path(src_path)
            if not src.is_file():
                logger.warning("Skipping non-existent file: {}", src_path)
                continue
            if src.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                logger.warning("Skipping unsupported file type: {}", src_path)
                continue
            dst = image_dir / src.name
            # Avoid name collisions by appending a counter
            if dst.exists():
                dst = image_dir / f"{src.stem}_{copied}{src.suffix}"
            shutil.move(str(src), str(dst))
            copied += 1

        if copied == 0:
            # Clean up the empty directory
            shutil.rmtree(str(image_dir), ignore_errors=True)
            raise ValueError("No valid images found in the provided paths")

        logger.info("Moved {} images to {}", copied, image_dir)

        # Pre-compute statistics
        stats = self._compute_stats(str(image_dir))

        # Persist metadata to the database
        now = datetime.now().isoformat()
        with self.db._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO reference_sets
                    (id, name, description, domain_id, image_dir, image_count, stats_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    set_id,
                    name,
                    description,
                    domain_id,
                    str(image_dir),
                    copied,
                    json.dumps(stats.model_dump()),
                    now,
                ),
            )

        logger.info(
            "Created reference set '{}' (id={}, images={})", name, set_id, copied
        )

        return ReferenceImageSet(
            set_id=set_id,
            name=name,
            description=description,
            domain_id=domain_id,
            image_count=copied,
            image_dir=str(image_dir),
            stats=stats,
            created_at=datetime.fromisoformat(now),
        )

    def get_set(self, set_id: str) -> Optional[ReferenceImageSet]:
        """Retrieve a reference set by ID.

        Returns:
            The ``ReferenceImageSet`` or ``None`` if not found.
        """
        with self.db._cursor() as cursor:
            cursor.execute("SELECT * FROM reference_sets WHERE id = ?", (set_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_model(row)

    def list_sets(self, domain_id: Optional[str] = None) -> List[ReferenceImageSet]:
        """List reference sets, optionally filtered by domain.

        Args:
            domain_id: If provided, only return sets for this domain.

        Returns:
            A list of ``ReferenceImageSet`` objects ordered by creation time
            (newest first).
        """
        if domain_id is not None:
            query = "SELECT * FROM reference_sets WHERE domain_id = ? ORDER BY created_at DESC"
            params: tuple = (domain_id,)
        else:
            query = "SELECT * FROM reference_sets ORDER BY created_at DESC"
            params = ()

        with self.db._cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_model(row) for row in rows]

    def delete_set(self, set_id: str) -> bool:
        """Delete a reference set and its images from disk.

        Returns:
            ``True`` if the set was found and deleted, ``False`` otherwise.
        """
        ref_set = self.get_set(set_id)
        if ref_set is None:
            logger.warning("Reference set not found for deletion: {}", set_id)
            return False

        # Remove images from disk
        image_dir = Path(ref_set.image_dir)
        if image_dir.exists():
            shutil.rmtree(str(image_dir), ignore_errors=True)
            logger.info("Removed image directory: {}", image_dir)

        # Remove metadata from the database
        with self.db._cursor() as cursor:
            cursor.execute("DELETE FROM reference_sets WHERE id = ?", (set_id,))
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info("Deleted reference set: {}", set_id)
        return deleted

    def get_set_stats(self, set_id: str) -> Optional[ReferenceImageStats]:
        """Return the pre-computed statistics for a reference set.

        Returns:
            ``ReferenceImageStats`` or ``None`` if the set does not exist.
        """
        with self.db._cursor() as cursor:
            cursor.execute(
                "SELECT stats_json FROM reference_sets WHERE id = ?", (set_id,)
            )
            row = cursor.fetchone()
        if row is None:
            return None

        stats_raw = row["stats_json"]
        if stats_raw is None:
            return None
        return ReferenceImageStats(**json.loads(stats_raw))

    # ------------------------------------------------------------------
    # Chunked upload API
    # ------------------------------------------------------------------

    def create_empty_set(
        self, name: str, description: str, domain_id: str
    ) -> Tuple[str, str]:
        """Create an empty reference set shell (phase 1 of chunked upload).

        Returns:
            Tuple of ``(set_id, image_dir_path)``.
        """
        set_id = str(uuid4())
        image_dir = self.base_dir / set_id
        image_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now().isoformat()
        with self.db._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO reference_sets
                    (id, name, description, domain_id, image_dir, image_count, stats_json, created_at)
                VALUES (?, ?, ?, ?, ?, 0, NULL, ?)
                """,
                (set_id, name, description, domain_id, str(image_dir), now),
            )

        logger.info("Created empty reference set '{}' (id={})", name, set_id)
        return set_id, str(image_dir)

    def get_image_dir(self, set_id: str) -> Optional[str]:
        """Return the image directory path for a set, or None."""
        with self.db._cursor() as cursor:
            cursor.execute(
                "SELECT image_dir FROM reference_sets WHERE id = ?", (set_id,)
            )
            row = cursor.fetchone()
        return row["image_dir"] if row else None

    def add_images_to_set(self, set_id: str, file_paths: List[str]) -> int:
        """Add already-saved image files to a set and update the count.

        Args:
            set_id: The reference set ID.
            file_paths: Paths to images already written to the set's image_dir.

        Returns:
            The new total image count for the set.
        """
        added = len(file_paths)
        with self.db._cursor() as cursor:
            cursor.execute(
                "UPDATE reference_sets SET image_count = image_count + ? WHERE id = ?",
                (added, set_id),
            )
            cursor.execute(
                "SELECT image_count FROM reference_sets WHERE id = ?", (set_id,)
            )
            row = cursor.fetchone()
        total = row["image_count"] if row else added
        logger.info("Added {} images to set {} (total: {})", added, set_id, total)
        return total

    def finalize_set(self, set_id: str) -> ReferenceImageSet:
        """Compute stats and mark the set as finalized (phase 3 of chunked upload).

        Returns:
            The finalized ``ReferenceImageSet``.

        Raises:
            ValueError: If the set does not exist or has no images.
        """
        ref = self.get_set(set_id)
        if ref is None:
            raise ValueError(f"Reference set {set_id} not found")
        if ref.image_count == 0:
            raise ValueError(f"Reference set {set_id} has no images")

        stats = self._compute_stats(ref.image_dir)

        with self.db._cursor() as cursor:
            cursor.execute(
                "UPDATE reference_sets SET stats_json = ?, image_count = ? WHERE id = ?",
                (json.dumps(stats.model_dump()), stats.image_count, set_id),
            )

        logger.info("Finalized reference set {} ({} images)", set_id, stats.image_count)
        return self.get_set(set_id)  # type: ignore[return-value]

    def create_from_directory(
        self,
        name: str,
        description: str,
        domain_id: str,
        source_dir: str,
    ) -> ReferenceImageSet:
        """Create a reference set from an existing server-side directory.

        Images are copied into the managed storage directory.

        Raises:
            ValueError: If the directory does not exist or has no images.
        """
        src_path = Path(source_dir)
        if not src_path.is_dir():
            raise ValueError(f"Directory not found: {source_dir}")

        image_files = sorted(
            p
            for p in src_path.rglob("*")
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )
        if not image_files:
            raise ValueError(f"No valid images found in {source_dir}")

        set_id = str(uuid4())
        image_dir = self.base_dir / set_id
        image_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for src in image_files:
            dst = image_dir / src.name
            if dst.exists():
                dst = image_dir / f"{src.stem}_{copied}{src.suffix}"
            shutil.copy2(str(src), str(dst))
            copied += 1

        logger.info("Copied {} images from {} to {}", copied, source_dir, image_dir)

        stats = self._compute_stats(str(image_dir))

        now = datetime.now().isoformat()
        with self.db._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO reference_sets
                    (id, name, description, domain_id, image_dir, image_count, stats_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    set_id, name, description, domain_id,
                    str(image_dir), copied,
                    json.dumps(stats.model_dump()), now,
                ),
            )

        logger.info("Created reference set '{}' from directory (id={}, images={})", name, set_id, copied)
        return self.get_set(set_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_table(self) -> None:
        """Create the ``reference_sets`` table if it does not already exist."""
        with self.db._cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_sets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    domain_id TEXT NOT NULL,
                    image_dir TEXT NOT NULL,
                    image_count INTEGER NOT NULL DEFAULT 0,
                    stats_json TEXT,
                    created_at TIMESTAMP NOT NULL
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_refsets_domain ON reference_sets(domain_id)"
            )
        logger.debug("reference_sets table initialized")

    @staticmethod
    def _process_single_image(img_path: Path) -> Optional[Dict]:
        """Compute per-image statistics (thread-safe, no shared state)."""
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        return {
            "lab_mean": np.array([lab[:, :, c].mean() for c in range(3)]),
            "lab_std": np.array([lab[:, :, c].std() for c in range(3)]),
            "rgb_mean": np.array([rgb[:, :, c].mean() for c in range(3)]),
            "rgb_std": np.array([rgb[:, :, c].std() for c in range(3)]),
            "lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "brightness": float(lab[:, :, 0].mean()),
        }

    def _compute_stats(self, image_dir: str, max_workers: int = 4) -> ReferenceImageStats:
        """Compute aggregate statistics over all images in a directory.

        Uses a thread pool for parallel image processing.
        """
        dir_path = Path(image_dir)
        image_files = sorted(
            p
            for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )

        if not image_files:
            raise ValueError(f"No valid images found in {image_dir}")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_single_image, p): p
                for p in image_files
            }
            for future in as_completed(futures):
                img_path = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    else:
                        logger.warning("Could not read image, skipping: {}", img_path)
                except Exception as exc:
                    logger.warning("Error processing {}: {}", img_path, exc)

        if not results:
            raise ValueError(f"No images could be read from {image_dir}")

        # Aggregate across images
        channel_means_lab = np.mean([r["lab_mean"] for r in results], axis=0).tolist()
        channel_stds_lab = np.mean([r["lab_std"] for r in results], axis=0).tolist()
        channel_means_rgb = np.mean([r["rgb_mean"] for r in results], axis=0).tolist()
        channel_stds_rgb = np.mean([r["rgb_std"] for r in results], axis=0).tolist()
        avg_edge_variance = float(np.mean([r["lap_var"] for r in results]))
        avg_brightness = float(np.mean([r["brightness"] for r in results]))

        logger.info(
            "Computed stats over {} images: avg_brightness={:.1f}, avg_edge_var={:.1f}",
            len(results),
            avg_brightness,
            avg_edge_variance,
        )

        return ReferenceImageStats(
            channel_means_lab=channel_means_lab,
            channel_stds_lab=channel_stds_lab,
            channel_means_rgb=channel_means_rgb,
            channel_stds_rgb=channel_stds_rgb,
            avg_edge_variance=avg_edge_variance,
            avg_brightness=avg_brightness,
            image_count=len(results),
        )

    @staticmethod
    def _row_to_model(row) -> ReferenceImageSet:
        """Convert a database row into a ``ReferenceImageSet`` model."""
        stats = None
        stats_raw = row["stats_json"]
        if stats_raw:
            try:
                stats = ReferenceImageStats(**json.loads(stats_raw))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Failed to deserialize stats for set {}: {}", row["id"], exc)

        return ReferenceImageSet(
            set_id=row["id"],
            name=row["name"],
            description=row["description"],
            domain_id=row["domain_id"],
            image_count=row["image_count"],
            image_dir=row["image_dir"],
            stats=stats,
            created_at=row["created_at"],
        )
