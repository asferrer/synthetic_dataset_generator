"""
COCO Segmentation format exporter - COCO with polygon segmentations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

from .coco_exporter import COCOExporter
from .base_exporter import ExportConfig, ExportResult

logger = logging.getLogger(__name__)


class COCOSegmentationExporter(COCOExporter):
    """
    Exporter for COCO format with polygon segmentations.

    Can generate segmentation polygons from:
    1. Existing mask images
    2. Bounding boxes (rectangular polygons)
    """

    def __init__(self, config: ExportConfig,
                 polygon_tolerance: float = 2.0,
                 min_polygon_area: int = 100,
                 masks_dir: Optional[str] = None):
        """
        Initialize COCO Segmentation exporter.

        Args:
            config: Export configuration
            polygon_tolerance: Tolerance for polygon simplification (cv2.approxPolyDP)
            min_polygon_area: Minimum area for valid polygons
            masks_dir: Optional directory containing mask images
        """
        super().__init__(config)
        self.polygon_tolerance = polygon_tolerance
        self.min_polygon_area = min_polygon_area
        self.masks_dir = masks_dir

    def get_format_name(self) -> str:
        """Return format name."""
        return "COCO_Segmentation"

    def export(self,
               coco_data: Dict[str, Any],
               images_dir: str,
               output_name: str = "dataset",
               masks_dir: Optional[str] = None) -> ExportResult:
        """
        Export to COCO format with segmentation polygons.

        Args:
            coco_data: Dataset in COCO format
            images_dir: Directory containing source images
            output_name: Base name for output
            masks_dir: Optional directory with mask images (overrides init)

        Returns:
            ExportResult with export details
        """
        errors = []
        warnings = []

        # Use provided masks_dir or fall back to init value
        masks_directory = masks_dir or self.masks_dir

        # Validate input
        if not self.validate_input(coco_data):
            return ExportResult(
                success=False,
                output_path="",
                num_images=0,
                num_annotations=0,
                format_name=self.get_format_name(),
                errors=["Invalid COCO data structure"]
            )

        try:
            # Build image lookup
            images_dict = self._build_image_lookup(coco_data['images'])

            # Process annotations to add/update segmentations
            enhanced_annotations = []
            segmentations_added = 0

            for ann in coco_data['annotations']:
                enhanced_ann = ann.copy()

                # Check if already has valid segmentation
                if self._has_valid_segmentation(ann):
                    enhanced_annotations.append(enhanced_ann)
                    continue

                # Try to generate segmentation from mask
                if masks_directory:
                    img_info = images_dict.get(ann['image_id'])
                    if img_info:
                        polygon = self._mask_to_polygon(ann, masks_directory, img_info)
                        if polygon:
                            enhanced_ann['segmentation'] = polygon
                            segmentations_added += 1
                            enhanced_annotations.append(enhanced_ann)
                            continue

                # Fall back to bbox polygon
                if 'bbox' in ann:
                    enhanced_ann['segmentation'] = self._bbox_to_polygon(ann['bbox'])
                    segmentations_added += 1
                    warnings.append(f"Generated bbox polygon for annotation {ann.get('id', 'unknown')}")

                enhanced_annotations.append(enhanced_ann)

            # Create enhanced data with segmentations
            enhanced_data = coco_data.copy()
            enhanced_data['annotations'] = enhanced_annotations

            logger.info(f"Added/updated {segmentations_added} segmentations")

            # Use parent class export with enhanced data
            result = super().export(enhanced_data, images_dir, output_name)

            # Update result with our warnings
            result.warnings.extend(warnings)
            result.errors.extend(errors)

            return result

        except Exception as e:
            logger.error(f"COCO Segmentation export failed: {e}")
            return ExportResult(
                success=False,
                output_path="",
                num_images=0,
                num_annotations=0,
                format_name=self.get_format_name(),
                errors=[str(e)]
            )

    def _has_valid_segmentation(self, ann: Dict) -> bool:
        """
        Check if annotation has valid segmentation.

        Args:
            ann: Annotation dict

        Returns:
            True if has valid segmentation
        """
        if 'segmentation' not in ann:
            return False

        seg = ann['segmentation']

        # Empty segmentation
        if not seg:
            return False

        # Check for polygon format (list of lists)
        if isinstance(seg, list) and len(seg) > 0:
            if isinstance(seg[0], list) and len(seg[0]) >= 6:
                return True

        # RLE format (dict with 'counts' and 'size')
        if isinstance(seg, dict) and 'counts' in seg:
            return True

        return False

    def _mask_to_polygon(self, ann: Dict, masks_dir: str,
                         img_info: Dict) -> Optional[List[List[float]]]:
        """
        Convert binary mask to polygon segmentation.

        Args:
            ann: Annotation dict
            masks_dir: Directory with mask images
            img_info: Image info dict

        Returns:
            List of polygon coordinates or None
        """
        try:
            # Try different mask naming conventions
            base_name = Path(img_info['file_name']).stem
            mask_candidates = [
                f"{base_name}_mask.png",
                f"{base_name}_mask_{ann['id']}.png",
                f"mask_{ann['id']}.png",
                f"{base_name}.png"
            ]

            mask_path = None
            for candidate in mask_candidates:
                path = Path(masks_dir) / candidate
                if path.exists():
                    mask_path = path
                    break

            if not mask_path:
                return None

            # Read mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return None

            # If annotation has bbox, crop mask to that region
            if 'bbox' in ann:
                x, y, w, h = [int(v) for v in ann['bbox']]
                # Ensure bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, mask.shape[1] - x)
                h = min(h, mask.shape[0] - y)

                roi_mask = mask[y:y+h, x:x+w]
            else:
                roi_mask = mask
                x, y = 0, 0

            # Threshold
            _, binary = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            polygons = []
            for contour in contours:
                # Simplify contour
                epsilon = self.polygon_tolerance
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Check minimum area
                if cv2.contourArea(approx) < self.min_polygon_area:
                    continue

                # Convert to COCO format [x1,y1,x2,y2,...]
                # Add offset back if we cropped
                polygon = []
                for point in approx:
                    px, py = point[0]
                    polygon.extend([float(px + x), float(py + y)])

                if len(polygon) >= 6:  # Minimum 3 points
                    polygons.append(polygon)

            return polygons if polygons else None

        except Exception as e:
            logger.warning(f"Failed to convert mask to polygon: {e}")
            return None

    def _bbox_to_polygon(self, bbox: List[float]) -> List[List[float]]:
        """
        Convert bounding box to rectangular polygon.

        Args:
            bbox: COCO bbox [x, y, width, height]

        Returns:
            Polygon as list of coordinates [x1,y1,x2,y2,x3,y3,x4,y4]
        """
        x, y, w, h = bbox

        # Rectangle corners: top-left, top-right, bottom-right, bottom-left
        polygon = [
            x, y,           # Top-left
            x + w, y,       # Top-right
            x + w, y + h,   # Bottom-right
            x, y + h        # Bottom-left
        ]

        return [polygon]

    def _rle_to_polygon(self, rle: Dict, img_height: int, img_width: int) -> Optional[List[List[float]]]:
        """
        Convert RLE segmentation to polygon.

        Args:
            rle: RLE encoded segmentation
            img_height: Image height
            img_width: Image width

        Returns:
            List of polygons or None
        """
        try:
            # Decode RLE to binary mask
            if 'counts' not in rle or 'size' not in rle:
                return None

            # Simple RLE decoding
            counts = rle['counts']
            if isinstance(counts, str):
                # Compressed RLE - would need pycocotools
                logger.warning("Compressed RLE requires pycocotools, falling back to bbox")
                return None

            # Uncompressed RLE
            mask = np.zeros(img_height * img_width, dtype=np.uint8)
            position = 0
            for i, count in enumerate(counts):
                if i % 2 == 1:  # Odd indices are foreground
                    mask[position:position + count] = 255
                position += count

            mask = mask.reshape((img_height, img_width), order='F')

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            polygons = []
            for contour in contours:
                approx = cv2.approxPolyDP(contour, self.polygon_tolerance, True)
                if cv2.contourArea(approx) >= self.min_polygon_area:
                    polygon = approx.flatten().tolist()
                    if len(polygon) >= 6:
                        polygons.append(polygon)

            return polygons if polygons else None

        except Exception as e:
            logger.warning(f"Failed to convert RLE to polygon: {e}")
            return None
