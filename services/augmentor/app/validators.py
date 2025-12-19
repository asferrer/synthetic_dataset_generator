"""
Validators Module
=================
Quality and Physics validation for synthetic images.

Features:
- LPIPS perceptual quality assessment
- Isolation Forest anomaly detection
- Physics plausibility checks (gravity, buoyancy, occlusion)
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from app.models.schemas import (
    AnnotationBox,
    QualityScoreInfo,
    PhysicsViolationInfo,
)


# =============================================================================
# Quality Validator
# =============================================================================

class QualityValidator:
    """
    Validates synthetic image quality using perceptual metrics.

    Features:
    - LPIPS perceptual quality (requires lpips package)
    - Isolation Forest anomaly detection
    - Composition plausibility checks
    """

    def __init__(
        self,
        use_gpu: bool = True,
        use_lpips: bool = True,
        use_anomaly_detection: bool = True,
    ):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.use_lpips = use_lpips
        self.use_anomaly_detection = use_anomaly_detection

        # Initialize LPIPS
        self.lpips_net = None
        if use_lpips:
            self._init_lpips()

        # Initialize anomaly detector
        self.anomaly_detector = None
        self._anomaly_trained = False
        if use_anomaly_detection:
            self._init_anomaly_detector()

        logger.info(f"QualityValidator initialized (GPU: {use_gpu}, LPIPS: {self.lpips_net is not None})")

    def _init_lpips(self):
        """Initialize LPIPS network"""
        try:
            import torch
            if self.device == 'cuda' and not torch.cuda.is_available():
                self.device = 'cpu'
                logger.warning("CUDA not available, using CPU")

            import lpips
            self.lpips_net = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_net.eval()
            logger.info("LPIPS network loaded")
        except ImportError:
            logger.warning("lpips package not found, quality validation limited")
            self.use_lpips = False
        except Exception as e:
            logger.error(f"Failed to load LPIPS: {e}")
            self.use_lpips = False

    def _init_anomaly_detector(self):
        """Initialize Isolation Forest anomaly detector"""
        try:
            from sklearn.ensemble import IsolationForest
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            logger.info("Anomaly detector initialized")
        except ImportError:
            logger.warning("sklearn not found, anomaly detection disabled")
            self.use_anomaly_detection = False

    async def validate(
        self,
        image_path: str,
        annotations: List[AnnotationBox],
        reference_images: Optional[List[str]] = None,
        check_quality: bool = True,
        check_anomalies: bool = True,
        min_perceptual_quality: float = 0.7,
        min_anomaly_score: float = 0.6,
    ) -> QualityScoreInfo:
        """
        Validate image quality.

        Args:
            image_path: Path to image to validate
            annotations: Object annotations
            reference_images: Optional paths to reference images
            check_quality: Run LPIPS quality check
            check_anomalies: Run anomaly detection
            min_perceptual_quality: Minimum acceptable quality
            min_anomaly_score: Minimum acceptable anomaly score

        Returns:
            QualityScoreInfo with all metrics
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load reference images if provided
        ref_imgs = []
        if reference_images:
            for ref_path in reference_images[:5]:  # Limit to 5 refs
                ref_img = cv2.imread(ref_path)
                if ref_img is not None:
                    ref_imgs.append(ref_img)

        # Calculate scores
        perceptual_quality = await self._compute_perceptual_quality(image, ref_imgs) if check_quality else 1.0
        anomaly_score = await self._compute_anomaly_score(image) if check_anomalies else 1.0
        composition_score = self._check_composition(image, annotations)

        # Overall score
        overall_score = (
            perceptual_quality * 0.4 +
            anomaly_score * 0.3 +
            composition_score * 0.3
        )

        # Check if passes thresholds
        overall_pass = (
            perceptual_quality >= min_perceptual_quality and
            anomaly_score >= min_anomaly_score and
            composition_score >= 0.5
        )

        return QualityScoreInfo(
            perceptual_quality=round(perceptual_quality, 3),
            distribution_match=1.0,  # FID not implemented in microservice
            anomaly_score=round(anomaly_score, 3),
            composition_score=round(composition_score, 3),
            overall_score=round(overall_score, 3),
            overall_pass=overall_pass,
        )

    async def _compute_perceptual_quality(
        self,
        image: np.ndarray,
        reference_imgs: List[np.ndarray],
    ) -> float:
        """Compute LPIPS-based perceptual quality"""
        if not self.use_lpips or self.lpips_net is None:
            return 1.0

        if not reference_imgs:
            # No references, use self-similarity check
            return self._check_self_consistency(image)

        try:
            import torch

            lpips_scores = []
            for ref_img in reference_imgs:
                # Resize to same dimensions
                h, w = image.shape[:2]
                ref_resized = cv2.resize(ref_img, (w, h))

                # Convert to tensors
                img_tensor = self._to_tensor(image)
                ref_tensor = self._to_tensor(ref_resized)

                # Compute LPIPS
                with torch.no_grad():
                    score = self.lpips_net(img_tensor, ref_tensor)
                    lpips_scores.append(score.item())

            if lpips_scores:
                # Lower LPIPS = more similar = better
                avg_lpips = np.mean(lpips_scores)
                return max(0.0, 1.0 - min(avg_lpips, 1.0))

        except Exception as e:
            logger.warning(f"LPIPS computation failed: {e}")

        return 1.0

    async def _compute_anomaly_score(self, image: np.ndarray) -> float:
        """Compute anomaly detection score"""
        if not self.use_anomaly_detection or self.anomaly_detector is None:
            return 1.0

        try:
            features = self._extract_anomaly_features(image)

            # If not trained, use heuristic-based detection
            if not self._anomaly_trained:
                return self._heuristic_anomaly_check(image, features)

            # Use trained model
            raw_score = self.anomaly_detector.score_samples([features])[0]
            return max(0.0, min(1.0, (raw_score + 0.5) / 1.0))

        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return 1.0

    def _check_composition(
        self,
        image: np.ndarray,
        annotations: List[AnnotationBox],
    ) -> float:
        """Check composition plausibility"""
        score = 1.0
        h, w = image.shape[:2]
        image_area = h * w

        # Check each annotation
        for ann in annotations:
            # Size check
            obj_area = ann.width * ann.height
            area_ratio = obj_area / image_area

            if area_ratio < 0.001:  # Too small
                score -= 0.1
            elif area_ratio > 0.5:  # Too large
                score -= 0.2

            # Position check (objects at edges)
            center_x = ann.x + ann.width / 2
            center_y = ann.y + ann.height / 2

            margin_x = min(center_x, w - center_x) / w
            margin_y = min(center_y, h - center_y) / h

            if margin_x < 0.05 or margin_y < 0.05:
                score -= 0.1  # Too close to edge

            # Aspect ratio check
            aspect = ann.width / max(ann.height, 1)
            if aspect > 10 or aspect < 0.1:
                score -= 0.15  # Extreme aspect ratio

        # Check overall density
        if len(annotations) > 10:
            score -= 0.1  # Too many objects

        return max(0.0, min(1.0, score))

    def _check_self_consistency(self, image: np.ndarray) -> float:
        """Check image self-consistency without references"""
        try:
            # Check for artifacts
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size

            # High edge density might indicate artifacts
            if edge_density > 0.3:
                return 0.7

            # Check for color consistency
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_std = lab[:, :, 0].std()
            a_std = lab[:, :, 1].std()
            b_std = lab[:, :, 2].std()

            # Very low variance might indicate unnatural uniformity
            if l_std < 10 or (a_std < 5 and b_std < 5):
                return 0.8

            return 1.0

        except Exception as e:
            logger.warning(f"Self-consistency check failed: {e}")
            return 1.0

    def _extract_anomaly_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features for anomaly detection"""
        features = []

        # Color statistics
        for channel in cv2.split(image):
            features.extend([
                channel.mean(),
                channel.std(),
                np.percentile(channel, 25),
                np.percentile(channel, 75),
            ])

        # Edge statistics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.count_nonzero(edges) / edges.size)

        # Laplacian variance (focus measure)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(laplacian.var())

        # Texture statistics (GLCM-like)
        h, w = gray.shape
        features.append(gray[:h//2, :].mean() - gray[h//2:, :].mean())  # Top-bottom diff
        features.append(gray[:, :w//2].mean() - gray[:, w//2:].mean())  # Left-right diff

        return np.array(features)

    def _heuristic_anomaly_check(
        self,
        image: np.ndarray,
        features: np.ndarray,
    ) -> float:
        """Heuristic-based anomaly detection without training"""
        score = 1.0

        # Check for extreme color values
        b, g, r = cv2.split(image)
        for channel in [b, g, r]:
            if channel.mean() < 10 or channel.mean() > 245:
                score -= 0.1
            if channel.std() < 5:
                score -= 0.1

        # Check Laplacian variance (blur detection)
        laplacian_var = features[-3] if len(features) > 3 else 100
        if laplacian_var < 50:
            score -= 0.15  # Too blurry

        return max(0.0, min(1.0, score))

    def _to_tensor(self, image: np.ndarray):
        """Convert BGR image to LPIPS-compatible tensor"""
        import torch

        # BGR to RGB, normalize to [-1, 1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 127.5 - 1.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)


# =============================================================================
# Physics Validator
# =============================================================================

class PhysicsValidator:
    """
    Validates physical plausibility of synthetic compositions.

    Features:
    - Gravity/buoyancy checks for underwater scenes
    - Material density validation
    - Scale plausibility
    """

    # Material densities relative to water (1.0)
    MATERIAL_DENSITIES = {
        # Marine life
        'fish': 1.05, 'shark': 1.08, 'ray': 1.06, 'jellyfish': 1.02,
        'coral': 1.10, 'starfish': 1.05, 'crab': 1.15, 'octopus': 1.03,
        # Plastics (float)
        'plastic': 0.90, 'plastic_bag': 0.92, 'plastic_bottle': 0.95,
        'styrofoam': 0.05, 'foam': 0.10, 'bottle': 0.95,
        # Metals (sink)
        'metal': 7.80, 'aluminum': 2.70, 'can': 2.70,
        # Glass (sink)
        'glass': 2.50, 'glass_bottle': 2.50,
        # Other
        'wood': 0.60, 'debris': 1.50, 'trash': 1.20,
    }

    def __init__(
        self,
        density_threshold_float: float = 0.95,
        density_threshold_sink: float = 1.15,
        surface_zone: float = 0.25,
        bottom_zone: float = 0.75,
    ):
        self.density_threshold_float = density_threshold_float
        self.density_threshold_sink = density_threshold_sink
        self.surface_zone = surface_zone
        self.bottom_zone = bottom_zone

        logger.info("PhysicsValidator initialized")

    async def validate(
        self,
        annotations: List[AnnotationBox],
        depth_map_path: Optional[str] = None,
        image_height: int = 1080,
        scene_type: str = 'underwater',
    ) -> List[PhysicsViolationInfo]:
        """
        Validate physics plausibility of object placements.

        Args:
            annotations: Object annotations
            depth_map_path: Optional depth map for occlusion checks
            image_height: Image height for position calculations
            scene_type: 'underwater', 'surface', 'ground'

        Returns:
            List of physics violations
        """
        violations = []

        # Check gravity/buoyancy
        gravity_violations = self._check_gravity(annotations, image_height, scene_type)
        violations.extend(gravity_violations)

        # Check scale plausibility
        scale_violations = self._check_scale_plausibility(annotations, image_height)
        violations.extend(scale_violations)

        # Check occlusion if depth map available
        if depth_map_path:
            try:
                depth_map = np.load(depth_map_path)
                occlusion_violations = self._check_occlusion(annotations, depth_map)
                violations.extend(occlusion_violations)
            except Exception as e:
                logger.warning(f"Occlusion check failed: {e}")

        return violations

    def _check_gravity(
        self,
        annotations: List[AnnotationBox],
        image_height: int,
        scene_type: str,
    ) -> List[PhysicsViolationInfo]:
        """Check gravity/buoyancy violations"""
        violations = []

        for ann in annotations:
            # Get material density
            material = self._classify_material(ann.class_name)
            density = self.MATERIAL_DENSITIES.get(material, 1.0)

            # Calculate vertical position (0=top, 1=bottom)
            center_y = ann.y + ann.height / 2
            relative_depth = center_y / image_height

            if scene_type == 'underwater':
                # Heavy objects floating near surface
                if density > self.density_threshold_sink and relative_depth < self.surface_zone:
                    violations.append(PhysicsViolationInfo(
                        violation_type='gravity',
                        object_class=ann.class_name,
                        severity='high' if density > 2.0 else 'medium',
                        description=f"{ann.class_name} (density={density:.2f}) floating near surface",
                        suggested_fix="Move object lower in frame or replace with lighter material",
                    ))

                # Light objects at seafloor
                if density < self.density_threshold_float and relative_depth > self.bottom_zone:
                    violations.append(PhysicsViolationInfo(
                        violation_type='buoyancy',
                        object_class=ann.class_name,
                        severity='medium' if density > 0.5 else 'high',
                        description=f"{ann.class_name} (density={density:.2f}) at seafloor",
                        suggested_fix="Move object higher in frame or add anchoring context",
                    ))

                # Very buoyant objects not at surface
                if density < 0.20 and relative_depth > 0.15:
                    violations.append(PhysicsViolationInfo(
                        violation_type='buoyancy',
                        object_class=ann.class_name,
                        severity='high',
                        description=f"{ann.class_name} (very buoyant) not at surface",
                        suggested_fix="Position object at water surface (top of frame)",
                    ))

        return violations

    def _check_scale_plausibility(
        self,
        annotations: List[AnnotationBox],
        image_height: int,
    ) -> List[PhysicsViolationInfo]:
        """Check for implausible scale relationships"""
        violations = []

        # Known size ranges (relative to image)
        SIZE_RANGES = {
            'fish': (0.01, 0.3),
            'shark': (0.05, 0.5),
            'jellyfish': (0.02, 0.25),
            'plastic_bottle': (0.01, 0.15),
            'can': (0.005, 0.1),
        }

        for ann in annotations:
            material = self._classify_material(ann.class_name)
            if material in SIZE_RANGES:
                min_ratio, max_ratio = SIZE_RANGES[material]
                actual_ratio = ann.height / image_height

                if actual_ratio < min_ratio * 0.5:
                    violations.append(PhysicsViolationInfo(
                        violation_type='scale',
                        object_class=ann.class_name,
                        severity='low',
                        description=f"{ann.class_name} appears too small",
                        suggested_fix="Increase object scale or move to background",
                    ))
                elif actual_ratio > max_ratio * 1.5:
                    violations.append(PhysicsViolationInfo(
                        violation_type='scale',
                        object_class=ann.class_name,
                        severity='low',
                        description=f"{ann.class_name} appears too large",
                        suggested_fix="Decrease object scale or move to foreground",
                    ))

        return violations

    def _check_occlusion(
        self,
        annotations: List[AnnotationBox],
        depth_map: np.ndarray,
    ) -> List[PhysicsViolationInfo]:
        """Check occlusion correctness based on depth"""
        violations = []

        # Sort by position (front to back based on depth)
        annotated_depths = []
        for ann in annotations:
            # Get depth at object center
            cy = min(ann.y + ann.height // 2, depth_map.shape[0] - 1)
            cx = min(ann.x + ann.width // 2, depth_map.shape[1] - 1)
            depth = depth_map[cy, cx]
            annotated_depths.append((ann, depth))

        # Sort by depth (smaller = closer)
        annotated_depths.sort(key=lambda x: x[1])

        # Check for overlapping objects with wrong depth order
        for i, (ann1, depth1) in enumerate(annotated_depths):
            for ann2, depth2 in annotated_depths[i+1:]:
                # Check if bboxes overlap
                if self._boxes_overlap(ann1, ann2):
                    # ann1 should be in front (smaller depth)
                    if depth1 > depth2 * 1.1:  # Tolerance
                        violations.append(PhysicsViolationInfo(
                            violation_type='occlusion',
                            object_class=f"{ann1.class_name} vs {ann2.class_name}",
                            severity='medium',
                            description=f"Depth order inconsistency between overlapping objects",
                            suggested_fix="Adjust object z-order or placement",
                        ))

        return violations

    def _boxes_overlap(self, ann1: AnnotationBox, ann2: AnnotationBox) -> bool:
        """Check if two bounding boxes overlap"""
        x1_1, y1_1 = ann1.x, ann1.y
        x2_1, y2_1 = ann1.x + ann1.width, ann1.y + ann1.height
        x1_2, y1_2 = ann2.x, ann2.y
        x2_2, y2_2 = ann2.x + ann2.width, ann2.y + ann2.height

        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

    def _classify_material(self, category: str) -> str:
        """Classify object category to material type"""
        category_lower = category.lower()

        # Direct matches
        if category_lower in self.MATERIAL_DENSITIES:
            return category_lower

        # Keyword matching
        keywords = {
            'fish': ['fish', 'tuna', 'salmon', 'bass', 'trout'],
            'shark': ['shark'],
            'plastic': ['plastic', 'wrapper', 'film'],
            'plastic_bottle': ['bottle', 'container'],
            'can': ['can', 'tin'],
            'glass': ['glass', 'jar'],
            'metal': ['metal', 'steel', 'iron'],
            'debris': ['debris', 'waste', 'garbage'],
            'coral': ['coral', 'reef'],
        }

        for material, words in keywords.items():
            if any(word in category_lower for word in words):
                return material

        return 'debris'  # Default
