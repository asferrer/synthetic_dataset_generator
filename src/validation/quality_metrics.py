"""
Quality Metrics for Synthetic Image Validation

Automated quality assessment using:
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Fréchet Inception Distance)
- Isolation Forest anomaly detection
- Composition plausibility checks
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.ensemble import IsolationForest
from scipy import linalg

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality assessment scores for a synthetic image"""
    perceptual_quality: float  # 0-1, LPIPS-based (higher = better)
    distribution_match: float  # 0-1, FID-based (higher = better)
    anomaly_score: float       # 0-1, inlier probability (higher = better)
    composition_score: float   # 0-1, layout plausibility (higher = better)
    overall_score: float       # 0-1, weighted average
    overall_pass: bool         # True if meets all thresholds

    def __str__(self):
        return (f"QualityScore(perceptual={self.perceptual_quality:.3f}, "
                f"distribution={self.distribution_match:.3f}, "
                f"anomaly={self.anomaly_score:.3f}, "
                f"composition={self.composition_score:.3f}, "
                f"overall={self.overall_score:.3f}, "
                f"pass={self.overall_pass})")


@dataclass
class Anomaly:
    """Detected anomaly in an image"""
    type: str          # 'color_bleeding', 'sharp_edges', 'scale_implausible', etc.
    severity: float    # 0-1, how severe (higher = worse)
    location: Optional[Tuple[int, int]] = None  # (y, x) if localizable
    description: str = ""

    def __str__(self):
        loc_str = f" at {self.location}" if self.location else ""
        return f"Anomaly({self.type}, severity={self.severity:.2f}{loc_str})"


class QualityValidator:
    """Automated quality assessment for synthetic images"""

    def __init__(self,
                 reference_dataset_path: Optional[str] = None,
                 device: str = 'cuda',
                 use_lpips: bool = True,
                 use_fid: bool = True,
                 use_anomaly_detection: bool = True):
        """
        Initialize quality validator

        Args:
            reference_dataset_path: Path to real images for FID/LPIPS comparison
            device: 'cuda' or 'cpu'
            use_lpips: Enable LPIPS perceptual quality
            use_fid: Enable FID distribution matching
            use_anomaly_detection: Enable anomaly detection
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")

        self.reference_dataset_path = reference_dataset_path
        self.use_lpips = use_lpips
        self.use_fid = use_fid
        self.use_anomaly_detection = use_anomaly_detection

        # Initialize LPIPS network
        self.lpips_net = None
        if use_lpips:
            try:
                import lpips
                self.lpips_net = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_net.eval()
                logger.info("LPIPS network loaded successfully")
            except ImportError:
                logger.warning("lpips package not found. Install with: pip install lpips")
                self.use_lpips = False
            except Exception as e:
                logger.error(f"Failed to load LPIPS network: {e}")
                self.use_lpips = False

        # Initialize FID reference features
        self.reference_features = None
        self.fid_model = None
        if use_fid and reference_dataset_path:
            try:
                self._initialize_fid(reference_dataset_path)
                logger.info("FID reference features computed")
            except Exception as e:
                logger.error(f"Failed to initialize FID: {e}")
                self.use_fid = False

        # Initialize anomaly detector
        self.anomaly_detector = None
        if use_anomaly_detection:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self._anomaly_trained = False
            logger.info("Anomaly detector initialized")

    def _initialize_fid(self, reference_path: str):
        """Initialize FID by computing reference features"""
        from torchvision.models import inception_v3
        from torchvision import transforms

        # Load InceptionV3 for FID
        self.fid_model = inception_v3(pretrained=True, transform_input=False)
        self.fid_model.fc = torch.nn.Identity()  # Remove final layer
        self.fid_model = self.fid_model.to(self.device)
        self.fid_model.eval()

        # Compute reference features
        ref_path = Path(reference_path)
        if ref_path.exists():
            reference_images = list(ref_path.glob('*.jpg')) + list(ref_path.glob('*.png'))
            if len(reference_images) > 0:
                self.reference_features = self._extract_fid_features(
                    [cv2.imread(str(img)) for img in reference_images[:100]]
                )
            else:
                logger.warning(f"No images found in {reference_path}")
        else:
            logger.warning(f"Reference path does not exist: {reference_path}")

    def validate_image(self,
                      synthetic_img: np.ndarray,
                      reference_imgs: Optional[List[np.ndarray]] = None) -> QualityScore:
        """
        Validate a synthetic image against quality metrics

        Args:
            synthetic_img: BGR image to validate
            reference_imgs: Optional list of reference images for LPIPS

        Returns:
            QualityScore with all metrics
        """
        # 1. Perceptual quality (LPIPS)
        perceptual_quality = 1.0  # Default perfect score
        if self.use_lpips and self.lpips_net is not None and reference_imgs:
            lpips_scores = []
            for ref_img in reference_imgs[:5]:  # Sample 5 random refs
                try:
                    score = self._compute_lpips(synthetic_img, ref_img)
                    lpips_scores.append(score)
                except Exception as e:
                    logger.warning(f"LPIPS computation failed: {e}")

            if lpips_scores:
                # Lower LPIPS = more similar = better (invert for 0-1 scale)
                avg_lpips = np.mean(lpips_scores)
                perceptual_quality = max(0.0, 1.0 - min(avg_lpips, 1.0))

        # 2. Distribution matching (FID)
        distribution_match = 1.0  # Default perfect score
        if self.use_fid and self.fid_model is not None and self.reference_features is not None:
            try:
                synthetic_features = self._extract_fid_features([synthetic_img])
                fid = self._compute_fid(synthetic_features, self.reference_features)
                # Normalize FID to 0-1 scale (lower FID = better)
                distribution_match = 1.0 / (1.0 + fid / 10.0)
            except Exception as e:
                logger.warning(f"FID computation failed: {e}")

        # 3. Anomaly detection
        anomaly_score = 1.0  # Default perfect score
        if self.use_anomaly_detection and self.anomaly_detector is not None and self._anomaly_trained:
            try:
                img_features = self._extract_anomaly_features(synthetic_img)
                anomaly_raw = self.anomaly_detector.score_samples([img_features])[0]
                # Normalize to 0-1 (higher = more normal)
                anomaly_score = max(0.0, min(1.0, (anomaly_raw + 0.5) / 1.0))
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")

        # 4. Composition plausibility
        composition_score = self._check_composition(synthetic_img)

        # Overall score (weighted average)
        weights = {
            'perceptual': 0.3,
            'distribution': 0.25,
            'anomaly': 0.2,
            'composition': 0.25
        }

        overall_score = (
            weights['perceptual'] * perceptual_quality +
            weights['distribution'] * distribution_match +
            weights['anomaly'] * anomaly_score +
            weights['composition'] * composition_score
        )

        # Pass criteria (configurable thresholds)
        overall_pass = (
            perceptual_quality > 0.70 and
            distribution_match > 0.65 and
            anomaly_score > 0.60 and
            composition_score > 0.70
        )

        return QualityScore(
            perceptual_quality=perceptual_quality,
            distribution_match=distribution_match,
            anomaly_score=anomaly_score,
            composition_score=composition_score,
            overall_score=overall_score,
            overall_pass=overall_pass
        )

    def detect_anomalies(self, image: np.ndarray) -> List[Anomaly]:
        """
        Detect obvious artifacts in an image

        Args:
            image: BGR image

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # 1. Color bleeding (extreme LAB variance)
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            a_std = np.std(lab[:, :, 1])
            b_std = np.std(lab[:, :, 2])

            if a_std > 50 or b_std > 50:
                severity = min(1.0, max(a_std, b_std) / 100.0)
                anomalies.append(Anomaly(
                    type='color_bleeding',
                    severity=severity,
                    description=f"High color variance (a={a_std:.1f}, b={b_std:.1f})"
                ))
        except Exception as e:
            logger.warning(f"Color bleeding check failed: {e}")

        # 2. Sharp edges (unrealistic compositing)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            max_edge = np.max(np.abs(laplacian))

            if max_edge > 200:
                severity = min(1.0, max_edge / 300.0)
                y, x = np.unravel_index(np.argmax(np.abs(laplacian)), laplacian.shape)
                anomalies.append(Anomaly(
                    type='sharp_edges',
                    severity=severity,
                    location=(int(y), int(x)),
                    description=f"Sharp edge detected (max={max_edge:.1f})"
                ))
        except Exception as e:
            logger.warning(f"Edge detection failed: {e}")

        # 3. Blur inconsistency
        try:
            blur_variance = self._estimate_blur_variance(image)
            if blur_variance > 100:
                severity = min(1.0, blur_variance / 200.0)
                anomalies.append(Anomaly(
                    type='blur_inconsistency',
                    severity=severity,
                    description=f"Inconsistent blur levels (var={blur_variance:.1f})"
                ))
        except Exception as e:
            logger.warning(f"Blur check failed: {e}")

        # 4. Extreme brightness/darkness
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_mean = np.mean(hsv[:, :, 2])
            v_std = np.std(hsv[:, :, 2])

            if v_mean < 50 or v_mean > 200:
                severity = 0.6
                anomalies.append(Anomaly(
                    type='extreme_brightness',
                    severity=severity,
                    description=f"Unusual brightness (mean={v_mean:.1f})"
                ))

            if v_std < 20:
                severity = 0.5
                anomalies.append(Anomaly(
                    type='low_contrast',
                    severity=severity,
                    description=f"Very low contrast (std={v_std:.1f})"
                ))
        except Exception as e:
            logger.warning(f"Brightness check failed: {e}")

        return anomalies

    def train_anomaly_detector(self, good_images: List[np.ndarray]):
        """
        Train anomaly detector on known good synthetic images

        Args:
            good_images: List of validated good synthetic images
        """
        if not self.use_anomaly_detection or self.anomaly_detector is None:
            logger.warning("Anomaly detection not enabled")
            return

        logger.info(f"Training anomaly detector on {len(good_images)} images...")

        # Extract features from all images
        features = []
        for img in good_images:
            try:
                feat = self._extract_anomaly_features(img)
                features.append(feat)
            except Exception as e:
                logger.warning(f"Feature extraction failed: {e}")

        if len(features) > 10:
            features_array = np.array(features)
            self.anomaly_detector.fit(features_array)
            self._anomaly_trained = True
            logger.info("Anomaly detector trained successfully")
        else:
            logger.warning(f"Not enough samples to train ({len(features)} < 10)")

    def _compute_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute LPIPS distance between two images"""
        # Preprocess for LPIPS
        img1_tensor = self._preprocess_lpips(img1)
        img2_tensor = self._preprocess_lpips(img2)

        with torch.no_grad():
            distance = self.lpips_net(img1_tensor, img2_tensor).item()

        return distance

    def _preprocess_lpips(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image for LPIPS (expects RGB [-1, 1])"""
        # BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 256x256 (LPIPS standard)
        img_resized = cv2.resize(img_rgb, (256, 256))

        # Normalize to [-1, 1]
        img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0

        # Convert to tensor
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        return img_tensor

    def _extract_fid_features(self, images: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract InceptionV3 features for FID computation"""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        features_list = []

        for img in images:
            # BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Transform
            img_tensor = transform(img_rgb).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.fid_model(img_tensor)

            features_list.append(features.cpu().numpy())

        # Concatenate all features
        features_array = np.concatenate(features_list, axis=0)

        # Compute mean and covariance
        mu = np.mean(features_array, axis=0)
        sigma = np.cov(features_array, rowvar=False)

        return {'mu': mu, 'sigma': sigma}

    def _compute_fid(self,
                     features1: Dict[str, np.ndarray],
                     features2: Dict[str, np.ndarray]) -> float:
        """Compute Fréchet Inception Distance between two feature distributions"""
        mu1, sigma1 = features1['mu'], features1['sigma']
        mu2, sigma2 = features2['mu'], features2['sigma']

        # Compute squared difference of means
        diff = mu1 - mu2

        # Compute sqrt of product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        # Handle numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Compute FID
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

        return float(fid)

    def _extract_anomaly_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features for anomaly detection"""
        features = []

        # Color features (LAB statistics)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        features.extend([
            np.mean(lab[:, :, 0]),  # L mean
            np.std(lab[:, :, 0]),   # L std
            np.mean(lab[:, :, 1]),  # a mean
            np.std(lab[:, :, 1]),   # a std
            np.mean(lab[:, :, 2]),  # b mean
            np.std(lab[:, :, 2]),   # b std
        ])

        # Texture features (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(np.var(laplacian))

        # Edge strength
        features.append(np.mean(np.abs(laplacian)))

        # Blur estimation
        features.append(self._estimate_blur(gray))

        # Brightness/contrast
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features.extend([
            np.mean(hsv[:, :, 2]),  # Brightness
            np.std(hsv[:, :, 2]),   # Contrast
        ])

        return np.array(features)

    def _check_composition(self, image: np.ndarray) -> float:
        """
        Check composition plausibility using heuristics

        Returns:
            Score 0-1 (higher = better composition)
        """
        score = 1.0

        h, w = image.shape[:2]

        # 1. Color balance (no extreme color dominance)
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_channel = hsv[:, :, 0]

            # Check if one hue dominates >80%
            hue_hist, _ = np.histogram(h_channel, bins=18, range=(0, 180))
            max_hue_ratio = np.max(hue_hist) / np.sum(hue_hist)

            if max_hue_ratio > 0.8:
                score *= 0.8  # Penalize extreme hue dominance
        except:
            pass

        # 2. Spatial distribution (check for extreme clustering)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Divide into quadrants
            quad_h, quad_w = h // 2, w // 2
            quadrants = [
                gray[:quad_h, :quad_w],
                gray[:quad_h, quad_w:],
                gray[quad_h:, :quad_w],
                gray[quad_h:, quad_w:]
            ]

            # Check variance in each quadrant
            variances = [np.var(q) for q in quadrants]

            # If one quadrant has very low variance (empty)
            if min(variances) < 100:
                score *= 0.9
        except:
            pass

        # 3. Overall complexity (not too simple, not too noisy)
        try:
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Ideal edge density: 0.05 - 0.20
            if edge_density < 0.02 or edge_density > 0.3:
                score *= 0.85
        except:
            pass

        return score

    def _estimate_blur(self, gray_img: np.ndarray) -> float:
        """Estimate blur using Laplacian variance"""
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        return float(np.var(laplacian))

    def _estimate_blur_variance(self, image: np.ndarray) -> float:
        """Estimate spatial variance in blur levels"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Split into patches
        patch_size = 64
        blur_values = []

        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = gray[y:y+patch_size, x:x+patch_size]
                blur = self._estimate_blur(patch)
                blur_values.append(blur)

        # Return variance of blur values
        return float(np.var(blur_values)) if blur_values else 0.0
