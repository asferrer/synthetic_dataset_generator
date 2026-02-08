"""
Advisor Engine - Domain Gap Analysis & Parameter Suggestion
============================================================
Analyzes domain gap metrics between synthetic and real image sets
and generates actionable parameter adjustment suggestions.

This engine is purely analytical (CPU-only, no GPU required).
It compares statistical distributions of color, edges, frequency,
and texture between two image sets and maps detected issues to
concrete pipeline parameter changes.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from app.models.schemas import (
    GapIssue,
    ImpactLevel,
    IssueCategory,
    IssueSeverity,
    ParameterSuggestion,
)

# Severity ordering for sorting (higher = more severe)
_SEVERITY_ORDER = {
    IssueSeverity.HIGH: 0,
    IssueSeverity.MEDIUM: 1,
    IssueSeverity.LOW: 2,
}

# Impact ordering for sorting (higher = more impactful)
_IMPACT_ORDER = {
    ImpactLevel.HIGH: 0,
    ImpactLevel.MEDIUM: 1,
    ImpactLevel.LOW: 2,
}

# Supported image extensions
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class AdvisorEngine:
    """
    Analyzes domain gap between synthetic and real image sets and
    generates parameter adjustment suggestions.

    All analysis is performed on CPU using OpenCV and NumPy.
    No GPU or deep learning models are required.
    """

    def __init__(self) -> None:
        logger.info("AdvisorEngine initialized (CPU-only analytical engine)")

    # =========================================================================
    # Public API
    # =========================================================================

    def analyze(
        self,
        synthetic_dir: str,
        real_dir: str,
        max_images: int = 50,
        current_config: Optional[Dict] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[List[GapIssue], List[ParameterSuggestion]]:
        """
        Run full domain gap analysis between synthetic and real image sets.

        Args:
            synthetic_dir: Directory containing synthetic images.
            real_dir: Directory containing real reference images.
            max_images: Maximum number of images to sample per set.
            current_config: Optional current pipeline configuration for
                            extracting current parameter values.

        Returns:
            Tuple of (issues, suggestions) sorted by severity/impact.

        Raises:
            ValueError: If either directory is empty or contains no valid images.
        """
        logger.info(
            "Starting domain gap analysis: synthetic={} real={} max_images={}",
            synthetic_dir,
            real_dir,
            max_images,
        )

        # Load image samples
        synthetic_images = self._load_images_sample(synthetic_dir, max_images)
        real_images = self._load_images_sample(real_dir, max_images)

        if len(synthetic_images) == 0:
            raise ValueError(
                f"No valid images found in synthetic directory: {synthetic_dir}"
            )
        if len(real_images) == 0:
            raise ValueError(
                f"No valid images found in real directory: {real_dir}"
            )

        if len(synthetic_images) < 3:
            logger.warning(
                "Very few synthetic images ({}). Results may be unreliable.",
                len(synthetic_images),
            )
        if len(real_images) < 3:
            logger.warning(
                "Very few real images ({}). Results may be unreliable.",
                len(real_images),
            )

        logger.info(
            "Loaded {} synthetic and {} real images for analysis",
            len(synthetic_images),
            len(real_images),
        )

        # Shorthand for progress reporting
        def _cb(phase: str, fraction: float) -> None:
            if progress_callback:
                progress_callback(phase, fraction)

        # Run all analysis passes
        issues: List[GapIssue] = []

        try:
            _cb("analyzing_color", 0.0)
            issues.extend(self._analyze_color(synthetic_images, real_images))
            _cb("analyzing_color", 1.0)
        except Exception as e:
            logger.error("Color analysis failed: {}", e)

        try:
            _cb("analyzing_edges", 0.0)
            issues.extend(self._analyze_edges(synthetic_images, real_images))
            _cb("analyzing_edges", 1.0)
        except Exception as e:
            logger.error("Edge analysis failed: {}", e)

        try:
            _cb("analyzing_frequency", 0.0)
            issues.extend(self._analyze_frequency(synthetic_images, real_images))
            _cb("analyzing_frequency", 1.0)
        except Exception as e:
            logger.error("Frequency analysis failed: {}", e)

        try:
            _cb("analyzing_texture", 0.0)
            issues.extend(self._analyze_texture(synthetic_images, real_images))
            _cb("analyzing_texture", 1.0)
        except Exception as e:
            logger.error("Texture analysis failed: {}", e)

        # Sort issues by severity (HIGH first)
        issues.sort(key=lambda i: _SEVERITY_ORDER.get(i.severity, 99))

        # Generate suggestions from detected issues
        _cb("generating_suggestions", 0.0)
        suggestions = self._generate_suggestions(issues, current_config)
        _cb("generating_suggestions", 1.0)

        logger.info(
            "Analysis complete: {} issues detected, {} suggestions generated",
            len(issues),
            len(suggestions),
        )

        return issues, suggestions

    # =========================================================================
    # Color Analysis
    # =========================================================================

    def _analyze_color(
        self,
        synthetic_images: List[np.ndarray],
        real_images: List[np.ndarray],
    ) -> List[GapIssue]:
        """
        Compare color distributions in LAB color space.

        Checks:
            - L channel mean difference (brightness)
            - L channel std difference (contrast)
            - A/B channel mean difference (color cast)
            - Overall saturation difference
        """
        issues: List[GapIssue] = []

        # Convert to LAB
        syn_lab_stats = self._compute_lab_stats(synthetic_images)
        real_lab_stats = self._compute_lab_stats(real_images)

        syn_l_mean, syn_l_std, syn_a_mean, syn_b_mean = syn_lab_stats
        real_l_mean, real_l_std, real_a_mean, real_b_mean = real_lab_stats

        # --- Brightness (L channel mean) ---
        l_mean_diff = abs(syn_l_mean - real_l_mean)
        if l_mean_diff > 15:
            direction = "brighter" if syn_l_mean > real_l_mean else "darker"
            issues.append(
                GapIssue(
                    category=IssueCategory.BRIGHTNESS,
                    severity=IssueSeverity.HIGH,
                    description=(
                        f"Synthetic images are significantly {direction} than real images. "
                        f"L channel mean difference: {l_mean_diff:.1f}"
                    ),
                    metric_name="l_channel_mean_diff",
                    metric_value=syn_l_mean,
                    reference_value=real_l_mean,
                )
            )

        # --- Contrast (L channel std) ---
        l_std_diff = abs(syn_l_std - real_l_std)
        if l_std_diff > 10:
            direction = "higher" if syn_l_std > real_l_std else "lower"
            issues.append(
                GapIssue(
                    category=IssueCategory.CONTRAST,
                    severity=IssueSeverity.MEDIUM,
                    description=(
                        f"Synthetic images have {direction} contrast than real images. "
                        f"L channel std difference: {l_std_diff:.1f}"
                    ),
                    metric_name="l_channel_std_diff",
                    metric_value=syn_l_std,
                    reference_value=real_l_std,
                )
            )

        # --- Color cast (A/B channel means) ---
        a_mean_diff = abs(syn_a_mean - real_a_mean)
        b_mean_diff = abs(syn_b_mean - real_b_mean)
        max_ab_diff = max(a_mean_diff, b_mean_diff)

        if max_ab_diff > 8:
            channel_name = "A (green-red)" if a_mean_diff >= b_mean_diff else "B (blue-yellow)"
            issues.append(
                GapIssue(
                    category=IssueCategory.COLOR,
                    severity=IssueSeverity.HIGH,
                    description=(
                        f"Significant color mismatch in {channel_name} channel. "
                        f"Mean difference: {max_ab_diff:.1f}"
                    ),
                    metric_name="ab_channel_mean_diff",
                    metric_value=max_ab_diff,
                    reference_value=0.0,
                )
            )

        # --- Saturation ---
        syn_sat = self._compute_avg_saturation(synthetic_images)
        real_sat = self._compute_avg_saturation(real_images)
        sat_diff = abs(syn_sat - real_sat)

        if sat_diff > 12:
            direction = "more saturated" if syn_sat > real_sat else "less saturated"
            issues.append(
                GapIssue(
                    category=IssueCategory.COLOR,
                    severity=IssueSeverity.MEDIUM,
                    description=(
                        f"Synthetic images are {direction} than real images. "
                        f"Saturation difference: {sat_diff:.1f}"
                    ),
                    metric_name="saturation_diff",
                    metric_value=syn_sat,
                    reference_value=real_sat,
                )
            )

        logger.debug(
            "Color analysis: L_mean_diff={:.1f}, L_std_diff={:.1f}, "
            "AB_diff={:.1f}, sat_diff={:.1f}",
            l_mean_diff,
            l_std_diff,
            max_ab_diff,
            sat_diff,
        )

        return issues

    # =========================================================================
    # Edge Analysis
    # =========================================================================

    def _analyze_edges(
        self,
        synthetic_images: List[np.ndarray],
        real_images: List[np.ndarray],
    ) -> List[GapIssue]:
        """
        Compare edge sharpness via Laplacian variance.

        A higher Laplacian variance indicates sharper edges.
        """
        issues: List[GapIssue] = []

        syn_edge_vars = []
        for img in synthetic_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            syn_edge_vars.append(lap_var)

        real_edge_vars = []
        for img in real_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            real_edge_vars.append(lap_var)

        syn_avg = float(np.mean(syn_edge_vars))
        real_avg = float(np.mean(real_edge_vars))

        if real_avg > 0:
            relative_diff = (syn_avg - real_avg) / real_avg
        else:
            relative_diff = 0.0

        logger.debug(
            "Edge analysis: syn_avg={:.1f}, real_avg={:.1f}, relative_diff={:.2%}",
            syn_avg,
            real_avg,
            relative_diff,
        )

        if relative_diff > 0.30:
            # Synthetic edges are 30%+ sharper
            issues.append(
                GapIssue(
                    category=IssueCategory.EDGES,
                    severity=IssueSeverity.HIGH,
                    description=(
                        f"Synthetic images have significantly sharper edges than real images "
                        f"({relative_diff:.0%} sharper). Consider more edge smoothing."
                    ),
                    metric_name="laplacian_variance_ratio",
                    metric_value=syn_avg,
                    reference_value=real_avg,
                )
            )
        elif relative_diff < -0.30:
            # Synthetic edges are 30%+ softer
            issues.append(
                GapIssue(
                    category=IssueCategory.EDGES,
                    severity=IssueSeverity.MEDIUM,
                    description=(
                        f"Synthetic images have significantly softer edges than real images "
                        f"({abs(relative_diff):.0%} softer). Consider less blur."
                    ),
                    metric_name="laplacian_variance_ratio",
                    metric_value=syn_avg,
                    reference_value=real_avg,
                )
            )

        return issues

    # =========================================================================
    # Frequency Analysis
    # =========================================================================

    def _analyze_frequency(
        self,
        synthetic_images: List[np.ndarray],
        real_images: List[np.ndarray],
    ) -> List[GapIssue]:
        """
        Compare high-frequency energy content via 2D FFT.

        High-frequency energy correlates with fine detail, noise,
        and sharpness.
        """
        issues: List[GapIssue] = []

        syn_hf_ratios = [self._compute_hf_ratio(img) for img in synthetic_images]
        real_hf_ratios = [self._compute_hf_ratio(img) for img in real_images]

        syn_avg_hf = float(np.mean(syn_hf_ratios))
        real_avg_hf = float(np.mean(real_hf_ratios))

        if real_avg_hf > 0:
            relative_diff = (syn_avg_hf - real_avg_hf) / real_avg_hf
        else:
            relative_diff = 0.0

        logger.debug(
            "Frequency analysis: syn_hf={:.4f}, real_hf={:.4f}, relative_diff={:.2%}",
            syn_avg_hf,
            real_avg_hf,
            relative_diff,
        )

        if relative_diff > 0.20:
            # Synthetic has 20%+ more high-frequency content
            issues.append(
                GapIssue(
                    category=IssueCategory.FREQUENCY,
                    severity=IssueSeverity.MEDIUM,
                    description=(
                        f"Synthetic images contain {relative_diff:.0%} more high-frequency "
                        f"energy than real images. They may appear too sharp or noisy."
                    ),
                    metric_name="hf_energy_ratio",
                    metric_value=syn_avg_hf,
                    reference_value=real_avg_hf,
                )
            )
        elif relative_diff < -0.20:
            # Synthetic has 20%+ less high-frequency content
            issues.append(
                GapIssue(
                    category=IssueCategory.FREQUENCY,
                    severity=IssueSeverity.MEDIUM,
                    description=(
                        f"Synthetic images contain {abs(relative_diff):.0%} less high-frequency "
                        f"energy than real images. They may appear too smooth or blurred."
                    ),
                    metric_name="hf_energy_ratio",
                    metric_value=syn_avg_hf,
                    reference_value=real_avg_hf,
                )
            )

        return issues

    # =========================================================================
    # Texture Analysis
    # =========================================================================

    def _analyze_texture(
        self,
        synthetic_images: List[np.ndarray],
        real_images: List[np.ndarray],
    ) -> List[GapIssue]:
        """
        Compare texture complexity using grayscale histogram entropy.

        Higher entropy indicates more complex/diverse texture content.
        """
        issues: List[GapIssue] = []

        syn_entropies = [self._compute_histogram_entropy(img) for img in synthetic_images]
        real_entropies = [self._compute_histogram_entropy(img) for img in real_images]

        syn_avg_entropy = float(np.mean(syn_entropies))
        real_avg_entropy = float(np.mean(real_entropies))

        entropy_diff = syn_avg_entropy - real_avg_entropy

        logger.debug(
            "Texture analysis: syn_entropy={:.2f}, real_entropy={:.2f}, diff={:.2f}",
            syn_avg_entropy,
            real_avg_entropy,
            entropy_diff,
        )

        # If synthetic entropy is notably lower, textures are too uniform
        if entropy_diff < -0.5:
            issues.append(
                GapIssue(
                    category=IssueCategory.TEXTURE,
                    severity=IssueSeverity.MEDIUM,
                    description=(
                        f"Synthetic images have lower texture complexity "
                        f"(entropy {syn_avg_entropy:.2f} vs {real_avg_entropy:.2f}). "
                        f"Textures appear too uniform. Consider adding noise or "
                        f"domain randomization."
                    ),
                    metric_name="histogram_entropy_diff",
                    metric_value=syn_avg_entropy,
                    reference_value=real_avg_entropy,
                )
            )

        # Check texture diversity (std of entropies)
        syn_entropy_std = float(np.std(syn_entropies)) if len(syn_entropies) > 1 else 0.0
        real_entropy_std = float(np.std(real_entropies)) if len(real_entropies) > 1 else 0.0

        if real_entropy_std > 0 and syn_entropy_std < real_entropy_std * 0.5:
            issues.append(
                GapIssue(
                    category=IssueCategory.TEXTURE,
                    severity=IssueSeverity.LOW,
                    description=(
                        f"Synthetic images lack texture diversity "
                        f"(entropy std {syn_entropy_std:.2f} vs {real_entropy_std:.2f}). "
                        f"Consider enabling domain randomization for more variation."
                    ),
                    metric_name="entropy_std_ratio",
                    metric_value=syn_entropy_std,
                    reference_value=real_entropy_std,
                )
            )

        return issues

    # =========================================================================
    # Suggestion Generation
    # =========================================================================

    def _generate_suggestions(
        self,
        issues: List[GapIssue],
        current_config: Optional[Dict],
    ) -> List[ParameterSuggestion]:
        """
        Map detected issues to concrete parameter adjustment suggestions.

        Uses the issue category, severity, and metric values to determine
        which pipeline parameters should be adjusted and by how much.
        """
        suggestions: List[ParameterSuggestion] = []

        for issue in issues:
            new_suggestions = self._map_issue_to_suggestions(issue, current_config)
            suggestions.extend(new_suggestions)

        # Deduplicate by parameter_path (keep highest impact)
        seen_paths: Dict[str, ParameterSuggestion] = {}
        for s in suggestions:
            if s.parameter_path not in seen_paths:
                seen_paths[s.parameter_path] = s
            else:
                existing = seen_paths[s.parameter_path]
                if _IMPACT_ORDER.get(s.expected_impact, 99) < _IMPACT_ORDER.get(
                    existing.expected_impact, 99
                ):
                    seen_paths[s.parameter_path] = s

        suggestions = list(seen_paths.values())

        # Sort by expected impact (HIGH first)
        suggestions.sort(key=lambda s: _IMPACT_ORDER.get(s.expected_impact, 99))

        return suggestions

    def _map_issue_to_suggestions(
        self,
        issue: GapIssue,
        current_config: Optional[Dict],
    ) -> List[ParameterSuggestion]:
        """Map a single issue to one or more parameter suggestions."""
        suggestions: List[ParameterSuggestion] = []

        if issue.category == IssueCategory.BRIGHTNESS:
            # Brightness too high/low -> adjust lighting intensity toward real mean
            current_val = self._extract_config_value(
                current_config, "effects.lighting.intensity"
            )
            # Compute suggested adjustment: move toward real reference value
            # metric_value = synthetic L mean, reference_value = real L mean
            if issue.metric_value > issue.reference_value:
                # Synthetic too bright -> decrease lighting intensity
                suggested = (current_val * 0.8) if current_val is not None else 0.7
                direction_text = "Decrease"
            else:
                # Synthetic too dark -> increase lighting intensity
                suggested = (current_val * 1.2) if current_val is not None else 1.3
                direction_text = "Increase"

            suggestions.append(
                ParameterSuggestion(
                    parameter_path="effects.lighting.intensity",
                    current_value=current_val,
                    suggested_value=round(suggested, 2),
                    reason=(
                        f"{direction_text} lighting intensity to match real image "
                        f"brightness (synthetic L={issue.metric_value:.1f}, "
                        f"real L={issue.reference_value:.1f})"
                    ),
                    expected_impact=ImpactLevel.HIGH,
                )
            )

        elif issue.category == IssueCategory.CONTRAST:
            # Contrast mismatch -> adjust augmentation contrast range
            current_val = self._extract_config_value(
                current_config, "augmentation.contrast"
            )
            if issue.metric_value > issue.reference_value:
                # Synthetic contrast too high -> narrow contrast range
                suggested = (current_val * 0.85) if current_val is not None else 0.85
            else:
                # Synthetic contrast too low -> widen contrast range
                suggested = (current_val * 1.15) if current_val is not None else 1.15

            suggestions.append(
                ParameterSuggestion(
                    parameter_path="augmentation.contrast",
                    current_value=current_val,
                    suggested_value=round(suggested, 2),
                    reason=(
                        f"Adjust contrast range to match real image contrast "
                        f"(synthetic std={issue.metric_value:.1f}, "
                        f"real std={issue.reference_value:.1f})"
                    ),
                    expected_impact=ImpactLevel.MEDIUM,
                )
            )

        elif issue.category == IssueCategory.COLOR:
            if issue.metric_name == "ab_channel_mean_diff":
                # Color cast mismatch -> increase color correction intensity
                current_val = self._extract_config_value(
                    current_config, "effects.color_correction.intensity"
                )
                # Suggest increasing intensity to reduce mismatch
                suggested = min(
                    (current_val + 0.2) if current_val is not None else 0.7,
                    1.0,
                )
                suggestions.append(
                    ParameterSuggestion(
                        parameter_path="effects.color_correction.intensity",
                        current_value=current_val,
                        suggested_value=round(suggested, 2),
                        reason=(
                            f"Increase color correction intensity to reduce "
                            f"color channel mismatch (difference: "
                            f"{issue.metric_value:.1f})"
                        ),
                        expected_impact=ImpactLevel.HIGH,
                    )
                )
            elif issue.metric_name == "saturation_diff":
                # Saturation mismatch -> also color correction
                current_val = self._extract_config_value(
                    current_config, "effects.color_correction.intensity"
                )
                suggested = min(
                    (current_val + 0.15) if current_val is not None else 0.6,
                    1.0,
                )
                suggestions.append(
                    ParameterSuggestion(
                        parameter_path="effects.color_correction.intensity",
                        current_value=current_val,
                        suggested_value=round(suggested, 2),
                        reason=(
                            f"Increase color correction to match saturation levels "
                            f"(synthetic={issue.metric_value:.1f}, "
                            f"real={issue.reference_value:.1f})"
                        ),
                        expected_impact=ImpactLevel.MEDIUM,
                    )
                )

        elif issue.category == IssueCategory.EDGES:
            if issue.metric_value > issue.reference_value:
                # Edges too sharp -> increase feather radius
                current_val = self._extract_config_value(
                    current_config, "effects.edge_smoothing.feather_radius"
                )
                suggested = (current_val + 2.0) if current_val is not None else 3.0
                suggestions.append(
                    ParameterSuggestion(
                        parameter_path="effects.edge_smoothing.feather_radius",
                        current_value=current_val,
                        suggested_value=round(suggested, 1),
                        reason=(
                            f"Increase edge smoothing feather radius to soften "
                            f"overly sharp synthetic edges (synthetic variance "
                            f"{issue.metric_value:.0f} vs real {issue.reference_value:.0f})"
                        ),
                        expected_impact=ImpactLevel.HIGH,
                    )
                )
            else:
                # Edges too soft -> decrease blur strength
                current_val = self._extract_config_value(
                    current_config, "effects.blur_matching.strength"
                )
                suggested = max(
                    (current_val * 0.6) if current_val is not None else 0.3,
                    0.0,
                )
                suggestions.append(
                    ParameterSuggestion(
                        parameter_path="effects.blur_matching.strength",
                        current_value=current_val,
                        suggested_value=round(suggested, 2),
                        reason=(
                            f"Decrease blur strength to preserve edge detail "
                            f"(synthetic variance {issue.metric_value:.0f} vs "
                            f"real {issue.reference_value:.0f})"
                        ),
                        expected_impact=ImpactLevel.MEDIUM,
                    )
                )

        elif issue.category == IssueCategory.FREQUENCY:
            if issue.metric_value > issue.reference_value:
                # Too much HF -> decrease noise
                current_val = self._extract_config_value(
                    current_config, "augmentation.noise"
                )
                suggested = max(
                    (current_val * 0.6) if current_val is not None else 0.01,
                    0.0,
                )
                suggestions.append(
                    ParameterSuggestion(
                        parameter_path="augmentation.noise",
                        current_value=current_val,
                        suggested_value=round(suggested, 3),
                        reason=(
                            f"Decrease noise to reduce excess high-frequency content "
                            f"(synthetic HF ratio={issue.metric_value:.4f}, "
                            f"real={issue.reference_value:.4f})"
                        ),
                        expected_impact=ImpactLevel.MEDIUM,
                    )
                )
            else:
                # Too smooth -> increase noise
                current_val = self._extract_config_value(
                    current_config, "augmentation.noise"
                )
                suggested = (current_val * 1.5) if current_val is not None else 0.03
                suggestions.append(
                    ParameterSuggestion(
                        parameter_path="augmentation.noise",
                        current_value=current_val,
                        suggested_value=round(suggested, 3),
                        reason=(
                            f"Increase noise to add missing high-frequency detail "
                            f"(synthetic HF ratio={issue.metric_value:.4f}, "
                            f"real={issue.reference_value:.4f})"
                        ),
                        expected_impact=ImpactLevel.MEDIUM,
                    )
                )

        elif issue.category == IssueCategory.TEXTURE:
            if issue.metric_name == "histogram_entropy_diff":
                # Low entropy -> increase noise for texture complexity
                current_val = self._extract_config_value(
                    current_config, "augmentation.noise"
                )
                suggested = (current_val * 1.4) if current_val is not None else 0.03
                suggestions.append(
                    ParameterSuggestion(
                        parameter_path="augmentation.noise",
                        current_value=current_val,
                        suggested_value=round(suggested, 3),
                        reason=(
                            f"Increase noise to improve texture complexity "
                            f"(synthetic entropy={issue.metric_value:.2f}, "
                            f"real={issue.reference_value:.2f})"
                        ),
                        expected_impact=ImpactLevel.MEDIUM,
                    )
                )
            elif issue.metric_name == "entropy_std_ratio":
                # Low texture diversity -> recommend domain randomization
                current_val = self._extract_config_value(
                    current_config, "domain_randomization.recommended"
                )
                suggestions.append(
                    ParameterSuggestion(
                        parameter_path="domain_randomization.recommended",
                        current_value=current_val,
                        suggested_value=1.0,
                        reason=(
                            f"Enable domain randomization to increase texture "
                            f"diversity (synthetic entropy std="
                            f"{issue.metric_value:.2f}, "
                            f"real={issue.reference_value:.2f})"
                        ),
                        expected_impact=ImpactLevel.LOW,
                    )
                )

        return suggestions

    # =========================================================================
    # Image Loading
    # =========================================================================

    def _load_images_sample(
        self, image_dir: str, max_images: int
    ) -> List[np.ndarray]:
        """
        Load a random sample of images from a directory.

        Args:
            image_dir: Path to directory containing images.
            max_images: Maximum number of images to load.

        Returns:
            List of BGR images as numpy arrays.
        """
        image_dir_path = Path(image_dir)

        if not image_dir_path.exists():
            logger.warning("Image directory does not exist: {}", image_dir)
            return []

        if not image_dir_path.is_dir():
            logger.warning("Path is not a directory: {}", image_dir)
            return []

        # Collect image file paths (recursive to support subdirectories)
        image_paths = [
            p
            for p in image_dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        ]

        if len(image_paths) == 0:
            logger.warning("No image files found in: {}", image_dir)
            return []

        # Random subsample if needed
        if len(image_paths) > max_images:
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(len(image_paths), size=max_images, replace=False)
            image_paths = [image_paths[i] for i in indices]

        # Load images
        images: List[np.ndarray] = []
        for path in image_paths:
            try:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning("Failed to read image (corrupted?): {}", path)
                    continue
                # Resize to a common resolution for fair comparison
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                images.append(img)
            except Exception as e:
                logger.warning("Error loading image {}: {}", path, e)
                continue

        logger.debug("Loaded {}/{} images from {}", len(images), len(image_paths), image_dir)
        return images

    # =========================================================================
    # Computation Helpers
    # =========================================================================

    @staticmethod
    def _compute_lab_stats(
        images: List[np.ndarray],
    ) -> Tuple[float, float, float, float]:
        """
        Compute mean L, std L, mean A, mean B across a set of images in LAB space.

        Returns:
            (l_mean, l_std, a_mean, b_mean)
        """
        l_means: List[float] = []
        l_stds: List[float] = []
        a_means: List[float] = []
        b_means: List[float] = []

        for img in images:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            l_channel = lab[:, :, 0]
            a_channel = lab[:, :, 1]
            b_channel = lab[:, :, 2]

            l_means.append(float(np.mean(l_channel)))
            l_stds.append(float(np.std(l_channel)))
            a_means.append(float(np.mean(a_channel)))
            b_means.append(float(np.mean(b_channel)))

        return (
            float(np.mean(l_means)),
            float(np.mean(l_stds)),
            float(np.mean(a_means)),
            float(np.mean(b_means)),
        )

    @staticmethod
    def _compute_avg_saturation(images: List[np.ndarray]) -> float:
        """Compute average saturation across a set of BGR images."""
        saturations: List[float] = []

        for img in images:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            s_channel = hsv[:, :, 1].astype(np.float32)
            saturations.append(float(np.mean(s_channel)))

        return float(np.mean(saturations))

    @staticmethod
    def _compute_hf_ratio(img: np.ndarray) -> float:
        """
        Compute high-frequency energy ratio from 2D FFT of grayscale image.

        High-frequency is defined as frequencies above Nyquist/4.

        Returns:
            Ratio of high-frequency energy to total energy.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shifted = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shifted)

        rows, cols = gray.shape
        center_r, center_c = rows // 2, cols // 2

        # Nyquist/4 radius
        nyquist_quarter = min(rows, cols) // 4

        # Create mask for high-frequency region (outside Nyquist/4 circle)
        y, x = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((y - center_r) ** 2 + (x - center_c) ** 2)

        total_energy = float(np.sum(magnitude ** 2))
        if total_energy == 0:
            return 0.0

        hf_mask = dist_from_center > nyquist_quarter
        hf_energy = float(np.sum(magnitude[hf_mask] ** 2))

        return hf_energy / total_energy

    @staticmethod
    def _compute_histogram_entropy(img: np.ndarray) -> float:
        """
        Compute Shannon entropy of the grayscale histogram.

        Higher entropy means more complex/diverse texture.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()

        # Normalize to probability distribution
        total = hist.sum()
        if total == 0:
            return 0.0

        prob = hist / total
        # Filter zero probabilities to avoid log(0)
        prob = prob[prob > 0]

        entropy = -float(np.sum(prob * np.log2(prob)))
        return entropy

    @staticmethod
    def _extract_config_value(
        config: Optional[Dict], dotted_path: str
    ) -> Optional[float]:
        """
        Extract a value from a nested config dict using dot-notation path.

        Args:
            config: Nested dictionary (e.g. pipeline configuration).
            dotted_path: Dot-separated key path, e.g. "effects.lighting.intensity".

        Returns:
            The value as float if found, None otherwise.
        """
        if config is None:
            return None

        keys = dotted_path.split(".")
        current = config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        try:
            return float(current)
        except (TypeError, ValueError):
            return None
