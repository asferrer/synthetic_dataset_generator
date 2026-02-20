"""
Automatic Domain Gap Optimizer Engine
=======================================
Iteratively reduces the domain gap by running:
    metrics → advisor → adjust config → apply technique → repeat

Supports early stopping when gap score reaches the target or when
successive iterations fail to improve the score.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


class OptimizerEngine:
    """
    Orchestrates iterative domain gap reduction by combining the
    MetricsEngine, AdvisorEngine, and one or more reduction techniques
    (randomization, style transfer) in a closed-loop pipeline.
    """

    def __init__(
        self,
        metrics_engine,
        advisor_engine,
        randomization_engine=None,
        style_transfer_engine=None,
    ) -> None:
        self.metrics = metrics_engine
        self.advisor = advisor_engine
        self.randomization = randomization_engine
        self.style_transfer = style_transfer_engine

    def optimize(
        self,
        synthetic_dir: str,
        real_dir: str,
        output_dir: str,
        current_config: Optional[Dict[str, Any]] = None,
        target_gap_score: float = 30.0,
        max_iterations: int = 5,
        max_images: int = 50,
        techniques: Optional[List[str]] = None,
        reference_set_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run iterative optimization to reduce domain gap.

        Args:
            synthetic_dir: Directory with synthetic images (input).
            real_dir: Directory with real reference images.
            output_dir: Directory to write optimized images.
            current_config: Current pipeline configuration (for advisor context).
            target_gap_score: Stop when gap score falls below this (0-100).
            max_iterations: Maximum optimization iterations.
            max_images: Max images to sample for metrics per iteration.
            techniques: List of techniques to apply. Options: "randomization",
                "style_transfer". Defaults to ["randomization"].
            reference_set_id: Reference set ID (for randomization histogram matching).
            progress_callback: Called with (iteration, max_iterations, gap_score, phase).

        Returns:
            Dictionary with optimization results including iteration history.
        """
        start_time = time.time()

        if techniques is None:
            techniques = ["randomization"]

        # Validate available techniques
        available = []
        if "randomization" in techniques and self.randomization is not None:
            available.append("randomization")
        if "style_transfer" in techniques and self.style_transfer is not None:
            available.append("style_transfer")

        if not available:
            raise ValueError(
                f"No available techniques from requested: {techniques}. "
                f"Randomization loaded: {self.randomization is not None}, "
                f"Style transfer loaded: {self.style_transfer is not None}"
            )

        os.makedirs(output_dir, exist_ok=True)

        # Working directory: start from the original synthetic images
        current_input_dir = synthetic_dir
        history: List[Dict[str, Any]] = []
        best_score = float("inf")
        stagnation_count = 0

        def _report(iteration: int, score: float, phase: str):
            if progress_callback:
                progress_callback(iteration, max_iterations, score, phase)

        for iteration in range(1, max_iterations + 1):
            logger.info(
                "Optimization iteration {}/{} - input: {}",
                iteration, max_iterations, current_input_dir,
            )
            _report(iteration, best_score if best_score != float("inf") else 100.0, "computing_metrics")

            # 1. Compute current metrics
            metrics = self.metrics.compute_metrics(
                synthetic_dir=current_input_dir,
                real_dir=real_dir,
                max_images=max_images,
                compute_fid=True,
                compute_kid=True,
                compute_color=True,
                compute_prdc=True,
            )

            gap_score = metrics.overall_gap_score
            logger.info(
                "Iteration {} - gap_score={:.1f} ({})",
                iteration, gap_score, metrics.gap_level.value,
            )

            iteration_result = {
                "iteration": iteration,
                "gap_score": gap_score,
                "gap_level": metrics.gap_level.value,
                "fid": metrics.fid_score,
                "kid": metrics.kid_score,
                "technique_applied": None,
            }

            # 2. Check if target reached
            if gap_score <= target_gap_score:
                logger.info(
                    "Target gap score {:.1f} reached at iteration {} (score={:.1f})",
                    target_gap_score, iteration, gap_score,
                )
                iteration_result["technique_applied"] = "none (target reached)"
                history.append(iteration_result)
                break

            # 3. Check for early stopping (stagnation)
            if gap_score >= best_score:
                stagnation_count += 1
                if stagnation_count >= 2:
                    logger.warning(
                        "Early stopping: gap score has not improved for 2 iterations"
                    )
                    iteration_result["technique_applied"] = "none (early stopping)"
                    history.append(iteration_result)
                    break
            else:
                best_score = gap_score
                stagnation_count = 0

            # 4. Run advisor analysis
            _report(iteration, gap_score, "analyzing_issues")
            issues, suggestions = self.advisor.analyze(
                synthetic_dir=current_input_dir,
                real_dir=real_dir,
                max_images=max_images,
                current_config=current_config,
            )

            # 5. Adjust config from suggestions
            adjusted_config = dict(current_config) if current_config else {}
            for suggestion in suggestions:
                _set_nested(adjusted_config, suggestion.parameter_path, suggestion.suggested_value)

            # 6. Apply technique
            iter_output_dir = os.path.join(output_dir, f"iter_{iteration:02d}")
            os.makedirs(iter_output_dir, exist_ok=True)

            technique_used = available[0]  # Primary technique

            if technique_used == "style_transfer":
                _report(iteration, gap_score, "applying_style_transfer")
                logger.info("Applying style transfer (iteration {})", iteration)
                try:
                    self.style_transfer.apply_batch(
                        images_dir=current_input_dir,
                        style_dir=real_dir,
                        output_dir=iter_output_dir,
                        color_only=True,  # Start with color-only (safer, faster)
                    )
                    iteration_result["technique_applied"] = "style_transfer"
                except Exception as e:
                    logger.error("Style transfer failed at iteration {}: {}", iteration, e)
                    # Fall back to randomization if available
                    if "randomization" in available:
                        technique_used = "randomization"
                    else:
                        iteration_result["technique_applied"] = f"failed: {e}"
                        history.append(iteration_result)
                        break

            if technique_used == "randomization":
                _report(iteration, gap_score, "applying_randomization")
                logger.info("Applying randomization (iteration {})", iteration)

                # Derive randomization params from advisor suggestions
                intensity = _extract_intensity_from_issues(issues)
                ref_stats = None
                if reference_set_id:
                    # The reference_manager is not available here directly;
                    # we pass ref_stats=None and rely on intensity-based randomization
                    pass

                self.randomization.apply_batch(
                    images_dir=current_input_dir,
                    output_dir=iter_output_dir,
                    num_variants=1,
                    intensity=intensity,
                    color_jitter=min(0.5, intensity * 0.8),
                    brightness_range=(max(0.7, 1.0 - intensity * 0.3), min(1.3, 1.0 + intensity * 0.3)),
                    contrast_range=(max(0.8, 1.0 - intensity * 0.2), min(1.2, 1.0 + intensity * 0.2)),
                    saturation_range=(max(0.7, 1.0 - intensity * 0.3), min(1.3, 1.0 + intensity * 0.3)),
                    noise_intensity=min(0.05, intensity * 0.04),
                    blur_range=(0.0, min(1.5, intensity * 1.0)),
                    reference_stats=ref_stats,
                )
                iteration_result["technique_applied"] = "randomization"

            # Update config and move to next iteration
            current_config = adjusted_config
            current_input_dir = iter_output_dir
            history.append(iteration_result)

        # Final metrics
        final_metrics = self.metrics.compute_metrics(
            synthetic_dir=current_input_dir,
            real_dir=real_dir,
            max_images=max_images,
            compute_fid=True,
            compute_kid=True,
            compute_color=True,
            compute_prdc=True,
        )

        processing_time = (time.time() - start_time) * 1000

        initial_score = history[0]["gap_score"] if history else 0.0
        final_score = final_metrics.overall_gap_score
        improvement_pct = (
            round((initial_score - final_score) / initial_score * 100, 2)
            if initial_score > 0 else 0.0
        )

        result = {
            "success": True,
            "initial_gap_score": initial_score,
            "final_gap_score": final_score,
            "final_gap_level": final_metrics.gap_level.value,
            "improvement_pct": improvement_pct,
            "iterations_completed": len(history),
            "target_reached": final_score <= target_gap_score,
            "output_dir": current_input_dir,
            "history": history,
            "final_metrics": {
                "fid": final_metrics.fid_score,
                "kid": final_metrics.kid_score,
                "precision": final_metrics.precision,
                "recall": final_metrics.recall,
                "density": final_metrics.density,
                "coverage": final_metrics.coverage,
            },
            "processing_time_ms": processing_time,
        }

        logger.info(
            "Optimization complete: {:.1f} -> {:.1f} ({:.1f}% improvement) in {} iterations",
            initial_score, final_score, improvement_pct, len(history),
        )

        return result


def _set_nested(d: dict, path: str, value: Any) -> None:
    """Set a value in a nested dict using dot-notation path."""
    keys = path.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _extract_intensity_from_issues(issues) -> float:
    """Derive a randomization intensity from detected gap issues."""
    if not issues:
        return 0.3

    severity_weights = {"high": 0.9, "medium": 0.6, "low": 0.3}
    max_severity = 0.3
    for issue in issues:
        weight = severity_weights.get(issue.severity.value if hasattr(issue.severity, 'value') else issue.severity, 0.3)
        max_severity = max(max_severity, weight)

    return min(0.8, max_severity)
