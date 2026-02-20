"""
Bayesian Optimization Engine for Domain Gap Minimization
=========================================================
Uses Optuna for intelligent hyperparameter search to find the optimal
effects configuration that minimizes domain gap.

Combines:
- Bayesian optimization (TPE sampler from Optuna)
- ML predictor (warm-start from historical data)
- Active learning (probe→measure→update loop)
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
from loguru import logger
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from .predictor_engine import PredictorEngine


class BayesianOptimizerEngine:
    """Bayesian optimization engine for finding optimal effects configuration.

    Uses Optuna's Tree-structured Parzen Estimator (TPE) to intelligently
    explore the parameter space and find configurations that minimize domain gap.
    """

    def __init__(
        self,
        predictor: Optional[PredictorEngine] = None,
        study_storage: Optional[str] = None,
    ):
        """
        Args:
            predictor: ML predictor for warm-starting (optional).
            study_storage: Optuna study storage URL. Defaults to SQLite in /shared/.
        """
        self.predictor = predictor
        self.study: Optional[optuna.Study] = None
        self.study_name: Optional[str] = None

        # Storage for Optuna studies (persists across restarts)
        if study_storage is None:
            storage_dir = Path("/shared/optuna_studies")
            storage_dir.mkdir(parents=True, exist_ok=True)
            study_storage = f"sqlite:///{storage_dir}/gap_optimization.db"
        self.study_storage = study_storage

        # Callback for real-world evaluation
        self._evaluation_callback: Optional[Callable] = None

    def optimize(
        self,
        n_trials: int,
        evaluation_callback: Callable[[Dict[str, Any]], float],
        timeout_seconds: Optional[int] = None,
        study_name: Optional[str] = None,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        initial_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run Bayesian optimization to find the best configuration.

        Args:
            n_trials: Number of optimization trials to run.
            evaluation_callback: Function that takes a config dict and returns
                                 the observed gap score (lower is better).
            timeout_seconds: Optional timeout for optimization.
            study_name: Name for this optimization study.
            parameter_ranges: Optional custom parameter ranges.
                             Format: {"param_name": (min, max), ...}
            initial_config: Optional starting configuration to evaluate first.

        Returns:
            Result dict with best_config, best_score, optimization_history, etc.
        """
        start_time = time.time()
        self._evaluation_callback = evaluation_callback

        # Create or load Optuna study
        if study_name is None:
            study_name = f"gap_opt_{int(time.time())}"
        self.study_name = study_name

        # Use TPE sampler with multivariate optimization
        sampler = TPESampler(
            n_startup_trials=3,  # Random trials before TPE kicks in
            multivariate=True,
            seed=42,
        )

        # Use median pruner to stop unpromising trials early
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0)

        try:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=self.study_storage,
                sampler=sampler,
                pruner=pruner,
                direction="minimize",  # Minimize gap score
                load_if_exists=True,
            )
            logger.info(f"Created/loaded Optuna study: {study_name}")
        except Exception as e:
            logger.error(f"Failed to create Optuna study: {e}")
            raise

        # Store parameter ranges
        self.parameter_ranges = parameter_ranges or self._default_parameter_ranges()

        # Enqueue initial config if provided
        if initial_config is not None:
            self.study.enqueue_trial(initial_config)
            logger.info("Enqueued initial configuration for evaluation")

        # Run optimization
        try:
            self.study.optimize(
                self._objective_function,
                n_trials=n_trials,
                timeout=timeout_seconds,
                show_progress_bar=False,
            )
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")
        except Exception as e:
            logger.exception(f"Optimization failed: {e}")
            raise

        # Extract results
        best_trial = self.study.best_trial
        best_config = best_trial.params
        best_score = best_trial.value

        # Build optimization history
        history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    "trial_number": trial.number,
                    "config": trial.params,
                    "gap_score": trial.value,
                    "datetime": trial.datetime_start.isoformat() if trial.datetime_start else None,
                })

        elapsed_time = (time.time() - start_time) * 1000

        # Update predictor with all completed trials
        if self.predictor is not None:
            for trial in self.study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    self.predictor.update_with_result(
                        config=trial.params,
                        gap_score=trial.value,
                        metadata={"trial_number": trial.number},
                        retrain=False,  # Don't retrain on every update
                    )
            # Retrain once at the end
            self.predictor.train(min_samples=5, retrain=True)

        result = {
            "success": True,
            "best_config": best_config,
            "best_gap_score": best_score,
            "n_trials_completed": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "optimization_history": history,
            "study_name": study_name,
            "processing_time_ms": elapsed_time,
        }

        logger.info(
            f"Optimization complete: best_score={best_score:.1f} after {result['n_trials_completed']} trials"
        )

        return result

    def _objective_function(self, trial: optuna.Trial) -> float:
        """Optuna objective function: sample a configuration and evaluate it.

        Args:
            trial: Optuna trial object.

        Returns:
            Gap score (lower is better).
        """
        # Sample configuration from parameter space
        config = self._sample_config(trial)

        # Evaluate using the provided callback
        gap_score = self._evaluation_callback(config)

        logger.info(f"Trial {trial.number}: gap_score={gap_score:.1f}, config={config}")

        return gap_score

    def _sample_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample a configuration from the search space.

        Args:
            trial: Optuna trial object.

        Returns:
            Configuration dict with sampled parameters.
        """
        config = {}

        # Continuous parameters
        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            if param_name.endswith("_type") or param_name.endswith("_clarity"):
                # Categorical parameter (skip, handled below)
                continue
            config[param_name] = trial.suggest_float(param_name, min_val, max_val)

        # Categorical parameters
        config["lighting_type"] = trial.suggest_categorical(
            "lighting_type", ["ambient", "directional", "underwater"]
        )
        config["water_clarity"] = trial.suggest_categorical(
            "water_clarity", ["clear", "moderate", "murky"]
        )

        return config

    def _default_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Default parameter search ranges for effects configuration.

        Returns:
            Dict mapping parameter names to (min, max) tuples.
        """
        return {
            # Effects intensities (0 = off, 1 = maximum)
            "color_intensity": (0.0, 0.5),
            "blur_strength": (0.0, 2.0),
            "underwater_intensity": (0.0, 0.5),
            "caustics_intensity": (0.0, 0.3),
            "shadow_opacity": (0.0, 0.3),
            "lighting_intensity": (0.2, 1.0),
            "motion_blur_probability": (0.0, 0.5),
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Analyze which parameters have the most impact on gap score.

        Uses Optuna's built-in feature importance calculation based on
        the observed trials.

        Returns:
            Dict mapping parameter names to importance scores, or None if not enough trials.
        """
        if self.study is None or len(self.study.trials) < 5:
            logger.warning("Not enough trials for feature importance analysis")
            return None

        try:
            from optuna.importance import get_param_importances

            importances = get_param_importances(self.study)
            return {k: float(v) for k, v in importances.items()}
        except Exception as e:
            logger.error(f"Failed to compute feature importance: {e}")
            return None

    def visualize_optimization_history(self, output_path: Optional[str] = None):
        """Generate visualization plots for optimization history.

        Requires optuna[visualization] and plotly. Saves plots to output_path.

        Args:
            output_path: Directory to save plots. If None, plots are not saved.
        """
        if self.study is None:
            logger.warning("No study to visualize")
            return

        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
            )

            if output_path is not None:
                output_dir = Path(output_path)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Optimization history plot
                fig = plot_optimization_history(self.study)
                fig.write_html(str(output_dir / "optimization_history.html"))

                # Parameter importance plot
                fig = plot_param_importances(self.study)
                fig.write_html(str(output_dir / "param_importances.html"))

                # Parallel coordinate plot
                fig = plot_parallel_coordinate(self.study)
                fig.write_html(str(output_dir / "parallel_coordinate.html"))

                logger.info(f"Saved optimization visualizations to {output_dir}")

        except ImportError:
            logger.warning("Optuna visualization not available (install optuna[visualization])")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")


# =============================================================================
# Hybrid Optimizer: Combines Bayesian Optimization + ML Predictor
# =============================================================================

class HybridOptimizer:
    """Hybrid optimizer combining Bayesian optimization with ML prediction.

    Workflow:
    1. Warm-start: Use ML predictor to suggest promising initial configs
    2. Explore: Run Bayesian optimization to find optimal config
    3. Learn: Update ML predictor with observed results
    """

    def __init__(self, predictor: Optional[PredictorEngine] = None):
        """
        Args:
            predictor: ML predictor engine (optional).
        """
        self.predictor = predictor or PredictorEngine()
        self.bayesian_optimizer = BayesianOptimizerEngine(predictor=self.predictor)

    def optimize(
        self,
        n_trials: int,
        evaluation_callback: Callable[[Dict[str, Any]], float],
        warm_start: bool = True,
        n_warmstart_candidates: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run hybrid optimization.

        Args:
            n_trials: Number of Bayesian optimization trials.
            evaluation_callback: Function to evaluate a config and return gap score.
            warm_start: Whether to use ML predictor for warm-starting.
            n_warmstart_candidates: Number of random candidates to evaluate with predictor.
            **kwargs: Additional arguments passed to BayesianOptimizerEngine.optimize().

        Returns:
            Optimization result dict.
        """
        initial_config = None

        # Warm-start: use predictor to find promising initial config
        if warm_start and self.predictor.trained:
            logger.info("Warm-starting with ML predictor")
            candidates = self._generate_random_configs(n_warmstart_candidates)
            best_candidates = self.predictor.suggest_best_from_candidates(candidates, top_k=1)

            if best_candidates:
                initial_config, predicted_score = best_candidates[0]
                logger.info(f"Warm-start initial config (predicted score: {predicted_score:.1f})")

        # Run Bayesian optimization
        result = self.bayesian_optimizer.optimize(
            n_trials=n_trials,
            evaluation_callback=evaluation_callback,
            initial_config=initial_config,
            **kwargs,
        )

        return result

    def _generate_random_configs(self, n: int) -> List[Dict[str, Any]]:
        """Generate n random configurations for warm-start evaluation.

        Returns:
            List of randomly sampled config dicts.
        """
        import random

        configs = []
        ranges = self.bayesian_optimizer._default_parameter_ranges()

        for _ in range(n):
            config = {}
            for param, (min_val, max_val) in ranges.items():
                config[param] = random.uniform(min_val, max_val)

            # Categorical parameters
            config["lighting_type"] = random.choice(["ambient", "directional", "underwater"])
            config["water_clarity"] = random.choice(["clear", "moderate", "murky"])

            configs.append(config)

        return configs

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from both Bayesian optimizer and ML predictor.

        Returns combined feature importance analysis.
        """
        from .predictor_engine import analyze_feature_importance

        bayesian_importance = self.bayesian_optimizer.get_feature_importance()
        ml_importance = analyze_feature_importance(self.predictor)

        if bayesian_importance is None and ml_importance is None:
            return None

        # Combine importances (average if both available)
        if bayesian_importance and ml_importance:
            combined = {}
            all_keys = set(bayesian_importance.keys()) | set(ml_importance.keys())
            for key in all_keys:
                b_val = bayesian_importance.get(key, 0.0)
                m_val = ml_importance.get(key, 0.0)
                combined[key] = (b_val + m_val) / 2.0
            return combined
        elif bayesian_importance:
            return bayesian_importance
        else:
            return ml_importance
