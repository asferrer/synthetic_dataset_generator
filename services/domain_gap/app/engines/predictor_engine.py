"""
Predictive Engine for Domain Gap Optimization
==============================================
ML-based predictor that learns from historical configurations and their
resulting domain gap scores to guide optimization.

Uses XGBoost regression to predict gap scores from parameter configurations.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import xgboost as xgb
from loguru import logger


class ConfigurationHistory:
    """Manages historical configuration → gap_score data for training."""

    def __init__(self, history_file: Optional[str] = None):
        """
        Args:
            history_file: Path to JSON file storing configuration history.
                         Defaults to /shared/config_history.json
        """
        if history_file is None:
            history_file = os.environ.get(
                "CONFIG_HISTORY_FILE",
                "/shared/config_history.json"
            )
        self.history_file = Path(history_file)
        self.history: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
                logger.info(f"Loaded {len(self.history)} historical configurations")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
                self.history = []
        else:
            logger.info("No history file found, starting fresh")
            self.history = []

    def _save(self):
        """Save history to disk."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def add_entry(
        self,
        config: Dict[str, Any],
        gap_score: float,
        fid: Optional[float] = None,
        kid: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a configuration and its result to history.

        Args:
            config: Effects/augmentation configuration dict.
            gap_score: Resulting domain gap score (0-100).
            fid: Optional FID score.
            kid: Optional KID score.
            metadata: Optional metadata (domain, reference_set_id, etc).
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "gap_score": gap_score,
            "fid": fid,
            "kid": kid,
            "metadata": metadata or {},
        }
        self.history.append(entry)
        self._save()
        logger.info(f"Added config to history: gap_score={gap_score:.1f}")

    def get_training_data(
        self,
        min_samples: int = 10,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract X (features) and y (gap scores) for training.

        Returns:
            (X, y) tuple if enough samples exist, else (None, None).
        """
        if len(self.history) < min_samples:
            logger.warning(
                f"Insufficient training data: {len(self.history)} < {min_samples}"
            )
            return None, None

        features = []
        targets = []

        for entry in self.history:
            config = entry["config"]
            gap_score = entry["gap_score"]

            # Extract numeric features from config
            feature_vec = self._config_to_features(config)
            features.append(feature_vec)
            targets.append(gap_score)

        X = np.array(features, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

        logger.info(f"Extracted training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y

    def _config_to_features(self, config: Dict[str, Any]) -> List[float]:
        """Convert a configuration dict to a fixed-length feature vector.

        This method defines the feature space for the predictor.
        All configs are converted to the same feature representation.
        """
        # Define standard effects parameters with defaults
        features = [
            config.get("color_intensity", 0.12),
            config.get("blur_strength", 0.5),
            config.get("underwater_intensity", 0.15),
            config.get("caustics_intensity", 0.10),
            config.get("shadow_opacity", 0.10),
            config.get("lighting_intensity", 0.5),
            config.get("motion_blur_probability", 0.2),
            # Encode lighting_type as one-hot
            1.0 if config.get("lighting_type") == "ambient" else 0.0,
            1.0 if config.get("lighting_type") == "directional" else 0.0,
            1.0 if config.get("lighting_type") == "underwater" else 0.0,
            # Encode water_clarity as one-hot
            1.0 if config.get("water_clarity") == "clear" else 0.0,
            1.0 if config.get("water_clarity") == "murky" else 0.0,
            1.0 if config.get("water_clarity") == "moderate" else 0.0,
        ]
        return features


class PredictorEngine:
    """ML-based predictor for domain gap scores from configurations.

    Uses XGBoost to learn the mapping: config → gap_score.
    """

    def __init__(self, history: Optional[ConfigurationHistory] = None):
        """
        Args:
            history: ConfigurationHistory instance. If None, creates one.
        """
        self.history = history or ConfigurationHistory()
        self.model: Optional[xgb.XGBRegressor] = None
        self.trained = False
        self.model_path = Path("/shared/gap_predictor_model.pkl")

        # Try to load pre-trained model
        self._load_model()

    def _load_model(self):
        """Load pre-trained model from disk if available."""
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.trained = True
                logger.info(f"Loaded pre-trained predictor model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self.model = None
                self.trained = False

    def _save_model(self):
        """Save trained model to disk."""
        if self.model is None:
            return
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, self.model_path)
            logger.info(f"Saved predictor model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def train(self, min_samples: int = 15, retrain: bool = False) -> bool:
        """Train the XGBoost model on historical data.

        Args:
            min_samples: Minimum samples required for training.
            retrain: Force retrain even if already trained.

        Returns:
            True if training succeeded, False otherwise.
        """
        if self.trained and not retrain:
            logger.info("Model already trained, skipping")
            return True

        X, y = self.history.get_training_data(min_samples=min_samples)
        if X is None or y is None:
            logger.warning("Insufficient data for training")
            return False

        try:
            # XGBoost regressor with conservative hyperparameters
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective="reg:squarederror",
                verbosity=0,
            )

            self.model.fit(X, y)
            self.trained = True
            self._save_model()

            # Log training metrics
            train_pred = self.model.predict(X)
            mae = np.mean(np.abs(train_pred - y))
            rmse = np.sqrt(np.mean((train_pred - y) ** 2))
            logger.info(f"Predictor trained: MAE={mae:.2f}, RMSE={rmse:.2f}")

            return True

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            return False

    def predict(self, config: Dict[str, Any]) -> Optional[float]:
        """Predict gap score for a given configuration.

        Args:
            config: Effects configuration dict.

        Returns:
            Predicted gap score (0-100), or None if model not trained.
        """
        if not self.trained or self.model is None:
            logger.warning("Predictor not trained, cannot predict")
            return None

        try:
            features = self.history._config_to_features(config)
            X = np.array([features], dtype=np.float32)
            pred = self.model.predict(X)[0]

            # Clip to valid range
            pred = max(0.0, min(100.0, float(pred)))
            return pred

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

    def predict_batch(self, configs: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Predict gap scores for a batch of configurations.

        Args:
            configs: List of configuration dicts.

        Returns:
            Array of predicted gap scores, or None if model not trained.
        """
        if not self.trained or self.model is None:
            return None

        try:
            features = [self.history._config_to_features(c) for c in configs]
            X = np.array(features, dtype=np.float32)
            preds = self.model.predict(X)

            # Clip to valid range
            preds = np.clip(preds, 0.0, 100.0)
            return preds

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return None

    def suggest_best_from_candidates(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Given a list of candidate configurations, predict their scores
        and return the top-k best (lowest gap score).

        Args:
            candidates: List of configuration dicts.
            top_k: Number of top candidates to return.

        Returns:
            List of (config, predicted_score) tuples, sorted by score (ascending).
        """
        preds = self.predict_batch(candidates)
        if preds is None:
            logger.warning("Cannot suggest best candidates without trained model")
            return [(candidates[0], 50.0)] if candidates else []

        # Sort by predicted score (lower is better)
        scored = list(zip(candidates, preds))
        scored.sort(key=lambda x: x[1])

        return [(config, float(score)) for config, score in scored[:top_k]]

    def update_with_result(
        self,
        config: Dict[str, Any],
        gap_score: float,
        fid: Optional[float] = None,
        kid: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        retrain: bool = True,
    ):
        """Update history with a new configuration result and optionally retrain.

        Args:
            config: Configuration dict.
            gap_score: Observed gap score.
            fid: Optional FID score.
            kid: Optional KID score.
            metadata: Optional metadata.
            retrain: Whether to retrain the model after adding this entry.
        """
        self.history.add_entry(config, gap_score, fid, kid, metadata)

        if retrain:
            # Retrain if we have enough samples
            self.train(min_samples=10, retrain=True)


# =============================================================================
# Feature Importance Analysis
# =============================================================================

def analyze_feature_importance(predictor: PredictorEngine) -> Optional[Dict[str, float]]:
    """Analyze which configuration parameters have the most impact on gap score.

    Returns:
        Dict mapping feature names to importance scores, or None if model not trained.
    """
    if not predictor.trained or predictor.model is None:
        return None

    # Feature names matching the order in _config_to_features
    feature_names = [
        "color_intensity",
        "blur_strength",
        "underwater_intensity",
        "caustics_intensity",
        "shadow_opacity",
        "lighting_intensity",
        "motion_blur_probability",
        "lighting_ambient",
        "lighting_directional",
        "lighting_underwater",
        "water_clear",
        "water_murky",
        "water_moderate",
    ]

    try:
        importances = predictor.model.feature_importances_
        importance_dict = {
            name: float(imp) for name, imp in zip(feature_names, importances)
        }

        # Sort by importance (descending)
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_importance

    except Exception as e:
        logger.error(f"Failed to analyze feature importance: {e}")
        return None
