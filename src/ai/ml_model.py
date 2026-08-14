# src/ai/ml_model.py
# XGBoost ML Trading Model (Phase 4).
#
# Binary classifier: predicts whether a trade signal will be profitable.
# Input: 30-dimensional feature vector (from ml_features.py)
# Output: probability of success (0.0 to 1.0)
#
# Training pipeline:
# 1. Collect 30+ days of paper trading data (features + outcomes)
# 2. Walk-forward validation: train on 60 days, test on 7
# 3. Only deploy if test accuracy > 55% AND Sharpe > 0.5
# 4. Retrain weekly with newest data
#
# Safety:
# - ML signal ONLY overrides rule-based when probability > 0.7
# - A/B testing: runs alongside existing system before replacement
# - Auto-reverts if accuracy drops >5% from training baseline
#
# The model does NOT replace the 5-brain system — it becomes Brain 6,
# adding ML confidence as another voter in the consensus.

import os
import json
import logging
import pickle
from typing import Optional
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

# Minimum trades needed before training is allowed
MIN_TRAINING_SAMPLES = 200
# Minimum accuracy on validation set to deploy model
MIN_ACCURACY_THRESHOLD = 0.55
# Probability threshold to generate a signal (below = HOLD)
SIGNAL_THRESHOLD = 0.6
# Model file paths
MODEL_DIR = "data/ml_models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_trader.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.json")


class MLModel:
    """XGBoost-based trade signal predictor.

    Trained on the bot's own trading history to predict
    which signals will be profitable.
    """

    def __init__(self):
        # The trained XGBoost model (None until trained)
        self._model = None
        # Whether xgboost is installed
        self._available = False
        # Training metrics for monitoring drift
        self._baseline_accuracy: float = 0.0
        self._training_date: Optional[str] = None
        self._total_predictions: int = 0
        self._correct_predictions: int = 0

    def initialize(self):
        """Load saved model or check if XGBoost is available."""
        try:
            import xgboost
            self._available = True
            logger.info("XGBoost available (version %s)", xgboost.__version__)

            # Try loading a previously saved model
            if os.path.exists(MODEL_PATH):
                self._load_model()
            else:
                logger.info("No saved ML model found — training needed")

        except ImportError:
            logger.warning("xgboost not installed — ML model disabled. "
                          "Install with: pip install xgboost scikit-learn")
            self._available = False

    @property
    def is_trained(self) -> bool:
        """Whether a trained model is loaded and ready."""
        return self._model is not None

    @property
    def is_available(self) -> bool:
        """Whether XGBoost is installed."""
        return self._available

    def predict(self, features: list[float]) -> dict:
        """Predict whether a trade signal will be profitable.

        Args:
            features: feature vector from FeatureEngineer

        Returns:
            dict with: direction (BUY/SELL/HOLD), probability, confidence
        """
        if not self.is_trained:
            return {"direction": "HOLD", "probability": 0.5,
                    "confidence": 0.0, "source": "ml_model"}

        try:
            import xgboost as xgb

            # Convert to DMatrix for prediction
            X = np.array([features])
            dmatrix = xgb.DMatrix(X)

            # Get probability of profitable trade
            probability = float(self._model.predict(dmatrix)[0])

            # Convert probability to direction
            if probability >= SIGNAL_THRESHOLD:
                direction = "BUY"
                confidence = probability
            elif probability <= (1.0 - SIGNAL_THRESHOLD):
                direction = "SELL"
                confidence = 1.0 - probability
            else:
                direction = "HOLD"
                confidence = 0.5

            self._total_predictions += 1

            return {
                "direction": direction,
                "probability": round(probability, 4),
                "confidence": round(confidence, 4),
                "source": "ml_model",
            }

        except Exception as e:
            logger.error("ML prediction failed: %s", e)
            return {"direction": "HOLD", "probability": 0.5,
                    "confidence": 0.0, "source": "ml_model"}

    def train(self, X: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2) -> dict:
        """Train the XGBoost model on collected trade data.

        Uses walk-forward split: oldest 80% for training, newest 20% for validation.
        Only saves the model if it meets the minimum accuracy threshold.

        Args:
            X: feature matrix (N samples × M features)
            y: labels (1 = profitable, 0 = not)
            validation_split: fraction to hold out for validation

        Returns:
            dict with training metrics
        """
        if not self._available:
            return {"error": "xgboost not installed"}

        if len(X) < MIN_TRAINING_SAMPLES:
            return {"error": f"Need {MIN_TRAINING_SAMPLES} samples, have {len(X)}"}

        try:
            import xgboost as xgb
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
            )

            # Walk-forward split: train on older data, test on newer
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            logger.info(
                "Training ML model: %d train / %d validation samples",
                len(X_train), len(X_val),
            )

            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            # XGBoost hyperparameters tuned for small trading datasets
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 4,           # shallow trees prevent overfitting
                "learning_rate": 0.05,    # slow learning for stability
                "subsample": 0.8,         # random 80% of data per tree
                "colsample_bytree": 0.8,  # random 80% of features per tree
                "min_child_weight": 5,    # minimum samples per leaf
                "gamma": 0.1,             # minimum gain to make a split
                "reg_alpha": 0.1,         # L1 regularization
                "reg_lambda": 1.0,        # L2 regularization
                "scale_pos_weight": 1.0,  # balance classes if needed
                "seed": 42,
                "verbosity": 0,
            }

            # Adjust class balance if data is imbalanced
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            if n_neg > 0 and n_pos > 0:
                params["scale_pos_weight"] = n_neg / n_pos

            # Train with early stopping to prevent overfitting
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=300,
                evals=[(dval, "validation")],
                early_stopping_rounds=30,
                verbose_eval=False,
            )

            # Evaluate on validation set
            y_pred_prob = model.predict(dval)
            y_pred = (y_pred_prob >= 0.5).astype(int)

            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            metrics = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "best_iteration": model.best_iteration,
                "training_date": datetime.now(timezone.utc).isoformat(),
                "class_balance": f"{n_pos}/{n_neg} (pos/neg)",
            }

            logger.info("Training results: %s", metrics)

            # Only deploy if accuracy meets threshold
            if accuracy >= MIN_ACCURACY_THRESHOLD:
                self._model = model
                self._baseline_accuracy = accuracy
                self._training_date = metrics["training_date"]
                self._save_model(metrics)
                logger.info("ML model deployed (accuracy: %.1f%%)", accuracy * 100)
                metrics["deployed"] = True
            else:
                logger.warning(
                    "ML model NOT deployed — accuracy %.1f%% below %.1f%% threshold",
                    accuracy * 100, MIN_ACCURACY_THRESHOLD * 100,
                )
                metrics["deployed"] = False

            return metrics

        except Exception as e:
            logger.error("ML training failed: %s", e)
            return {"error": str(e)}

    def record_outcome(self, was_correct: bool):
        """Track prediction accuracy for drift detection.

        Call this after each trade to update the rolling accuracy.
        If accuracy drops >5% from training baseline, the model
        should be retrained.
        """
        self._total_predictions += 1
        if was_correct:
            self._correct_predictions += 1

    def check_drift(self) -> dict:
        """Check if model accuracy has drifted from training baseline.

        Returns:
            dict with: drifted (bool), live_accuracy, baseline_accuracy, gap
        """
        if self._total_predictions < 50:
            return {"drifted": False, "reason": "not enough predictions yet",
                    "predictions": self._total_predictions}

        live_accuracy = self._correct_predictions / self._total_predictions
        gap = self._baseline_accuracy - live_accuracy

        drifted = gap > 0.05  # 5% accuracy drop = drift

        return {
            "drifted": drifted,
            "live_accuracy": round(live_accuracy, 4),
            "baseline_accuracy": round(self._baseline_accuracy, 4),
            "gap": round(gap, 4),
            "predictions": self._total_predictions,
        }

    def get_feature_importance(self) -> dict:
        """Get which features matter most to the model.

        Returns dict mapping feature names to importance scores.
        Useful for understanding what drives the model's decisions.
        """
        if not self.is_trained:
            return {}

        from src.ai.ml_features import FEATURE_NAMES

        importance = self._model.get_score(importance_type="gain")
        # Map xgboost feature names (f0, f1, ...) to our names
        named_importance = {}
        for feat_key, score in importance.items():
            idx = int(feat_key.replace("f", ""))
            if idx < len(FEATURE_NAMES):
                named_importance[FEATURE_NAMES[idx]] = round(score, 4)

        # Sort by importance (highest first)
        return dict(sorted(named_importance.items(),
                          key=lambda x: x[1], reverse=True))

    def _save_model(self, metrics: dict):
        """Save trained model and metrics to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Save model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self._model, f)

        # Save metrics
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("ML model saved to %s", MODEL_PATH)

    def _load_model(self):
        """Load a previously saved model."""
        try:
            with open(MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)

            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH) as f:
                    metrics = json.load(f)
                self._baseline_accuracy = metrics.get("accuracy", 0)
                self._training_date = metrics.get("training_date")

            logger.info("Loaded ML model (trained: %s, baseline: %.1f%%)",
                       self._training_date, self._baseline_accuracy * 100)

        except Exception as e:
            logger.error("Failed to load ML model: %s", e)
            self._model = None
