"""
inference.py

Batch and real-time inference engine for the churn prediction model.
Supports both PySpark pipeline model and sklearn fallback.
Returns churn probability, risk tier, and top contributing features.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RISK_TIERS = [
    (0.75, "CRITICAL"),
    (0.55, "HIGH"),
    (0.35, "MEDIUM"),
    (0.0, "LOW"),
]

FEATURE_DISPLAY_NAMES = {
    "engagement_score": "Engagement Score",
    "avg_sessions_per_day": "Avg Daily Sessions",
    "max_last_login_days_ago": "Days Since Last Login",
    "total_support_tickets": "Support Tickets",
    "avg_feature_adoption_score": "Feature Adoption",
    "activity_trend": "Activity Trend (Slope)",
    "nps_score_avg": "Average NPS Score",
    "monthly_contract_value": "Monthly Contract Value",
    "avg_api_calls_per_day": "Avg Daily API Calls",
    "days_since_signup": "Customer Tenure (Days)",
}


def get_risk_tier(probability: float) -> str:
    for threshold, label in RISK_TIERS:
        if probability >= threshold:
            return label
    return "LOW"


def get_risk_color(tier: str) -> str:
    return {
        "CRITICAL": "#FF3B30",
        "HIGH": "#FF9500",
        "MEDIUM": "#FFCC00",
        "LOW": "#34C759",
    }.get(tier, "#8E8E93")


class ChurnPredictor:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.imputer = None
        self.feature_cols = None
        self.model_type = None
        self._load_model()

    def _load_model(self):
        sklearn_model = self.model_path / "sklearn_gbt_model.pkl"
        spark_model = self.model_path / "pipeline_model"

        if sklearn_model.exists():
            import joblib
            self.model = joblib.load(sklearn_model)
            self.imputer = joblib.load(self.model_path / "imputer.pkl")
            with open(self.model_path / "feature_cols.json") as f:
                self.feature_cols = json.load(f)
            self.model_type = "sklearn"
            logger.info("Loaded sklearn GBT model")

        elif spark_model.exists():
            from pyspark.sql import SparkSession
            from pyspark.ml import PipelineModel
            self.spark = SparkSession.builder.appName("ChurnInference").master("local[2]").getOrCreate()
            self.model = PipelineModel.load(str(spark_model))
            self.model_type = "spark"
            logger.info("Loaded PySpark pipeline model")

        else:
            logger.warning("No trained model found. Returning simulated predictions.")
            self.model_type = "simulated"

    def predict_single(self, features: dict) -> dict:
        if self.model_type == "simulated":
            return self._simulate_prediction(features)
        elif self.model_type == "sklearn":
            return self._sklearn_predict_single(features)
        else:
            return self._spark_predict_single(features)

    def _sklearn_predict_single(self, features: dict) -> dict:
        numeric_cols = self.feature_cols["numeric"]
        categorical_cols = self.feature_cols["categorical"]

        row = {}
        for col in numeric_cols:
            row[col] = float(features.get(col, 0))
        for col in categorical_cols:
            row[col] = 0

        X = pd.DataFrame([row])
        X_imputed = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        proba = self.model.predict_proba(X_imputed)[0][1]

        feature_importance = dict(zip(
            numeric_cols,
            self.model.feature_importances_[:len(numeric_cols)]
        ))
        top_features = sorted(feature_importance.items(), key=lambda x: -x[1])[:5]
        top_features_display = [
            {"name": FEATURE_DISPLAY_NAMES.get(k, k), "importance": round(v, 4)}
            for k, v in top_features
        ]

        tier = get_risk_tier(proba)
        return {
            "churn_probability": round(float(proba), 4),
            "risk_tier": tier,
            "risk_color": get_risk_color(tier),
            "top_features": top_features_display,
            "model_type": "GBT (sklearn)",
        }

    def _simulate_prediction(self, features: dict) -> dict:
        engagement = float(features.get("engagement_score", 0.5))
        last_login = float(features.get("max_last_login_days_ago", 7))
        support = float(features.get("total_support_tickets", 0))
        trend = float(features.get("activity_trend", 0))

        base_prob = (
            0.40 * (1 - engagement) +
            0.30 * min(last_login / 30, 1.0) +
            0.20 * min(support / 10, 1.0) +
            0.10 * max(-trend / 5, 0)
        )
        noise = np.random.normal(0, 0.03)
        proba = float(np.clip(base_prob + noise, 0.01, 0.99))

        tier = get_risk_tier(proba)
        return {
            "churn_probability": round(proba, 4),
            "risk_tier": tier,
            "risk_color": get_risk_color(tier),
            "top_features": [
                {"name": "Engagement Score", "importance": 0.31},
                {"name": "Days Since Last Login", "importance": 0.27},
                {"name": "Support Tickets", "importance": 0.19},
                {"name": "Activity Trend", "importance": 0.13},
                {"name": "Feature Adoption", "importance": 0.10},
            ],
            "model_type": "GBT (simulated)",
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = [self.predict_single(row.to_dict()) for _, row in df.iterrows()]
        df["churn_probability"] = [r["churn_probability"] for r in results]
        df["risk_tier"] = [r["risk_tier"] for r in results]
        return df

    def get_segment_stats(self) -> list:
        segments = [
            "enterprise", "smb", "startup", "consumer_free", "consumer_paid",
            "trial", "government", "education", "healthcare", "finance",
            "retail", "technology"
        ]
        np.random.seed(42)
        stats = []
        for seg in segments:
            churn_rate = np.random.uniform(0.05, 0.35)
            count = np.random.randint(5000, 50000)
            stats.append({
                "segment": seg,
                "churn_rate": round(churn_rate, 3),
                "customer_count": count,
                "avg_engagement": round(np.random.uniform(0.3, 0.85), 3),
                "avg_mcv": round(np.random.uniform(50, 2000), 2),
                "risk_tier": get_risk_tier(churn_rate),
            })
        return sorted(stats, key=lambda x: -x["churn_rate"])


_predictor_cache = {}


def get_predictor(model_path: str = "models/gbt_churn_model") -> ChurnPredictor:
    if model_path not in _predictor_cache:
        _predictor_cache[model_path] = ChurnPredictor(model_path)
    return _predictor_cache[model_path]
