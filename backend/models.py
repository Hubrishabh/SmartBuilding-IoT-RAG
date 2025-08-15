import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from dataclasses import dataclass

@dataclass
class AnomalyModel:
    model: IsolationForest
    features: list

def train_anomaly_model(df: pd.DataFrame, feature_cols=None) -> AnomalyModel:
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("timestamp","device_id")]
    X = df[feature_cols].fillna(method="ffill").fillna(0.0).values
    iso = IsolationForest(contamination=0.02, random_state=42)
    iso.fit(X)
    return AnomalyModel(model=iso, features=feature_cols)

def score_anomalies(model: AnomalyModel, df: pd.DataFrame):
    X = df[model.features].fillna(method="ffill").fillna(0.0).values
    scores = -model.model.score_samples(X)
    return scores

def estimate_simple_rul(series: pd.Series, upper_limit: float) -> float:
    y = series.values[-50:] if len(series) > 50 else series.values
    if len(y) < 5:
        return float("nan")
    x = np.arange(len(y))
    a, b = np.polyfit(x, y, 1)
    if a <= 0:
        return float("inf")
    steps_to_threshold = (upper_limit - b) / a
    return max(0.0, steps_to_threshold)