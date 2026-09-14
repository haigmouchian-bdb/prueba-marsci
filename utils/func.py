"""Production functions consumed by the mmm_demo inference pipeline."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

SPEND_CHANNELS = ["tv_spend", "digital_spend", "ooh_spend"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a sales/media CSV and parse the date column."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def apply_adstock(series: pd.Series, decay: float) -> np.ndarray:
    """Geometric adstock: today's effect carries over into future weeks at `decay` rate."""
    adstocked = np.zeros(len(series))
    carryover = 0.0
    for i, value in enumerate(series.to_numpy()):
        carryover = value + decay * carryover
        adstocked[i] = carryover
    return adstocked


def engineer_features(df: pd.DataFrame, decay: float = 0.5) -> pd.DataFrame:
    """Add adstocked spend columns and a linear time trend."""
    df = df.copy()
    for channel in SPEND_CHANNELS:
        df[f"{channel}_adstock"] = apply_adstock(df[channel], decay=decay)
    df["trend"] = np.arange(len(df))
    return df


def load_model(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(df_features: pd.DataFrame, model) -> np.ndarray:
    feature_cols = [f"{channel}_adstock" for channel in SPEND_CHANNELS] + ["trend"]
    return model.predict(df_features[feature_cols])


def save_predictions(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
