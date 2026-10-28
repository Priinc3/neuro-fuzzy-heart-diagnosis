"""Frozen feature contract for the ANFIS heart-disease app (Phase 01).

Phases 02/03/04 READ this file — do not edit it without CEO approval.
Contracts (from prompts/00-plan.md):
  - ANFIS_FEATURES: the 5 ANFIS inputs, min-max scaled to [0,1] with RANGES.
  - FULL_FEATURES: the 17 raw dataset columns (no HeartDisease) shared by
    the classical-ML baselines, the FCM dashboard plot and the form.
  - raw_from_form(): 1-row raw DataFrame from the Streamlit form dict.
  - full_preprocessor(): StandardScaler(numeric) + OneHotEncoder(categorical).
  - load_split(): (df_train, df_test) from the frozen data artifacts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ANFIS_FEATURES = ["Age", "GenHealth", "BMI", "PhysicalHealth", "Comorbidity"]

# Fixed min/max for min-max scaling to [0,1] (clipped). Never computed from data.
RANGES = {
    "Age": (18.0, 85.0),
    "GenHealth": (0.0, 4.0),
    "BMI": (12.0, 60.0),
    "PhysicalHealth": (0.0, 30.0),
    "Comorbidity": (0.0, 5.0),
}

# AgeCategory -> midpoint years ("80 or older" -> 85 per contract).
AGE_MIDPOINTS = {
    "18-24": 21.0, "25-29": 27.0, "30-34": 32.0, "35-39": 37.0,
    "40-44": 42.0, "45-49": 47.0, "50-54": 52.0, "55-59": 57.0,
    "60-64": 62.0, "65-69": 67.0, "70-74": 72.0, "75-79": 77.0,
    "80 or older": 85.0,
}

# General health rank: Excellent=0 (best) ... Poor=4 (worst).
GENHEALTH_MAP = {
    "Excellent": 0.0, "Very good": 1.0, "Good": 2.0, "Fair": 3.0, "Poor": 4.0,
}

FULL_FEATURES = [
    "BMI", "Smoking", "AlcoholDrinking", "Stroke", "PhysicalHealth",
    "MentalHealth", "DiffWalking", "Sex", "AgeCategory", "Race",
    "Diabetic", "PhysicalActivity", "GenHealth", "SleepTime",
    "Asthma", "KidneyDisease", "SkinCancer",
]

_NUMERIC = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]
_CATEGORICAL = [c for c in FULL_FEATURES if c not in _NUMERIC]

_COMORB_BINARY = ["Stroke", "DiffWalking", "KidneyDisease", "Smoking"]


def _yes(v) -> float:
    """1.0 for the exact string 'Yes', else 0.0."""
    return 1.0 if str(v).strip() == "Yes" else 0.0


def _comorbidity(row: pd.Series) -> float:
    """0-5 comorbidity count: Stroke/DiffWalking/KidneyDisease/Smoking Yes=1;
    Diabetic Yes=1, borderline=0.5 ('Yes (during pregnancy)' counts as Yes=1)."""
    d = str(row["Diabetic"]).strip()
    if d.startswith("Yes"):
        diabetic = 1.0
    elif d.startswith("No, borderline"):
        diabetic = 0.5
    else:
        diabetic = 0.0
    return float(_yes(row["Stroke"]) + _yes(row["DiffWalking"])
                 + _yes(row["KidneyDisease"]) + _yes(row["Smoking"]) + diabetic)


def _age_midpoints(series: pd.Series) -> pd.Series:
    """AgeCategory strings -> midpoint years; raises on unknown categories."""
    out = series.astype(str).str.strip().map(AGE_MIDPOINTS)
    if out.isnull().any():
        bad = sorted(set(series[out.isnull()].astype(str)))
        raise ValueError(f"Unknown AgeCategory values {bad}; expected {sorted(AGE_MIDPOINTS)}")
    return out


def _genhealth(series: pd.Series) -> pd.Series:
    """GenHealth strings -> 0..4 (Excellent=0 ... Poor=4); raises on unknown categories."""
    out = series.astype(str).str.strip().map(GENHEALTH_MAP)
    if out.isnull().any():
        bad = sorted(set(series[out.isnull()].astype(str)))
        raise ValueError(f"Unknown GenHealth values {bad}; expected {sorted(GENHEALTH_MAP)}")
    return out


def anfis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw DataFrame -> unscaled 5-column ANFIS feature table (ANFIS_FEATURES order)."""
    feats = pd.DataFrame(index=df.index)
    feats["Age"] = _age_midpoints(df["AgeCategory"])
    feats["GenHealth"] = _genhealth(df["GenHealth"])
    feats["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    feats["PhysicalHealth"] = pd.to_numeric(df["PhysicalHealth"], errors="coerce")
    feats["Comorbidity"] = df.apply(_comorbidity, axis=1)
    if feats.isnull().any().any():
        raise ValueError(f"NaN in ANFIS features: {feats.isnull().sum().to_dict()}")
    return feats[ANFIS_FEATURES]


def anfis_matrix(df: pd.DataFrame) -> np.ndarray:
    """Scaled (n, 5) ANFIS input matrix: min-max with RANGES, clipped to [0,1]."""
    feats = anfis_features(df).to_numpy(dtype=float)
    lo = np.array([RANGES[f][0] for f in ANFIS_FEATURES])
    hi = np.array([RANGES[f][1] for f in ANFIS_FEATURES])
    return np.clip((feats - lo) / (hi - lo), 0.0, 1.0)


def raw_from_form(d: dict) -> pd.DataFrame:
    """Streamlit form dict -> 1-row DataFrame with the original 17 raw columns."""
    missing = [c for c in FULL_FEATURES if c not in d]
    if missing:
        raise ValueError(f"raw_from_form: missing form keys {missing}")
    row = {c: d[c] for c in FULL_FEATURES}
    for c in _NUMERIC:
        row[c] = float(row[c])
    return pd.DataFrame([row], columns=FULL_FEATURES)


def full_preprocessor() -> ColumnTransformer:
    """Preprocessor for the classical-ML baselines: StandardScaler + OneHotEncoder(handle_unknown='ignore')."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _CATEGORICAL),
        ],
        remainder="drop",
    )


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(df_train, df_test) read from data/balanced.parquet + data/processed/split.npz."""
    root = Path(__file__).resolve().parents[1]
    df = pd.read_parquet(root / "data" / "balanced.parquet")
    split = np.load(root / "data" / "processed" / "split.npz")
    return df.iloc[split["train_idx"]], df.iloc[split["test_idx"]]
