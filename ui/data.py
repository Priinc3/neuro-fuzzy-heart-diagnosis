"""Cached, defensive loaders for every artifact the dashboard needs.

Each loader returns `(value, message)`; value is None when the artifact is
absent or unreadable, and message is a one-line, user-safe hint naming the
script that produces it. Nothing here ever raises to the Streamlit page.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from src.anfis import ANFIS
from src.features import ANFIS_FEATURES

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

ANFIS_MSG = "Model not trained yet — run `python scripts/train_anfis.py`"
BASELINES_MSG = "Baselines not trained yet — run `python scripts/train_baselines.py`"
LR_MSG = "Logistic Regression baseline not trained yet — run `python scripts/train_baselines.py`"
FCM_MSG = "Clusters not computed yet — run `python scripts/run_fcm.py`"
HISTORY_MSG = "No training history — run `python scripts/train_anfis.py`"


def _load_json(name: str, message: str) -> tuple[dict | None, str | None]:
    path = ART / name
    if not path.exists():
        return None, message
    try:
        with path.open() as f:
            return json.load(f), None
    except Exception:
        return None, f"Could not read `{artifacts_rel(name)}` — re-run the script that writes it."


def artifacts_rel(name: str) -> str:
    return f"artifacts/{name}"


@st.cache_resource(show_spinner=False)
def load_anfis() -> tuple[ANFIS | None, str | None]:
    """The trained ANFIS, or (None, hint)."""
    path = ART / "anfis_model.npz"
    if not path.exists():
        return None, ANFIS_MSG
    try:
        return ANFIS.load(path), None
    except Exception:
        return None, "Could not load the ANFIS model — re-run `python scripts/train_anfis.py`."


@st.cache_resource(show_spinner=False)
def load_lr() -> tuple[object | None, str | None]:
    """The classical-ML Pipeline (preprocessor + LogisticRegression), or (None, hint)."""
    path = ART / "baseline_lr.joblib"
    if not path.exists():
        return None, LR_MSG
    try:
        pipe = joblib.load(path)
        if not hasattr(pipe, "predict_proba"):
            return None, LR_MSG
        return pipe, None
    except Exception:
        return None, "Could not load the baseline — re-run `python scripts/train_baselines.py`."


@st.cache_data(show_spinner=False)
def load_init_mf() -> tuple[dict | None, str | None]:
    """Membership functions *before* training."""
    return _load_json("anfis_init_mf.json", "No initial MFs saved — run `python scripts/train_anfis.py`")


@st.cache_data(show_spinner=False)
def load_history() -> tuple[dict | None, str | None]:
    """Training curve {"loss": [...], "val_auc": [...]}."""
    return _load_json("anfis_history.json", HISTORY_MSG)


@st.cache_data(show_spinner=False)
def load_metrics_anfis() -> tuple[dict | None, str | None]:
    """METRICS dict for ANFIS (under key "ANFIS")."""
    data, msg = _load_json("metrics_anfis.json", ANFIS_MSG)
    if data is None:
        return None, msg
    inner = data.get("ANFIS", data)  # tolerate flat or nested schema
    return (inner if isinstance(inner, dict) else None), (None if isinstance(inner, dict) else ANFIS_MSG)


@st.cache_data(show_spinner=False)
def load_metrics_baselines() -> tuple[dict | None, str | None]:
    """METRICS dict for the 3 baselines {name: METRICS}."""
    data, msg = _load_json("metrics_baselines.json", BASELINES_MSG)
    if data is None:
        return None, msg
    return (data if isinstance(data, dict) and data else None), (None if data else BASELINES_MSG)


@st.cache_data(show_spinner=False)
def load_fcm() -> tuple[dict | None, str | None]:
    """Fuzzy C-Means summary from run_fcm.py."""
    return _load_json("fcm.json", FCM_MSG)


# ---------------------------------------------------------------- safe calls
def _try_shapes(model, method: str, x_row: np.ndarray, **kw):
    """Call a model method tolerating 1-D or 2-D row conventions.

    The contract says `x_row` but doesn't pin the shape, so try the common
    ones and return the first success — never crash on a sibling's choice.
    """
    row = np.asarray(x_row, dtype=float).reshape(-1)
    for candidate in (row, row.reshape(1, -1)):
        try:
            return getattr(model, method)(candidate, **kw)
        except Exception:
            continue
    return None


def memberships(model: ANFIS, x_row) -> dict | None:
    """{feature: {label: degree}} or None on any failure."""
    try:
        out = _try_shapes(model, "memberships", x_row)
    except Exception:
        out = None
    if not isinstance(out, dict):
        return None
    return out


def top_rules(model: ANFIS, x_row, k: int = 5) -> list[dict] | None:
    """List of fired-rule dicts, or None on any failure."""
    try:
        out = _try_shapes(model, "top_rules", x_row, k=k)
    except Exception:
        out = None
    if not isinstance(out, list) or not out:
        return None
    return out


def predict_proba(model: ANFIS, X) -> np.ndarray | None:
    """(n,) probabilities or None on failure."""
    try:
        out = np.asarray(model.predict_proba(np.asarray(X, dtype=float)), dtype=float)
        return out.reshape(-1)
    except Exception:
        return None


def points_frame(fcm: dict):
    """fcm.json sample_points -> (DataFrame, None) or (None, reason)."""
    import pandas as pd

    rows = fcm.get("sample_points")
    if not isinstance(rows, list) or not rows:
        return None, "no sample points in fcm.json"
    width = len(ANFIS_FEATURES) + 2  # 5 features + cluster + disease
    if any(len(r) != width for r in rows):
        return None, f"expected {width} values per row in sample_points"
    df = pd.DataFrame(rows, columns=ANFIS_FEATURES + ["cluster", "disease"])
    df["cluster"] = df["cluster"].astype(int)
    df["disease"] = df["disease"].astype(float)
    return df, None
