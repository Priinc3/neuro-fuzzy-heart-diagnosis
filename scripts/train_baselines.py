"""Phase 03 — classical-ML baselines (LogReg / RandomForest / MLP) on the frozen split.

Writes:
  artifacts/metrics_baselines.json  (METRICS schema from 00-plan.md, one key per model)
  artifacts/baseline_lr.joblib      (LogReg Pipeline only — dashboard side-by-side prediction)

Read-only inputs: src/features.py contract, data/balanced.parquet, data/processed/split.npz.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import FULL_FEATURES, full_preprocessor, load_split  # noqa: E402

SEED = 42
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(
        n_estimators=150, max_depth=12, n_jobs=-1, random_state=SEED),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(32, 16), early_stopping=True, random_state=SEED),
}


def _roc_points(y_true: np.ndarray, proba: np.ndarray, max_pts: int = 200) -> dict:
    """ROC curve capped at max_pts (uniform index subsampling of the full curve)."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(max_pts, len(fpr))).astype(int))
    return {"fpr": np.round(fpr[idx], 6).tolist(),
            "tpr": np.round(tpr[idx], 6).tolist()}


def main() -> None:
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)

    df_train, df_test = load_split()
    X_tr, y_tr = df_train[FULL_FEATURES], (df_train["HeartDisease"] == "Yes").to_numpy(int)
    X_te, y_te = df_test[FULL_FEATURES], (df_test["HeartDisease"] == "Yes").to_numpy(int)
    print(f"train={X_tr.shape} test={X_te.shape} pos_rate_train={y_tr.mean():.3f}")

    metrics: dict[str, dict] = {}
    for name, model in MODELS.items():
        pipe = Pipeline([("prep", full_preprocessor()), ("clf", model)])
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        seconds = round(time.perf_counter() - t0, 2)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "roc": _roc_points(y_te, proba),
            "train_seconds": seconds,
        }
        print(f"{name:20s} auc={metrics[name]['roc_auc']:.4f} f1={metrics[name]['f1']:.4f} "
              f"({seconds}s)")
        if name == "LogisticRegression":
            joblib.dump(pipe, out_dir / "baseline_lr.joblib")

    path = out_dir / "metrics_baselines.json"
    path.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
