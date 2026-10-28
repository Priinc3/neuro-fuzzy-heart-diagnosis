"""Phase 02 — train the ANFIS heart-disease model and write its artifacts.

Usage: python scripts/train_anfis.py [--mfs 2]
Writes (one writer per file):
  artifacts/anfis_init_mf.json   membership params BEFORE training
  artifacts/anfis_model.npz      trained model (via ANFIS.save)
  artifacts/anfis_history.json   {"loss": [...], "val_auc": [...]}
  artifacts/metrics_anfis.json   METRICS schema, key "ANFIS"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anfis import ANFIS
from src.features import anfis_matrix, load_split

ART = ROOT / "artifacts"


def _label(df) -> np.ndarray:
    """y column as float array ('y' int written by Phase 01, HeartDisease fallback)."""
    if "y" in df.columns:
        return df["y"].to_numpy(dtype=float)
    return (df["HeartDisease"].astype(str).str.strip() == "Yes").astype(float).to_numpy()


def _downsample_roc(fpr: np.ndarray, tpr: np.ndarray, max_points: int = 200):
    """ROC curve downsampled to <= max_points (endpoints kept)."""
    if len(fpr) <= max_points:
        return fpr.tolist(), tpr.tolist()
    idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
    return fpr[idx].tolist(), tpr[idx].tolist()


def evaluate(y_true: np.ndarray, proba: np.ndarray, train_seconds: float) -> dict:
    """METRICS schema from 00-plan.md, threshold 0.5."""
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_true, proba)
    fpr_d, tpr_d = _downsample_roc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "roc": {"fpr": fpr_d, "tpr": tpr_d},
        "train_seconds": float(train_seconds),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the ANFIS model (Phase 02).")
    ap.add_argument("--mfs", type=int, default=2, help="membership functions per input (2 or 3)")
    args = ap.parse_args()

    df_train, df_test = load_split()
    Xtr, ytr = anfis_matrix(df_train), _label(df_train)
    Xte, yte = anfis_matrix(df_test), _label(df_test)
    print(f"data: train={Xtr.shape}, test={Xte.shape}, pos_rate={ytr.mean():.3f}")

    model = ANFIS(n_mfs=args.mfs)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "anfis_init_mf.json").write_text(json.dumps(model.mf_params(), indent=2))
    print(f"init MFs saved: {model._n_rules} rules, centers={model.mf_params()['centers'][0]}")

    t0 = time.perf_counter()
    history = model.fit(Xtr, ytr, epochs=40, lr=0.01, batch_size=512, X_val=Xte, y_val=yte)
    seconds = time.perf_counter() - t0

    model.save(ART / "anfis_model.npz")
    (ART / "anfis_history.json").write_text(json.dumps(history))
    proba = model.predict_proba(Xte)
    metrics = {"ANFIS": evaluate(yte, proba, seconds)}
    (ART / "metrics_anfis.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(
        f"epochs={len(history['loss'])} last_loss={history['loss'][-1]:.4f} "
        f"last_val_auc={history['val_auc'][-1]:.4f} train_seconds={seconds:.1f}"
    )

    sample = model.top_rules(Xte[0], 3)
    print("sample top_rules for test row 0:")
    for r in sample:
        print(f"  [{r['strength']:.3f}] {r['text']}")

    # self-check: reload from disk and compare with the in-memory model
    loaded = ANFIS.load(ART / "anfis_model.npz")
    p_loaded = loaded.predict_proba(Xte)
    assert np.allclose(proba, p_loaded, atol=1e-6), "reloaded model differs from in-memory model"
    print("self-check OK: ANFIS.load reproduces predict_proba (atol 1e-6)")


if __name__ == "__main__":
    main()
