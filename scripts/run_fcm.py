"""Phase 03 — Fuzzy C-Means soft patient clusters on the ANFIS 5-feature space.

Runs skfuzzy cmeans for c=2..6 on a 10k subsample of the scaled train matrix,
picks best_c by Fuzzy Partition Coefficient (FPC), writes artifacts/fcm.json:
  {"fpc_by_c": {"2": ..}, "best_c": int, "centers": [[..5 scaled..]],
   "cluster_disease_rate": [..], "sample_points": [[x5..., cluster, disease]] (<=2000),
   "feature_names": [...]}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import skfuzzy as fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import ANFIS_FEATURES, anfis_matrix, load_split  # noqa: E402

SEED = 42
SUBSAMPLE = 10_000
N_SAMPLE_POINTS = 2000
CS = range(2, 7)


def main() -> None:
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)

    df_train, _ = load_split()
    X = anfis_matrix(df_train)                      # (n, 5) scaled [0,1]
    y = (df_train["HeartDisease"] == "Yes").to_numpy(int)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(X), size=min(SUBSAMPLE, len(X)), replace=False))
    Xs, ys = X[idx], y[idx]
    data = Xs.T                                     # skfuzzy wants (features, samples)
    print(f"train={X.shape} subsample={data.shape}")

    fpc_by_c: dict[str, float] = {}
    best = {"fpc": -1.0, "c": None, "cntr": None, "u": None}
    for c in CS:
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            data, c, m=2, error=0.005, maxiter=300, init=None, seed=SEED)
        fpc_by_c[str(c)] = round(float(fpc), 4)
        print(f"c={c} fpc={fpc:.4f}")
        if fpc > best["fpc"]:
            best = {"fpc": float(fpc), "c": c, "cntr": cntr, "u": u}

    c, cntr, u = best["c"], best["cntr"], best["u"]
    hard = np.argmax(u, axis=0)                     # (n_samples,) cluster 0..c-1
    rate = [round(float(ys[hard == k].mean()), 4) for k in range(c)]
    print(f"best_c={c} fpc={best['fpc']:.4f} disease_rate={rate}")

    sp_idx = rng.choice(len(Xs), size=min(N_SAMPLE_POINTS, len(Xs)), replace=False)
    sample_points = np.column_stack([
        Xs[sp_idx], hard[sp_idx].astype(float), ys[sp_idx].astype(float),
    ]).round(6).tolist()

    payload = {
        "fpc_by_c": fpc_by_c,
        "best_c": int(c),
        "centers": np.round(cntr, 6).tolist(),
        "cluster_disease_rate": rate,
        "sample_points": sample_points,
        "feature_names": list(ANFIS_FEATURES),
    }
    path = out_dir / "fcm.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
