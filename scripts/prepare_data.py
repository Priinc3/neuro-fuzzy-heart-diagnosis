"""Prepare data for the neuro-fuzzy heart-disease project (Phase 01, frozen pipeline).

1. Downloads the raw CDC BRFSS heart-disease CSV once -> data/raw/final.csv (git-ignored).
2. Drops rows with nulls (keeps duplicates, same as the original project).
3. Balanced undersample: all HeartDisease=="Yes" + equal count of random "No" (seed 42).
4. Adds label column y (Yes=1, No=0), saves data/balanced.parquet (COMMITTED).
5. Stratified 80/20 split (seed 42) -> data/processed/split.npz with train_idx/test_idx
   (row indices into data/balanced.parquet).

Run:  python scripts/prepare_data.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_URL = "https://raw.githubusercontent.com/Priinc3/Heartattack_ML/main/final.csv"
RAW = ROOT / "data" / "raw" / "final.csv"
BALANCED = ROOT / "data" / "balanced.parquet"
SPLIT = ROOT / "data" / "processed" / "split.npz"
SEED = 42


def download_raw() -> None:
    """Download the fixed public raw CSV if it is not on disk yet."""
    if RAW.exists():
        print(f"[skip] {RAW} already exists ({RAW.stat().st_size / 1e6:.1f} MB)")
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {RAW_URL} -> {RAW} (~25 MB) ...")
    urllib.request.urlretrieve(RAW_URL, RAW)
    print(f"[done] {RAW.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    download_raw()
    df = pd.read_csv(RAW)
    n_raw = len(df)
    n_null = int(df.isnull().any(axis=1).sum())
    df = df.dropna().reset_index(drop=True)
    print(f"[load] raw rows={n_raw} null_rows_dropped={n_null} kept={len(df)}")
    print(f"[load] columns={list(df.columns)}")
    print(f"[load] HeartDisease values={df['HeartDisease'].value_counts(dropna=False).to_dict()}")

    # Balanced undersample: ALL positives + same count of random negatives.
    pos = df[df["HeartDisease"] == "Yes"]
    neg = df[df["HeartDisease"] == "No"].sample(n=len(pos), random_state=SEED)
    bal = pd.concat([pos, neg], ignore_index=True)
    bal["y"] = (bal["HeartDisease"] == "Yes").astype(np.int8)

    BALANCED.parent.mkdir(parents=True, exist_ok=True)
    bal.to_parquet(BALANCED, index=False)

    # Stratified 80/20 split, indices into the saved parquet.
    train_idx, test_idx = train_test_split(
        np.arange(len(bal)), test_size=0.2, random_state=SEED, stratify=bal["y"]
    )
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SPLIT, train_idx=train_idx, test_idx=test_idx)

    print(f"[balanced] rows={len(bal)}  y=1: {int((bal['y'] == 1).sum())}  "
          f"y=0: {int((bal['y'] == 0).sum())}  y.mean={bal['y'].mean():.4f}")
    print(f"[split] train={len(train_idx)} ({train_idx.mean():.0f} avg idx)  test={len(test_idx)}  "
          f"train y.mean={bal['y'].iloc[train_idx].mean():.4f}  test y.mean={bal['y'].iloc[test_idx].mean():.4f}")
    print(f"[saved] {BALANCED} ({BALANCED.stat().st_size / 1e6:.2f} MB)")
    print(f"[saved] {SPLIT}")


if __name__ == "__main__":
    main()
