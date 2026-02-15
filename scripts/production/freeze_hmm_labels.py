from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def _best_seed_dir_for_k(root: Path, k: int) -> Path | None:
    # choose seed with best (lowest) BIC from per-seed CSVs
    best_seed: int | None = None
    best_bic: float = float("inf")
    for seed_dir in sorted((root).glob("seed_*")):
        sel_csv = seed_dir / f"HMM_K_SELECTION_seed{seed_dir.name.split('_')[-1]}.csv"
        if not sel_csv.exists():
            continue
        try:
            df = pd.read_csv(sel_csv)
            row = df[df["K"] == k]
            if not row.empty:
                bic = float(row.iloc[0]["BIC"])
                if bic < best_bic:
                    best_bic = bic
                    best_seed = int(seed_dir.name.split('_')[-1])
        except Exception:
            continue
    if best_seed is None:
        return None
    return root / f"seed_{best_seed}"


def main() -> int:
    # HMM multi-seed outputs on fused dataset
    hmm_root = Path("outputs/fourier/hmm/BTC_FUSED_2h")
    out_dir = Path("outputs/fourier/labels_frozen/BTC_FUSED_2h")
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in (2, 3, 4, 5, 8, 10):
        seed_dir = _best_seed_dir_for_k(hmm_root, k)
        if seed_dir is None:
            print(f"No per-seed selection found for K={k} under {hmm_root}")
            continue
        src = seed_dir / f"HMM_PRED_{k}.csv"
        if not src.exists():
            print(f"Missing prediction CSV for K={k} at {src}")
            continue
        df = pd.read_csv(src, parse_dates=["timestamp"]).rename(columns={"state": "label"})
        df = df[["timestamp", "label"]].copy()
        dst = out_dir / f"K{k}.csv"
        df.to_csv(dst, index=False)
        print(f"Frozen labels written: {dst} rows={len(df)} range={df['timestamp'].min()} -> {df['timestamp'].max()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


