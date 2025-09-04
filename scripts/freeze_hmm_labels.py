from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> int:
    base = Path("outputs/fourier/hmm/BTC_USDT_2h")
    out_dir = Path("outputs/fourier/labels_frozen/BTC_FUSED_2h")
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in (3, 5):
        src = base / f"HMM_PRED_{k}.csv"
        if not src.exists():
            print(f"Missing {src}")
            return 1
        df = pd.read_csv(src, parse_dates=["timestamp"])  # columns: timestamp,state
        # Normalize columns
        df = df.rename(columns={"state": "label"})
        df = df[["timestamp", "label"]].copy()
        dst = out_dir / f"K{k}.csv"
        df.to_csv(dst, index=False)
        print(f"Frozen labels written: {dst} rows={len(df)} range={df['timestamp'].min()} -> {df['timestamp'].max()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


