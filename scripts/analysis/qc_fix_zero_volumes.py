from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


def fix_zero_volumes(path: Path) -> int:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        # keep original order if already sorted
        pass
    if "volume" not in df.columns:
        print(f"No volume column in {path}")
        return 1
    n0 = (pd.to_numeric(df["volume"], errors="coerce") <= 0).sum()
    if n0 == 0:
        print(f"OK: no zero/neg volumes in {path}")
        return 0
    # Simple fix: forward-fill then back-fill volumes at 0 with nearest non-zero
    v = pd.to_numeric(df["volume"], errors="coerce")
    v = v.mask(v <= 0)
    v = v.ffill().bfill()
    df["volume"] = v
    out = path.with_name(path.stem + "_clean.csv")
    df.to_csv(out, index=False)
    print(f"Fixed {n0} rows in {path.name} -> {out.name}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/qc_fix_zero_volumes.py data/<FILE>.csv")
        return 2
    p = Path(argv[1])
    if not p.exists():
        print(f"Missing file: {p}")
        return 1
    return fix_zero_volumes(p)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


