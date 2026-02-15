from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> int:
    p_bs = Path("data/BTC_USD_2h.csv")   # Bitstamp (ancienne histoire)
    p_bn = Path("data/BTC_USDT_2h.csv")  # Binance (récente histoire)
    out = Path("data/BTC_FUSED_2h.csv")

    bs = (
        pd.read_csv(p_bs, parse_dates=["timestamp"]).set_index("timestamp")
        if p_bs.exists()
        else pd.DataFrame()
    )
    bn = (
        pd.read_csv(p_bn, parse_dates=["timestamp"]).set_index("timestamp")
        if p_bn.exists()
        else pd.DataFrame()
    )

    df = pd.concat([bs, bn]) if not bs.empty or not bn.empty else pd.DataFrame()
    if df.empty:
        print("No data to fuse: missing data/BTC_USD_2h.csv and data/BTC_USDT_2h.csv")
        return 1

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Colonnes standard attendues par les autres scripts
    # (close, open, high, low, volume, etc.). On conserve ce qui existe.

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(
        f"Fused H2 written: {out} rows={len(df)} range={df.index.min()} -> {df.index.max()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


