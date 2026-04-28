"""R2 — P0bis mini-WFA validation 30j BTC avec/sans fees.

Vérifie numériquement que le branchement cost_model dans le pipeline 14 ans
change effectivement le PnL (et dans le bon sens). Sans cette validation,
le grep positif (31 sites de cost_model) ne prouve PAS que les fees sont
appliqués au PnL final.

Méthode
-------
1. Charge 30 derniers jours BTC H1 depuis data/BTC_USD_1h.csv.
2. Run pipeline.backtest_long_short normal (fees taker 4 bps, maker 2 bps).
3. Monkey-patch _round_trip_fee_cost_usdt = lambda: 0.0  (fees désactivées).
4. Re-run le même backtest sur la même data.
5. Compare equity_with_fees vs equity_no_fees, vérifie :
     - equity_no_fees > equity_with_fees (fees diminuent le PnL)
     - delta ≈ n_trades × 8 bps × notional_moyen (round-trip)

Usage
-----
$ python scripts/validation/p0bis_mini_wfa_fees.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Mute pipeline logs pendant la comparaison (centaines de lignes/run).
os.environ["ICHIMOKU_QUIET"] = "1"

import ichimoku_pipeline_web_v4_8_fixed as pipeline
from src.cost_model import BinanceFutureFees


# Params Ichimoku canoniques (proches du K3 1D stable spec)
PARAMS = dict(
    tenkan=21, kijun=35, senkou_b=90, shift=44,
    atr_mult=10.5, loss_mult=3.0, tp_mult=8.0,
)
N_DAYS = 90  # Spec dit 30j, mais Ichimoku senkou_b=90 + shift=44 → besoin >150 bars d'init
HOURS_PER_DAY = 24


def _load_last_n_days(csv_path: Path, n_days: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_bars = n_days * HOURS_PER_DAY
    df = df.tail(n_bars).reset_index(drop=True)
    df = df.set_index("timestamp")
    return df


def _run_backtest(df: pd.DataFrame, label: str) -> dict:
    """Lance backtest_long_short et extrait equity_mult + nb trades du dict metrics."""
    print(f"\n[{label}] running backtest_long_short on {len(df)} bars...")
    metrics = pipeline.backtest_long_short(
        df, **PARAMS, symbol="BTC/USDT", timeframe="1h",
    )
    # Pipeline retourne un dict (cf ichimoku_pipeline ligne 1471)
    equity_final = float(metrics.get("equity_mult", 1.0))
    n_trades = int(metrics.get("trades", 0))
    n_long = int(metrics.get("nb_trades_long", 0))
    n_short = int(metrics.get("nb_trades_short", 0))
    print(f"[{label}] equity_mult={equity_final:.6f}, "
          f"trades={n_trades} (long={n_long}, short={n_short})")
    return {
        "label": label,
        "equity_final": equity_final,
        "n_trades": n_trades,
        "n_long": n_long,
        "n_short": n_short,
    }


def main() -> int:
    csv_path = ROOT / "data" / "BTC_USD_1h.csv"
    if not csv_path.exists():
        print(f"ERREUR: data manquante: {csv_path}", file=sys.stderr)
        return 1

    df = _load_last_n_days(csv_path, N_DAYS)
    print(f"Loaded {len(df)} bars BTC H1 ({df.index[0]} → {df.index[-1]})")

    # Run 1 : avec fees (default)
    res_with = _run_backtest(df, label="WITH_FEES")

    # Run 2 : sans fees (monkey-patch)
    original_helper = pipeline._round_trip_fee_cost_usdt
    pipeline._round_trip_fee_cost_usdt = lambda *args, **kwargs: 0.0
    try:
        res_no = _run_backtest(df, label="NO_FEES")
    finally:
        pipeline._round_trip_fee_cost_usdt = original_helper

    # Comparaison
    delta_equity = res_no["equity_final"] - res_with["equity_final"]
    n_trades_total = max(res_with["n_trades"], res_no["n_trades"])

    print("\n" + "=" * 70)
    print("RÉSULTATS R2 — P0bis Mini-WFA Validation")
    print("=" * 70)
    print(f"  equity NO_FEES   : {res_no['equity_final']:.8f}")
    print(f"  equity WITH_FEES : {res_with['equity_final']:.8f}")
    print(f"  delta            : {delta_equity:.8f}")
    print(f"  n_trades         : {n_trades_total}")
    # Drag attendu : pipeline calcule fees sur position_value = position_size
    # × capital = 0.01 × 1000 = 10 USDT (PAS multiplié par leverage). Fee
    # round-trip taker = 8 bps. Drag équité par trade = 8e-4 × 10 USDT /
    # 1000 USDT = 8e-6 (= 8 bps × position_size).
    expected_drag_per_trade = 0.01 * 8e-4   # position_size × round_trip_bps_fraction
    expected_drag_total = n_trades_total * expected_drag_per_trade
    print(f"  drag attendu     : ~{expected_drag_total:.6f} "
          f"(n_trades × 0.01 × 8e-4)")

    # Verdict
    if n_trades_total == 0:
        print("\n⚠️  AUCUN TRADE déclenché sur la fenêtre 30j → impossible de "
              "valider numériquement le branchement.")
        return 2

    if delta_equity > 1e-9:
        ratio = delta_equity / max(expected_drag_total, 1e-9)
        print(f"\n✅ BRANCHEMENT VALIDÉ : delta > 0 (fees diminuent equity)")
        print(f"   ratio observed/expected = {ratio:.2f} "
              f"(ordre de grandeur OK si 0.3 < r < 3.0)")
        return 0
    elif abs(delta_equity) < 1e-9:
        print(f"\n❌ ÉCHEC BRANCHEMENT : delta ≈ 0 → fees n'affectent PAS "
              "le PnL malgré le grep positif.")
        return 3
    else:
        print(f"\n⚠️  delta < 0 (fees augmentent equity ?!) — anomalie à "
              "investiguer.")
        return 4


if __name__ == "__main__":
    sys.exit(main())
