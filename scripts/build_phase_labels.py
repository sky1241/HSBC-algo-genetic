#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build three labelings on the same timeline:
  - 3-regimes (up/down/range) from M and LFP
  - 5-phases (accumulation/expansion/euphoria/distribution/bear) from phase_aware_module
  - 6-phases (5-phases + capitulation) based on extreme drawdown/vol/momentum

Outputs:
  - outputs/fourier/phase_labels/<sym>_<tf>_labels.csv
  - outputs/fourier/phase_labels/<sym>_<tf>_confusion_3v5.csv
  - outputs/fourier/phase_labels/<sym>_<tf>_confusion_3v6.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.phase_aware_module import phase_snapshot  # type: ignore


OHLCV_PATHS: Dict[tuple[str,str], Path] = {
    ("BTC_USDT","2h"): Path('data') / 'BTC_USDT_2h.csv',
    ("BTC_USDT","1d"): Path('data') / 'BTC_USDT_1d.csv',
    ("BTC_USD","2h"): Path('data') / 'BTC_USD_2h.csv',
    ("BTC_USD","1d"): Path('data') / 'BTC_USD_1d.csv',
}


def read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    return df[['open','high','low','close','volume']]


def load_daily_lfp(sym: str, tf: str) -> pd.DataFrame:
    p = Path('outputs') / 'fourier' / f'DAILY_SUMMARY_{sym}_{tf}.csv'
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    return df[['LFP']]


def label_3_regimes(M: float, LFP: float, m_up: float = 0.05, m_down: float = -0.05, lfp_trend: float = 0.80) -> str:
    if np.isnan(M) or np.isnan(LFP):
        return 'unknown'
    if (M >= m_up) and (LFP >= lfp_trend):
        return 'up'
    if (M <= m_down) and (LFP >= lfp_trend):
        return 'down'
    return 'range'


def label_6_phases(phase5: str, M: float, V_ann: float, DD: float) -> str:
    # Capitulation as extreme bear
    if not (np.isnan(M) or np.isnan(V_ann) or np.isnan(DD)):
        if (M <= -0.15) and (V_ann >= 1.0) and (DD <= -0.50):
            return 'capitulation'
    return phase5


def main() -> int:
    ap = argparse.ArgumentParser(description='Build labels: 3-regimes, 5-phases, 6-phases (capitulation)')
    ap.add_argument('--symbol', default='BTC_USDT')
    ap.add_argument('--timeframe', default='2h')
    args = ap.parse_args()

    sym, tf = args.symbol, args.timeframe
    ohlcv_path = OHLCV_PATHS.get((sym, tf))
    if ohlcv_path is None or not ohlcv_path.exists():
        raise FileNotFoundError(f'OHLCV not found for {(sym, tf)}')
    df = read_ohlcv(ohlcv_path)

    feats = phase_snapshot(df)
    # daily alignment
    feats_d = feats.resample('1D').last()
    lfp_d = load_daily_lfp(sym, tf)
    joined = feats_d.join(lfp_d, how='inner')

    # 3-regimes
    joined['regime3'] = [label_3_regimes(m, l) for m, l in zip(joined['M'], joined['LFP'])]
    # 5-phases
    joined['phase5'] = joined['phase']
    # 6-phases (bear -> capitulation if extreme)
    joined['phase6'] = [label_6_phases(p5, m, v, dd) for p5, m, v, dd in zip(joined['phase5'], joined['M'], joined['V_ann'], joined['DD'])]

    out_dir = Path('outputs') / 'fourier' / 'phase_labels'
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = out_dir / f'{sym}_{tf}_labels.csv'
    cols = ['M','V_ann','DD','LFP','regime3','phase5','phase6']
    joined.reset_index().rename(columns={'timestamp':'date'}).to_csv(labels_csv, index=False, columns=['date']+cols)

    # Confusions
    conf_3v5 = pd.crosstab(joined['regime3'], joined['phase5']).reset_index()
    conf_3v6 = pd.crosstab(joined['regime3'], joined['phase6']).reset_index()
    conf_3v5.to_csv(out_dir / f'{sym}_{tf}_confusion_3v5.csv', index=False)
    conf_3v6.to_csv(out_dir / f'{sym}_{tf}_confusion_3v6.csv', index=False)

    print('Wrote:', labels_csv)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


