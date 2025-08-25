import numpy as np
import pandas as pd
try:
    import optuna
    from optuna.pruners import SuccessiveHalvingPruner
except Exception:  # pragma: no cover - optuna optional
    optuna = None

from .data import make_annual_folds

def sample_params_optuna(trial):
    """Sample Ichimoku parameters under constraints."""
    tenkan = trial.suggest_int("tenkan", 5, 30)
    r_kijun = trial.suggest_int("r_kijun", 1, 5)
    r_senkou = trial.suggest_int("r_senkou", 1, 9)
    kijun = max(tenkan, r_kijun * tenkan)
    senkou_b = max(kijun, r_senkou * tenkan)
    shift = trial.suggest_int("shift", 20, 30)
    atr_mult = trial.suggest_float("atr_mult", 0.5, 6.0, step=0.1)
    return {
        "tenkan": int(tenkan),
        "kijun": int(kijun),
        "senkou_b": int(senkou_b),
        "shift": int(shift),
        "atr_mult": float(atr_mult),
    }

def compute_score_optuna(cagr_list, sharpe_list, dd_list, trades_list):
    mean_sharpe = float(np.mean(sharpe_list)) if sharpe_list else 0.0
    mean_cagr = float(np.mean(cagr_list)) if cagr_list else 0.0
    mean_dd = float(np.mean(dd_list)) if dd_list else 0.0
    stab_penalty = float(np.std(sharpe_list) + 0.5 * np.std(cagr_list)) if sharpe_list else 0.0
    trade_penalty = 0.0 if (float(np.mean(trades_list)) if trades_list else 0.0) >= 30.0 else 0.5
    return 0.6 * mean_sharpe + 0.3 * mean_cagr - 0.3 * mean_dd - 0.5 * stab_penalty - trade_penalty

def objective(trial, market_data: dict, timeframe: str, start_year: int | None, end_year: int | None, fast_ratio: float, backtest_fn):
    params = sample_params_optuna(trial)
    cagr_list, sharpe_list, dd_list, trades_list = [], [], [], []
    for sym, df in market_data.items():
        folds = make_annual_folds(df, start_year, end_year)
        for (start, end) in folds:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            if 0.0 < fast_ratio < 1.0:
                horizon = end_ts - start_ts
                cut_ts = start_ts + pd.Timedelta(seconds=int(horizon.total_seconds() * fast_ratio))
                sub = df.loc[start_ts:cut_ts]
            else:
                sub = df.loc[start_ts:end_ts]
            if sub.empty:
                continue
            m = backtest_fn(sub, params, symbol=sym, timeframe=timeframe)
            cagr_list.append(float(m.get("CAGR", 0.0)))
            sharpe_list.append(float(m.get("sharpe_proxy", 0.0)))
            dd_list.append(float(m.get("max_drawdown", 0.0)))
            trades_list.append(int(m.get("trades", 0)))
    return compute_score_optuna(cagr_list, sharpe_list, dd_list, trades_list)

def create_study(n_trials: int, objective_fn, seed: int | None = None, jobs: int = 1):
    if optuna is None:
        raise RuntimeError("Optuna non disponible. Installez-le (pip install optuna).")
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective_fn, n_trials=int(n_trials), n_jobs=int(jobs))
    return study
