from .data import fetch_ohlcv_range
from .exchange import load_binance_filters, binance_constraints_ok, apply_binance_rounding
from .optimization import objective, create_study


def run_pipeline(exchange, symbol: str, timeframe: str, since_ms: int, until_ms: int,
                  outputs_dir: str = "outputs", use_cache: bool = True,
                  backtest_fn=None, optuna_trials: int = 0):
    """Orchestrate data loading, exchange filters and optional optimization."""
    df = fetch_ohlcv_range(exchange, symbol, timeframe, since_ms, until_ms,
                           cache_dir="data", use_cache=use_cache)
    filters = load_binance_filters(outputs_dir)
    study = None
    if backtest_fn and optuna_trials > 0:
        market_data = {symbol: df}
        obj = lambda tr: objective(tr, market_data, timeframe, None, None, 1.0, backtest_fn)
        study = create_study(optuna_trials, obj)
    return df, filters, study
