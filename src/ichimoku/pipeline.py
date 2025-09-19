from ..io_loader import align_funding_to_ohlcv, load_funding, load_ohlcv
from .exchange import load_binance_filters
from .optimization import objective, create_study


def run_pipeline(
    ohlcv_csv,
    funding_csv,
    symbol: str,
    timeframe: str,
    *,
    outputs_dir: str = "outputs",
    tz: str | None = "UTC",
    backtest_fn=None,
    optuna_trials: int = 0,
):
    """Load cached data through :mod:`src.io_loader` and optionally optimise."""

    df_prices = load_ohlcv(ohlcv_csv, tz=tz)
    df_funding = load_funding(funding_csv)
    aligned = align_funding_to_ohlcv(df_prices, df_funding, freq=timeframe.upper())
    if not aligned["close"].equals(df_prices["close"]):
        raise ValueError("Les clôtures ne sont pas alignées après chargement")
    df = df_prices.copy()
    df["funding"] = aligned["funding"]

    filters = load_binance_filters(outputs_dir)
    study = None
    if backtest_fn and optuna_trials > 0:
        market_data = {symbol: df}
        obj = lambda tr: objective(tr, market_data, timeframe, None, None, 1.0, backtest_fn)
        study = create_study(optuna_trials, obj)
    return df, filters, study
