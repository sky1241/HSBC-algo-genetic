import numpy as np

def daily_loss_threshold(atr: float, k: float) -> float:
    """Return daily loss threshold as k * ATR.

    If ATR is NaN or None, return infinity to disable the limit.
    """
    if atr is None or (isinstance(atr, float) and np.isnan(atr)):
        return float("inf")
    return k * atr
