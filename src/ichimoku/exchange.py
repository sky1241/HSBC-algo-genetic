import os
import json
import numpy as np

def symbol_to_binance(sym: str) -> str:
    try:
        return sym.replace("/", "")
    except Exception:
        return sym

def round_to_step(value: float, step: float) -> float:
    if step is None or step <= 0:
        return float(value)
    return float(np.floor((value + 1e-15) / step) * step)

def load_binance_filters(outputs_dir: str) -> dict:
    """Load exchangeInfo from outputs folder. Returns map symbol-> {tickSize, stepSize, minNotional}."""
    path = os.path.join(outputs_dir, "binance_usdm_exchangeInfo.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    result: dict[str, dict] = {}
    if isinstance(data, dict) and isinstance(data.get("symbols"), list):
        for s in data["symbols"]:
            sym = s.get("symbol")
            if not sym:
                continue
            tick = None
            step = None
            min_notional = None
            for f in s.get("filters", []):
                ftype = f.get("filterType")
                if ftype == "PRICE_FILTER":
                    try:
                        tick = float(f.get("tickSize"))
                    except Exception:
                        tick = None
                elif ftype in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                    try:
                        step = float(f.get("stepSize"))
                    except Exception:
                        step = step or None
                elif ftype == "MIN_NOTIONAL":
                    try:
                        min_notional = float(f.get("notional"))
                    except Exception:
                        min_notional = None
            result[sym] = {
                "tickSize": tick if (tick and tick > 0) else 0.01,
                "stepSize": step if (step and step > 0) else 0.001,
                "minNotional": min_notional if (min_notional and min_notional > 0) else 5.0,
            }
    return result

def binance_constraints_ok(symbol_ccxt: str, price: float, qty_notional: float, filters: dict) -> bool:
    sym_b = symbol_to_binance(symbol_ccxt)
    f = filters.get(sym_b)
    if not f:
        return qty_notional >= 5.0
    return qty_notional >= float(f.get("minNotional", 5.0))

def apply_binance_rounding(symbol_ccxt: str, price: float, qty: float, filters: dict) -> tuple[float, float]:
    sym_b = symbol_to_binance(symbol_ccxt)
    f = filters.get(sym_b)
    if not f:
        return float(price), float(qty)
    price_rounded = round_to_step(price, float(f.get("tickSize", 0.01)))
    qty_rounded = round_to_step(qty, float(f.get("stepSize", 0.001)))
    return price_rounded, qty_rounded
