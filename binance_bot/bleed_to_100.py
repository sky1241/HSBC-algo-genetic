"""Bleed testnet futures balance to ~$100 via fee round-trips.

Open market long ~$45k notional → close immediately. Round-trip cost ~$36 (8 bps fees + slippage).
Stops as soon as balance ≤ $100 + small buffer for fine-tune phase.
"""
import os
import sys
import time
from dotenv import load_dotenv
import ccxt

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

TARGET = 100.0
SAFETY_BUFFER_FROM_TARGET = 50.0  # when bal <= 150, switch to small trades
MAX_ITERS = 250
BIG_NOTIONAL = 45000.0  # $45k under $50k testnet cap at 125x BTC

ex = ccxt.binance({
    "apiKey": os.environ["BINANCE_API_KEY"],
    "secret": os.environ["BINANCE_API_SECRET"],
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})
ex.set_sandbox_mode(True)

# Detect hedge vs one-way
try:
    dual = ex.fapiPrivateGetPositionSideDual()
    is_hedge = dual.get("dualSidePosition") in (True, "true")
except Exception as e:
    print(f"[warn] dualSide fetch failed: {e} — assume one-way")
    is_hedge = False
print(f"position_mode: {'hedge' if is_hedge else 'oneway'}")

# Make sure leverage is 125x for max notional headroom
try:
    ex.set_leverage(125, "BTC/USDT")
except Exception as e:
    print(f"[warn] leverage set: {e}")


def get_balance():
    b = ex.fetch_balance()
    return float(b["total"].get("USDT", 0.0))


def round_trip(qty: float) -> tuple[float, float]:
    """Open long market then close. Returns (bal_before, bal_after)."""
    bal_before = get_balance()
    if is_hedge:
        ex.create_order("BTC/USDT", "market", "buy", qty, params={"positionSide": "LONG"})
        time.sleep(0.4)
        ex.create_order("BTC/USDT", "market", "sell", qty, params={"positionSide": "LONG"})
    else:
        ex.create_market_order("BTC/USDT", "buy", qty)
        time.sleep(0.4)
        ex.create_market_order("BTC/USDT", "sell", qty, params={"reduceOnly": True})
    time.sleep(0.4)
    return bal_before, get_balance()


def main():
    initial = get_balance()
    print(f"Starting balance: ${initial:,.2f}")
    print(f"Target:           ${TARGET:,.2f}")
    print(f"Need to lose:     ${initial - TARGET:,.2f}\n")

    if initial <= TARGET:
        print("Already at or below target.")
        return 0

    for i in range(MAX_ITERS):
        bal = get_balance()
        if bal <= TARGET:
            print(f"\n✅ Reached target. Final balance: ${bal:,.2f}")
            return 0

        ticker = ex.fetch_ticker("BTC/USDT")
        price = float(ticker["last"])

        # Adaptive sizing
        if bal > TARGET + SAFETY_BUFFER_FROM_TARGET:
            notional = BIG_NOTIONAL
        else:
            # Fine-tune phase: smaller trades to land precisely at $100
            excess = bal - TARGET
            notional = max(500.0, excess * 100.0)  # 8 bps × 100 = 0.8% of excess per trade
            notional = min(notional, BIG_NOTIONAL)

        qty = round(notional / price, 3)
        if qty < 0.002:
            print(f"qty too small ({qty}) — stopping")
            break

        try:
            b_before, b_after = round_trip(qty)
            delta = b_after - b_before
            print(f"[{i:3d}] bal ${b_before:>9,.2f} -> ${b_after:>9,.2f} ({delta:+8.2f}) qty={qty} notional=${qty*price:,.0f}")
        except Exception as e:
            print(f"[{i:3d}] ERROR: {e}")
            time.sleep(2)
            continue

    print(f"\nDone. Final balance: ${get_balance():,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
