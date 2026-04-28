"""R9 E2E test : valider le funding-close runner sur testnet Binance.

Workflow :
  1. Charge .env (BINANCE_TESTNET=true, TRADE_MODE=live).
  2. Ouvre une position minimale BTC long sur testnet ($5 notional × leverage 1).
  3. Wait 3s pour confirmation Binance.
  4. Patche `funding_close_runner` :
       - is_settlement_window → True (force fenêtre)
       - _fetch_funding_rate_bps → 8.0 bps (force trigger)
  5. Set env HSBC_FUNDING_CLOSE_LIVE=1.
  6. Run main() — devrait close la position via TradeManager live.
  7. Wait 3s.
  8. Vérifie via Binance API que la position est bien close.

Usage :
  $ python scripts/validation/r9_e2e_funding_close_testnet.py

Sortie attendue :
  [OK] position opened on testnet
  [OK] funding_close runner triggered
  [OK] position closed (final qty=0)

Si le résultat est en rouge → investiguer (logs/funding_close.log).
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "binance_bot"))

# Charger .env si dispo
_env_path = ROOT / "binance_bot" / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        if k and v:
            os.environ.setdefault(k.strip(), v.strip())


def _open_test_position() -> str | None:
    """Ouvre une petite position long BTC sur testnet. Retourne order_id."""
    from services.data_fetcher import DataFetcher
    from bot.trade_manager import TradeManager

    print("[step 1] init data_fetcher (testnet=", os.environ.get("BINANCE_TESTNET"), ")")
    fetcher = DataFetcher(symbol="BTC/USDT", timeframe="1h")

    # Auto-detect position mode (testnet est généralement oneway)
    pos_mode = "oneway"
    try:
        info = fetcher.exchange.fapiPrivateGetPositionSideDual()
        if info.get("dualSidePosition"):
            pos_mode = "hedge"
    except Exception:
        pass
    print(f"[step 2.5] position_mode={pos_mode}")
    trade_mgr = TradeManager(
        exchange=fetcher.exchange,
        symbol="BTC/USDT",
        mode="live",
        leverage=1.0,  # leverage 1x pour minimiser risque
        position_mode=pos_mode,
    )

    # Récupère prix actuel pour calculer qty minimal
    df_recent = fetcher.get_ohlcv(limit=10)
    if len(df_recent) == 0:
        # Fallback : ticker fetch
        ticker = fetcher.exchange.fetch_ticker("BTC/USDT")
        last_close = float(ticker.get("last") or ticker.get("close") or 0)
    else:
        last_close = float(df_recent["close"].iloc[-1])
    print(f"[step 2] BTC last close: ${last_close:.2f}")

    # Position $60 notional pour respecter Binance min notional 50 USDT.
    capital_usdt = 1000.0  # capital fictif
    size_pct = 60.0 / capital_usdt  # 6%
    signal = {
        "action": "open_long",
        "entry": last_close,
        "stop": last_close * 0.95,
        "tp": last_close * 1.05,
        "size": size_pct,
    }
    print(f"[step 3] open_long size={size_pct} (${size_pct * capital_usdt} notional)")
    order_id = trade_mgr.execute_signal(signal, capital_usdt)
    print(f"[step 4] order_id = {order_id}")
    return order_id


def _check_position_open() -> bool:
    """Query Binance testnet pour vérifier qu'on a une position BTC ouverte."""
    from services.data_fetcher import DataFetcher
    fetcher = DataFetcher(symbol="BTC/USDT", timeframe="1h")
    try:
        positions = fetcher.exchange.fetch_positions(["BTC/USDT"])
        for p in positions:
            qty = float(p.get("contracts", 0) or 0)
            if abs(qty) > 1e-9:
                print(f"[check] open position: {p.get('side')} qty={qty}")
                return True
        return False
    except Exception as e:
        print(f"[check] error: {e}")
        return False


def _inject_state_for_test(order_id: str) -> Path:
    """Injecte une position dans state.json pour que funding_close la voie."""
    import json
    state_path = ROOT / "binance_bot" / "data" / "state.json"
    backup = state_path.with_suffix(".json.r9bak")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    backup.write_text(state_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Ajoute la position fictive
    sym_state = state.setdefault("symbols", {}).setdefault("BTC/USDT", {})
    sym_state["positions_long"] = [{
        "id": order_id or f"r9_test_{int(time.time())}",
        "entry": 100000.0, "stop": 95000.0, "tp": 105000.0,
        "size": 0.05,
    }]
    sym_state["positions_short"] = []
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"[state] injected position into state.json (backup at {backup.name})")
    return backup


def _restore_state(backup: Path) -> None:
    """Restaure state.json depuis backup."""
    state_path = backup.with_suffix("")
    state_path = state_path.with_name(state_path.name.replace(".r9bak", ""))
    state_path = backup.parent / "state.json"
    state_path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    backup.unlink(missing_ok=True)
    print(f"[state] restored from backup")


def _run_funding_close_with_patches() -> int:
    """Run le runner avec patches pour forcer window + funding élevé.

    Patches :
      - runner.is_settlement_window (check d'entrée main())
      - services.funding_close.is_settlement_window (check interne
        evaluate_close_signals)
      - runner._fetch_funding_rate_bps → 8.0 bps
      - runner.yaml.safe_load → injecte trade_mode=live (override
        bot_settings.yaml qui est en simulation par défaut)
    """
    from routines import funding_close_runner
    from services import funding_close as fc_module
    import yaml as _yaml

    os.environ["HSBC_FUNDING_CLOSE_LIVE"] = "1"

    # Override settings pour forcer trade_mode=live
    real_safe_load = _yaml.safe_load
    def _patched_safe_load(s):
        cfg = real_safe_load(s)
        if isinstance(cfg, dict):
            cfg["trade_mode"] = "live"
        return cfg

    print("[run] patching window=True + funding=8bps + trade_mode=live + main()...")
    with patch.object(funding_close_runner, "is_settlement_window", return_value=True), \
         patch.object(fc_module, "is_settlement_window", return_value=True), \
         patch.object(funding_close_runner, "_fetch_funding_rate_bps", return_value=8.0), \
         patch.object(funding_close_runner.yaml, "safe_load", _patched_safe_load):
        rc = funding_close_runner.main()
    print(f"[run] main() rc={rc}")
    return rc


def main() -> int:
    if os.environ.get("BINANCE_TESTNET", "").lower() != "true":
        print("[ERROR] BINANCE_TESTNET != 'true' — abort (refuse to touch live trading)")
        return 2
    print("=" * 70)
    print(f"R9 E2E funding-close testnet validation @ {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    order_id = None
    backup = None
    try:
        order_id = _open_test_position()
        time.sleep(3)
        if not _check_position_open():
            print("[WARN] position non confirmée sur Binance — abort")
            return 3

        backup = _inject_state_for_test(order_id)
        time.sleep(1)

        rc = _run_funding_close_with_patches()
        time.sleep(3)

        still_open = _check_position_open()
        if still_open:
            print("[FAIL] position still open after funding_close run")
            return 4
        else:
            print("[OK] position closed by funding_close runner ✓")
            return 0
    finally:
        if backup is not None:
            _restore_state(backup)


if __name__ == "__main__":
    sys.exit(main())
