"""Tests unitaires pour TradeManager (BUG-001 et suivants).

On mocke ccxt.binance — pas d'appel réseau réel.

Depuis l'introduction du middleware BinanceClient (B9 + B11), les tests qui
exercent les algo orders mockent `bot.trade_manager.BinanceClient` plutôt que
`requests.post/get/delete` directement.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Permettre l'import du package binance_bot depuis tests/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from bot.trade_manager import TradeManager


def _make_exchange_mock():
    """Construit un mock d'exchange ccxt avec valeurs par défaut sûres."""
    ex = MagicMock()
    ex.fetch_open_orders.return_value = []
    ex.fetch_positions.return_value = []
    ex.cancel_order.return_value = {'id': 'cancel_id'}
    ex.create_market_sell_order.return_value = {'id': 'mkt_sell_id'}
    ex.create_market_buy_order.return_value = {'id': 'mkt_buy_id'}
    # Précision passthrough — comportement historique (round 3 décimales BTC).
    # Sur ccxt réel, amount_to_precision arrondit selon stepSize du symbole.
    ex.amount_to_precision = lambda sym, qty: f"{round(float(qty), 3)}"
    ex.price_to_precision = lambda sym, p: f"{round(float(p), 2)}"
    return ex


def _make_client_mock(get_return=None, post_return=None, delete_return=None):
    """Crée un BinanceClient mocké avec valeurs par défaut sûres."""
    client = MagicMock()
    client.signed_get.return_value = get_return if get_return is not None else []
    client.signed_post.return_value = post_return if post_return is not None else {}
    client.signed_delete.return_value = (
        delete_return if delete_return is not None else {"code": "200", "msg": "success"}
    )
    return client


@pytest.fixture(autouse=True)
def _safe_binance_client(monkeypatch):
    """Évite tout appel réseau réel : remplace `BinanceClient` dans
    `bot.trade_manager` par une factory qui retourne un mock neutre.
    Les tests qui veulent un comportement spécifique réécrivent
    `tm.client` après construction OU passent `binance_client=...` explicitement.
    """
    def _factory(*args, **kwargs):
        return _make_client_mock()

    monkeypatch.setattr("bot.trade_manager.BinanceClient", _factory)
    yield


# ============================================================
# BUG-001 — close_long
# ============================================================

def test_close_long_simulation_does_not_touch_exchange():
    """En mode simulation: aucun appel à l'exchange, retourne sim_close_*."""
    ex = _make_exchange_mock()
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="simulation")

    result = tm.execute_signal(
        {"action": "close_long", "exit": 50000, "reason": "take_profit"},
        capital_usdt=1000,
    )

    assert result is not None
    assert result.startswith("sim_close_")
    ex.fetch_open_orders.assert_not_called()
    ex.fetch_positions.assert_not_called()
    ex.create_market_sell_order.assert_not_called()


def test_close_long_cancels_pending_and_market_closes_oneway():
    """One-way mode: annule SL/TP + market sell reduceOnly."""
    ex = _make_exchange_mock()
    ex.fetch_open_orders.return_value = [
        {'id': 'sl_order_1'},
        {'id': 'tp_order_1'},
    ]
    ex.fetch_positions.return_value = [
        {'symbol': 'BTC/USDT', 'side': 'long', 'contracts': 0.05},
    ]

    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live", position_mode="oneway")
    result = tm.execute_signal(
        {"action": "close_long", "exit": 78000, "reason": "trailing_stop"},
        capital_usdt=1000,
    )

    assert ex.cancel_order.call_count == 2
    cancelled_ids = {c.args[0] for c in ex.cancel_order.call_args_list}
    assert cancelled_ids == {'sl_order_1', 'tp_order_1'}

    ex.create_market_sell_order.assert_called_once()
    args = ex.create_market_sell_order.call_args
    assert args.args[0] == "BTC/USDT"
    assert args.args[1] == 0.05
    assert args.kwargs['params']['reduceOnly'] is True
    assert result == 'mkt_sell_id'


def test_close_long_hedge_mode_uses_position_side():
    """Hedge mode: market sell avec positionSide=LONG (pas reduceOnly)."""
    ex = _make_exchange_mock()
    ex.fetch_positions.return_value = [
        {'symbol': 'BTC/USDT', 'side': 'long', 'contracts': 0.05},
    ]

    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live", position_mode="hedge")
    tm.execute_signal(
        {"action": "close_long", "exit": 78000, "reason": "tp"},
        capital_usdt=1000,
    )

    args = ex.create_market_sell_order.call_args
    params = args.kwargs['params']
    assert params.get('positionSide') == 'LONG'
    assert 'reduceOnly' not in params


def test_close_long_no_position_returns_none_no_market_order():
    """Si aucune position long côté Binance: pas de market order."""
    ex = _make_exchange_mock()
    ex.fetch_positions.return_value = [
        {'symbol': 'BTC/USDT', 'side': 'short', 'contracts': 0.02},  # short uniquement
    ]
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live", position_mode="oneway")

    result = tm.execute_signal(
        {"action": "close_long", "exit": 78000, "reason": "manual"},
        capital_usdt=1000,
    )

    ex.create_market_sell_order.assert_not_called()
    assert result is None


def test_close_long_filters_other_symbols():
    """fetch_positions peut retourner d'autres paires — on ignore."""
    ex = _make_exchange_mock()
    ex.fetch_positions.return_value = [
        {'symbol': 'ETH/USDT', 'side': 'long', 'contracts': 1.0},
        {'symbol': 'BTC/USDT', 'side': 'long', 'contracts': 0.03},
    ]
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live", position_mode="oneway")

    tm.execute_signal(
        {"action": "close_long", "exit": 78000, "reason": "tp"},
        capital_usdt=1000,
    )

    args = ex.create_market_sell_order.call_args
    assert args.args[1] == 0.03  # qty BTC, pas ETH


def test_close_long_continues_when_cancel_fails():
    """Si un cancel fail, on continue à fermer la position."""
    ex = _make_exchange_mock()
    ex.fetch_open_orders.return_value = [{'id': 'sl_1'}, {'id': 'tp_1'}]
    ex.cancel_order.side_effect = [Exception("network"), {'id': 'cancel_id'}]
    ex.fetch_positions.return_value = [
        {'symbol': 'BTC/USDT', 'side': 'long', 'contracts': 0.04},
    ]
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live", position_mode="oneway")

    result = tm.execute_signal(
        {"action": "close_long", "exit": 78000, "reason": "sl"},
        capital_usdt=1000,
    )

    # Les deux cancels ont été tentés, le market order a quand même eu lieu
    assert ex.cancel_order.call_count == 2
    ex.create_market_sell_order.assert_called_once()
    assert result == 'mkt_sell_id'


# ============================================================
# BUG-001 — close_short
# ============================================================

def test_close_short_simulation_does_not_touch_exchange():
    ex = _make_exchange_mock()
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="simulation")

    result = tm.execute_signal(
        {"action": "close_short", "exit": 78000, "reason": "tp"},
        capital_usdt=1000,
    )

    assert result is not None
    assert result.startswith("sim_close_")
    ex.create_market_buy_order.assert_not_called()


def test_close_short_cancels_pending_and_market_closes():
    """En mode live one-way: annule SL/TP + market buy reduceOnly."""
    ex = _make_exchange_mock()
    ex.fetch_open_orders.return_value = [{'id': 'sl_s1'}]
    ex.fetch_positions.return_value = [
        {'symbol': 'BTC/USDT', 'side': 'short', 'contracts': 0.02},
    ]
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live", position_mode="oneway")

    result = tm.execute_signal(
        {"action": "close_short", "exit": 78000, "reason": "trailing_stop"},
        capital_usdt=1000,
    )

    ex.cancel_order.assert_called_once_with('sl_s1', "BTC/USDT")
    args = ex.create_market_buy_order.call_args
    assert args.args[1] == 0.02
    assert args.kwargs['params']['reduceOnly'] is True
    assert result == 'mkt_buy_id'


def test_close_short_no_position_returns_none():
    ex = _make_exchange_mock()
    ex.fetch_positions.return_value = []
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live", position_mode="oneway")

    result = tm.execute_signal(
        {"action": "close_short", "exit": 78000, "reason": "manual"},
        capital_usdt=1000,
    )

    ex.create_market_buy_order.assert_not_called()
    assert result is None


# ============================================================
# BUG-002 — apply_leverage_on_exchange
# ============================================================

def test_apply_leverage_simulation_skips_exchange_call():
    ex = _make_exchange_mock()
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="simulation", leverage=10.0)

    ok = tm.apply_leverage_on_exchange()

    assert ok is False
    ex.set_leverage.assert_not_called()


def test_apply_leverage_live_calls_set_leverage_with_int():
    """En live, on appelle exchange.set_leverage(int_leverage, symbol)."""
    ex = _make_exchange_mock()
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live", leverage=10.0)

    ok = tm.apply_leverage_on_exchange()

    assert ok is True
    ex.set_leverage.assert_called_once_with(10, "BTC/USDT")


def test_apply_leverage_handles_exchange_exception_gracefully():
    """Si set_leverage lève (ex: déjà configuré), on log mais on ne crash pas."""
    ex = _make_exchange_mock()
    ex.set_leverage.side_effect = Exception("leverage already set")
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live", leverage=10.0)

    ok = tm.apply_leverage_on_exchange()

    assert ok is False  # geste tenté mais pas confirmé OK


def test_default_leverage_is_one():
    ex = _make_exchange_mock()
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live")
    assert tm.leverage == 1.0


# ============================================================
# BUG-005 — sizing avec levier
# ============================================================

def test_open_long_sizing_with_leverage_simulation():
    """Avec leverage=10 et size_pct=0.01, qty = capital * 0.01 * 10 / price."""
    ex = _make_exchange_mock()
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="simulation", leverage=10.0)

    tm.execute_signal(
        {"action": "open_long", "entry": 50000, "stop": 49000, "tp": 60000, "size": 0.01},
        capital_usdt=1000,
    )
    # En simulation pas d'appel exchange — on vérifie qu'aucun create_market_buy_order n'a eu lieu
    ex.create_market_buy_order.assert_not_called()


def test_open_long_sizing_with_leverage_live():
    """En live, qty doit être (capital * size * leverage) / entry, arrondi à 3 décimales."""
    ex = _make_exchange_mock()
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live", leverage=10.0)

    tm.execute_signal(
        {"action": "open_long", "entry": 50000, "stop": 49000, "tp": 60000, "size": 0.01},
        capital_usdt=1000,
    )

    # qty = 1000 * 0.01 * 10 / 50000 = 0.002 → round 3 = 0.002
    args = ex.create_market_buy_order.call_args
    assert args.args[0] == "BTC/USDT"
    assert args.args[1] == 0.002


def test_open_long_sizing_no_leverage_default():
    """Sans levier (default=1.0), qty = capital * size / entry."""
    ex = _make_exchange_mock()
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live")

    tm.execute_signal(
        {"action": "open_long", "entry": 50000, "stop": 49000, "tp": 60000, "size": 0.1},
        capital_usdt=1000,
    )

    # qty = 1000 * 0.1 * 1 / 50000 = 0.002
    args = ex.create_market_buy_order.call_args
    assert args.args[1] == 0.002


def test_open_short_sizing_with_leverage_live():
    ex = _make_exchange_mock()
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live", leverage=5.0)

    tm.execute_signal(
        {"action": "open_short", "entry": 50000, "stop": 51000, "tp": 40000, "size": 0.02},
        capital_usdt=1000,
    )

    # qty = 1000 * 0.02 * 5 / 50000 = 0.002
    args = ex.create_market_sell_order.call_args
    assert args.args[1] == 0.002


# ============================================================
# BUG-ALGO — _place_algo_order via BinanceClient
# ============================================================

def test_place_algo_order_oneway_signs_and_uses_reduceOnly():
    """One-way mode: pas de positionSide, reduceOnly=true."""
    ex = _make_exchange_mock()
    client = _make_client_mock(post_return={"algoId": 1000000058811835})
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live",
                      api_key="K", api_secret="S",
                      algo_base_url="https://testnet.binancefuture.com",
                      position_mode="oneway",
                      binance_client=client)

    algo_id = tm._place_algo_order("SELL", "STOP_MARKET", 0.05, 76500.0)

    assert algo_id == "1000000058811835"
    client.signed_post.assert_called_once()
    path = client.signed_post.call_args.args[0]
    assert path == "/fapi/v1/algoOrder"
    params = client.signed_post.call_args.kwargs.get("params") or client.signed_post.call_args.args[1]
    assert params["algotype"] == "CONDITIONAL"
    assert params["symbol"] == "BTCUSDT"
    assert params["side"] == "SELL"
    assert params["type"] == "STOP_MARKET"
    assert params["quantity"] == 0.05
    assert params["triggerprice"] == 76500.0
    assert params["reduceOnly"] == "true"
    assert "positionSide" not in params
    assert params["workingType"] == "MARK_PRICE"


def test_place_algo_order_hedge_uses_positionSide_not_reduceOnly():
    """Hedge mode: positionSide=LONG/SHORT, pas de reduceOnly."""
    ex = _make_exchange_mock()
    client = _make_client_mock(post_return={"algoId": 99})
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live",
                      api_key="K", api_secret="S",
                      position_mode="hedge",
                      binance_client=client)

    tm._place_algo_order("SELL", "STOP_MARKET", 0.05, 76500.0, position_side="LONG")

    params = client.signed_post.call_args.kwargs.get("params")
    assert params["positionSide"] == "LONG"
    assert "reduceOnly" not in params


def test_place_algo_order_returns_none_on_error_response():
    ex = _make_exchange_mock()
    client = _make_client_mock(post_return={"code": -4120, "msg": "TIF cannot be set..."})
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live",
                      api_key="K", api_secret="S",
                      binance_client=client)
    result = tm._place_algo_order("SELL", "STOP_MARKET", 0.05, 76500)
    assert result is None


def test_place_algo_order_handles_request_exception():
    ex = _make_exchange_mock()
    client = MagicMock()
    client.signed_post.side_effect = Exception("connection reset")
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live",
                      api_key="K", api_secret="S",
                      binance_client=client)
    result = tm._place_algo_order("SELL", "STOP_MARKET", 0.05, 76500)
    assert result is None


def test_place_algo_order_skips_when_no_credentials():
    ex = _make_exchange_mock()
    ex.apiKey = None
    ex.secret = None
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live")
    # tm.client devrait être None car pas de credentials
    result = tm._place_algo_order("SELL", "STOP_MARKET", 0.05, 76500)
    assert result is None


def test_open_long_live_calls_algo_order_for_sl_and_tp():
    """Quand on ouvre un LONG en live, on poste 2 algoOrders (STOP_MARKET + TAKE_PROFIT_MARKET)."""
    ex = _make_exchange_mock()
    client = _make_client_mock(post_return={"algoId": 12345})
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live", leverage=10.0,
                      api_key="K", api_secret="S",
                      binance_client=client)

    tm.execute_signal(
        {"action": "open_long", "entry": 50000, "stop": 49000, "tp": 60000, "size": 0.01},
        capital_usdt=1000,
    )

    # 2 algoOrders (SL + TP)
    assert client.signed_post.call_count == 2

    # Premier appel = SL (STOP_MARKET, side SELL, trigger 49000)
    sl_params = client.signed_post.call_args_list[0].kwargs.get("params")
    assert sl_params["type"] == "STOP_MARKET"
    assert sl_params["side"] == "SELL"
    assert sl_params["triggerprice"] == 49000

    # Deuxième appel = TP (TAKE_PROFIT_MARKET, side SELL, trigger 60000)
    tp_params = client.signed_post.call_args_list[1].kwargs.get("params")
    assert tp_params["type"] == "TAKE_PROFIT_MARKET"
    assert tp_params["side"] == "SELL"
    assert tp_params["triggerprice"] == 60000


def test_fetch_open_algo_orders_returns_list_and_signs_request():
    """GET /fapi/v1/openAlgoOrders signé via BinanceClient."""
    ex = _make_exchange_mock()
    client = _make_client_mock(get_return=[
        {"algoId": 111, "symbol": "BTCUSDT", "orderType": "STOP_MARKET"},
        {"algoId": 222, "symbol": "BTCUSDT", "orderType": "TAKE_PROFIT_MARKET"},
    ])
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live",
                      api_key="K", api_secret="S",
                      algo_base_url="https://testnet.binancefuture.com",
                      binance_client=client)

    result = tm.fetch_open_algo_orders()

    assert len(result) == 2
    assert result[0]["algoId"] == 111
    client.signed_get.assert_called_once()
    path = client.signed_get.call_args.args[0]
    assert path == "/fapi/v1/openAlgoOrders"
    params = client.signed_get.call_args.kwargs.get("params")
    assert params["symbol"] == "BTCUSDT"


def test_fetch_open_algo_orders_returns_empty_on_error_dict():
    """Si Binance renvoie un dict d'erreur, on retourne []."""
    ex = _make_exchange_mock()
    client = _make_client_mock(get_return={"code": -5000, "msg": "Path invalid"})
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live",
                      api_key="K", api_secret="S",
                      binance_client=client)
    result = tm.fetch_open_algo_orders()
    assert result == []


def test_fetch_open_algo_orders_no_credentials_returns_empty():
    """Sans api_key/secret, on n'appelle pas le client et retourne []."""
    ex = _make_exchange_mock()
    ex.apiKey = None
    ex.secret = None
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live")
    result = tm.fetch_open_algo_orders()
    assert result == []


def test_cancel_algo_order_returns_true_on_success():
    """DELETE /fapi/v1/algoOrder avec algoId; succès si code='200' ou msg='success'."""
    ex = _make_exchange_mock()
    client = _make_client_mock(delete_return={"algoId": 123, "code": "200", "msg": "success"})
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live",
                      api_key="K", api_secret="S",
                      algo_base_url="https://testnet.binancefuture.com",
                      binance_client=client)

    ok = tm.cancel_algo_order(123)

    assert ok is True
    client.signed_delete.assert_called_once()
    path = client.signed_delete.call_args.args[0]
    assert path == "/fapi/v1/algoOrder"
    params = client.signed_delete.call_args.kwargs.get("params")
    assert params["algoId"] == 123


def test_cancel_algo_order_returns_false_on_error():
    """Code négatif → retourne False."""
    ex = _make_exchange_mock()
    client = _make_client_mock(delete_return={"code": -2011, "msg": "Unknown order sent."})
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live",
                      api_key="K", api_secret="S",
                      binance_client=client)
    ok = tm.cancel_algo_order(999)
    assert ok is False


def test_cancel_algo_order_handles_request_exception():
    ex = _make_exchange_mock()
    client = MagicMock()
    client.signed_delete.side_effect = Exception("net")
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live",
                      api_key="K", api_secret="S",
                      binance_client=client)
    assert tm.cancel_algo_order(1) is False


def test_cancel_all_algo_orders_counts_successes():
    """cancel_all_algo_orders annule chaque order listé et compte les succès."""
    ex = _make_exchange_mock()
    client = MagicMock()
    client.signed_get.return_value = [
        {"algoId": 1}, {"algoId": 2}, {"algoId": 3},
    ]
    # 1er & 3e succès, 2e échec
    client.signed_delete.side_effect = [
        {"code": "200", "msg": "success"},
        {"code": -2011, "msg": "Unknown order"},
        {"code": "200", "msg": "success"},
    ]
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live",
                      api_key="K", api_secret="S",
                      binance_client=client)

    n = tm.cancel_all_algo_orders()

    assert n == 2  # 2 cancels réussis sur 3
    assert client.signed_delete.call_count == 3


def test_cancel_all_algo_orders_empty_when_no_open():
    ex = _make_exchange_mock()
    client = _make_client_mock(get_return=[])
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live",
                      api_key="K", api_secret="S",
                      binance_client=client)
    assert tm.cancel_all_algo_orders() == 0
    client.signed_delete.assert_not_called()


def test_close_long_calls_cancel_all_algo_orders():
    """_close_long doit annuler les algo orders en plus des ordres classiques."""
    ex = _make_exchange_mock()
    ex.fetch_positions.return_value = [
        {'symbol': 'BTC/USDT', 'side': 'long', 'contracts': 0.05},
    ]
    client = MagicMock()
    client.signed_get.return_value = [{"algoId": 999}]
    client.signed_delete.return_value = {"code": "200", "msg": "success"}
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live",
                      api_key="K", api_secret="S", position_mode="oneway",
                      binance_client=client)

    tm.execute_signal(
        {"action": "close_long", "exit": 78000, "reason": "tp"},
        capital_usdt=1000,
    )

    # Au moins un DELETE algo et le market sell
    assert client.signed_delete.call_count == 1
    ex.create_market_sell_order.assert_called_once()


def test_close_short_calls_cancel_all_algo_orders():
    ex = _make_exchange_mock()
    ex.fetch_positions.return_value = [
        {'symbol': 'BTC/USDT', 'side': 'short', 'contracts': 0.02},
    ]
    client = MagicMock()
    client.signed_get.return_value = [{"algoId": 888}, {"algoId": 889}]
    client.signed_delete.return_value = {"code": "200", "msg": "success"}
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live",
                      api_key="K", api_secret="S", position_mode="oneway",
                      binance_client=client)

    tm.execute_signal(
        {"action": "close_short", "exit": 78000, "reason": "tp"},
        capital_usdt=1000,
    )

    assert client.signed_delete.call_count == 2
    ex.create_market_buy_order.assert_called_once()


def test_open_short_live_calls_algo_order_for_sl_and_tp():
    """Pour un SHORT, le SL et le TP sont des BUY (cover)."""
    ex = _make_exchange_mock()
    client = _make_client_mock(post_return={"algoId": 12345})
    tm = TradeManager(exchange=ex, symbol="BTC/USDT", mode="live", leverage=10.0,
                      api_key="K", api_secret="S",
                      binance_client=client)

    tm.execute_signal(
        {"action": "open_short", "entry": 50000, "stop": 51000, "tp": 40000, "size": 0.01},
        capital_usdt=1000,
    )

    assert client.signed_post.call_count == 2
    sl_params = client.signed_post.call_args_list[0].kwargs.get("params")
    assert sl_params["type"] == "STOP_MARKET"
    assert sl_params["side"] == "BUY"
    tp_params = client.signed_post.call_args_list[1].kwargs.get("params")
    assert tp_params["type"] == "TAKE_PROFIT_MARKET"
    assert tp_params["side"] == "BUY"
