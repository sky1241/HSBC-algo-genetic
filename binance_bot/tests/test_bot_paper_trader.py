"""Tests pour PaperTrader — B8."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from bot.paper_trader import PaperTrader, CSV_HEADERS


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "paper_log.csv"


def test_paper_trader_creates_file_with_headers(log_path):
    pt = PaperTrader(log_path)
    assert log_path.exists()
    content = log_path.read_text()
    for h in CSV_HEADERS:
        assert h in content


def test_paper_trader_does_not_overwrite_existing_file(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("dummy content\n")
    PaperTrader(log_path)
    # Le fichier existant n'est PAS écrasé
    assert log_path.read_text().startswith("dummy")


def test_log_open_appends_row_with_costs(log_path):
    pt = PaperTrader(log_path, taker_bps=4.0, slippage_bps=2.0)
    pt.log_open("open_long", qty=0.001, price=78000, live_order_id="oid1", signal_id="sig1")
    rep = pt.report()
    # qty=0.001 * 78000 = 78 USDT notional
    # fees = 78 * 4/10000 = 0.0312
    # slip = 78 * 2/10000 = 0.0156
    assert rep.n_trades == 0          # open ne compte pas comme trade fermé
    assert rep.open_positions == 1
    assert rep.total_fees == pytest.approx(-0.0312, rel=0.01)
    assert rep.total_slippage == pytest.approx(-0.0156, rel=0.01)


def test_log_open_then_close_long_computes_sim_and_live_pnl(log_path):
    pt = PaperTrader(log_path, taker_bps=4.0, slippage_bps=2.0, funding_per_8h=0.0)
    pt.log_open("open_long", qty=0.001, price=78000, live_order_id="o1", signal_id="s1")
    # +1000 mouvement de prix
    pt.log_close("close_long", qty=0.001, exit_price=79000, entry_price=78000,
                 live_order_id="c1", signal_id="s1", held_seconds=3600)
    rep = pt.report()
    # sim_pnl long = (79000 - 78000) * 0.001 = 1.0
    assert rep.total_sim_pnl == pytest.approx(1.0, abs=0.01)
    # fees ouverture+fermeture = 78*4/10000 + 79*4/10000 ≈ 0.0628
    # slip ouverture+fermeture ≈ 0.0314
    # live_pnl = sim - close_fees - close_slip ≈ 1 - 0.0316 - 0.0158 ≈ 0.9526
    assert rep.total_live_pnl == pytest.approx(0.9526, abs=0.01)
    assert rep.n_trades == 1
    assert rep.open_positions == 0  # paire complète


def test_log_short_close_inverts_pnl_sign(log_path):
    pt = PaperTrader(log_path, taker_bps=0.0, slippage_bps=0.0, funding_per_8h=0.0)
    pt.log_open("open_short", qty=0.001, price=80000, live_order_id="o", signal_id="s")
    # Prix BAISSE → short profit
    pt.log_close("close_short", qty=0.001, exit_price=79000, entry_price=80000,
                 live_order_id="c", signal_id="s", held_seconds=0)
    rep = pt.report()
    # sim_pnl short = (80000 - 79000) * 0.001 = 1.0
    assert rep.total_sim_pnl == pytest.approx(1.0, abs=0.01)
    # Pas de coûts → live = sim
    assert rep.total_live_pnl == pytest.approx(1.0, abs=0.01)


def test_funding_long_pays_short_receives(log_path):
    """Avec funding rate positif, long paye, short reçoit."""
    pt_long = PaperTrader(log_path, taker_bps=0.0, slippage_bps=0.0, funding_per_8h=0.001)
    # Position détenue 8h, funding 0.1% sur notional 78
    f_long = pt_long.compute_funding_cost(qty=0.001, price=78000, position_side="long",
                                           held_seconds=8 * 3600)
    f_short = pt_long.compute_funding_cost(qty=0.001, price=78000, position_side="short",
                                            held_seconds=8 * 3600)
    # Long paye 78 * 0.001 = 0.078
    assert f_long == pytest.approx(0.078, abs=0.001)
    assert f_short == pytest.approx(-0.078, abs=0.001)


def test_unmatched_open_counted_as_open_position(log_path):
    pt = PaperTrader(log_path)
    pt.log_open("open_long", qty=0.001, price=78000, live_order_id="o", signal_id="abc")
    rep = pt.report()
    assert rep.open_positions == 1
    assert rep.n_trades == 0


def test_tracking_error_zero_with_single_trade(log_path):
    pt = PaperTrader(log_path)
    pt.log_open("open_long", qty=0.001, price=78000, live_order_id="o", signal_id="s")
    pt.log_close("close_long", qty=0.001, exit_price=79000, entry_price=78000,
                 live_order_id="c", signal_id="s", held_seconds=3600)
    rep = pt.report()
    assert rep.tracking_error == 0.0


def test_tracking_error_nonzero_with_multiple_trades(log_path):
    pt = PaperTrader(log_path, taker_bps=4.0, slippage_bps=2.0, funding_per_8h=0.0)
    # Trade 1: petit
    pt.log_open("open_long", qty=0.001, price=78000, live_order_id="o1", signal_id="s1")
    pt.log_close("close_long", qty=0.001, exit_price=78500, entry_price=78000,
                 live_order_id="c1", signal_id="s1", held_seconds=3600)
    # Trade 2: gros (notional 10x)
    pt.log_open("open_long", qty=0.01, price=78000, live_order_id="o2", signal_id="s2")
    pt.log_close("close_long", qty=0.01, exit_price=78500, entry_price=78000,
                 live_order_id="c2", signal_id="s2", held_seconds=3600)
    rep = pt.report()
    assert rep.tracking_error > 0
    assert rep.n_trades == 2


def test_report_empty_log_returns_zeros(log_path):
    pt = PaperTrader(log_path)  # crée le header
    rep = pt.report()
    assert rep.n_trades == 0
    assert rep.total_live_pnl == 0
    assert rep.tracking_error == 0
