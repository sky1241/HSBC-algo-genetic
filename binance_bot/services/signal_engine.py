#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Signal Engine: détecte signaux EXACT comme backtest_long_short.

P0 (2026-04-28): support `portfolio_state` pour moduler le `size` du signal
d'entrée selon le risk agrégé multi-symbole déjà utilisé.

P1 (2026-04-28): daily caps soft/hard sur PnL signed du jour.
- Soft loss/gain : flat positions internes + block_until_iso = next UTC midnight
- Hard loss : trigger kill_switch + Telegram CRITICAL + freeze
Sorties TP/SL/trailing restent actives même si block actif (le bot doit pouvoir
fermer ses positions pendant le block).
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


class SignalEngine:
    """
    Reproduit la logique EXACTE de backtest_long_short pour détecter signaux.

    État interne maintenu: positions_long, positions_short (max 3 chacun).

    P0: optionnellement un `portfolio_scale_fn` callable est invoqué juste
    avant d'émettre un signal `open_long`/`open_short` pour moduler `size`
    selon le risk agrégé déjà utilisé multi-symbole. Si non fourni → comportement
    legacy avec size=position_size_pct constant.

    P1: daily_loss_soft_cap_pct, daily_gain_soft_cap_pct, daily_loss_hard_cap_pct
    déclenchent flat positions + block jusqu'à 00:00 UTC suivant (soft) ou
    kill_switch (hard).
    """

    def __init__(
        self,
        max_positions: int = 3,
        daily_loss_threshold: float = 0.10,
        atr_trailing_mult: float = 2.0,
        portfolio_scale_fn: Optional[Callable[[], float]] = None,
        daily_loss_soft_cap_pct: float = 0.0,
        daily_gain_soft_cap_pct: float = 0.0,
        daily_loss_hard_cap_pct: float = 0.0,
        kill_switch_path: Optional[Path] = None,
        notifier: Optional[Any] = None,
        drawdown_scale_fn: Optional[Callable[[], float]] = None,
        var_gate_fn: Optional[Callable[[str, float], Tuple[bool, str]]] = None,
    ):
        """
        Args:
            max_positions: nombre max de positions par côté (3 dans backtest)
            daily_loss_threshold: seuil perte journalière legacy (10% backtest)
            atr_trailing_mult: multiplicateur ATR pour le trailing stop (cliquet)
            portfolio_scale_fn: P0 — callable sans args, scale [0,1] portfolio-aware.
            daily_loss_soft_cap_pct: P1 — perte signed (e.g. 0.02 = -2%) qui flat
                positions + block nouvelles entrées jusqu'au prochain 00:00 UTC.
                0.0 désactive.
            daily_gain_soft_cap_pct: P1 — gain (e.g. 0.03 = +3%) qui flat positions
                + block. 0.0 désactive.
            daily_loss_hard_cap_pct: P1 — perte hard (e.g. 0.10 = -10%) qui
                déclenche kill_switch + Telegram CRITICAL. 0.0 désactive.
            kill_switch_path: P1 — chemin vers data/.killed pour trigger_kill().
                None = pas de kill switch (mode test/backtest).
            notifier: P1 — TelegramNotifier (méthodes critical/warn/info).
            drawdown_scale_fn: P2 — callable sans args qui retourne un facteur
                [0, 1] selon le drawdown depuis equity_high (anti-martingale,
                cf src/risk_sizing.drawdown_size_multiplier). Combiné multiplicativement
                avec portfolio_scale dans le sizing final.
            var_gate_fn: P3 — callable (side, notional_pct) -> (allowed, reason)
                qui BLOQUE l'émission d'un signal d'entrée si la VaR95 projetée
                du portefeuille dépasse le seuil. Distinct de portfolio_scale_fn
                qui module la taille — ici on bloque carrément.
        """
        self.max_positions = max_positions
        self.daily_loss_threshold = daily_loss_threshold
        self.atr_trailing_mult = float(atr_trailing_mult)
        self.portfolio_scale_fn = portfolio_scale_fn
        # P1 daily caps
        self.daily_loss_soft_cap_pct = float(daily_loss_soft_cap_pct)
        self.daily_gain_soft_cap_pct = float(daily_gain_soft_cap_pct)
        self.daily_loss_hard_cap_pct = float(daily_loss_hard_cap_pct)
        self.kill_switch_path = kill_switch_path
        self.notifier = notifier
        # P2 anti-martingale
        self.drawdown_scale_fn = drawdown_scale_fn
        # P3 portfolio risk gate VaR95
        self.var_gate_fn = var_gate_fn
        # État runtime
        self.positions_long: List[Dict] = []
        self.positions_short: List[Dict] = []
        self.daily_loss = 0.0
        # P1 — PnL signed du jour (positif = gain, négatif = perte) en fraction capital
        self.daily_pnl_pct = 0.0
        # P1 — timestamp ISO UTC jusqu'auquel block_new_entries est actif (None = pas de block)
        self.block_until_iso: Optional[str] = None

    def _ratchet_trailing_stops(self, current_price: float, atr: float):
        """Met à jour les stops des positions ouvertes (cliquet — jamais desserré).

        LONG : stop = max(stop_actuel, prix - atr * mult)
        SHORT: stop = min(stop_actuel, prix + atr * mult)
        """
        if atr <= 0:
            return
        offset = atr * self.atr_trailing_mult
        for pos in self.positions_long:
            new_stop = current_price - offset
            if new_stop > pos.get("stop", float("-inf")):
                pos["stop"] = new_stop
        for pos in self.positions_short:
            new_stop = current_price + offset
            if new_stop < pos.get("stop", float("inf")):
                pos["stop"] = new_stop
    
    def load_state(
        self,
        positions_long: List[Dict],
        positions_short: List[Dict],
        daily_loss: float,
        daily_pnl_pct: float = 0.0,
        block_until_iso: Optional[str] = None,
    ):
        """Charge positions + daily_pnl + block timestamp depuis state_manager.

        P1: daily_pnl_pct et block_until_iso ajoutés (defaults backward-compat).
        """
        self.positions_long = positions_long
        self.positions_short = positions_short
        self.daily_loss = daily_loss
        self.daily_pnl_pct = float(daily_pnl_pct)
        self.block_until_iso = block_until_iso

    # ===== P1 daily caps helpers =====

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _next_utc_midnight_iso(self) -> str:
        """Retourne ISO timestamp du prochain 00:00 UTC."""
        now = self._now_utc()
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return tomorrow.isoformat()

    def _is_blocked_now(self) -> bool:
        """True si block_until_iso > now (= entries bloquées). Auto-clear si expiré."""
        if not self.block_until_iso:
            return False
        try:
            until = datetime.fromisoformat(self.block_until_iso)
        except (TypeError, ValueError):
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        if self._now_utc() >= until:
            # Block expiré → clear
            self.block_until_iso = None
            return False
        return True

    def _flat_all_internal_positions(
        self, current_price: float, reason: str
    ) -> List[Dict]:
        """Émet signaux close pour toutes positions internes. Vide les listes."""
        out: List[Dict] = []
        for pos in list(self.positions_long):
            out.append({
                "action": "close_long",
                "pos_id": pos["id"],
                "exit": current_price,
                "reason": reason,
            })
        self.positions_long = []
        for pos in list(self.positions_short):
            out.append({
                "action": "close_short",
                "pos_id": pos["id"],
                "exit": current_price,
                "reason": reason,
            })
        self.positions_short = []
        return out

    def _hard_cap_triggered(self) -> bool:
        if self.daily_loss_hard_cap_pct <= 0:
            return False
        return self.daily_pnl_pct <= -self.daily_loss_hard_cap_pct

    def _soft_loss_cap_triggered(self) -> bool:
        if self.daily_loss_soft_cap_pct <= 0:
            return False
        return self.daily_pnl_pct <= -self.daily_loss_soft_cap_pct

    def _soft_gain_cap_triggered(self) -> bool:
        if self.daily_gain_soft_cap_pct <= 0:
            return False
        return self.daily_pnl_pct >= self.daily_gain_soft_cap_pct

    def _trigger_hard_cap(self, current_price: float) -> List[Dict]:
        """Hard cap: flat all + kill switch + Telegram critical.

        Retourne les signaux close. Kill switch + alerte sont side-effects.
        """
        signals = self._flat_all_internal_positions(current_price, reason="hard_loss_cap")
        msg = (
            f"HARD DAILY LOSS CAP triggered: pnl={self.daily_pnl_pct*100:+.3f}% "
            f"<= -{self.daily_loss_hard_cap_pct*100:.2f}% (freeze until manual audit)"
        )
        # Kill switch (best effort)
        if self.kill_switch_path is not None:
            try:
                from bot.kill_switch import trigger_kill  # type: ignore
                trigger_kill(self.kill_switch_path, msg)
            except Exception:
                pass
        # Telegram alert (best effort)
        if self.notifier is not None:
            try:
                self.notifier.critical(msg)
            except Exception:
                pass
        return signals

    def _var_gate_blocks(self, side: str, notional_pct: float) -> bool:
        """P3 — True si var_gate_fn refuse l'entrée. False si pas de gate ou autorisé.

        Sécurisation: si le callback raise, on AUTORISE par défaut (fallback safe
        symétrique aux autres callbacks: ne jamais bloquer le bot par exception).
        """
        if self.var_gate_fn is None:
            return False
        try:
            allowed, reason = self.var_gate_fn(side, float(notional_pct))
            if not allowed and self.notifier is not None:
                try:
                    self.notifier.info(
                        f"VaR gate blocked entry side={side} sized={notional_pct:.5f}: {reason}"
                    )
                except Exception:
                    pass
            return not bool(allowed)
        except Exception:
            return False  # safe fallback: autoriser

    def _trigger_soft_cap(
        self, current_price: float, reason: str
    ) -> List[Dict]:
        """Soft cap: flat all + set block_until = next UTC midnight."""
        signals = self._flat_all_internal_positions(current_price, reason=reason)
        self.block_until_iso = self._next_utc_midnight_iso()
        if self.notifier is not None:
            try:
                pnl_str = f"{self.daily_pnl_pct*100:+.3f}%"
                self.notifier.warn(
                    f"Daily {reason} hit at PnL={pnl_str}. Block new entries until {self.block_until_iso}."
                )
            except Exception:
                pass
        return signals
    
    def detect_signals(
        self,
        df_ichimoku: pd.DataFrame,
        params: Dict[str, float],
        current_price: float
    ) -> List[Dict]:
        """
        Détecte signaux d'entrée/sortie sur la dernière bougie.
        
        Args:
            df_ichimoku: DataFrame avec Ichimoku calculé
            params: dict avec atr_mult, tp_mult
            current_price: prix actuel (close de la dernière bougie)
        
        Returns:
            Liste de signaux: [
                {"action": "open_long", "entry": ..., "stop": ..., "tp": ..., "size": ...},
                {"action": "close_long", "pos_id": ..., "exit": ..., "reason": ...},
                ...
            ]
        """
        if len(df_ichimoku) == 0:
            return []

        signals = []
        last = df_ichimoku.iloc[-1]
        atr = float(last['ATR']) if pd.notna(last['ATR']) else 0.0

        # P1 — HARD CAP en premier (priorité absolue, kill switch + freeze)
        if self._hard_cap_triggered():
            return self._trigger_hard_cap(current_price)

        # Vérifier si daily_loss dépasse seuil legacy (stop trading pour aujourd'hui)
        if self.daily_loss >= self.daily_loss_threshold:
            return signals

        # Mettre à jour les trailing stops AVANT de tester les sorties
        self._ratchet_trailing_stops(current_price, atr)

        # === SORTIES (TP ou Trailing Stop) ===
        
        # Sorties LONG
        for pos in self.positions_long[:]:
            tp_hit = current_price >= pos.get("tp", np.inf)
            stop_hit = current_price <= pos.get("stop", -np.inf)
            
            if tp_hit or stop_hit:
                signals.append({
                    "action": "close_long",
                    "pos_id": pos["id"],
                    "exit": current_price,
                    "reason": "take_profit" if tp_hit else "trailing_stop"
                })
                self.positions_long.remove(pos)
        
        # Sorties SHORT
        for pos in self.positions_short[:]:
            tp_hit = current_price <= pos.get("tp", -np.inf)
            stop_hit = current_price >= pos.get("stop", np.inf)
            
            if tp_hit or stop_hit:
                signals.append({
                    "action": "close_short",
                    "pos_id": pos["id"],
                    "exit": current_price,
                    "reason": "take_profit" if tp_hit else "trailing_stop"
                })
                self.positions_short.remove(pos)
        
        # P1 — Soft caps APRÈS sorties TP/SL/trailing (les ferme proprement),
        # AVANT toute nouvelle entrée. Si soft cap atteint :
        #   1. Flat positions internes restantes (en plus des TP/SL déjà sortis)
        #   2. Set block_until_iso = next UTC midnight
        #   3. Return immédiat (pas de nouvelle entrée ni rotation opposite)
        if self._soft_loss_cap_triggered():
            signals.extend(self._trigger_soft_cap(current_price, "soft_loss_cap"))
            return signals
        if self._soft_gain_cap_triggered():
            signals.extend(self._trigger_soft_cap(current_price, "soft_gain_cap"))
            return signals

        # P1 — Block check : si bloqué, pas de nouvelle entrée mais sorties OK.
        # Les sorties TP/SL/trailing au-dessus se sont déjà exécutées; on saute
        # juste le bloc d'entrées en dessous.
        if self._is_blocked_now():
            return signals

        # === ENTRÉES (si signal Ichimoku) ===

        # P0 portfolio-aware: scale [0,1] selon risk agrégé multi-symbole déjà utilisé
        # Default = 1.0 (sizing legacy inchangé) si callback None ou échec.
        portfolio_scale = 1.0
        if self.portfolio_scale_fn is not None:
            try:
                _scale = self.portfolio_scale_fn()
                if _scale is not None and np.isfinite(_scale):
                    portfolio_scale = float(max(0.0, min(1.0, _scale)))
            except Exception:
                portfolio_scale = 1.0  # safe fallback

        # P2 anti-martingale: scale [0,1] selon drawdown depuis equity high.
        # Combiné MULTIPLICATIVEMENT avec portfolio_scale → si l'un est 0, sizing = 0.
        dd_scale = 1.0
        if self.drawdown_scale_fn is not None:
            try:
                _dd = self.drawdown_scale_fn()
                if _dd is not None and np.isfinite(_dd):
                    dd_scale = float(max(0.0, min(1.0, _dd)))
            except Exception:
                dd_scale = 1.0  # safe fallback

        sized = 0.01 * portfolio_scale * dd_scale  # 1% × scale composite

        # Signal LONG: bull_cross + close > nuage + pas de SHORT ouverts
        if last.get('signal_long', False) and len(self.positions_short) == 0:
            if len(self.positions_long) < self.max_positions:
                # P3 — VaR95 gate : bloque l'émission si VaR projetée > seuil
                if self._var_gate_blocks("long", sized):
                    pass  # signal d'entrée filtré par le gate, pas d'émission
                else:
                    atr_stop_mult = params.get('atr_mult', 10.0) * 2.0
                    tp_mult = params.get('tp_mult', 20.0)
                    entry = current_price  # Simplifié (en réel: next open + slippage)
                    stop = entry - (atr * atr_stop_mult)
                    tp = entry + (atr * tp_mult)
                    signals.append({
                        "action": "open_long",
                        "entry": entry,
                        "stop": stop,
                        "tp": tp,
                        "size": sized,
                    })

        # Signal SHORT: bear_cross + close < nuage + pas de LONG ouverts
        if last.get('signal_short', False) and len(self.positions_long) == 0:
            if len(self.positions_short) < self.max_positions:
                if self._var_gate_blocks("short", sized):
                    pass  # filtré par le gate
                else:
                    atr_stop_mult = params.get('atr_mult', 10.0) * 2.0
                    tp_mult = params.get('tp_mult', 20.0)
                    entry = current_price
                    stop = entry + (atr * atr_stop_mult)
                    tp = entry - (atr * tp_mult)
                    signals.append({
                        "action": "open_short",
                        "entry": entry,
                        "stop": stop,
                        "tp": tp,
                        "size": sized,
                    })
        
        # Fermer LONG si signal SHORT opposé (et vice-versa)
        if last.get('signal_short', False) and len(self.positions_long) > 0:
            for pos in self.positions_long[:]:
                signals.append({
                    "action": "close_long",
                    "pos_id": pos["id"],
                    "exit": current_price,
                    "reason": "opposite_signal"
                })
                self.positions_long.remove(pos)
        
        if last.get('signal_long', False) and len(self.positions_short) > 0:
            for pos in self.positions_short[:]:
                signals.append({
                    "action": "close_short",
                    "pos_id": pos["id"],
                    "exit": current_price,
                    "reason": "opposite_signal"
                })
                self.positions_short.remove(pos)
        
        return signals
    
    def register_open_position(self, side: str, pos_id: str, entry: float, stop: float, tp: float, size: float):
        """Enregistre une position ouverte."""
        pos = {
            "id": pos_id,
            "entry": float(entry),
            "stop": float(stop),
            "tp": float(tp),
            "size": float(size)
        }
        if side == "long":
            self.positions_long.append(pos)
        elif side == "short":
            self.positions_short.append(pos)
    
    def get_positions_state(self) -> Tuple[List[Dict], List[Dict]]:
        """Retourne (positions_long, positions_short)."""
        return self.positions_long, self.positions_short


if __name__ == "__main__":
    # Test simulation
    engine = SignalEngine()
    
    # Simuler DataFrame Ichimoku
    df = pd.DataFrame({
        'close': [50000],
        'ATR': [400],
        'signal_long': [True],
        'signal_short': [False]
    })
    
    params = {"atr_mult": 11.8, "tp_mult": 20.0}
    signals = engine.detect_signals(df, params, current_price=50000)
    
    print(f"Signaux détectés: {len(signals)}")
    for sig in signals:
        print(sig)

