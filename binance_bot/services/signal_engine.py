#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Signal Engine: détecte signaux EXACT comme backtest_long_short.

P0 (2026-04-28): support `portfolio_state` pour moduler le `size` du signal
d'entrée selon le risk agrégé multi-symbole déjà utilisé. Évite l'over-bet
quand BTC+ETH+SOL sont co-corrélés.
"""
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


class SignalEngine:
    """
    Reproduit la logique EXACTE de backtest_long_short pour détecter signaux.

    État interne maintenu: positions_long, positions_short (max 3 chacun).

    P0: optionnellement un `portfolio_scale_fn` callable est invoqué juste
    avant d'émettre un signal `open_long`/`open_short` pour moduler `size`
    selon le risk agrégé déjà utilisé multi-symbole. Si non fourni → comportement
    legacy avec size=position_size_pct constant.
    """

    def __init__(
        self,
        max_positions: int = 3,
        daily_loss_threshold: float = 0.10,
        atr_trailing_mult: float = 2.0,
        portfolio_scale_fn: Optional[Callable[[], float]] = None,
    ):
        """
        Args:
            max_positions: nombre max de positions par côté (3 dans backtest)
            daily_loss_threshold: seuil perte journalière (10% dans backtest)
            atr_trailing_mult: multiplicateur ATR pour le trailing stop (cliquet)
            portfolio_scale_fn: P0 — callable sans args qui retourne un facteur
                [0, 1] basé sur le risk agrégé multi-symbole. Si None ou retourne
                None, le sizing reste 1.0 (legacy).
        """
        self.max_positions = max_positions
        self.daily_loss_threshold = daily_loss_threshold
        self.atr_trailing_mult = float(atr_trailing_mult)
        self.portfolio_scale_fn = portfolio_scale_fn
        self.positions_long: List[Dict] = []
        self.positions_short: List[Dict] = []
        self.daily_loss = 0.0

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
    
    def load_state(self, positions_long: List[Dict], positions_short: List[Dict], daily_loss: float):
        """Charge positions depuis state_manager."""
        self.positions_long = positions_long
        self.positions_short = positions_short
        self.daily_loss = daily_loss
    
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
        
        # Vérifier si daily_loss dépasse seuil (stop trading pour aujourd'hui)
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

        sized = 0.01 * portfolio_scale  # 1% × scale portfolio-aware

        # Signal LONG: bull_cross + close > nuage + pas de SHORT ouverts
        if last.get('signal_long', False) and len(self.positions_short) == 0:
            if len(self.positions_long) < self.max_positions:
                # Calculer stop et TP
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
                    "size": sized,  # P0: 1% × portfolio_scale
                })

        # Signal SHORT: bear_cross + close < nuage + pas de LONG ouverts
        if last.get('signal_short', False) and len(self.positions_long) == 0:
            if len(self.positions_short) < self.max_positions:
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
                    "size": sized,  # P0: 1% × portfolio_scale
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

