#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""State Manager: gère l'état du bot (positions, equity, phase courante)."""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import os


class StateManager:
    """Gère état persistant du bot dans state.json."""
    
    def __init__(self, state_file: str = "data/state.json"):
        """
        Args:
            state_file: chemin vers state.json
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_initialize()
    
    def _load_or_initialize(self) -> Dict[str, Any]:
        """Charge state.json ou initialise si inexistant."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                return state
            except Exception as e:
                print(f"⚠️ Erreur lecture state.json: {e}. Initialisation nouveau state.")
        
        # État initial
        return {
            "date": datetime.now().date().isoformat(),
            "phase_today": None,
            "params_today": None,
            "positions_long": [],
            "positions_short": [],
            "equity": 1.0,
            "initial_capital_usdt": 1000.0,
            "max_drawdown": 0.0,
            "daily_loss": 0.0,
            "total_trades": 0,
            "last_update": datetime.now().isoformat()
        }
    
    def save(self):
        """Sauvegarde atomic du state (tmp + rename)."""
        self.state["last_update"] = datetime.now().isoformat()
        tmp = str(self.state_file) + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur du state."""
        return self.state.get(key, default)
    
    def set(self, key: str, value: Any):
        """Définit une valeur dans le state."""
        self.state[key] = value
    
    def update_phase(self, phase: int, params: Dict[str, float]):
        """Mise à jour phase quotidienne (appelé par daily_phase_job)."""
        self.state["date"] = datetime.now().date().isoformat()
        self.state["phase_today"] = int(phase)
        self.state["params_today"] = params
        self.state["daily_loss"] = 0.0  # Reset daily loss
        self.save()
    
    def add_position(self, side: str, entry: float, stop: float, tp: float, size: float):
        """Ajoute une position (long ou short)."""
        pos = {
            "id": f"{side}_{datetime.now().timestamp()}",
            "entry": float(entry),
            "stop": float(stop),
            "tp": float(tp),
            "size": float(size),
            "opened_at": datetime.now().isoformat()
        }
        if side == "long":
            self.state["positions_long"].append(pos)
        elif side == "short":
            self.state["positions_short"].append(pos)
        self.save()
    
    def remove_position(self, side: str, pos_id: str):
        """Supprime une position fermée."""
        if side == "long":
            self.state["positions_long"] = [p for p in self.state["positions_long"] if p["id"] != pos_id]
        elif side == "short":
            self.state["positions_short"] = [p for p in self.state["positions_short"] if p["id"] != pos_id]
        self.save()
    
    def get_positions(self, side: str) -> List[Dict]:
        """Retourne positions ouvertes (long ou short)."""
        if side == "long":
            return self.state.get("positions_long", [])
        elif side == "short":
            return self.state.get("positions_short", [])
        return []
    
    def update_equity(self, new_equity: float):
        """Met à jour equity et max drawdown."""
        self.state["equity"] = float(new_equity)
        dd = 1.0 - new_equity
        if dd > self.state.get("max_drawdown", 0.0):
            self.state["max_drawdown"] = float(dd)
        self.save()
    
    def add_daily_loss(self, loss: float):
        """Accumule perte journalière (pour limiter re-entry)."""
        self.state["daily_loss"] = self.state.get("daily_loss", 0.0) + float(loss)
        self.save()


if __name__ == "__main__":
    # Test rapide
    sm = StateManager("../data/state_test.json")
    print(f"État initial: equity={sm.get('equity')}, phase={sm.get('phase_today')}")
    
    sm.update_phase(phase=1, params={"tenkan": 29, "atr_mult": 19.3})
    print(f"Après update phase: {sm.state}")
    
    sm.add_position("long", entry=50000, stop=48000, tp=55000, size=0.01)
    print(f"Positions LONG: {sm.get_positions('long')}")

