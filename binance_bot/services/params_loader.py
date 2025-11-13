#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Params Loader: charge paramètres Ichimoku/ATR/TP par phase."""
import json
from pathlib import Path
from typing import Dict


class ParamsLoader:
    """Charge et retourne les paramètres optimisés par phase depuis JSON."""
    
    def __init__(self, json_path: str):
        """
        Args:
            json_path: chemin vers phase_params_K3.json
        """
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"Paramètres introuvables: {json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convertir clés en int
        self.params_by_phase = {int(k): v for k, v in data.items()}
    
    def get_params(self, phase: int) -> Dict[str, float]:
        """
        Retourne les paramètres pour une phase donnée.
        
        Args:
            phase: numéro de phase (0, 1 ou 2 pour K3)
        
        Returns:
            dict avec clés: tenkan, kijun, senkou_b, shift, atr_mult, tp_mult
        """
        if phase not in self.params_by_phase:
            raise ValueError(f"Phase {phase} introuvable dans {list(self.params_by_phase.keys())}")
        
        return self.params_by_phase[phase]
    
    def list_phases(self):
        """Liste toutes les phases disponibles."""
        return sorted(self.params_by_phase.keys())


if __name__ == "__main__":
    # Test rapide
    loader = ParamsLoader("../configs/phase_params_K3.json")
    print(f"Phases disponibles: {loader.list_phases()}")
    for phase in loader.list_phases():
        params = loader.get_params(phase)
        print(f"\nPhase {phase}: {params}")

