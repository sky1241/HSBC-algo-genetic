#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Phase Job: mise à jour quotidienne de la phase (1×/jour à 00:05 UTC).

Logique:
1. Charge labels K3 1D stable
2. Détermine phase de la veille (J-1)
3. Assigne cette phase pour trader aujourd'hui (J)
4. Charge paramètres Ichimoku/ATR/TP correspondants
5. Sauvegarde dans state.json
"""
import sys
from pathlib import Path
from datetime import datetime

# Ajouter binance_bot au path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.phase_labeller import PhaseLabeller
from services.params_loader import ParamsLoader
from bot.state_manager import StateManager


def main():
    print("="*70)
    print(f"📅 DAILY PHASE UPDATE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Chemins
    labels_csv = ROOT / "data" / "K3_1d_stable.csv"
    params_json = ROOT / "configs" / "phase_params_K3.json"
    state_file = ROOT / "data" / "state.json"
    
    # Charger modules
    try:
        labeller = PhaseLabeller(str(labels_csv))
        loader = ParamsLoader(str(params_json))
        state_mgr = StateManager(str(state_file))
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        return 1
    
    # Déterminer phase pour aujourd'hui
    today = datetime.now().date()
    phase = labeller.get_phase_for_trading(today)
    params = loader.get_params(phase)
    
    print(f"\n🎯 Phase du jour: {phase}")
    print(f"📊 Paramètres:")
    for k, v in params.items():
        print(f"   {k}: {v}")
    
    # Mise à jour state
    state_mgr.update_phase(phase, params)
    
    print(f"\n✅ State mis à jour: {state_file}")
    print(f"   Date: {today}")
    print(f"   Phase: {phase}")
    print(f"   Daily loss réinitialisé à 0")
    
    print("="*70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

