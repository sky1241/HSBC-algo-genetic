#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase Labeller: détermine la phase du jour depuis labels K3 1D stable."""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


class PhaseLabeller:
    """Charge labels K3 1D stable et retourne phase pour une date donnée."""
    
    def __init__(self, csv_path: str):
        """
        Args:
            csv_path: chemin vers K3_1d_stable.csv
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Labels introuvables: {csv_path}")
        
        df = pd.read_csv(self.csv_path, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['date'] = df['timestamp'].dt.date
        
        # Grouper par jour et prendre label majoritaire (cohérent avec downsample)
        self.labels_daily = df.groupby('date')['label'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
    
    def get_phase_for_trading(self, trading_date: datetime.date) -> int:
        """
        Retourne la phase à utiliser pour trader le jour `trading_date`.
        
        Logique 1D stable: utilise le label de la veille (J-1) pour trader le jour J.
        Pas de lookahead: on décide avec le passé.
        
        Args:
            trading_date: date pour laquelle on veut trader
        
        Returns:
            phase (int): 0, 1 ou 2 pour K3
        """
        # Utiliser phase du jour précédent (J-1) pour trader J
        label_date = trading_date - timedelta(days=1)
        
        try:
            phase = int(self.labels_daily.loc[label_date])
        except KeyError:
            # Si date introuvable (weekend, jour férié), prendre label le plus récent disponible
            available_dates = self.labels_daily.index
            recent = available_dates[available_dates <= label_date]
            if len(recent) > 0:
                phase = int(self.labels_daily.loc[recent[-1]])
            else:
                # Fallback: première phase disponible
                phase = int(self.labels_daily.iloc[0])
        
        return phase
    
    def get_last_known_phase(self) -> int:
        """Retourne la dernière phase connue (utile pour initialisation)."""
        return int(self.labels_daily.iloc[-1])


if __name__ == "__main__":
    # Test rapide
    labeller = PhaseLabeller("../data/K3_1d_stable.csv")
    today = datetime.now().date()
    phase = labeller.get_phase_for_trading(today)
    print(f"Phase pour trading aujourd'hui ({today}): {phase}")

