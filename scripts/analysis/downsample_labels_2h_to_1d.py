#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downsample labels 2h vers stabilité 1D pour stratégie hybride.

Principe:
- Les labels K3/K5/K8 existent sur grid 2h (12 barres/jour)
- On calcule le label MAJORITAIRE de chaque jour J
- On applique ce label à TOUTES les barres 2h du jour J+1
- Résultat: phases changent max 1×/jour au lieu de 12×/jour

No Lookahead:
- Jour J 23:59: calcul label majoritaire du jour J (passé)
- Jour J+1: application sur toutes barres 2h (futur)
- Conforme WFA: utilise passé pour décider futur

Usage:
    python scripts/downsample_labels_2h_to_1d.py --k 3
    python scripts/downsample_labels_2h_to_1d.py --k 5 --method rolling

Arguments:
    --k: K value (3, 5, ou 8)
    --method: 'daily' (défaut) ou 'rolling' (24h glissantes)
    --input-dir: Dossier labels source (défaut: outputs/fourier/labels_frozen/BTC_FUSED_2h)
    --output-suffix: Suffixe fichier sortie (défaut: _1d_stable)
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(description="Downsample 2h labels to daily stability")
    p.add_argument("--k", type=int, required=True, choices=[2, 3, 4, 5, 8, 10], help="K value")
    p.add_argument("--method", choices=["daily", "rolling"], default="daily",
                   help="daily=label majoritaire jour J pour J+1, rolling=24h glissantes")
    p.add_argument("--input-dir", default="outputs/fourier/labels_frozen/BTC_FUSED_2h",
                   help="Dossier contenant les labels K originaux")
    p.add_argument("--output-suffix", default="_1d_stable",
                   help="Suffixe ajouté au nom du fichier de sortie")
    return p.parse_args()


def majority_label(labels):
    """Retourne le label majoritaire dans une série."""
    if len(labels) == 0:
        return np.nan
    # Exclure NaN
    valid = labels[~pd.isna(labels)]
    if len(valid) == 0:
        return np.nan
    # Counter pour trouver le plus fréquent
    counts = Counter(valid)
    most_common = counts.most_common(1)[0][0]
    return most_common


def downsample_daily(df):
    """
    Méthode daily: label majoritaire du jour J appliqué au jour J+1.
    
    No lookahead:
    - À la fin du jour J, on connaît les 12 labels 2h du jour J
    - On calcule le majoritaire
    - On applique ce label à TOUTES les barres 2h du jour J+1
    """
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    
    # Calculer label majoritaire par jour
    daily_majority = df.groupby('date')['label'].apply(majority_label).reset_index()
    daily_majority.columns = ['date', 'label_daily']
    
    # Shifter de 1 jour (jour J détermine jour J+1)
    daily_majority['date'] = pd.to_datetime(daily_majority['date']) + pd.Timedelta(days=1)
    daily_majority['date'] = daily_majority['date'].dt.date
    
    # Merger avec les données 2h
    df = df.merge(daily_majority, on='date', how='left')
    
    # Forward fill pour les premiers jours sans historique
    df['label_stable'] = df['label_daily'].fillna(method='ffill')
    
    # Si toujours NaN au début, prendre le label original
    df['label_stable'] = df['label_stable'].fillna(df['label'])
    
    return df[['timestamp', 'label_stable']].rename(columns={'label_stable': 'label'})


def downsample_rolling(df):
    """
    Méthode rolling: label majoritaire sur fenêtre glissante 24h (12 barres).
    
    Plus lisse que daily, mais toujours pas de lookahead:
    - À chaque barre 2h, on regarde les 12 barres PRÉCÉDENTES (24h passées)
    - On prend le label majoritaire
    """
    df = df.copy()
    
    # Rolling window de 12 barres (24h)
    window_size = 12
    
    def rolling_majority(series):
        return majority_label(series.values)
    
    df['label_stable'] = df['label'].rolling(window=window_size, min_periods=1).apply(
        rolling_majority, raw=False
    )
    
    return df[['timestamp', 'label_stable']].rename(columns={'label_stable': 'label'})


def main():
    args = parse_args()
    
    # Chemins
    input_dir = Path(args.input_dir)
    input_file = input_dir / f"K{args.k}.csv"
    output_file = input_dir / f"K{args.k}{args.output_suffix}.csv"
    
    if not input_file.exists():
        print(f"❌ Fichier introuvable: {input_file}")
        return 1
    
    print(f"📊 Downsampling K{args.k} labels (méthode: {args.method})")
    print(f"   Input:  {input_file}")
    print(f"   Output: {output_file}")
    
    # Charger labels 2h
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\n📈 Stats originales (2h):")
    print(f"   Lignes: {len(df):,}")
    print(f"   Période: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   Labels uniques: {df['label'].nunique()}")
    
    # Calculer changements de phase
    changes_2h = (df['label'] != df['label'].shift(1)).sum()
    print(f"   Changements de phase: {changes_2h:,} ({changes_2h/len(df)*100:.2f}%)")
    
    # Downsampling
    if args.method == "daily":
        df_stable = downsample_daily(df)
    else:
        df_stable = downsample_rolling(df)
    
    # Stats après downsampling
    changes_stable = (df_stable['label'] != df_stable['label'].shift(1)).sum()
    reduction = (1 - changes_stable/changes_2h) * 100 if changes_2h > 0 else 0
    
    print(f"\n✅ Stats après downsampling 1D stable:")
    print(f"   Lignes: {len(df_stable):,} (inchangé)")
    print(f"   Changements de phase: {changes_stable:,} ({changes_stable/len(df_stable)*100:.2f}%)")
    print(f"   Réduction changements: {reduction:.1f}%")
    print(f"   Switches/jour moyen: {changes_stable / (len(df_stable)/12):.2f}")
    
    # Distribution des phases
    print(f"\n📊 Distribution phases (1D stable):")
    for phase, count in df_stable['label'].value_counts().sort_index().items():
        pct = count / len(df_stable) * 100
        print(f"   Phase {int(phase)}: {count:,} barres ({pct:.1f}%)")
    
    # Sauvegarder
    df_stable.to_csv(output_file, index=False)
    print(f"\n💾 Sauvegardé: {output_file}")
    
    # Exemple de comparaison sur une semaine
    print(f"\n🔍 Exemple comparaison (première semaine):")
    sample_period = df['timestamp'] < (df['timestamp'].min() + pd.Timedelta(days=7))
    df_sample = df[sample_period].copy()
    df_stable_sample = df_stable[sample_period].copy()
    
    df_sample['date'] = df_sample['timestamp'].dt.date
    df_stable_sample['date'] = df_stable_sample['timestamp'].dt.date
    
    print("\n   Original 2h (changements fréquents):")
    for date in df_sample['date'].unique()[:3]:
        day_labels = df_sample[df_sample['date'] == date]['label'].values
        print(f"   {date}: {day_labels}")
    
    print("\n   1D stable (label constant par jour):")
    for date in df_stable_sample['date'].unique()[:3]:
        day_labels = df_stable_sample[df_stable_sample['date'] == date]['label'].values
        print(f"   {date}: {day_labels}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

