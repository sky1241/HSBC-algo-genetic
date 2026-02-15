#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcule les probabilités de transition empiriques à partir des labels HMM.

Sortie: matrice de transition P(phase_j | phase_i) et probabilités de changement.

Usage:
    python scripts/compute_transition_proba.py --k 5
    python scripts/compute_transition_proba.py --k 3 --labels-csv path/to/labels.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Compute HMM transition probabilities")
    p.add_argument("--k", type=int, default=5, help="K value (3, 5, 8)")
    p.add_argument("--labels-csv", type=str, default=None,
                   help="Path to labels CSV (default: outputs/fourier/labels_frozen/BTC_FUSED_2h/K{k}_1d_stable.csv)")
    return p.parse_args()


def compute_transition_matrix(labels: pd.Series) -> pd.DataFrame:
    """
    Calcule la matrice de transition empirique.

    Returns:
        DataFrame avec P(to_phase | from_phase)
    """
    # Obtenir les phases uniques
    phases = sorted(labels.dropna().unique())
    n_phases = len(phases)

    # Matrice de comptage
    counts = np.zeros((n_phases, n_phases))

    # Compter les transitions
    prev_label = None
    for label in labels:
        if pd.isna(label):
            prev_label = None
            continue
        if prev_label is not None:
            i = phases.index(prev_label)
            j = phases.index(label)
            counts[i, j] += 1
        prev_label = label

    # Normaliser en probabilités
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Eviter division par zero
    probs = counts / row_sums

    # Créer DataFrame
    df = pd.DataFrame(probs, index=[f"from_{int(p)}" for p in phases],
                      columns=[f"to_{int(p)}" for p in phases])

    return df


def add_transition_proba_to_labels(df: pd.DataFrame, trans_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les colonnes de probabilité de transition aux labels.

    Colonnes ajoutées:
    - proba_stay: P(rester dans la même phase)
    - proba_change: P(changer de phase) = 1 - proba_stay
    - next_phase_likely: phase la plus probable au step suivant
    - proba_next_phase: proba de la phase la plus probable
    """
    df = df.copy()

    phases = sorted(df['label'].dropna().unique())
    n_phases = len(phases)

    # Extraire la matrice numpy
    trans_np = trans_matrix.values

    # Calculer les probas pour chaque ligne
    proba_stay = []
    proba_change = []
    next_phase_likely = []
    proba_next_phase = []

    for label in df['label']:
        if pd.isna(label):
            proba_stay.append(np.nan)
            proba_change.append(np.nan)
            next_phase_likely.append(np.nan)
            proba_next_phase.append(np.nan)
        else:
            i = phases.index(label)
            p_stay = trans_np[i, i]
            p_change = 1.0 - p_stay

            # Phase la plus probable
            j_max = np.argmax(trans_np[i])
            p_next = trans_np[i, j_max]

            proba_stay.append(p_stay)
            proba_change.append(p_change)
            next_phase_likely.append(phases[j_max])
            proba_next_phase.append(p_next)

    df['proba_stay'] = proba_stay
    df['proba_change'] = proba_change
    df['next_phase_likely'] = next_phase_likely
    df['proba_next_phase'] = proba_next_phase

    return df


def main():
    args = parse_args()

    # Chemins
    if args.labels_csv:
        labels_path = Path(args.labels_csv)
    else:
        labels_path = Path(f"outputs/fourier/labels_frozen/BTC_FUSED_2h/K{args.k}_1d_stable.csv")

    if not labels_path.exists():
        print(f"Fichier introuvable: {labels_path}")
        return 1

    print(f"Chargement labels: {labels_path}")
    df = pd.read_csv(labels_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Lignes: {len(df):,}")
    print(f"Phases uniques: {sorted(df['label'].dropna().unique())}")

    # Calculer matrice de transition
    print("\nCalcul matrice de transition...")
    trans_matrix = compute_transition_matrix(df['label'])

    print("\nMatrice de transition P(to | from):")
    print(trans_matrix.round(3).to_string())

    # Stats
    print("\nProbabilités de RESTER dans la même phase:")
    for i, phase in enumerate(sorted(df['label'].dropna().unique())):
        p_stay = trans_matrix.iloc[i, i]
        print(f"  Phase {int(phase)}: {p_stay:.1%}")

    # Sauvegarder matrice
    out_dir = labels_path.parent
    trans_path = out_dir / f"K{args.k}_transition_matrix.csv"
    trans_matrix.to_csv(trans_path)
    print(f"\nMatrice sauvegardée: {trans_path}")

    # Ajouter probas aux labels
    print("\nAjout des probabilités de transition aux labels...")
    df_enriched = add_transition_proba_to_labels(df, trans_matrix)

    # Sauvegarder labels enrichis
    enriched_path = out_dir / f"K{args.k}_1d_stable_with_proba.csv"
    df_enriched.to_csv(enriched_path, index=False)
    print(f"Labels enrichis sauvegardés: {enriched_path}")

    # Stats finales
    print("\nStats probabilités de changement:")
    print(f"  Moyenne: {df_enriched['proba_change'].mean():.1%}")
    print(f"  Médiane: {df_enriched['proba_change'].median():.1%}")
    print(f"  Max: {df_enriched['proba_change'].max():.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
