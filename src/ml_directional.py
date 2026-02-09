#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Directional Predictor - LightGBM pour prediction de direction BTC.

Remplace le NHHM casse par un classifier supervise robuste.

Target: y = 1 si forward_return_24h > 0, sinon 0
Features: momentum, vol_ratio, halving_phase, rsi, Fourier
Output: Labels (1=LONG, 0=CASH, -1=SHORT) bases sur P(bull)

Walk-Forward: Train sur annees passees, predict annee suivante.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

# Import features builder from regime_nhhm
from src.regime_nhhm import build_nhhm_features


@dataclass
class MLDirectionalConfig:
    """Configuration pour le modele ML directionnel."""

    # Target
    forward_horizon: int = 12  # 12 bars H2 = 24h

    # Thresholds pour labels
    long_threshold: float = 0.55   # P(up) > 0.55 => LONG
    short_threshold: float = 0.45  # P(up) < 0.45 => SHORT

    # LightGBM params
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    })
    num_boost_round: int = 200
    early_stopping_rounds: int = 30

    # Features
    feature_cols: List[str] = field(default_factory=lambda: [
        'momentum_6', 'momentum_12', 'momentum_24',
        'vol_ratio',
        'rsi_centered',
        'dist_ma20', 'dist_ma50',
        'halving_phase', 'halving_direction',
        'P1_period', 'LFP_ratio',
    ])

    # Walk-forward
    min_train_years: int = 2  # Minimum 2 ans de training


@dataclass
class MLDirectionalResult:
    """Resultat de prediction ML."""
    p_up: pd.Series            # P(up) pour chaque bar (avec index)
    labels: pd.Series          # 1=LONG, 0=CASH, -1=SHORT (avec index)
    hit_rate: float            # Taux de bonnes predictions OOS
    feature_importance: Dict[str, float]
    train_years: List[int]
    test_years: List[int]
    valid_index: pd.DatetimeIndex  # Index des bars valides


class MLDirectional:
    """
    Predicteur de direction base sur LightGBM.

    Usage:
        ml = MLDirectional(config)
        result = ml.walk_forward_predict(df)
        labels_df = ml.to_labels_csv(df.index, result)
    """

    def __init__(self, config: Optional[MLDirectionalConfig] = None):
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm required. Install with: pip install lightgbm")

        self.config = config or MLDirectionalConfig()
        self.models: Dict[int, lgb.Booster] = {}  # year -> model
        self._feature_importance: Dict[str, float] = {}

    def _compute_target(self, close: pd.Series) -> pd.Series:
        """
        Compute binary target: 1 si prix monte dans forward_horizon bars.

        ATTENTION: shift(-N) = lookahead. On l'utilise UNIQUEMENT pour le training,
        jamais pour les features.
        """
        fwd_return = close.shift(-self.config.forward_horizon) / close - 1
        target = (fwd_return > 0).astype(int)
        return target

    def _prepare_features(
        self,
        df: pd.DataFrame,
        include_funding: bool = False
    ) -> pd.DataFrame:
        """
        Prepare features pour le model.

        Reutilise build_nhhm_features() et selectionne les colonnes pertinentes.
        """
        # Build all features
        df_feat = build_nhhm_features(
            df,
            include_funding=include_funding,
            include_fourier=True,
            include_halving=True,
        )

        # Select only configured feature columns
        available_cols = [c for c in self.config.feature_cols if c in df_feat.columns]
        missing_cols = [c for c in self.config.feature_cols if c not in df_feat.columns]

        if missing_cols:
            warnings.warn(f"Features manquantes (ignores): {missing_cols}")

        if not available_cols:
            raise ValueError("Aucune feature disponible!")

        return df_feat[available_cols]

    def _get_year(self, idx: pd.DatetimeIndex) -> pd.Series:
        """Extract year from DatetimeIndex."""
        return pd.Series(idx.year, index=idx)

    def walk_forward_predict(
        self,
        df: pd.DataFrame,
        start_year: int = 2013,
        end_year: Optional[int] = None,
        verbose: bool = True
    ) -> MLDirectionalResult:
        """
        Walk-forward training et prediction.

        Pour chaque annee T:
        - Train sur annees [start_year, T-1]
        - Predict sur annee T

        Retourne predictions combinees pour toutes les annees OOS.
        """
        if end_year is None:
            end_year = df.index.max().year

        if verbose:
            print(f"Walk-Forward ML: {start_year} -> {end_year}")
            print(f"Features: {self.config.feature_cols}")

        # Prepare features et target
        X = self._prepare_features(df)
        y = self._compute_target(df['close'])

        # Align X and y (drop NaN)
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        if verbose:
            print(f"Donnees valides: {len(X)} bars ({len(X)/12/365:.1f} ans)")

        # Years
        years = self._get_year(X.index)
        all_years = sorted(years.unique())

        # Results storage
        all_proba = pd.Series(index=X.index, dtype=float)
        all_proba[:] = np.nan

        train_years_used = []
        test_years_used = []
        hit_counts = []
        total_counts = []

        # Walk-forward loop
        for test_year in range(start_year, end_year + 1):
            # Train years: all years before test_year with min_train_years
            train_mask = years < test_year
            test_mask = years == test_year

            n_train = train_mask.sum()
            n_test = test_mask.sum()

            if n_train < self.config.min_train_years * 12 * 365:
                if verbose:
                    print(f"  {test_year}: Skip (train insuffisant: {n_train} bars)")
                continue

            if n_test == 0:
                if verbose:
                    print(f"  {test_year}: Skip (pas de donnees test)")
                continue

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            # Train model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = lgb.train(
                    self.config.lgb_params,
                    train_data,
                    num_boost_round=self.config.num_boost_round,
                    valid_sets=[valid_data],
                    callbacks=[
                        lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )

            # Predict probabilities
            proba = model.predict(X_test)
            all_proba[test_mask] = proba

            # Calculate hit rate for this year
            predictions = (proba > 0.5).astype(int)
            hits = (predictions == y_test).sum()
            hit_rate_year = hits / len(y_test)

            hit_counts.append(hits)
            total_counts.append(len(y_test))
            train_years_used.append(list(range(all_years[0], test_year)))
            test_years_used.append(test_year)

            # Store model and feature importance
            self.models[test_year] = model
            for feat, imp in zip(X.columns, model.feature_importance('gain')):
                self._feature_importance[feat] = self._feature_importance.get(feat, 0) + imp

            if verbose:
                print(f"  {test_year}: Train={n_train}, Test={n_test}, Hit={hit_rate_year:.1%}")

        # Calculate overall hit rate
        overall_hit_rate = sum(hit_counts) / sum(total_counts) if total_counts else 0.0

        if verbose:
            print(f"\nHit Rate Global OOS: {overall_hit_rate:.2%}")

        # Generate labels from probabilities
        labels_arr = self._proba_to_labels(all_proba.values)
        labels_series = pd.Series(labels_arr, index=X.index)

        # Normalize feature importance
        total_imp = sum(self._feature_importance.values())
        if total_imp > 0:
            self._feature_importance = {
                k: v / total_imp for k, v in self._feature_importance.items()
            }

        return MLDirectionalResult(
            p_up=all_proba,
            labels=labels_series,
            hit_rate=overall_hit_rate,
            feature_importance=self._feature_importance,
            train_years=train_years_used,
            test_years=test_years_used,
            valid_index=X.index,
        )

    def _proba_to_labels(self, p_up: np.ndarray) -> np.ndarray:
        """
        Convertir probabilites en labels trading.

        - P(up) > long_threshold => LONG (1)
        - P(up) < short_threshold => SHORT (-1)
        - Sinon => CASH (0)
        """
        labels = np.zeros(len(p_up), dtype=int)

        # LONG si confiance haute
        labels[p_up > self.config.long_threshold] = 1

        # SHORT si confiance basse (donc haute proba down)
        labels[p_up < self.config.short_threshold] = -1

        # NaN => CASH
        labels[np.isnan(p_up)] = 0

        return labels

    def to_labels_df(
        self,
        index: pd.DatetimeIndex,
        result: MLDirectionalResult
    ) -> pd.DataFrame:
        """
        Convertir resultat en DataFrame format labels CSV.

        Format attendu par WFA:
        - timestamp: DatetimeIndex
        - label: 1 (LONG), 0 (CASH), -1 (SHORT)
        """
        # Create full labels array initialized to CASH (0)
        labels_full = pd.Series(0, index=index, dtype=int)

        # Fill in predictions where we have them
        # result.labels has valid_index as its index
        labels_full.loc[result.valid_index] = result.labels.values

        df = pd.DataFrame({
            'timestamp': index,
            'label': labels_full.values,
        })

        # Add regime name for compatibility
        regime_map = {1: 'bull', 0: 'neutral', -1: 'bear'}
        df['regime'] = df['label'].map(regime_map)

        return df


def generate_ml_labels(
    data_path: str = "data/BTC_FUSED_2h.csv",
    output_path: str = "data/ML_directional.csv",
    start_year: int = 2013,
    verbose: bool = True
) -> Tuple[pd.DataFrame, MLDirectionalResult]:
    """
    Fonction principale pour generer les labels ML.

    Usage:
        labels_df, result = generate_ml_labels()
        print(f"Hit rate: {result.hit_rate:.2%}")
    """
    if verbose:
        print("=" * 60)
        print("GENERATION LABELS ML DIRECTIONAL")
        print("=" * 60)

    # Load data
    if verbose:
        print(f"\nChargement: {data_path}")

    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)

    if verbose:
        print(f"Donnees: {df.index.min()} -> {df.index.max()}")
        print(f"Bars: {len(df)}")

    # Create predictor
    config = MLDirectionalConfig()
    ml = MLDirectional(config)

    # Walk-forward predict
    if verbose:
        print("\n" + "-" * 40)
    result = ml.walk_forward_predict(df, start_year=start_year, verbose=verbose)

    # Convert to labels DataFrame
    labels_df = ml.to_labels_df(df.index, result)

    # Save
    labels_df.to_csv(output_path, index=False)
    if verbose:
        print(f"\nLabels sauvegardes: {output_path}")
        print(f"Total bars: {len(labels_df)}")
        print(f"LONG: {(labels_df['label'] == 1).sum()}")
        print(f"CASH: {(labels_df['label'] == 0).sum()}")
        print(f"SHORT: {(labels_df['label'] == -1).sum()}")

    # Feature importance
    if verbose and result.feature_importance:
        print("\nFeature Importance:")
        for feat, imp in sorted(result.feature_importance.items(), key=lambda x: -x[1])[:5]:
            print(f"  {feat}: {imp:.3f}")

    # CHECKPOINT: Hit rate
    if verbose:
        print("\n" + "=" * 60)
        print("CHECKPOINT AUTO-CONTROLE")
        print("=" * 60)
        print(f"Hit Rate OOS: {result.hit_rate:.2%}")
        if result.hit_rate > 0.53:
            print("  [OK] Hit rate > 53%")
        else:
            print("  [WARN] Hit rate < 53% - Objectif non atteint")

    return labels_df, result


if __name__ == "__main__":
    # Test direct
    labels_df, result = generate_ml_labels(verbose=True)
