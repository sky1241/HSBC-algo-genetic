# ALPHA-3 — Regime-Aware LightGBM Backtest

## 1. Contexte

ALPHA-2 a invalide la piste Ichimoku K3 (Sharpe -1.91, deploy=no). Suite a la
litterature MDPI Electronics 15(6) 1334 (2025) "Regime-Aware LightGBM
Walk-Forward Framework" qui rapporte Sharpe portfolio crypto = 1.18
(IC95% [0.53, 1.84]), on tente une approche supervisee combinant un HMM
gaussien 2 etats avec un classifier LightGBM directional.

**Mise en garde** : l'IC95% [0.53, 1.84] du paper original est tres large.
Cela signifie que l'edge mesure est **fragile** — le borne basse 0.53 est a
peine au-dessus du seuil acceptable (0.5), donc une replication peut tres bien
tomber en dessous de zero apres frais reels.

## 2. Methodologie

- **Donnees** : BTC/USDT 2h, periode 2020-01-01 -> 2025-08-01 (19985 bars OOS).
- **HMM** : `GaussianHMM(n_states=2)` fit sur (log_returns, vol_20) — fit
  uniquement sur le train fold, pas de regen sur le test.
- **Features (causaux)** : returns 1/4/12, vol_20, range%, ADX(14), ER(30),
  BBW(20)+squeeze, Hurst(100), encoding cyclique (hour, dow), funding current
  + change, momentum 6/12/24, plus la proba HMM `regime_p`.
- **Label** : sign(close[t+3]/close[t] - 1) en 3 classes {-1, 0, +1}, seuil
  flat = 0.2%. (Lookahead UNIQUEMENT sur le label, conformement aux best
  practices du supervised learning.)
- **Walk-forward** : 28 folds, train expanding, test OOS = 2 mois,
  purge = 3 bars (= horizon du label, anti-leakage Lopez de Prado).
- **Couts** : `cost_model.apply_costs_to_returns` (fees 4 bps taker, slippage
  vol-aware, funding empirique).

## 3. Resultats OOS

### Metriques principales

| Metrique | Regime-LGBM | ALPHA-2 V1 | HODL |
| --- | --- | --- | --- |
| Sharpe (full OOS) | -6.583 | -1.230 | 0.485 |
| Sortino | -6.141 | -0.961 | 0.484 |
| Calmar | -0.954 | -0.378 | 0.159 |
| CAGR | -95.37% | -37.33% | 12.30% |
| MDD | -100.00% | -98.68% | -77.20% |
| Ulcer | 0.9898 | 0.9109 | 0.4256 |
| dd_duration (bars) | 19972 | 19975 | 10188 |

### Walk-forward stability

- **Median fold Sharpe** : -7.566
- IQR : [-8.187, -5.426]
- Folds calcules : 28 / 28

### Significativite statistique

- **DSR(10 trials)** : 0.000  (seuil 0.70 — NOT OK)
- **Hansen SPA p_consistent** : 0.4960  (seuil 0.05 — NOT OK)
- **Hansen SPA p_lower** : 1.0000
- **Hansen SPA p_upper** : 0.4960

### Quality du classifier

- **AUC ROC (direction up vs down)** : 0.532
- **Brier score** : 0.2578

### Top 10 features (gain cumule)

- **regime_p**: 132123
- **range_pct**: 82243
- **vol_20**: 63707
- **hurst_100**: 52347
- **ret_1**: 47272
- **bbw_20**: 45461
- **ret_12**: 45028
- **funding_current**: 43453
- **ret_4**: 42270
- **adx_14**: 41936

## 4. Verdict honnete

**Decision : NO**

Raison : Sharpe OOS -6.58 < 0.5; DSR 0.00 < 0.7; Hansen SPA p=0.496 > 0.05

### Garde-fous appliques

- Sharpe OOS < 0.5 : TRIGGERED
- DSR < 0.7 : TRIGGERED
- Hansen SPA p > 0.05 : TRIGGERED

### Notes critiques

1. Le paper MDPI 2025 (Sharpe 1.18, IC [0.53, 1.84]) a un IC tres large.
   L'edge rapporte est **fragile**. Une replication independante peut tomber
   sous zero apres frais reels.
2. Pas de SOPR ni MVRV utilise (on-chain non integre dans cette pipeline) —
   le paper original les utilisait. Une replication complete devrait les
   rajouter avant tout deploiement.
3. La purge de 3 bars (= label horizon) est minimale; certains auteurs
   (Lopez de Prado) recommandent une purge + embargo > 1 horizon pour
   crypto. A reverifier en sensibilite.
4. Le HMM 2 etats est volontairement simple (paper recommande 2-3). Tester
   3 etats peut etre utile mais ajoute du data-snooping si on optimise.

### Recommandation pour la suite

Si verdict = NO : ne PAS deployer en live. Garder ce backtest comme reference
negative et iterer (a) features on-chain (SOPR/MVRV), (b) horizon plus long
(12-24 bars = 1-2 jours), (c) ensembling avec strategy momentum simple.

Si verdict = YES : forward-test 30 jours testnet AVANT live, monitor le hit
rate par fold et le DSR mensuel.

## 5. Fichiers produits

- `equity_curve.png` — courbes equity LGBM vs V1 vs HODL
- `feature_importance.png` — top 10 features (gain)
- `recommendation.json` — verdict machine-readable
- `fold_sharpes.csv` — distribution Sharpe par fold OOS
- `RAPPORT.md` — ce document
