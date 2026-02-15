# PROMPT DE PASSATION - Projet HSBC-algo-genetic

**A donner au prochain Claude pour reprendre le travail.**

---

## CONTEXTE RAPIDE

Tu reprends un projet de trading algorithmique sur BTC. L'objectif est de generer de l'**alpha** (rendements superieurs au marche) via une strategie Ichimoku+ATR optimisee par WFA (Walk-Forward Analysis).

**PROBLEME ACTUEL:** Le systeme survit (Sharpe 0.99) mais ne genere pas d'alpha. On a merde sur l'implementation du NHHM qui devait predire la direction.

---

## PROMPT A COPIER-COLLER

```
Je reprends le projet HSBC-algo-genetic. Voici le contexte:

## ETAT ACTUEL
- Strategie: Ichimoku + ATR sur BTC (timeframe H2)
- Optimisation: WFA annuel avec Optuna (50-300 trials)
- Labels actuels: CYCLE_cash_bear.csv (filtre halving 4 ans)
- Resultat: Sharpe OOS = 0.99, Equity x1.217 sur 14 ans

## LE PROBLEME
Le NHHM (Non-Homogeneous HMM) devait predire la direction (P(bull), P(bear))
mais l'implementation a echoue:
- statsmodels.MarkovRegression ne converge pas
- Fallback silencieux vers mode degrade
- On a abandonne et utilise un simple filtre CYCLE

Resultat: PAS D'ALPHA, juste de la survie.

## CE QUE JE VEUX
Implementer un predicteur de direction qui fonctionne VRAIMENT.

Option recommandee: ML supervise (LightGBM/XGBoost)
- Target: y = 1 si forward_return_24h > 0, else 0
- Features: momentum, vol_ratio, halving_phase, rsi, funding_rate, Fourier
- Output: Labels (1=LONG, 0=CASH, -1=SHORT) bases sur P(bull)

## FICHIERS CLES
- src/regime_nhhm.py - Implementation cassee (A REFAIRE ou remplacer)
- scripts/run_scheduler_wfa_phase.py - WFA principal (--labels-csv)
- data/CYCLE_cash_bear.csv - Labels actuels (baseline Sharpe 0.99)
- docs/POST_MORTEM_NHHM_ECHEC.md - Analyse de l'echec

## CRITERES DE SUCCES
- Hit rate prediction > 53%
- Sharpe OOS > 1.5 (vs 0.99 actuel)
- Generer data/ML_directional.csv compatible avec WFA

## CONTRAINTES
- Donnees BTC depuis 2011 (H2 timeframe)
- Pas de lookahead bias (features t pour predire t+1)
- Walk-forward: train sur annees passees, test sur annee suivante

Commence par lire docs/POST_MORTEM_NHHM_ECHEC.md puis propose un plan.
```

---

## FICHIERS A LIRE EN PRIORITE

1. `docs/POST_MORTEM_NHHM_ECHEC.md` - Comprendre l'echec
2. `src/regime_nhhm.py` - Voir ce qui a merde (lignes 280-327)
3. `scripts/run_scheduler_wfa_phase.py` - Comprendre le WFA
4. `data/CYCLE_cash_bear.csv` - Format des labels attendu

---

## ARCHITECTURE DU PROJET

```
HSBC-algo-genetic/
├── src/
│   ├── regime_nhhm.py      # NHHM casse - A REFAIRE
│   ├── regime_hmm.py       # HMM volatilite (fonctionne)
│   ├── features_fourier.py # Features spectrales
│   ├── funding_rate.py     # Funding rate Binance
│   └── ...
├── scripts/
│   ├── run_scheduler_wfa_phase.py  # WFA principal
│   ├── generate_cycle_labels.py    # Genere labels CYCLE
│   ├── launch_30_seeds_cycle.ps1   # Lance 30 seeds WFA
│   └── ...
├── data/
│   ├── BTC_FUSED_2h.parquet       # Donnees OHLCV
│   ├── CYCLE_cash_bear.csv        # Labels actuels
│   └── ...
├── outputs/
│   └── wfa_phase_cycle_test/      # Resultats WFA
└── docs/
    ├── POST_MORTEM_NHHM_ECHEC.md  # Ce document
    └── PROMPT_PASSATION_CLAUDE.md # Ce prompt
```

---

## FEATURES DISPONIBLES

Les features sont construites dans `src/regime_nhhm.py:build_nhhm_features()`:

| Feature | Description | Disponible depuis |
|---------|-------------|-------------------|
| momentum_6/12/24 | Returns sur N bars | 2011 |
| vol_ratio | Vol court/long | 2011 |
| rsi_centered | RSI - 50 | 2011 |
| dist_ma20/50 | Distance aux MAs | 2011 |
| halving_phase | Phase du cycle 4 ans | 2011 |
| halving_direction | Signal directionnel cycle | 2011 |
| P1_period | Periode dominante Fourier | 2011 |
| LFP_ratio | Low-frequency power | 2011 |
| funding_rate | Funding Binance | 2019+ seulement |

---

## FORMAT DES LABELS ATTENDU

```csv
timestamp,label
2011-08-18 00:00:00,1
2011-08-18 02:00:00,1
2011-08-18 04:00:00,0
...
```

- `timestamp`: DatetimeIndex (UTC)
- `label`: 1 (LONG), 0 (CASH), -1 (SHORT optionnel)

Le WFA lit ces labels et optimise les parametres Ichimoku pour chaque regime.

---

## COMMANDES UTILES

```powershell
# Tester un script Python
py -3 scripts/nom_script.py

# Lancer WFA test (1 seed, rapide)
py -3 scripts/run_scheduler_wfa_phase.py --labels-csv data/NOUVEAU_LABELS.csv --trials 50 --seed 101 --out-dir outputs/test

# Voir progression WFA
Get-Content outputs/test/seed_101/PROGRESS.json

# Lancer 30 seeds production
.\scripts\launch_30_seeds_cycle.ps1
```

---

## PIEGES A EVITER

1. **Lookahead bias** - Ne jamais utiliser info future dans les features
2. **Overfitting** - Toujours valider OOS (walk-forward)
3. **Fallback silencieux** - Si ca crash, LEVER UNE EXCEPTION
4. **Confondre filtre et predicteur** - Un filtre n'est pas de l'alpha

---

## OBJECTIF FINAL

Generer des labels predictifs qui permettent d'obtenir:
- **Sharpe OOS > 1.5** (vs 0.99 avec CYCLE)
- **Equity > x2.0 sur 14 ans** (vs x1.217 avec CYCLE)

Bonne chance!
