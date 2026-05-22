# HSBC-algo-genetic

Systeme de trading algorithmique sur BTC utilisant l'analyse Fourier, la detection de regimes HMM et l'optimisation Walk-Forward.

> **Statut honnete** : ce repo est un projet de recherche, pas une strategie deployable. Voir [`algo_trading_haute_frequence/RAPPORT_ALPHA_FINAL_2026-04-26.md`](algo_trading_haute_frequence/RAPPORT_ALPHA_FINAL_2026-04-26.md) et [`algo_trading_haute_frequence/RAPPORT_BILAN_QUANT_2026-04-27.md`](algo_trading_haute_frequence/RAPPORT_BILAN_QUANT_2026-04-27.md) avant toute decision.

## Resultats

| Approche | Sharpe | MDD | Statut OOS net de frais |
|----------|--------|-----|--------------------------|
| **HODL spot BTC** (benchmark) | **1.09** | — | Reference 2020-2025 |
| K3 (3 regimes) alone | **-1.91** | — | Pas d'edge significatif (DSR 0.040 << 0.95) |
| K3 in-sample | 0.99 | 13% | Avant frais et OOS — ne pas confondre |
| CYCLE+ML (COMBINED) | 0.91 | 5.8% | Seed 101 uniquement ; seed 102 = -0.65 (seed-sensitive, multi-seed non valide) |
| K5 (5 regimes) | - | - | En cours |

**Bilan documente** : aucune strategie testee ne bat HODL spot BTC sur 2020-2025 net de frais Binance VIP0 (4 bps taker + funding + slippage ATR-aware). Le bot mainnet K3 actuel a perdu environ $66, coherent avec le Sharpe -1.91 OOS.

## Quick Start

```powershell
# Verifier l'installation
py -3 scripts/production/run_scheduler_wfa_phase.py --help

# Lancer un test WFA
py -3 scripts/production/run_scheduler_wfa_phase.py --label K5 --trials 30 --seed 42
```

## Ou en est-on ?

Voir **[docs/README_ETAT.md](docs/README_ETAT.md)** pour l'etat actuel du projet, les tests en cours et les prochaines etapes.

## Structure

```
HSBC-algo-genetic/
├── src/                    # Code principal
├── scripts/
│   ├── production/         # Scripts valides
│   ├── analysis/           # Scripts d'analyse
│   └── experimental/       # Scripts en test
├── data/                   # Donnees et labels
├── docs/
│   ├── README_ETAT.md      # Etat actuel du projet
│   ├── guides/             # Documentation technique
│   └── reports/            # Rapports d'analyse
└── outputs/                # Resultats WFA
```

## Documentation

- [Etat du projet](docs/README_ETAT.md) - Ou on en est, quoi faire ensuite
- [Methodologie](docs/guides/METHODOLOGIE_COMPLETE.md) - Comment ca marche
- [Guide validation](docs/guides/GUIDE_VALIDATION_AUTO.md) - Valider les resultats
- [Arbre de decision](docs/guides/ARBRE_DECISION_ALPHA.md) - Prendre des decisions

## Pipeline

1. **Features Fourier** - Extraction spectrale des cycles
2. **Detection HMM** - Classification des regimes de marche
3. **Optimisation** - Parametres Ichimoku + ATR par regime
4. **Walk-Forward** - Donnees historiques 2011-2025 (~14 ans), fenetre de backtest finale 2020-2025 (5 ans), 949 dossiers de runs WFA, DSR (Bailey-Lopez de Prado) + Hansen SPA (1000 bootstrap blocks)

## Licence

MIT - Usage educatif et experimental. Le trading comporte des risques.
