# HSBC-algo-genetic

Systeme de trading algorithmique sur BTC utilisant l'analyse Fourier, la detection de regimes HMM et l'optimisation Walk-Forward.

## Resultats

| Approche | Sharpe | MDD | Status |
|----------|--------|-----|--------|
| K3 (3 regimes) | 0.99 | 13% | Valide |
| CYCLE (halving) | 0.99 | ~15% | Valide |
| K5 (5 regimes) | - | - | En cours |

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
4. **Walk-Forward** - Validation sur 14 ans glissants

## Licence

MIT - Usage educatif et experimental. Le trading comporte des risques.
