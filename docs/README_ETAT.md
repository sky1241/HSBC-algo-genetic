# Etat du Projet HSBC-algo-genetic

**Date:** 2026-02-11
**Derniere mise a jour:** Decouverte COMBINED (CYCLE+ML) - meilleur resultat projet

---

## Resume en 3 lignes

Le projet genere de l'alpha sur BTC via analyse Fourier + HMM pour detecter les regimes de marche, puis optimise les parametres Ichimoku + ATR par regime. La **meilleure approche COMBINED (CYCLE+ML)** atteint Equity 1.72x avec MDD 5.8% sur 14 ans de WFA.

---

## Resultats Valides

| Approche | Sharpe | Equity | MDD | Status |
|----------|--------|--------|-----|--------|
| **★ COMBINED (CYCLE+ML)** | **0.91** | **1.72x** | **5.8%** | **★ MEILLEUR** |
| K3 (3 regimes HMM) | 0.99 | - | 13% | VALIDE |
| CYCLE (halving) | 0.99 | 1.22x | ~15% | VALIDE |
| ML seul | 0.12 | 1.23x | 4.4% | Teste |
| K5 (5 regimes) | - | - | - | En cours (12 seeds ~87%) |

---

## ★ Decouverte COMBINED (2026-02-10)

Fusion CYCLE (filtre bear) + ML (direction bull) = meilleur resultat du projet.

**Logique:** Si CYCLE=0 (bear) → CASH force / Si CYCLE=1 (bull) → ML decide
**Labels:** `data/COMBINED_labels.csv` (54879 CASH, 6456 LONG, 130 SHORT)
**Resultats:** `outputs/wfa_combined_test/`
**Rapport:** `docs/reports/RAPPORT_DECOUVERTE_COMBINED.md`

---

## Tests en Cours

- **K5 WFA Phase**: 12 seeds a ~87% de completion
- **18 seeds supplementaires** en attente
- **COMBINED**: 1 seed valide, multi-seed a lancer

### Monitoring K5

```powershell
# Verifier processus Python actifs
Get-Process python* | Measure-Object

# Voir progression par seed
Get-ChildItem 'outputs\wfa_phase_k5\seed_*\PROGRESS.json' | % {
  $j = Get-Content $_.FullName | ConvertFrom-Json
  "$($_.Directory.Name): $($j.percent)%"
}
```

---

## Prochaines Etapes

1. **★ Multi-seed COMBINED** - Lancer 5-10 seeds pour confirmer robustesse
2. **Attendre fin K5** - Laisser les 12 seeds terminer
3. **Comparer** K5 vs COMBINED vs K3 → choisir candidat production

---

## Structure du Projet

```
HSBC-algo-genetic/
├── README.md              # Ce fichier
├── src/                   # Code principal (NE PAS TOUCHER)
│   ├── regime_hmm.py      # Detection regimes HMM
│   ├── features_fourier.py # Features spectrales
│   ├── optimizer.py       # Optimisation parametres
│   ├── wfa.py             # Walk-Forward Analysis
│   └── ...
├── scripts/
│   ├── production/        # Scripts valides et utilises
│   ├── analysis/          # Scripts d'analyse
│   ├── experimental/      # Scripts en test
│   └── archived/          # Anciens scripts
├── data/
│   ├── BTC_FUSED_2h.csv   # Donnees principales
│   ├── CYCLE_cash_bear.csv # Labels CYCLE (VALIDE)
│   ├── ML_directional.csv # Labels ML
│   └── ...
├── docs/
│   ├── README_ETAT.md     # CE FICHIER - ou on en est
│   ├── guides/            # METHODOLOGIE_COMPLETE.md, GUIDE_VALIDATION_AUTO.md
│   ├── prompts/           # Prompts de session
│   ├── reports/           # Rapports d'analyse
│   ├── journals/          # Journaux de session
│   └── archive/           # Ancienne doc
└── outputs/               # Resultats WFA (NE PAS TOUCHER)
    └── wfa_phase_k5/      # Tests K5 en cours
```

---

## Fichiers Critiques

### Code (ne pas modifier sans raison)
- `src/regime_hmm.py` - Detection regimes HMM
- `src/features_fourier.py` - Features spectrales
- `src/optimizer.py` - Optimisation Optuna
- `src/wfa.py` - Walk-Forward Analysis

### Scripts Production
- `scripts/production/run_scheduler_wfa_phase.py` - Scheduler principal
- `scripts/production/freeze_hmm_labels.py` - Gel des labels HMM
- `scripts/production/launch_30_seeds_k5.ps1` - Lancement multi-seeds

### Labels Valides
- `outputs/fourier/labels_frozen/BTC_FUSED_2h/K3.csv` - Labels K3 valides
- `outputs/fourier/labels_frozen/BTC_FUSED_2h/K5_1d_stable.csv` - Labels K5
- `data/CYCLE_cash_bear.csv` - Labels CYCLE valides
- `data/ML_directional.csv` - Labels ML
- **`data/COMBINED_labels.csv`** - ★ Labels COMBINED (CYCLE+ML)

### Documentation Cle
- `docs/guides/METHODOLOGIE_COMPLETE.md` - Methodologie detaillee
- `docs/guides/GUIDE_VALIDATION_AUTO.md` - Guide de validation
- `docs/guides/ARBRE_DECISION_ALPHA.md` - Arbre de decision

---

## Commandes Utiles

### Lancer un test WFA
```powershell
py -3 scripts/production/run_scheduler_wfa_phase.py --help
```

### Verifier imports Python
```powershell
py -3 -c "from src.regime_hmm import RegimeHMM; print('OK')"
py -3 -c "from src.wfa import WFA; print('OK')"
```

### Agreger resultats K5
```powershell
py -3 scripts/production/aggregate_k5_results.py
```

---

## Historique Recent

- **2026-02-11**: Documentation decouverte COMBINED
- **2026-02-10**: ★ Decouverte COMBINED (Equity 1.72x, MDD 5.8%) - Reorganisation structure
- **2026-02-09**: Test ML seul (Sharpe 0.12, MDD 4.4%)
- **2026-02-07**: Implementation NHHM, CYCLE labels
- **2025-10**: Validation K3 (Sharpe 0.99)

---

## Contact / Reprise

Pour reprendre le projet:
1. Lire ce fichier (README_ETAT.md)
2. Consulter `docs/guides/METHODOLOGIE_COMPLETE.md`
3. Verifier l'etat des tests K5 avec les commandes ci-dessus
4. Suivre l'arbre de decision dans `docs/guides/ARBRE_DECISION_ALPHA.md`
