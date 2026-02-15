# PROMPT DE REPRISE - Projet NHHM (Non-Homogeneous HMM)

**Copie-colle ce prompt au début de ta prochaine session Claude.**

---

## CONTEXTE DU PROJET

Je travaille sur un système de trading algorithmique BTC. Mon système actuel a un problème identifié :

```
PROBLÈME: Mon HMM détecte les RÉGIMES (volatilité) mais pas la DIRECTION (bull/bear)
RÉSULTAT: 100% survie mais seulement +3.6%/an vs +85%/an Buy&Hold
```

## CE QU'ON A FAIT

### Session précédente (2026-02-07)

1. **Analysé la Deep Research** → 5 recommandations validées avec sources académiques
2. **Créé le module NHHM** → `src/regime_nhhm.py` (Markov-switching avec statsmodels)
3. **Créé le module Funding Rate** → `src/funding_rate.py` (télécharge depuis Binance API)
4. **Testé le NHHM** → Hit rate 47% → 49% avec funding (légère amélioration)

### Fichiers créés

| Fichier | Description | Statut |
|---------|-------------|--------|
| `src/regime_nhhm.py` | Module NHHM (Markov-switching) | ✅ Fonctionne |
| `src/funding_rate.py` | Features funding rate | ✅ Fonctionne |
| `scripts/test_nhhm_quick.py` | Test baseline | ✅ OK |
| `scripts/test_nhhm_with_funding.py` | Test avec funding | ✅ OK |
| `data/funding_rate_BTCUSDT.csv` | Données funding 2019-2026 | ✅ Téléchargé |
| `docs/DEEP_RESEARCH_ANALYSIS_2026-02-07.md` | Analyse complète | ✅ Référence "1" |

### Résultats actuels

```
NHHM Baseline (sans funding, 2011-2025):
- Hit rate: 47.36%
- Sharpe: 0.01
- Problème: Pas assez prédictif

NHHM + Funding (2019-2025 seulement):
- Hit rate: 49.03% (+1.67%)
- Mais moins de données (funding existe depuis 2019)
```

## CE QU'IL RESTE À FAIRE

### Priorité 1: Intégrer les features Fourier existantes

Le NHHM actuel n'utilise PAS mes features Fourier (`P1_period`, `LFP_ratio`, `volatility`).
Il faut les ajouter comme covariables pour avoir le modèle complet sur 14 ans.

**Action:** Modifier `build_nhhm_features()` pour inclure les features spectrales.

### Priorité 2: Remplacer HMM par NHHM dans le pipeline

```
AVANT: Données → Fourier → HMM → Label dur → Ichimoku
APRÈS: Données → Fourier → NHHM → P(bull), P(bear) → Ichimoku
```

**Action:** Créer un script qui génère des labels NHHM comme `K3_nhhm.csv`.

### Priorité 3: Intégrer au WFA

Tester avec le même framework que K3/K5 (30 seeds, 300 trials).

## FICHIERS À LIRE EN PREMIER

```
1. docs/DEEP_RESEARCH_ANALYSIS_2026-02-07.md  → Le diagnostic complet (référence "1")
2. src/regime_nhhm.py                         → Le code NHHM actuel
3. src/regime_hmm.py                          → L'ancien HMM (pour comprendre le format)
4. src/spectral/hmm_features.py               → Les features Fourier existantes
```

## COMMANDES UTILES

```powershell
# Tester NHHM rapidement
py -3 scripts/test_nhhm_quick.py

# Tester avec funding
py -3 scripts/test_nhhm_with_funding.py

# Voir progression K5 (peut-être encore en cours)
Get-ChildItem 'outputs\wfa_phase_k5\seed_*\PROGRESS.json' | % {
    $j = Get-Content $_.FullName | ConvertFrom-Json
    "$($_.Directory.Name): $($j.percent)%"
}
```

## DONNÉES DISPONIBLES

| Fichier | Période | Usage |
|---------|---------|-------|
| `data/BTC_FUSED_2h.csv` | 2011-2025 (14 ans) | Données OHLCV principales |
| `data/funding_rate_BTCUSDT.csv` | 2019-2026 | Funding rate (feature bonus) |
| `outputs/fourier/labels_frozen/BTC_FUSED_2h/K5_1d_stable.csv` | 2011-2025 | Labels HMM actuels |

## ARCHITECTURE DU SYSTÈME

```
Pipeline actuel:
┌─────────────┐    ┌─────────┐    ┌─────┐    ┌────────┐    ┌──────────┐    ┌─────┐
│ BTC H2 Data │ → │ Fourier │ → │ HMM │ → │ Labels │ → │ Ichimoku │ → │ WFA │
└─────────────┘    └─────────┘    └─────┘    └────────┘    └──────────┘    └─────┘
                                    ↓
                              Label = 0,1,2
                              (régime seulement)

Pipeline cible (NHHM):
┌─────────────┐    ┌─────────┐    ┌──────┐    ┌─────────────┐    ┌──────────┐    ┌─────┐
│ BTC H2 Data │ → │ Fourier │ → │ NHHM │ → │ P(bull/bear)│ → │ Ichimoku │ → │ WFA │
└─────────────┘    └─────────┘    └──────┘    └─────────────┘    └──────────┘    └─────┘
                                    ↓
                              P(bull) = 0.72
                              P(bear) = 0.28
                              Signal = LONG
```

## CONTRAINTES TECHNIQUES

- Python 3.10+ (utiliser `py -3` sur Windows)
- Pas de GPU (CPU seulement)
- statsmodels pour MarkovRegression/MarkovAutoregression
- 60,000+ bougies H2 (14 ans)
- Frais: 0.04% par trade

## CE QUE JE NE COMPRENDS PAS (aide-moi)

Je suis pas un quant, je fais du copier-coller entre AIs. Explique-moi les choses simplement et code pour moi. Quand tu fais du code, teste-le pour vérifier qu'il marche.

## OBJECTIF FINAL

Améliorer le Sharpe de ~0.3 (actuel) vers ~0.8-1.0 en:
1. Prédisant la DIRECTION (pas juste le régime)
2. Filtrant les mauvais trades
3. Utilisant le funding rate comme signal directionnel (quand dispo)

---

**FIN DU PROMPT DE REPRISE**
