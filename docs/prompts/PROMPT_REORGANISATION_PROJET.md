# PROMPT - Reorganisation Architecture Projet HSBC-algo-genetic

**Date:** 2026-02-09
**Objectif:** Reorganiser le projet sans rien supprimer, juste clarifier

---

## CONTEXTE

Le projet est devenu complexe avec beaucoup de fichiers eparpilles.
On ne supprime RIEN, on reorganise pour y voir clair.

---

## PROMPT A COPIER-COLLER

```
Je veux reorganiser le projet HSBC-algo-genetic.

## REGLES STRICTES
1. NE RIEN SUPPRIMER - aucun fichier, aucun dossier
2. NE PAS TOUCHER aux processus en cours (K5 tourne)
3. Reorganiser uniquement via deplacements et renommages
4. Creer un README principal clair

## ETAT ACTUEL DU PROJET

Le projet genere de l'alpha sur BTC via:
- Analyse Fourier (features spectrales)
- HMM/NHHM (detection regimes)
- Ichimoku + ATR (strategie trading)
- WFA (validation walk-forward)

### Resultats connus
- K3: Sharpe 0.99, MDD 13% (VALIDE)
- CYCLE: Sharpe 0.99 (VALIDE)
- ML seul: Sharpe 0.12, MDD 4.4%
- K5: En cours de validation

### Tests en cours
- 12 seeds K5 a ~85-87%
- 18 seeds K5 en attente

## STRUCTURE ACTUELLE (bordel)

```
HSBC-algo-genetic/
├── src/                    # Code principal (OK)
├── scripts/                # 50+ scripts (BORDEL)
├── data/                   # Donnees + labels (OK mais mixte)
├── docs/                   # 70+ docs (BORDEL)
├── outputs/                # Resultats WFA (OK)
└── README.md               # Obsolete
```

## STRUCTURE CIBLE PROPOSEE

```
HSBC-algo-genetic/
├── README.md               # NOUVEAU - etat du projet, comment demarrer
├── src/                    # Code principal (ne pas toucher)
│   ├── core/               # Moteurs principaux
│   └── experimental/       # Code en test (NHHM, ML)
├── scripts/
│   ├── production/         # Scripts valides et utilises
│   ├── analysis/           # Scripts d'analyse
│   ├── experimental/       # Scripts en test
│   └── archived/           # Anciens scripts (ne pas supprimer)
├── data/
│   ├── raw/                # Donnees brutes (OHLCV)
│   ├── labels/             # Labels regimes (K3, K5, CYCLE, ML)
│   └── features/           # Features calculees
├── docs/
│   ├── README_ETAT.md      # Ou on en est maintenant
│   ├── guides/             # Guides (validation, decision)
│   ├── prompts/            # Prompts de session
│   ├── reports/            # Rapports d'analyse
│   └── archive/            # Ancienne doc (ne pas supprimer)
├── outputs/                # Resultats WFA (ne pas toucher)
└── config/                 # Fichiers de config
```

## TACHES A FAIRE

### 1. Creer README_ETAT.md (priorite 1)
Un seul fichier qui dit:
- Ou on en est
- Quoi faire ensuite
- Quels fichiers sont importants

### 2. Reorganiser docs/ (priorite 2)
Deplacer les fichiers dans les sous-dossiers:
- guides/ : GUIDE_VALIDATION_AUTO.md, ARBRE_DECISION_ALPHA.md
- prompts/ : tous les PROMPT_*.md
- reports/ : tous les rapports d'analyse
- archive/ : le reste

### 3. Reorganiser scripts/ (priorite 3)
Identifier:
- Scripts utilises en production
- Scripts d'analyse one-shot
- Scripts experimentaux
- Scripts obsoletes

### 4. Mettre a jour README.md principal
- Resume du projet en 10 lignes
- Quick start
- Lien vers README_ETAT.md

## CE QU'IL NE FAUT PAS FAIRE

- Supprimer des fichiers
- Modifier du code
- Toucher aux outputs/
- Interrompre les processus K5
- Renommer src/ ou ses fichiers principaux

## VALIDATION

Apres reorganisation, verifier que:
1. `py -3 scripts/run_scheduler_wfa_phase.py --help` fonctionne
2. Les imports Python marchent encore
3. Les processus K5 tournent toujours

## LIVRABLES

1. README.md mis a jour
2. docs/README_ETAT.md cree
3. Sous-dossiers crees dans docs/ et scripts/
4. Fichiers deplaces (pas supprimes)
5. Liste des fichiers deplaces dans un log
```

---

## FICHIERS IMPORTANTS A NE PAS TOUCHER

### Code critique
- src/regime_hmm.py
- src/features_fourier.py
- src/optimizer.py
- src/wfa.py
- scripts/run_scheduler_wfa_phase.py
- scripts/freeze_hmm_labels.py

### Labels valides
- data/CYCLE_cash_bear.csv
- outputs/fourier/labels_frozen/BTC_FUSED_2h/K3.csv
- outputs/fourier/labels_frozen/BTC_FUSED_2h/K5_1d_stable.csv

### Documentation cle
- docs/METHODOLOGIE_COMPLETE.md
- docs/GUIDE_VALIDATION_AUTO.md
- docs/ARBRE_DECISION_ALPHA.md

---

## MONITORING K5 PENDANT REORGANISATION

```powershell
# Verifier que K5 tourne toujours
Get-Process python* | Measure-Object

# Voir progression
Get-ChildItem 'outputs\wfa_phase_k5\seed_*\PROGRESS.json' | % {
  $j = Get-Content $_.FullName | ConvertFrom-Json
  "$($_.Directory.Name): $($j.percent)%"
}
```

---

Cree: 2026-02-09
