# PROMPT IMPLEMENTATION CYCLE HALVING

## CONTEXTE RAPIDE

On a decouvert que le HMM (NHHM) detecte la **volatilite**, pas la **direction**. Ca ne marche pas pour predire bull/bear.

La VRAIE solution: utiliser le **cycle de 4 ans du halving BTC** qui est 100% deterministe.

## RESULTATS PROUVES

| Strategie | Hit Rate | Sharpe |
|-----------|----------|--------|
| **Cycle V2 (cash en bear)** | 55.70% | **1.61** |
| Momentum baseline | 47.82% | 0.26 |

**+8% hit rate, +1.35 Sharpe** - C'est enorme.

## LA LOGIQUE DU CYCLE (a implementer)

```python
HALVINGS = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]

# Jours depuis halving -> Direction
CYCLE_PHASES = {
    (0, 180): 1,      # accumulation: LONG
    (180, 365): 1,    # early_bull: LONG
    (365, 540): 1,    # parabolic: LONG
    (540, 730): 0,    # distribution: CASH (pas short!)
    (730, 1095): 0,   # bear: CASH (pas short!)
    (1095, 1460): 1,  # late_bear: LONG
}
```

**IMPORTANT:** On ne short PAS en bear market (les bounces tuent le short). On reste CASH.

## FICHIERS EXISTANTS

### Labels deja generees
- `data/CYCLE_cash_bear.csv` - **UTILISER CELUI-CI** (Sharpe 1.61)
- `data/CYCLE_selective.csv` - Alternative (Sharpe 1.28)
- `data/CYCLE_original.csv` - Short en bear (Sharpe 1.10)

### Scripts de test
- `scripts/test_cycle_variants.py` - Compare toutes les variantes
- `scripts/generate_cycle_labels.py` - Regenere les labels

## CE QU'IL FAUT FAIRE

### 1. Verifier que les labels sont OK
```powershell
py -3 scripts/test_cycle_variants.py
```
Doit afficher V2 avec Sharpe ~1.61.

### 2. Integrer dans le pipeline WFA

Le pipeline WFA utilise des labels pour definir les regimes. Il faut:

a) Verifier le format des labels CYCLE vs K5:
```powershell
# Comparer les formats
head data/CYCLE_cash_bear.csv
head data/K5_1d_stable.csv
```

b) Si format different, adapter `generate_cycle_labels.py` pour matcher le format K5.

c) Lancer un test WFA avec les labels CYCLE:
```powershell
# Modifier le script de lancement pour utiliser CYCLE_cash_bear.csv au lieu de K5
# Puis lancer 1-2 seeds de test
```

### 3. Comparer avec K5

Une fois le WFA termine:
- Sharpe CYCLE vs Sharpe K5
- Si CYCLE > K5: utiliser CYCLE pour les 30 seeds finaux

## STRUCTURE DU CODE EXISTANT

Le pipeline WFA est dans:
- `src/wfa_pipeline.py` ou similaire
- Les labels sont charges via un parametre `label_file` ou `regime_file`

Chercher:
```powershell
# Trouver ou les labels sont charges
rg "K5" src/
rg "label" src/wfa
```

## QUESTIONS A RESOUDRE

1. Le WFA utilise-t-il directement les labels ou recalcule-t-il les regimes?
2. Faut-il modifier le code source ou juste changer le fichier de labels?
3. Les labels CYCLE ont 3 valeurs (1, 0, -1) - le pipeline supporte-t-il le 0 (CASH)?

## VALIDATION FINALE

Avant de lancer les 30 seeds:
1. Faire un WFA test sur 2-3 seeds avec labels CYCLE
2. Verifier que le Sharpe OOS est > 1.0
3. Si OK, lancer les 30 seeds avec `CYCLE_cash_bear.csv`

## FICHIERS CLES A LIRE

1. `scripts/test_cycle_variants.py` - Comprendre la logique
2. `data/CYCLE_cash_bear.csv` - Format des labels
3. `src/wfa_pipeline.py` (ou equivalent) - Comment les labels sont utilises
4. `scripts/launch_30_seeds_k5.ps1` - Script de lancement a adapter

## COMMANDES UTILES

```powershell
# Tester la logique cycle
py -3 scripts/test_cycle_variants.py

# Regenerer les labels si besoin
py -3 scripts/generate_cycle_labels.py

# Voir le format des labels
Get-Content data/CYCLE_cash_bear.csv | Select-Object -First 10

# Chercher ou les labels sont utilises
rg "label" src/ --type py
rg "regime" src/ --type py
```

## RESUME EN 1 PHRASE

**Remplacer les labels K5/K3 par `CYCLE_cash_bear.csv` dans le pipeline WFA et lancer 30 seeds pour valider le Sharpe 1.61.**
