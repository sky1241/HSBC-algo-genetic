# Arbre de Decision - Generateur Alpha HSBC

**Objectif:** Savoir ou on en est et quelle branche suivre

---

## ARBRE PRINCIPAL

```
                    GENERATEUR ALPHA
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    DETECTION         STRATEGIE         VALIDATION
    REGIMES           TRADING           ROBUSTESSE
        │                 │                 │
   ┌────┴────┐       Ichimoku+ATR      WFA 30 seeds
   │         │            │
 HMM      NHHM/ML    (stable, OK)
   │         │
┌──┴──┐   ┌──┴──┐
K3 K5 K8  ML  NHHM
   │         │
   OK?    EN COURS
```

---

## OU ON EN EST (2026-02-09)

### Branche actuelle: DETECTION REGIMES → K5

```
DETECTION REGIMES
      │
      ├─ HMM Classique
      │     ├─ K3 ✅ VALIDE (Sharpe 0.99, MDD 13%)
      │     ├─ K5 🔄 EN COURS (12 seeds a ~81%)
      │     └─ K8 ⏳ A FAIRE
      │
      └─ NHHM/ML (prediction direction)
            ├─ NHHM ❌ ECHEC (statsmodels ne converge pas)
            ├─ ML LightGBM ⚠️ TESTE (Sharpe 0.12, MDD 4.4%)
            └─ CYCLE+ML ⏳ A COMBINER
```

### Status K5 actuel

| Seed | % | Status |
|------|---|--------|
| 107 | ~82% | En cours |
| 108-110 | ~81-82% | En cours |
| 201-202 | ~81% | En cours |
| 101-106 | ~79% | En cours |
| **Total** | **12/30** | **~81% moyen** |

---

## DECISION TREE COMPLET

```
START: Je veux generer de l'alpha sur BTC
           │
           ▼
    As-tu des labels de regime?
           │
     ┌─────┴─────┐
     │           │
    NON         OUI
     │           │
     ▼           ▼
  Generer    Quel type?
  labels         │
     │     ┌─────┼─────┐
     ▼     │     │     │
  HMM?    HMM  NHHM   ML
     │     │     │     │
     ▼     ▼     ▼     ▼
  K=?    K3/5/8  P(dir) Labels
     │     │     │     │
┌────┼────┐│     │     │
K3  K5  K8 │     │     │
│    │   │ │     │     │
▼    ▼   ▼ ▼     ▼     ▼
VALIDE  ?   OK?   OK?
│           │     │
▼           ▼     ▼
WFA       ECHEC  TEST
30 seeds    │     │
│           ▼     ▼
▼        ABANDON COMBINER
PRODUCTION  ou    avec
            ML    CYCLE
```

---

## FICHIERS PAR BRANCHE

### Branche HMM (stable)
```
src/regime_hmm.py           ← Modele HMM
src/features_fourier.py     ← Features spectrales
scripts/freeze_hmm_labels.py ← Genere K3/K5/K8.csv
outputs/fourier/labels_frozen/BTC_FUSED_2h/K*.csv
```

### Branche NHHM (echec)
```
src/regime_nhhm.py          ← CASSE (statsmodels fail)
docs/POST_MORTEM_NHHM_ECHEC.md ← Analyse echec
```

### Branche ML (en test)
```
src/ml_directional.py       ← LightGBM directionnel
data/ML_directional.csv     ← Labels ML
data/CYCLE_cash_bear.csv    ← Labels CYCLE
```

### Branche WFA (validation)
```
scripts/run_scheduler_wfa_phase.py  ← WFA principal
outputs/wfa_phase_k5/seed_*/        ← Resultats K5
scripts/launch_30_seeds_k5.ps1      ← Lancement
```

---

## QUELLE BRANCHE SUIVRE?

### Si K5 reussit (8/12+ survivent):
```
K5 OK → Tester K8 (Type C: 10 seeds × 100 trials)
     → Si K8 mieux → Production K8
     → Sinon → Production K5
```

### Si K5 echoue (<8/12 survivent):
```
K5 FAIL → Rester sur K3 (deja valide)
       → Tester CYCLE+ML (Type D: 15 seeds × 150 trials)
       → Si mieux → Production CYCLE+ML
```

### Prochaines etapes recommandees:
1. **Attendre fin K5** (12 seeds → resultats)
2. **Analyser survie** (critere: 8/12 OK)
3. **Decider**: K8 ou rester K5 ou K3
4. **Optionnel**: Tester CYCLE+ML en parallele

---

## CRITERES DE DECISION

| Metrique | Seuil OK | Seuil Excellent |
|----------|----------|-----------------|
| Survie | >66% | >80% |
| Sharpe median | >0.5 | >1.0 |
| MDD median | <25% | <15% |
| Monthly return | >0.2% | >0.5% |

---

## COMMANDES UTILES

```powershell
# Voir progression K5
Get-ChildItem 'outputs\wfa_phase_k5\seed_*\PROGRESS.json' | % {
  $j = Get-Content $_.FullName | ConvertFrom-Json
  "$($_.Directory.Name): $($j.percent)%"
}

# Analyser resultats (quand termine)
py -3 scripts/analyze_k5_results.py

# Lancer K8 (apres K5)
.\scripts\launch_10_seeds_k8.ps1
```

---

## HISTORIQUE DECISIONS

| Date | Decision | Resultat |
|------|----------|----------|
| 2025-10 | Tester K3 | ✅ Sharpe 0.99 |
| 2026-02-07 | Implementer NHHM | ❌ Echec |
| 2026-02-08 | Fallback CYCLE | ✅ Sharpe 0.99 |
| 2026-02-09 | Tester K5 | 🔄 En cours |

---

Cree: 2026-02-09
