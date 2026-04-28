# MISSION REWFA — Re-Walk-Forward Analysis du pipeline historique

> Mission séparée pour traiter BUG-C / BUG-D / BUG-E identifiés par l'audit
> du 27/04/2026 matin. **Hors scope P0bis-P12** (mission précédente). Ouverte
> mais **non bloquante pour testnet** (la stack actuelle est déployable
> indépendamment).

## Pré-requis

- ☐ 30 jours testnet stable sur la stack P0bis-P12 (stabilité validée
  via PSR daily + flow logs + meta_logger sans incident).
- ☐ Pas de modification du pipeline `ichimoku_pipeline_web_v4_8_fixed.py`
  pendant la mission (ce fichier est désormais **figé** comme référence
  historique).

## Bugs critiques à traiter (rappel audit 27/04/2026 matin)

### BUG-C — Deux systèmes Sharpe parallèles non-réconciliés

- **Symptôme** : pipeline historique calcule Sharpe ligne 1309 via
  `mean/std × √252` sur returns par-trade (formule potentiellement buggée
  pour trades H2 de durée variable). Module récent `src/stats_eval.py:compute_metrics`
  utilise une formule paramétrable correcte.
- **Conséquence** : K3 1D stable 21 seeds (le référent gagnant) n'a JAMAIS
  été ré-évalué avec le module correct.
- **Action** : ré-extraire les returns OOS des seeds K3 1D stable, les
  passer dans `compute_metrics` avec `periods_per_year` correct selon le
  timeframe réel des trades (H2 → variable), produire un nouveau Sharpe
  qui sera la **vraie** référence.

### BUG-D — Survivor bias par pruning (commit `d1b68c5`, 2025-09-04)

- **Symptôme** : `ichimoku_pipeline_web_v4_8_fixed.py` élimine des
  agrégats les seeds liquidés / margin_calls / `min_equity<0.6` /
  `MDD>0.6`. Les médianes publiées sont **survivor-biased**.
- **Conséquence** : statistiques de performance optimistes par construction.
- **Action** : ré-agréger TOUS les seeds (incluant liquidés) pour
  recalculer la médiane / quantiles honnêtes. Quantifier le biais
  introduit par le pruning historique (différence entre pruned vs full).

### BUG-E — WFA_SUMMARY_ANNUAL Sharpe = 1.16 vs MONTHLY = -18 sur MÊMES seeds

- **Symptôme** : énorme divergence numérique selon le mode d'évaluation.
- **Conséquence** : bug d'annualisation dans l'un des deux pipelines.
  Au moins un des deux est faux.
- **Action** : tracer mathématiquement la différence entre le calcul
  annual et monthly. Reproduire les 2 sur seed unique. Identifier la
  formule correcte (probablement le monthly si les returns sont à grain
  fin) et écraser l'autre.

## Plan d'attaque proposé

**Principe** : NE PAS toucher `ichimoku_pipeline_web_v4_8_fixed.py`
(fichier figé). Construire un **nouveau pipeline propre**
`scripts/rewfa/run_rewfa.py` qui :

1. **Réutilise** `src/wfa.py`, `src/stats_eval.py`, `src/cost_model.py`,
   `src/reality_check.py` (modules quant validés mission P0bis-P12).
2. **Re-fait** la WFA sur les mêmes données BTC (et ETH+SOL si possible)
   avec :
   - Formule Sharpe correcte paramétrable (BUG-C)
   - **Aucun pruning** sur seeds — agrégation full inclusive (BUG-D)
   - Annualisation cohérente : `√(periods_per_year)` calculé selon le
     timeframe réel des returns (BUG-E)
   - **DSR déflaté** avec `n_trials = 200` ou plus (selon le nombre
     réel de configs Optuna testées historiquement) — au lieu du 50
     hardcodé pour P10
3. **Compare** les nouveaux résultats avec les anciens pour quantifier
   l'écart (combien de bps Sharpe perdus par le pruning ? quelle vraie
   significativité après DSR honnête ?).

## Estimation effort

- **5-10 jours** en sessions concentrées :
  - Day 1-2 : reproduction de l'ancien WFA + identification de la
    formule Sharpe utilisée historiquement.
  - Day 3-4 : nouveau pipeline propre + re-run sur BTC.
  - Day 5-6 : ré-évaluation seeds K3 1D stable + comparaison.
  - Day 7-8 : DSR déflaté n_trials=200, calcul significativité.
  - Day 9-10 : rapport comparatif + décision (la stratégie tient-elle
    après correction des 3 bugs ?).

## Statut

- **NON DÉMARRÉE**. À planifier après bascule testnet stable + retour
  d'expérience 30j.
- **Non bloquante pour testnet** : la stack actuelle (P0bis-P12) est
  fonctionnelle indépendamment. Le pipeline historique restera figé et
  documenté comme référence buggée connue.

## Liens

- Audit du 27/04/2026 matin (référence des 6 BUG-A à BUG-F).
- BUG-A et BUG-B : déjà fixés par P0bis et P0 (mission précédente).
- BUG-F (DSR non déflaté) : partiellement adressé par P5 + P10
  (n_trials=50 max), mais le pipeline historique 14 ans n'a pas été
  ré-évalué — sera traité par REWFA.
