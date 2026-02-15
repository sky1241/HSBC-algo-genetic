# QC des gaps H2 (BTC_FUSED_2h)

- Contexte: la pipeline a signalé "Gaps extrêmes détectés" sur BTC/USDT pendant des backtests. Nous avons quantifié et documenté l'impact potentiel.

## Détection
- Fichier source: `data/BTC_FUSED_2h.csv` (UTC, tz-naive après normalisation).
- Script: `scripts/qc_detect_and_fix_gaps.py` (détection > 2h entre bougies).
- Résumé (12-09): ~24 gaps sur ~60k bougies H2 (< 0,05%).

## Impact attendu
- Ordre de grandeur: < 0,2% sur l’équity finale; cas défavorable rare ≲ 0,5%. 1% très improbable.
- Justification: très faible proportion de barres affectées; déclencheurs rares sur ces instants.

## Procédure A/B (original vs clean)
1. Baseline sur le CSV original (sans remplissage de gaps)
2. Création du CSV "clean" avec insertion de barres neutres (open=high=low=close=close_précédent; volume=0)
3. Baseline sur le CSV clean
4. Comparaison des métriques (equity, MDD, trades, Sharpe)

Commandes (PowerShell):
```
$env:USE_FUSED_H2='1'
# A) Original
py -3 scripts\run_btc_baseline_fixed.py --use-fused --tenkan 9 --kijun 26 --senkou-b 52 --shift 26 --atr 5 --out-dir outputs\ab_gap\A_original 2>&1 | Tee-Object -FilePath outputs\ab_gap\A_original\RUN.txt
# Clean
py -3 scripts\qc_detect_and_fix_gaps.py --path data\BTC_FUSED_2h.csv --out data\BTC_FUSED_2h_clean.csv --fill ffill
# B) Clean
py -3 scripts\run_btc_baseline_fixed.py --use-fused --tenkan 9 --kijun 26 --senkou-b 52 --shift 26 --atr 5 --out-dir outputs\ab_gap\B_clean 2>&1 | Tee-Object -FilePath outputs\ab_gap\B_clean\RUN.txt
```

## Décision
- Pour l’optimisation/tri (Optuna, WFA): on continue sans correction (on logue les gaps).
- Pour les baselines "publication": on documente et on fournit l’A/B; si l’écart > 0,5% on retient la version "clean" et on note la correction.
