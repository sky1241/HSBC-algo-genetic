# Phase ↔ Ichimoku presets

Ce fichier centralise les réglages **Ichimoku** utilisés pour chaque phase
du moteur halving.

## Contenu
- `scripts/phase_ichimoku_settings.py` : dictionnaire `PHASE_ICHIMOKU`
  et fonction `ichimoku_params_for_phase()` pour récupérer
  `(tenkan, kijun, senkou_b)`.
- `scripts/phase_aware_module.py` : paramètre `use_ichimoku=True` dans
  `phase_snapshot()` calcule `M` à partir de Tenkan/Kijun et ajoute les
  colonnes `tenkan`, `kijun`, `cloud_top`, `cloud_bot` aux features.

## Exemple
```python
from scripts.phase_aware_module import phase_snapshot
from scripts.phase_ichimoku_settings import ichimoku_params_for_phase

feats = phase_snapshot(df, use_ichimoku=True)
phase = feats.iloc[-1]["phase"]
params = ichimoku_params_for_phase(phase)
```

La configuration est isolée afin de faciliter les ajustements ultérieurs
sans modifier le code principal.
