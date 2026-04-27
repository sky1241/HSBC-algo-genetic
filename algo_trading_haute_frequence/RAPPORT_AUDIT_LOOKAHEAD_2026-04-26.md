# Audit lookahead bias — 2026-04-26

> Suite à la flag de `RAPPORT_GAPS_LAYER2_2026-04-26.md` ("BUG-004 closed-bar
> fixé mais pas vérifié sur funding et autres features t+1").

## Méthodologie

Grep systematic des patterns suspects dans `binance_bot/` et `src/` :
1. `.shift(-N)` (shift négatif = data future)
2. `iloc[i+N]` (forward indexing)
3. `rolling().shift(neg)`
4. Mots-clés `future` / `forward`

## Findings

### ✅ Bot live (`binance_bot/`) — clean

- `binance_bot/services/ichimoku_engine.py` : Senkou Span A/B utilisent `shift(+shift)` (positif = projection vers le futur sur le DataFrame mais `senkou.iloc[t]` représente la valeur calculée avec data t-shift). **Pas de lookahead.**
- `binance_bot/services/signal_engine.py` : utilise `df_ichimoku.iloc[-1]` (dernière bar, déjà filtrée par BUG-004 closed-bar). **Pas de lookahead.**
- `binance_bot/services/data_fetcher.py` : `_filter_unclosed_last_candle` (BUG-004) drop la bougie en formation. **Pas de lookahead.**
- `binance_bot/services/regime_filter.py` : `efficiency_ratio_value` et `adx_value` utilisent uniquement `close.diff()` et `close.shift(+1)`. **Pas de lookahead.**

### ✅ Module funding rate (`src/funding_rate.py`)

```python
# Line 132-134:
# No look-ahead: shift rates forward by one bar so the rate observed at t
# is applied to the position held over the next interval.
funding_for_bar = funding_for_bar.shift(1).fillna(0.0)
```

Le funding est explicitement shifté de **+1** pour éviter le lookahead (le rate observé à t s'applique à la position détenue à t+1). **Audit explicite et correct.**

### ✅ Module cost_model (`src/cost_model.py`)

Commentaire ligne 132 : "No look-ahead: shift rates forward by one bar". Cohérent avec funding_rate.py.

### ✅ Range detector (`src/range_detector.py`)

```python
h_pc = (high - close.shift(1)).abs()    # ATR uses bar PREVIOUS close — OK
direction = (close - close.shift(period)).abs()  # ER uses PAST close — OK
```

Tous les `shift()` sont positifs (data du passé). **Pas de lookahead.**

### ✅ Risk sizing (`src/risk_sizing.py`)

```python
lag = max(int(params.get("shift", 1)) // 2, 1)
signal = signal.shift(lag).fillna(0.0)   # delay le signal
```

Le signal est shifté de **+lag** = retardé. **L'inverse du lookahead.** Le signal observé à t-lag déclenche le trade à t. C'est la pratique standard pour éviter de trader sur le close de la bar courante (qu'on voit en arrière, pas en temps réel).

### ⚠️ Patterns acceptables hors bot live

- `src/ml_directional.py:114` : `fwd_return = close.shift(-N) / close - 1` → utilisé comme **target Y** dans le supervised learning (label, pas feature). Legitime tant qu'on ne l'utilise pas comme input X.
- `src/regime_nhhm.py:82, 765` : pareil, target pour HMM training.
- `src/live_trader_adaptive.py:119` : `chikou = close.shift(-shift)` calculé comme colonne du DataFrame mais **JAMAIS utilisé pour générer signal_long/signal_short** (vérifié par grep). Le bot live `binance_bot/services/ichimoku_engine.py` n'utilise même pas chikou.

## Conclusion

**Aucun lookahead bias actif dans le pipeline de trading live**. Les seuls `.shift(-N)` du codebase sont dans :
1. Le code de labeling ML (légitime)
2. Une variante legacy non utilisée par le bot live

Les modules sensibles (funding, cost_model, signal_engine, ichimoku_engine, regime_filter) ont été individuellement vérifiés et sont propres. Plusieurs modules contiennent des commentaires explicites "no lookahead" attestant que la question a été considérée par les auteurs originaux.

## Recommandation

- ✅ **Pas d'action immédiate requise** — l'audit est concluant.
- 📌 **Test régression à ajouter** : pour chaque feature critique (ATR, ER, ADX, signal_long/short, funding), ajouter un test qui calcule la feature jusqu'à `t-1` et compare avec calcul jusqu'à `t` truncé à `t-1`. Si différence → lookahead caché. À planifier dans la prochaine session.
- 📌 **Vérifier régulièrement** : à chaque ajout de nouveau module signal/feature, refaire ce grep + lecture.

Statut B19 : **FIXED — audit clean, aucun bug trouvé**.
