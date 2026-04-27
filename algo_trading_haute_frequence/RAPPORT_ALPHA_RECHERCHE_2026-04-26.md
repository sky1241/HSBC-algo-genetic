# Rapport recherche ALPHA — Ichimoku + range filter (2026-04-26)

> Synthèse de recherche académique et industrielle sur la génération d'alpha
> avec Ichimoku en marché crypto en évitant ou exploitant les ranges.
> Ce fichier liste les hypothèses validables et celles à écarter.
> Le backtest empirique sur données BTC réelles est dans
> `outputs/wfa_alpha_2026-04-26/` (en cours, agent ALPHA-2).

## TL;DR — recommandations

| Priorité | Action | Confiance | Edge espéré |
|---|---|---|---|
| **P1** | Filtre **ER > 0.30 ET ADX > 22** avant entrée Ichimoku | **HAUTE** | Sharpe +0.3 à +0.6 |
| P2 | Veto **CHOP > 61.8** sur 6 bars suivants | MOYENNE | marginal cumulatif |
| P3 | **HMM 2-état** sur returns log H2 → P(trend) → sizing | MOY-HAUTE | Sortino +0.2 à +0.4 |
| ❌ P4 | Range trading actif (RSI extrêmes + BB reversion) | HAUTE pour NE PAS faire | edge net 5-15 bps trop fragile |
| ❌ P5 | Breakout TTM Squeeze / Donchian additionnel | HAUTE pour NE PAS faire | slippage brutal H2, gap backtest/live 20-40% |

**Edge total combiné P1+P2+P3 espéré** : +0.5 à +1.0 Sharpe vs baseline Ichimoku K3 nu, drawdown -20-30%.

Le gain principal vient de **NE PAS TRADER** quand le marché est en range, plus que d'extraire de l'alpha en range.

## 1. Range detection — qui marche en crypto

### Indicators classés par utilité crypto BTC

| Indicator | Formule | Seuils | Verdict crypto BTC |
|---|---|---|---|
| **ADX** (Wilder 1978) | Wilder smooth de DX | <20 range, >25 trend, >40 strong trend | Retard 14 périodes mais robuste. Combiné c'est un must. |
| **Choppiness Index** (Dreiss 1991) | `100·log₁₀(Σ TR / (HH-LL)) / log₁₀(n)` | >61.8 range, <38.2 trend (Fibonacci) | Bruité sur BTC daily. Veto secondaire OK. |
| **Efficiency Ratio** (Kaufman) | `\|close-close_n\|/Σ\|Δclose\|` | >0.30 trend, <0.20 range | **Le meilleur sans paramètre subjectif.** |
| **Bollinger BBW + squeeze** | `(BB_up - BB_low)/BB_mid` | bottom-percentile = squeeze | Robuste mais retardée. |
| **Hurst exponent** (R/S) | slope log-log de std vs lag | <0.5 mean-rev, >0.5 trend | BTC oscille autour de 0.5 — diagnostic offline, pas signal H2. |
| **Variance Ratio** (Lo-MacKinlay) | `Var(r_q)/[q·Var(r_1)]` | =1 random, ≠1 rejet | **Diagnostic offline only**, pas fiable H2. |

### Source clé
- Macrosynergy : "Detecting trends and mean reversion with Hurst exponent"
- MDPI Mathematics 13(10) 1577 (mai 2025) : HMM bayésien BTC stylized facts
- arXiv 2402.11930 : "Stylized Facts of HF Bitcoin time series"

## 2. Ichimoku + filtres — état académique

**État de la littérature : famélique académiquement.** Pas de papier peer-reviewed sérieux sur Ichimoku + regime filter en crypto.

- Liberatedstocktrader : "15 024 trades, win rate Ichimoku ~10%, sous-perf B&H 90% du temps" — chiffre frappant mais méthodologie opaque (à prendre avec prudence).
- Patton & Timmermann 2019 + Bailey/López de Prado **DSR** = références méthodo incontournables.
- Repo `ChakshuGupta13/technical-indicators-backtesting` (combo BIRA: ADX+Ichimoku+RSI) = point de départ utilisable, pas de claim d'edge net.

**Conclusion** : Ichimoku seul n'a pas d'edge documenté en crypto net de frais. Le filtre de régime est nécessaire.

## 3. Trading des ranges — alpha possible mais fragile

- **BB BTC/USDT (Efe Arda, SSRN 5775962, nov 2025)** : régime-dépendant. Mean reversion gagne en accumulation/bull, échoue en bear. Breakout solide partout sauf distribution. Chiffres exacts derrière paywall partiel.
- **RSI extrêmes + BB reversion** classique : edge brut 10-30 bps par trade. Net de fees Binance Futures (4 bps taker) + slippage 1-3 bps : **edge net plausible 5-25 bps** mais fragile.
- **Range expansion** (short top, long bottom) : ranges crypto rarement "propres" plus de quelques jours. Edge marginal.
- **Variance/vol arbitrage** : inaccessible retail (options Deribit + execution pro).

**Verdict** : reportable. Trop de complexité/fragilité pour le gain modéré.

## 4. Breakout strategies — slippage tue l'edge

- **TTM Squeeze** : profitable en backtest mais 50-65% de faux breakouts en BTC H2.
- **Donchian breakout (Turtle 20/55)** : profitable historiquement BTC daily. Sur H2 : dégradation forte par slippage.
- **NR4/NR7 (Crabel)** : adapté daily/intraday actions, peu pertinent BTC 24/7.
- **ATR breakout (Schwager)** : variante. Mêmes limites.

Live performance < backtest de 20-40% typique à cause du slippage taker forcé.

**Verdict** : reportable, à reconsidérer après P1-P3 validés.

## 5. Régime ML / HMM — voie la plus prometteuse

- **HMM Bayésien BTC** (MDPI Mathematics 2025) : 2-3 régimes robustes 2016-2024.
- **HMM > Markov-Switching > Threshold** sur détection bull/bear/neutre (Asian J Probability 2025).
- **Regime-Aware LightGBM** (MDPI Electronics 15(6) 1334) : HMM rolling + WFA 100 folds, **Sharpe portfolio 1.18 [CI 0.53, 1.84]** — edge réel mais modeste, intervalle large. Méthodo la plus propre identifiée.
- **Wavelet decomposition** : théoriquement intéressant mais peu de gains documentés vs HMM.

**Verdict** : HMM 2-3 états sur returns + volatility = la meilleure approche académique pour crypto. Implémentation pragmatique = HMM offline mensuel + LightGBM features (ADX, ER, Hurst, ATR%, BBW) en walk-forward.

## 6. Garde-fous méthodologiques (Bailey/López de Prado)

- Walk-forward strict, purge ≥ horizon de prédiction.
- **Deflated Sharpe Ratio** (déjà implémenté `src/stats_eval.py`) sur tout backtest avant deploy.
- **Hansen SPA** (déjà implémenté `src/reality_check.py`) pour multi-strat comparison.
- Frais Binance Futures réels : 4 bps taker, 2 bps maker, slippage 1-3 bps selon profondeur.
- Min track record (López de Prado) : ≥ 6 mois live testnet avant taille pleine.

## Sources principales

- [Choppiness Index — QuantifiedStrategies](https://www.quantifiedstrategies.com/choppiness-index/)
- [Detecting trends Hurst — Macrosynergy](https://macrosynergy.com/research/detecting-trends-and-mean-reversion-with-the-hurst-exponent/)
- [HMM Bitcoin Bayesian MCMC — MDPI 2025](https://www.mdpi.com/2227-7390/13/10/1577)
- [Regime-Aware LightGBM WFA — MDPI Electronics 2025](https://www.mdpi.com/2079-9292/15/6/1334)
- [Bollinger Bands BTC under regimes — Efe Arda SSRN 2025](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5775962)
- [Deflated Sharpe Ratio — Bailey/Lopez de Prado SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [Stylized Facts HF Bitcoin — arXiv 2402.11930](https://arxiv.org/html/2402.11930v2)
- [BIRA: ADX+Ichimoku+RSI — GitHub ChakshuGupta13](https://github.com/ChakshuGupta13/technical-indicators-backtesting)

## État implémentation

✅ Indicators implémentés dans `src/range_detector.py` :
- `adx`, `choppiness_index`, `efficiency_ratio`, `bollinger_band_width`, `bbw_squeeze`, `hurst_exponent`
- `range_score` composite avec poids ajustables
- 16 tests unitaires verts (`tests/test_range_detector.py`)

🔄 Backtest ALPHA-2 en cours (agent) :
- 4 stratégies comparées sur BTC H2 2020-2026 avec coûts réalistes
- Métriques DSR + Hansen SPA + Sortino + Ulcer + DD-duration
- Rapport dans `outputs/wfa_alpha_2026-04-26/`

⏳ À implémenter si validé empiriquement :
- Module `src/regime_filter.py` qui combine ER+ADX+CHOP
- Hook dans `signal_engine.py` pour bloquer signaux en range
- Test live sur testnet 2 semaines minimum avant scaling
