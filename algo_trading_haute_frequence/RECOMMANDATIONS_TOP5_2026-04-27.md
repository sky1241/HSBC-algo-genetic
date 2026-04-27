# Recommandations Top 5 — Audit Layer 3 — 2026-04-27

> Synthèse actionnable issue de `RAPPORT_GAPS_LAYER3_2026-04-27.md` (audit professionnel, 16 sections nouvelles).
> Focus : ce qui manque encore après le refactor multi-symbole BTC/ETH/SOL d'aujourd'hui, pour atteindre des standards "boutique quant fund junior".
>
> **Verdict transverse** : ces ajouts ne CRÉENT PAS d'alpha. Ils permettent de :
> - (a) détecter plus vite l'absence d'edge
> - (b) gérer le risque agrégé multi-symbole correctement
> - (c) réduire les coûts cachés
>
> Cohérent avec le bilan quant : K3 nu Sharpe -1.91, aucune stratégie ne bat HODL spot.

## Tableau de priorisation

| # | Action | Priorité | Effort | Impact |
|---|--------|----------|--------|--------|
| 1 | `portfolio_risk.py` (corrélation + CVaR portefeuille + leverage cap 4x) | CRITICAL | 1-2j | ★★★★★ |
| 2 | `psr_live.py` (Probabilistic Sharpe Ratio live vs backtest) | CRITICAL | <1j | ★★★★★ |
| 3 | Funding-aware close-before-settlement | IMPORTANT | heures | ★★★★ |
| 4 | `pre_trade_gate.py` (5 règles FIA fat-finger) | CRITICAL | heures | ★★★★ |
| 5 | Implementation Shortfall complet (Perold) | IMPORTANT | 1j | ★★★ |

---

## #1 — `src/portfolio_risk.py` — gestion du risque agrégé multi-symbole

### Problème
Avec le refactor multi-symbole d'aujourd'hui :
- Worst case = 9 positions LONG (3 BTC + 3 ETH + 3 SOL) × 1% margin × levier moyen 92 = **8.3× equity en notional total**
- Avec rho ~0.85 BTC-ETH et ~0.80 BTC-SOL en stress (MDPI 2025), c'est **mathématiquement équivalent à 9 positions BTC** corrélées, pas à 9 positions indépendantes
- Sur drawdown -10% du panier crypto : -83% equity totale

La règle "1% margin par trade" est correcte par-trade mais NE PLAFONNE PAS le notional agrégé. Pas de "portfolio-level VaR" au sens FRTB.

### Solution
Créer `src/portfolio_risk.py` :

```python
class PortfolioRiskManager:
    rolling_corr_window_days = 30      # rho_ij(t) sur returns 1d
    portfolio_cvar_target = 0.05       # ES 97.5% < 5% equity
    aggregate_leverage_cap = 4.0       # somme(notional) / equity <= 4
    per_cluster_concentration_cap = 0.6 # max 60% du book dans 1 cluster

    def can_open(self, new_pos) -> Decision:
        # Recalcule covariance matrix BTC/ETH/SOL daily returns 720 bars
        # Stress: scale rho à max(observed, 0.85) si pump+5σ
        # Vérifie: portfolio_es_97_5 + new_pos_es <= cvar_target
        # Vérifie: agg_leverage_post <= cap
```

Brancher sur `RiskManager.calculate_position_size` → remplace le 1% statique par `min(1%, hrp_weight × leverage_budget)`.

### Sources
- López de Prado "Building Diversified Portfolios that Outperform Out-of-Sample", SSRN 2708678 (HRP)
- Burggraf, *Finance Research Letters* vol. 38, 2021 — HRP appliqué crypto, bat IV et minvar
- MDPI 2025 "Cryptocurrency Market Maturation and Evolving Risk Profiles" — diversification illusoire en stress
- Bucher & Osterrieder, SSRN 3858730 — Equal Risk Contribution sur 21 futures

---

## #2 — `src/psr_live.py` — Probabilistic Sharpe Ratio live vs backtest

### Problème
La stratégie K3 a un Sharpe backtest **-1.91**. En live, on veut savoir si :
- (a) le live RÉALISE bien ce Sharpe négatif (= modèle cohérent, juste pas d'edge)
- (b) le live est ENCORE PIRE que le backtest (= dégradation supplémentaire, à arrêter)

Sans PSR live, il faut **6 mois de données live** pour conclure statistiquement. Avec PSR rolling, **2-3 semaines suffisent**.

### Solution
Module `src/psr_live.py` :

```python
def psr(returns, sr_benchmark=0):
    """Bailey-López de Prado closed form."""
    sr_hat, skew, kurt, n = stats(returns)
    return norm.cdf(
        (sr_hat - sr_benchmark) * sqrt(n - 1) /
        sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2)
    )
```

Cron quotidien :
- `PSR_live = psr(returns_live_60j, sr_benchmark=0)`
- `PSR_bt = psr(returns_backtest_même_période_60j, 0)`
- Si `PSR_live < 0.5 PSR_bt` pendant **14j consécutifs** → alarme Telegram "live diverge backtest"

### Sources
- Bailey & López de Prado "The Sharpe Ratio Efficient Frontier", SSRN 1821643, 2012
- Bailey & López de Prado "Deflated Sharpe Ratio", SSRN 2460551

---

## #3 — Funding-aware close-before-settlement

### Problème
Settlements Binance à 00:00 / 08:00 / 16:00 UTC. Sur 3 LONG × 3 symboles × 3 settlements/jour = **9 funding payments potentiels par jour**.

Si funding moyen LONG positif (= on paie pour rester long) à 1 bps par 8h × 9 events = **~1% drag mensuel évitable**.

Le bot loggue le funding dans `paper_trader.csv` mais ne ferme PAS systématiquement avant settlement.

### Solution
Dans `binance_bot/bot/trade_manager.py`, ajouter :

```python
def funding_close_check(position, predicted_funding_rate):
    seconds_to_settlement = next_settlement_utc(now) - now
    if 30 < seconds_to_settlement < 300:
        cost_if_held = abs(predicted_funding_rate * notional)
        cost_to_flip = 2 * fees_taker * notional + slippage_estimate
        if cost_if_held > cost_to_flip * 1.5:  # safety margin
            return CloseAndReopen
    return Hold
```

Predicted funding via Binance `/fapi/v1/premiumIndex` → champ `lastFundingRate` (rate de la prochaine settlement, dispo 1h avant).

**À désactiver** si la stratégie ne tolère pas le close-reopen (impact signal). Ne s'applique que sur funding extrême (>5 bps annualisé par 8h).

### Sources
- Amberdata 2024 "Funding Rates: How They Impact Perpetual Swap Positions"
- BIS Working Paper 1087 "Crypto carry"

---

## #4 — `src/pre_trade_gate.py` — sanity checks fat-finger

### Problème
Aucun garde-fou pré-ordre côté bot. En cas de bug logique (signal forgé, bug arithmétique, données stale), le bot peut envoyer un ordre **catastrophiquement gros** ou **à un prix complètement aberrant**. Un prerequisite ABSOLU avant tout passage mainnet.

### Solution
Module `src/pre_trade_gate.py` avec **5 règles standard FIA** :

| Règle | Seuil |
|---|---|
| `max_qty_per_order` | 10× sizing nominal du symbole |
| `max_notional_per_symbol` | 30% du capital |
| `max_total_notional_portfolio` | aggregate_leverage_cap × equity |
| `price_band` | mid ± 5% |
| `stale_data_check` | refus si dernière candle > 4× timeframe |
| `message_rate_limit` | max 10 ordres/min par symbole |

Hook dans `TradeManager.execute_signal` AVANT `create_market_order`. Si gate refuse → log audit + Telegram critical + skip ordre.

### Sources
- FIA "Recommendations for Risk Controls for Trading Firms", 2024 white paper

---

## #5 — Implementation Shortfall complet (Perold)

### Problème
`paper_trader.py` loggue déjà fees, slippage, funding. Mais sans **décomposition Perold** complète, on ne sait pas si on perd à cause :
- (a) du **decision delay** (signal → submit, ex: 150ms)
- (b) du **execution slippage** (submit → fill, ex: 50ms)
- (c) des **fees**
- (d) du **funding**
- (e) de l'**opportunity cost** (signaux non-exécutés)

### Solution
Étendre le schema CSV de `paper_trader.py` avec :

```
signal_ts        # timestamp génération signal
submit_ts        # envoi ordre
fill_ts          # confirmation fill
decision_price   # prix au signal
arrival_price    # prix au submit
fill_price       # prix moyen rempli
next_bar_price   # prix bar suivante = paper price

IS_decision  = (arrival_price - decision_price) / decision_price * side
IS_execution = (fill_price - arrival_price) / arrival_price * side
IS_fees      = fees / notional
IS_funding   = cumulative_funding / notional
IS_total     = sum
```

Rapport hebdo : moyenne et p95 de chaque composante. Si `decision_delay > 2 bps` → optimiser la latence Python signal → submit.

### Sources
- Perold A.F. "The Implementation Shortfall: Paper versus Reality", *Journal of Portfolio Management* 1988
- Almgren et al. 2005 "Direct Estimation of Equity Market Impact"

---

## Hors top 5 — autres recommandations notables

Voir `RAPPORT_GAPS_LAYER3_2026-04-27.md` sections 8-16 pour le détail :

- **Section 8** : Liquidation cascade / ADL stress (priority IMPORTANT/NICE)
- **Section 10** : Counterparty risk Binance / Proof-of-Reserves (NICE testnet, CRITICAL mainnet)
- **Section 11** : Markov-Switching GARCH (NICE — K3 Fourier joue déjà ce rôle)
- **Section 14** : Vol targeting Moreira-Muir (module `vol_targeting.py` existe, à étendre)
- **Section 15** : Model risk governance Fed SR 11-7 (NICE, process)
- **Section 16** : VPIN / order flow toxicity (NICE)

**SKIP justifié** :
- Section 12 Almgren-Chriss : trop avancé pour $100 capital (utile dès $50k+ notional)
- Section 13 Cross-exchange arb : infra trop lourde pour retail

---

## Ordre d'implémentation suggéré

**Phase 1 (semaine 1)** — bloqueurs avant mainnet :
- #4 pre_trade_gate (heures)
- #3 funding_close_check (heures)
- #2 psr_live (1j)

**Phase 2 (semaine 2)** — risk multi-symbole :
- #1 portfolio_risk (1-2j)
- HRP weights weekly cron (1j)

**Phase 3 (semaine 3)** — observabilité fine :
- #5 Implementation Shortfall complet (1j)
- Alpha decay monitor CUSUM (1j)
- EWMA correlation regime detection (1j)

Total ≈ **2-3 semaines de dev** pour atteindre standard "boutique quant junior".

---

*Sources complètes (URLs vérifiées) : voir `RAPPORT_GAPS_LAYER3_2026-04-27.md`*
