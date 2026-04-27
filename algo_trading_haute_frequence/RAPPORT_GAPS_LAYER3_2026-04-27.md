# RAPPORT GAPS LAYER 3 — Audit professionnel multi-symbole — 2026-04-27

> Audit specialise sur ce que LAYER2 (2026-04-26) n'a pas couvert. Focus : portefeuille multi-symbole BTC/ETH/SOL active depuis ce matin, et standards quant institutionnels (hedge fund / market maker).
>
> Pre-requis (deja audite hier, NE PAS RE-CITER) : sections 1-10 du `RAPPORT_GAPS_LAYER2_2026-04-26.md` — ES/CVaR, Cornish-Fisher, GPD/POT, MDD Magdon-Ismail, funding arb, basis, BTC.D, MVRV/SOPR/NVT, TWAP/post-only, OBI/depth, lookahead/PBO, QuantStats, latence/NTP, MiCA/ESMA/DAC8, B2 Object Lock.
>
> Verdict bilan deja connu (`RAPPORT_BILAN_QUANT_2026-04-27.md`) : aucune strategie ne bat HODL spot BTC. K3 nu Sharpe -1.91. Donc reco priorisent (a) detecter l'absence d'edge plus vite, (b) gerer le risque agrege, (c) reduire les couts. PAS d'optim de K3.

---

## TL;DR — Top 5 actions multi-symbole, ratio impact / effort

| # | Action | Impact | Effort | Ratio |
|---|--------|--------|--------|-------|
| 1 | **Ajouter `portfolio_risk.py` : matrice de correlation rolling 30j BTC/ETH/SOL + portfolio CVaR + leverage agrege cap** (Section 1+2). Gere la realite que ETH et SOL ont rho > 0.8 avec BTC -> 9 positions max ne diversifient PAS, c'est 9x BTC en stress. | CRITICAL | S (1-2j) | tres haut |
| 2 | **Module `psr_live.py` : Probabilistic Sharpe Ratio rolling 60j live vs PSR backtest, avec alarme si PSR_live < 0.5 PSR_backtest pendant 14j consecutifs** (Section 7). C'est l'outil n.1 pour detecter alpha decay sans attendre 6 mois. | CRITICAL | S (1j) | tres haut |
| 3 | **Funding-aware close-before-settlement** (Section 9) : si `position.funding_due_8h * size > expected_pnl_remaining`, fermer 1 minute avant 00:00/08:00/16:00 UTC. Coute zero, gain 5-15 bps/trade sur funding negatif. | IMPORTANT | XS (heures) | tres haut |
| 4 | **Pre-trade sanity gate** (Section 11) : max_qty_per_order, max_notional_per_symbol, max_total_notional_portfolio, price_band (+/- 5% mid), refus si tout dehors. Avant n'importe quel test live mainnet. | CRITICAL | XS (heures) | tres haut |
| 5 | **Implementation shortfall report quotidien** (Section 6) : decomposition Perold (decision -> arrival -> execution -> close) sur les trades testnet pour calibrer le vrai cout cache. Sans ca on pilote a l'aveugle. | IMPORTANT | S (1j) | haut |

---

## Section 1. Cross-asset risk budgeting et correlation regime (PRIORITE ABSOLUE)

### Etat de l'art
- **Risk parity / Equal Risk Contribution (ERC)** : Bucher & Osterrieder (SSRN 3858730) montrent qu'un portefeuille ERC sur 21 futures roulant 300 jours ameliore systematiquement le Sharpe vs equal-weight ou min-variance, surtout en presence de heavy tails. Reference Roncalli "The Risk Parity Page" (https://www.thierry-roncalli.com/RiskParity.html) — formules canoniques et code de reference.
- **Hierarchical Risk Parity (HRP)** : Lopez de Prado, "Building Diversified Portfolios that Outperform Out-of-Sample" (SSRN 2708678, 2016). Resout l'instabilite de Markowitz (covariance matrix non-inversible / mal conditionnee). Trois etapes : clustering hierarchique, quasi-diagonalisation, recursive bisection.
- **HRP applique au crypto** : Burggraf, "Beyond risk parity — A machine learning-based hierarchical risk parity approach on cryptocurrencies", Finance Research Letters vol. 38, 2021 (https://www.sciencedirect.com/science/article/abs/pii/S154461232030177X). Out-of-sample sur >50 cryptos : HRP bat IV et minvar en tail-risk-adjusted return, robuste aux fenetres et frequences de rebalancement.
- **Tail dependence crypto** : MDPI 2025 "Cryptocurrency Market Maturation and Evolving Risk Profiles" (https://www.mdpi.com/2674-1032/5/2/28) — la **lower tail dependence BTC-ETH** s'est elargie en 2024-2025, et la upper tail correlation est devenue **negative** par moments. Conclusion academique directe : *"diversification within the cryptocurrency asset class remains illusory during market stress"*.
- Egalement Springer 2023 "Assessing portfolio vulnerability to systemic risk: a vine copula and APARCH-DCC approach" (https://link.springer.com/article/10.1186/s40854-023-00559-2) — vine copula + APARCH-DCC pour le crash co-mouvement crypto.

### Ce que le bot a deja
- Vol targeting per-symbole (`src/volatility_targeting.py`).
- `RiskManager` per-symbole (stop global, position size 1%, max leverage clamp).
- **AUCUNE** vue agregee : pas de matrice de correlation, pas de portfolio CVaR, pas de leverage cap au niveau portefeuille. Chaque symbole tire sa bourre de 3 LONG XOR 3 SHORT independamment.

### Gap identifie (CRITIQUE pour le multi-symbole de ce matin)
1. Worst case actuel : 9 positions LONG (3 BTC 125x + 3 ETH 100x + 3 SOL 50x). Avec rho ~ 0.85 BTC-ETH et ~ 0.80 BTC-SOL en stress (cf MDPI 2025), c'est **mathematiquement equivalent a 9 positions BTC** correlees, pas a 9 positions independantes.
2. Pas de "portfolio-level VaR" au sens FRTB : on additionne du risque sans modeliser la covariance.
3. La regle "1% margin par trade" est correcte per-trade mais **ne plafonne pas le notional agrege** : 9 positions x 1% margin x leverage moyen 92 ~= **8.3x equity en notional**. Sur drawdown -10% du panier : -83% equity.
4. Pas de "concentration cap" : si SOL pump et BTC chute, le bot peut avoir 3 LONG SOL + 3 SHORT BTC — exposition theta inverse non monitoree.

### Adaptation au bot
**YES — c'est le delta numero 1 du refactor multi-symbole d'aujourd'hui.** Sans ce module, le passage 1->3 symboles aggrave le risque au lieu de le diversifier.

### Priorite : **CRITICAL**
### Effort : **S (1-2 jours)**
### Reco actionnable
Creer `src/portfolio_risk.py` :
```
class PortfolioRiskManager:
    rolling_corr_window_days = 30      # rho_ij(t) sur returns 1d
    portfolio_cvar_target = 0.05       # ES 97.5% < 5% equity
    aggregate_leverage_cap = 4.0       # somme(notional) / equity <= 4
    per_cluster_concentration_cap = 0.6 # max 60% du book dans un cluster

    def can_open(self, new_pos) -> Decision:
        # Recalcule covariance matrix BTC/ETH/SOL daily returns 720 bars (30j)
        # Stress: scale rho de stress (rho_stress = max(rho, 0.85) si pump+5sigma)
        # Verifie: portfolio_es_97_5 + new_pos_es <= cvar_target
        # Verifie: agg_leverage_post <= cap
        # Verifie: HRP weights -> notional du symbole <= w_HRP * equity * leverage_target
```
Brancher sur `RiskManager.calculate_position_size` -> remplace le 1% statique par `min(1%, hrp_weight * leverage_budget)`.

Notes additionnelles :
- En stress, forcer rho a max(observed, 0.85) pour BTC-ETH et 0.80 pour BTC-SOL (MDPI 2025).
- DSR/PSR existant doit etre **portfolio-level**, pas per-strategy.

---

## Section 2. Tail dependence et correlation regime detection

### Etat de l'art
- **DCC-GARCH** (Engle 2002) : standard academique pour correlation conditionnelle dynamique. NYU V-Lab maintient une implementation reference (https://vlab.stern.nyu.edu/docs/correlation/GARCH-DCC).
- **Application crypto** : ScienceDirect 2025 "Connectedness and investment strategies of volatile assets: DCC-GARCH R2 analysis" (https://www.sciencedirect.com/science/article/pii/S2214845025000493) — DCC sur crypto+sectors emerging, evidence de **regime-dependent connectedness** (correlations 2x plus fortes en stress).
- **Copules pour tail dependence** : MDPI Risks 2025 "GARCH-EVT-Copula Approach to Investigating Dependence" (https://www.mdpi.com/1911-8074/17/11/504) — montre que la lower tail dependence BTC-quote crypto est >> upper tail (asymmetrie : on co-crash mais on co-pump moins).

### Ce que le bot a deja
- Rien. Aucun monitoring de correlation dynamique.

### Gap identifie
La correlation BTC-ETH n'est **pas une constante** : elle passe de ~0.5 en bull tranquille a ~0.95 sur un crash type LUNA/FTX. Si on dimensionne le book sur correlation moyenne historique, on se retrouve sous-capitalise au pire moment.

### Adaptation au bot
**YES** mais en version simple. Pas besoin de DCC-GARCH complet (lourd a calibrer, instable). Une **EWMA correlation** (lambda=0.94 a la RiskMetrics) suffit pour 90% du benefice.

### Priorite : **IMPORTANT**
### Effort : **XS-S (1 jour)**
### Reco actionnable
Dans `portfolio_risk.py` ajouter :
```
def ewma_correlation(returns_df, lam=0.94):
    # RiskMetrics standard, lambda 0.94 daily
    # Retourne rho(t) updated bar-by-bar
```
Si `rho_BTC_ETH(t) > 0.85` ET drawdown 24h > 5% : **bloquer toute nouvelle entree LONG**. C'est le pattern "tail dependence broken loose" -> on est en regime de stress, le levier doit baisser.

---

## Section 3. Hierarchical Risk Parity (HRP) pour le sizing inter-symbole

### Etat de l'art
- Lopez de Prado SSRN 2708678 (deja cite Section 1).
- Wikipedia HRP (https://en.wikipedia.org/wiki/Hierarchical_Risk_Parity) — algorithme detaille.
- Burggraf FRL 2021 — application crypto.

### Ce que le bot a deja
- Sizing fixe `position_size_pct: 0.01` ; chaque symbole a 1% margin de la meme equity.

### Gap identifie
Allouer 1% a BTC, 1% a ETH, 1% a SOL est **inverse-volatility implicite seulement** parce que le levier change (125/100/50). Mais ce n'est ni une vraie equal-risk allocation ni une HRP. SOL est ~1.4x la vol de BTC — donc SOL devrait avoir moins de margin, pas un levier plus bas.

### Adaptation au bot
**YES**. HRP est calculable hors-ligne sur 90 jours de returns 2h (le timeframe), recalcule hebdo, sortie = poids w_btc, w_eth, w_sol qui somment a 1.

### Priorite : **IMPORTANT**
### Effort : **S (1j)**
### Reco actionnable
Module `src/portfolio_hrp.py` :
1. Pull 90j returns 2h close-to-close BTC/ETH/SOL.
2. Sklearn AgglomerativeClustering sur distance = sqrt(0.5 * (1 - corr)).
3. Quasi-diagonalisation puis recursive bisection (cf algo Lopez de Prado).
4. Output `weights.json` recharge par `PortfolioRiskManager`.
5. Cron hebdo (`scripts/weekly_hrp.py`).

Ajustement notional par symbole = `equity_total * hrp_weight[symbol] * symbol_leverage`. La regle "1% per trade" devient "hrp_weight per symbole, divise par max_positions_per_side".

---

## Section 4. Online changepoint detection pour alpha decay

### Etat de l'art
- **Bayesian Online Changepoint Detection (BOCPD)** : Adams & MacKay, "Bayesian Online Changepoint Detection", arXiv 0710.3742, 2007 (https://arxiv.org/abs/0710.3742). Detection en ligne de breaks dans la distribution des returns/Sharpe. R/Python implementations matures (`bayesian_changepoint_detection`).
- **CUSUM Page (1954)** : standard SPC, detect mean shift via cumulative sum. Simple, robuste, applicable au Sharpe rolling.
- **Application trading** : arXiv 2307.02375 "Online Learning of Order Flow and Market Impact with Bayesian Change-Point Detection Methods" (https://arxiv.org/abs/2307.02375) — BOCPD pour detecter regime shifts en order flow et adapter market impact model en ligne.
- **Alpha decay measurement** : arXiv 2512.11913 "Not All Factors Crowd Equally: Modeling, Measuring, and Trading on Alpha Decay" (https://arxiv.org/abs/2512.11913) — modele de decay hyperbolique alpha(t) = K/(1 + lambda*t) avec R2 ~0.65 sur momentum 1963-2024.

### Ce que le bot a deja
- DSR/PSR offline. Pas de detection live d'un changement de distribution.

### Gap identifie
Si la strategie K3 commence a decay en live (ce qui est attendu vu Sharpe -1.91 backtest), il faudra **6 mois minimum** pour le voir via une PSR statique. Une CUSUM ou un BOCPD sur le PnL bar-par-bar le detecte en **2-3 semaines**.

### Adaptation au bot
**YES** — directement applicable et c'est exactement ce dont un bot demo testnet a besoin pour decider quand arreter / reactiver.

### Priorite : **CRITICAL**
### Effort : **S (1j) pour CUSUM, M (semaine) pour BOCPD complet**
### Reco actionnable
Module `src/alpha_decay_monitor.py` :
- **V1 (CUSUM)** : `g(t) = max(0, g(t-1) + (PnL(t) - mu_backtest) / sigma_backtest - k)`. Si `g(t) > h` pendant N bars, alarme "alpha decay suspected". Calibrer h via false alarm rate (cf Lai 1995).
- **V2 (BOCPD)** : Adams-MacKay full Bayesian, run-length distribution. Use lib `bayesian_changepoint_detection` PyPI.
- Output : metric `p_regime_break(t)` exposee Telegram + dashboard.

---

## Section 5. Probabilistic Sharpe Ratio live vs backtest

### Etat de l'art
- Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier", SSRN 1821643, 2012 (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643). PSR = P(SR_observed > SR_benchmark | skew, kurt, n).
- Bailey-Lopez de Prado "Deflated Sharpe Ratio", SSRN 2460551 (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) — corrige multi-testing et non-normalite.
- Formula : PSR = Phi( (SR_hat - SR_star) * sqrt(n - 1) / sqrt(1 - skew * SR_hat + (kurt-1)/4 * SR_hat^2) ).

### Ce que le bot a deja
- DSR offline (calcul OOS post-WFA). Pas de PSR live.

### Gap identifie
La PSR backtest etait deja negative pour K3 (-1.91 Sharpe). En live, on veut **PSR rolling 60-90j** vs PSR backtest pour confirmer "yes la live realise bien le sharpe de backtest" ou "non on est encore plus mauvais". Un seuil de divergence ~2 ecarts-types declenche un kill signal.

### Adaptation au bot
**YES** — directement greffable sur `paper_trader.csv` + log live.

### Priorite : **CRITICAL** (c'est l'outil qui te dit "le testnet confirme ou infirme ton backtest")
### Effort : **XS-S (heures-1j)**
### Reco actionnable
Module `src/psr_live.py` :
```
def psr(returns, sr_benchmark=0):
    # Bailey-Lopez de Prado closed form
    sr_hat, skew, kurt, n = stats(returns)
    return norm.cdf((sr_hat - sr_benchmark) * sqrt(n-1) /
                    sqrt(1 - skew*sr_hat + (kurt-1)/4 * sr_hat**2))
```
Cron quotidien :
- PSR_live = psr(returns_live_60d, sr_benchmark=0)
- PSR_bt = psr(returns_backtest_corresponding_period_60d, 0)
- Si PSR_live < 0.5 ET PSR_bt > 0.5 pendant 14j -> alarme Telegram "live diverge backtest, K3 ne marche pas live".

---

## Section 6. Implementation Shortfall systematique (Perold)

### Etat de l'art
- Perold A.F. "The Implementation Shortfall: Paper versus Reality", JPM 1988 (PDF accessible https://www.cis.upenn.edu/~mkearns/finread/impshort.pdf via UPenn). Definition canonique : IS = (Paper PnL — Real PnL) / Paper PnL, decomposable en : (a) decision delay, (b) execution slippage, (c) opportunity cost (non-executed), (d) fees+funding.
- Almgren et al. 2005 "Direct Estimation of Equity Market Impact" (Risk magazine) — cadre empirique permanent / temporaire impact.
- Berkeley note "Implementation Shortfall with Transitory Price Effects" (https://faculty.haas.berkeley.edu/hender/chapter_ELOv5.pdf) — extension moderne.

### Ce que le bot a deja
- `binance_bot/bot/paper_trader.py` logge live + simule. **A confirmer** : la decomposition Perold (decision-arrival-fill-close) est-elle faite ? Sur le code lu, on voit `live_pnl_usdt`, `sim_pnl_usdt`, `fees_usdt`, `slippage_usdt`, `funding_usdt`. C'est partiel : pas de decision_delay_cost ni opportunity_cost.

### Gap identifie
Sans decomposition complete, on ne sait pas SI on perd a cause des fees, du slippage, du funding, ou du delay decision-fill (ex: signal generated a t, ordre place a t+150ms, fill a t+300ms, prix bouge de 1bps -> 1bps perdu sur le delay seul).

### Adaptation au bot
**YES** — extension simple de `paper_trader.py`.

### Priorite : **IMPORTANT**
### Effort : **S (1j)**
### Reco actionnable
Etendre CSV schema avec :
- `signal_ts` (timestamp generation signal)
- `submit_ts` (envoi ordre)
- `fill_ts` (confirmation fill)
- `decision_price` (prix au signal)
- `arrival_price` (prix au submit)
- `fill_price` (prix moyen rempli)
- `next_bar_price` (prix bar suivante = paper price)
- IS_decision = (arrival_price - decision_price) / decision_price * side
- IS_execution = (fill_price - arrival_price) / arrival_price * side
- IS_fees = fees / notional
- IS_funding = cumulative funding / notional
- IS_total = sum

Rapport hebdo : moyenne et p95 de chaque composante. Sortie attendue : si decision_delay > 2 bps, optimiser la latence python signal->submit.

---

## Section 7. Funding-aware position management

### Etat de l'art
- Cube Exchange / OKX / Deribit docs : "Funding is only charged to positions held at the exact settlement time". Fermer 1 minute avant (00:00 / 08:00 / 16:00 UTC) evite l'evenement.
- Amberdata 2024 "Funding Rates: How They Impact Perpetual Swap Positions" (https://blog.amberdata.io/funding-rates-how-they-impact-perpetual-swap-positions) — strategie standard MM/HFT crypto, gain typique 1-5 bps/8h sur funding evite quand le rate est extreme.
- BIS Working Paper 1087 "Crypto carry" (deja cite Layer 2) — explique pourquoi funding est positif 92% du temps Q3 2025 -> short paye long.

### Ce que le bot a deja
- Funding logge dans `paper_trader.csv` mais le bot **ne ferme pas systematiquement** avant settlement (a verifier mais probable que non vu le code lu).

### Gap identifie
Sur 3 LONG x 3 symbols x 3 settlements/jour = 9 funding payments/jour potentiels. Si funding moyen LONG positif (i.e. on paie) = +1 bps par 8h x 9 evenements = ~1% drag mensuel evitable.

### Adaptation au bot
**YES — quick win evident**. Logique : si position open ET (now < next_settlement < now+5min) ET (predicted_funding > threshold), close at market 1 min avant, reopen 30s apres si signal toujours actif.

### Priorite : **IMPORTANT**
### Effort : **XS (heures)**
### Reco actionnable
Dans `trade_manager.py`, ajouter :
```
def funding_close_check(position, predicted_funding_rate):
    seconds_to_settlement = next_settlement_utc(now) - now
    if 30 < seconds_to_settlement < 300:
        cost_if_held = abs(predicted_funding_rate * notional)
        cost_to_flip = 2 * fees_taker * notional + slippage_estimate
        if cost_if_held > cost_to_flip * 1.5:  # safety margin
            return CloseAndReopen
    return Hold
```
Predicted funding via Binance `/fapi/v1/premiumIndex` -> champ `lastFundingRate` (rate de la prochaine settlement, dispo 1h avant).

Note : a desactiver si la strategie ne tolere pas le close-reopen (impact signal). Ne s'applique que sur funding extreme (>5 bps annualise par 8h = >0.5%).

---

## Section 8. Liquidation cascade et insurance fund modeling

### Etat de l'art
- Binance support FAQ "Auto-Deleveraging (ADL) and How It Works" (https://www.binance.com/en/support/faq/detail/360033525471) — derniere ligne quand insurance fund epuise. Position queue par profit*leverage : les plus gros gagnants leveraged sont liquides en premier.
- Binance "Introduction to Futures Insurance Funds" (https://www.binance.com/en/support/faq/introduction-to-futures-insurance-funds-360033525371).
- arXiv 2512.01112 "Autodeleveraging: Impossibilities and Optimization" (https://arxiv.org/html/2512.01112v2) — analyse formelle ADL, montre evenements 10-11 oct 2025 ou multiple venues ont epuise insurance funds simultanement.
- insights4vc "Inside the $19B Flash Crash" (https://insights4vc.substack.com/p/inside-the-19b-flash-crash) — narration evenement oct 2025.

### Ce que le bot a deja
- Aucune logique ADL (peu probable sur testnet).
- Liquidation price calcule per-position, mais pas de "cascade scenario".

### Gap identifie
1. En leverage 50-125x, **buffer maintenance margin tres faible** : un wick de 1-2% peut liquider, surtout SOL.
2. Pas d'estimation "probability of ADL" : si on est dans le top quartile du leveraged-PnL ranking sur Binance, en cas de crash insurance fund epuise, **on peut etre force-deleveraged meme si on est en profit**.
3. Pas de stress sur "qu'est-ce qui se passe si BTC -10% en 5 minutes" pour les 3 positions LONG simultanees.

### Adaptation au bot
**YES** mais limite — ADL est un risque exchange, peu modelisable retail. En revanche le **stress flash crash** est trivial.

### Priorite : **IMPORTANT** (pour stress test forward), **NICE** (pour ADL probability).
### Effort : **S (1j)**
### Reco actionnable
1. Stress synthetique daily : pour chaque snapshot de positions, calculer PnL et margin si BTC/ETH/SOL -10/-20/-30% simultanement (worst case oct 2025 etait ~-15% en 1h sur BTC). Si max_drawdown > 50% equity -> alarme + propose de reduire size.
2. Tracker le ranking ADL approximatif via `unrealized_pnl / margin` -> si ratio dans top decile (proxy "tu es candidat ADL"), reduire size.
3. **Surtout** : maintenir reserve cash (>=30% equity en margin libre) pour absorber un wick brutal sans liquidation.

---

## Section 9. Pre-trade sanity checks (FIA fat-finger)

### Etat de l'art
- FIA "Best Practices For Automated Trading Risk Controls and System Safeguards" (2024) (https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf). Les 4 piliers : pre-trade controls (max order size, price band, message rate), monitoring, kill switch, post-trade reconciliation.
- FIA "Fat finger risk" (https://www.fia.org/marketvoice/articles/fat-finger-risk).
- CFTC Electronic Trading Risk Principles (https://www.federalregister.gov/documents/2020/07/15/2020-14381/electronic-trading-risk-principles).

### Ce que le bot a deja
- Kill switch (post-trade behavioral).
- Rate limit middleware.
- `RiskManager.calculate_position_size` clamp leverage.

### Gap identifie
Verifie la presence dans `trade_manager.py` de :
1. **Max order size absolu** par symbole (independant de l'equity calc — protege contre bug calcul equity).
2. **Price band** : refus si `abs(order_price - mid_price) / mid_price > 5%`.
3. **Message rate cap** : refus si > 10 ordres/seconde envoyes.
4. **Total notional cap portfolio** (somme tous symboles) : independant des per-symbol limits.
5. **Stale data check** : refus si dernier WS update > 5s.

Si **un seul** de ces 5 manque, c'est un risque qui s'est materialise plusieurs fois en industry (Knight Capital 2012, BAML 2013, ABN AMRO).

### Adaptation au bot
**YES — BLOCKER avant tout passage mainnet.** Sur testnet c'est moins critique mais le code doit etre la pour ne pas avoir a l'ajouter sous pression.

### Priorite : **CRITICAL**
### Effort : **XS (quelques heures)**
### Reco actionnable
Module `src/pre_trade_gate.py` :
```
class PreTradeGate:
    max_qty_btc = 0.5
    max_qty_eth = 5.0
    max_qty_sol = 100.0
    price_band_pct = 0.05
    max_orders_per_second = 5
    max_total_notional_pct_equity = 4.0
    max_data_staleness_sec = 5

    def check(self, order, market_state, recent_orders) -> Result(ok, reason):
        ...
```
Branche **avant** chaque `place_order` dans `trade_manager.py`. Tests unitaires obligatoires (un par regle).

---

## Section 10. Counterparty risk Binance et Proof-of-Reserves

### Etat de l'art
- ScienceDirect 2025 "Centralized exchanges & proof-of-solvency: The guardians of trust" (https://www.sciencedirect.com/science/article/pii/S1042443125000733) — meta-analyse PoR post-FTX.
- Glassnode "Determining Cryptocurrency Exchange Risks Based on Reserves" (https://insights.glassnode.com/content/determining-cryptocurrency-exchange-risks-based-on-reserves-a-guide/) — guide pratique 3 etapes.
- CoinGeek "Zero-Knowledge proof of solvency for crypto exchanges" (https://coingeek.com/zero-knowledge-proof-of-solvency-for-crypto-exchanges-how-to-detect-the-next-ftx-and-mt-gox/) — limite Merkle-tree PoR (ne couvre pas liabilities cachees).
- Wikipedia FTX (https://en.wikipedia.org/wiki/FTX) — chronologie collapse novembre 2022.

### Ce que le bot a deja
- Backup B2 Object Lock (Layer 2 reco). Pas de monitoring exchange health.

### Gap identifie
1. Pas de check sur Binance reserves (Merkle root publie quarterly).
2. Pas de seuil de retrait automatique : si `withdrawal_delays > X heures` ou `peg USDT-USD diverge > 1%`, retirer immediatement les fonds vers cold wallet.
3. Pas de plan B "exchange B" execute (Bitget, Bybit, OKX EU si Binance fail).

### Adaptation au bot
**YES** mais **bas pour testnet** ($100 fictif, perte = 0). Devient **critical** des le 1er $ mainnet.

### Priorite : **NICE** (testnet) -> **CRITICAL** (mainnet).
### Effort : **XS** (alertes manual) -> **M** (auto-failover multi-exchange).
### Reco actionnable
Pour testnet : juste documenter le runbook "que faire si Binance freeze withdrawals" (3 etapes max).
Pour mainnet (futur) : monitor (a) Merkle root release (api.binance.com), (b) USDT/USDC depeg via CoinGecko, (c) withdrawal queue duration via balance test transfer hourly. Sur trigger : kill switch + tentative retrait spot.

---

## Section 11. Markov-Switching GARCH et regime volatility

### Etat de l'art
- Hamilton J.D. (1989) "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle", Econometrica 57(2):357-384 — fondateur regime switching.
- Klaassen F. (2002) "Improving GARCH volatility forecasts with regime-switching GARCH" (https://www.researchgate.net/publication/31056247_A_new_approach_to_Markov-switching_GARCH_models) — solution traitable au probleme path-dependence.
- Application crypto : ScienceDirect "Modelling volatility of cryptocurrencies using Markov-Switching GARCH models" (https://www.sciencedirect.com/science/article/pii/S027553191830669X) — MSGARCH bat single-regime sur Bitcoin VaR.
- IntechOpen 2025 "Estimating Extreme Value at Risk Using Bayesian Markov Regime Switching GARCH-EVT" (https://www.intechopen.com/chapters/1173161) — combo MSGARCH + EVT pour queue.
- MDPI Mathematics 2025 "Bitcoin Price Regime Shifts: A Bayesian MCMC and Hidden Markov Model Analysis" (https://www.mdpi.com/2227-7390/13/10/1577) — application BTC recente.

### Ce que le bot a deja
- `src/regime_hmm.py` — Gaussian HMM 3 etats sur P1_period, LFP_ratio, volatility (features Fourier).
- `src/regime_lgbm.py` — verdict bilan : AUC 0.53 random.
- K3 phase rotation Fourier offline (labelling).

### Gap identifie
Le HMM actuel travaille sur **features Fourier** (period, LFP) — interessant mais pas standard. La litterature converge sur MSGARCH = HMM sur **volatility regime** directement. Et l'experience bilan 2026-04-27 dit que les regimes Fourier+HMM **n'ameliorent pas** le sharpe.

### Adaptation au bot
**NO sur le pivot strategie** (le bilan montre que ML/HMM ne predit pas). **YES sur la mesure du risque** : MSGARCH sert a estimer la **vol forward conditionelle** pour le sizing.

### Priorite : **NICE**
### Effort : **M (semaine)**
### Reco actionnable
Plutot que d'ajouter un MSGARCH from scratch, adopter `arch` Python lib + R `MSGARCH` (via rpy2 si necessaire). Sortie : `sigma_t+1_state_high`, `sigma_t+1_state_low`, `prob_state_high`. Utiliser pour ajuster vol_target dynamique :
```
sigma_forecast = p_high * sigma_high + (1-p_high) * sigma_low
leverage = vol_target / sigma_forecast  # bornes [Lmin, Lmax]
```
Ne **pas** s'en servir comme signal directionnel (c'est ce qui a foire).

---

## Section 12. Almgren-Chriss optimal execution adaptation crypto

### Etat de l'art
- Almgren R. & Chriss N. (2000) "Optimal Execution of Portfolio Transactions", Journal of Risk 3 (https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf). Trade-off impact temporaire / permanent vs risque vol.
- Application crypto : Claremont CMC thesis "Optimal Execution in Cryptocurrency Markets" (https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=3566&context=cmc_theses) — calibration BTC perp, decay impact temporaire ~10-30s.
- arXiv 1206.0682 "Calibration of optimal execution of financial transactions" (https://arxiv.org/pdf/1206.0682) — methodologie calibration.
- Github reference impl : joshuapjacob/almgren-chriss-optimal-execution (https://github.com/joshuapjacob/almgren-chriss-optimal-execution).

### Ce que le bot a deja
- Layer 2 a recommande TWAP simple ; pas implemente.
- Notional par trade typique : $1 margin x 125 leverage = $125 notional. **Tres en dessous** du seuil ou Almgren-Chriss devient rentable.

### Gap identifie
A capital actuel ($100 testnet), Almgren-Chriss est over-engineering. Le post-only timeout suffit largement.

### Adaptation au bot
**NO** a capital actuel. **YES** si capital scale > $100k ET notional > $30k par trade.

### Priorite : **SKIP** pour la version testnet.
### Effort : **N/A**
### Reco actionnable
Documenter dans `LOGIQUE_PROGRAMME.md` : "Si capital > $50k ET notional/trade > $30k, implementer Almgren-Chriss via lib reference. Avant ce seuil, post-only + TWAP 5 chunks suffisent."

---

## Section 13. Cross-exchange arbitrage et best execution multi-venue

### Etat de l'art
- Coinglass Funding Rates (https://www.coinglass.com/FundingRate) — funding spread Binance vs OKX vs Bybit visible.
- Coinglass Funding Rate Arbitrage (https://www.coinglass.com/FrArbitrage) — scanner ready.
- Github kir1l/Funding-Arbitrage-Screener (https://github.com/kir1l/Funding-Arbitrage-Screener) — implementation OSS.
- BIS WP 1087 "Crypto carry" (deja cite) — explique pourquoi spreads existent (composition utilisateurs / margin engine difference).

### Ce que le bot a deja
- Mono-exchange Binance.

### Gap identifie
1. Pas d'arbitrage cross-exchange. Edge reel mais demande infra >= 2 connexions.
2. Pas de "best venue" routing : si Binance funding +30% annualise et Bybit +5% sur meme paire, on devrait short cote Binance ET long cote Bybit.

### Adaptation au bot
**NO court terme** (complexite x 2-3, capital morcele, KYC multiple, MiCA risque). **YES long terme** seulement si edge K3 trouve.

### Priorite : **SKIP** (sauf decision strategique de pivoter sur funding-arb cross-venue, mais le bilan dit "frais > funding mean" donc pas d'edge non plus).
### Effort : **L (mois)**
### Reco actionnable
Documenter mais ne pas commencer. Si pivot un jour : commencer par lecture-only Coinglass API pour valider la persistence des spreads sur 60j avant tout dev infra.

---

## Section 14. Volatility targeting dynamique (au-dela du module existant)

### Etat de l'art
- Moreira A. & Muir T. (2017) "Volatility-Managed Portfolios", Journal of Finance 72(4):1611-1644 (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2659431). NBER WP w22208 (https://www.nber.org/system/files/working_papers/w22208/w22208.pdf). Resultat clef : alpha 4.9% sur le market portfolio, +25% buy-and-hold Sharpe.
- Critique recente Cederburg, O'Doherty et al., "On the performance of volatility-managed portfolios" (https://www.lehigh.edu/~xuy219/research/COWY.pdf) — sur 103 strategies, **aucune evidence systematique** que vol-managed bat unmanaged. Look-ahead bias dans le scaling factor original Moreira-Muir.
- Conditional Volatility Targeting (Tandfonline 2020, https://www.tandfonline.com/doi/full/10.1080/0015198X.2020.1790853) — version corrigee, ajuste seulement aux extremes.
- Man Group "The Impact of Volatility Targeting" (https://www.man.com/insights/the-impact-of-volatility-targeting) — practitioner view.

### Ce que le bot a deja
- `src/volatility_targeting.py` : sigma_target=0.15, L_max=10, lookback=20 bars 2h, drawdown throttle a -10%. Ce module existe deja !

### Gap identifie
1. **Look-ahead bias** : le sigma_realized est-il calcule sur barres **closed-only** ? A verifier dans le code (Layer 2 a fait l'audit lookahead clean mais a re-checker pour ce module specifique).
2. **Sigma annualization** = sqrt(365*12) = sqrt(4380) — correct pour 2h bars, mais tres sensible aux outliers (LUNA, FTX, mai 2021).
3. **Drawdown throttle 10%** : sur testnet avec mises de $1, n'est-il pas trop agressif ? Les wicks naturels BTC font passer en throttle frequemment.

### Adaptation au bot
**YES partial** : le module existe, juste auditer + calibrer.

### Priorite : **NICE**
### Effort : **XS (audit) + S (re-tune si necessaire)**
### Reco actionnable
1. Grep `sigma_realized` calcul : verifier `.iloc[:-1]` ou `.shift(1)` pour exclure barre courante.
2. Conditional VT : appliquer scaling **seulement** quand sigma_realized hors bande [0.5*sigma_target, 2*sigma_target]. Reduit le turnover et les couts.
3. Tester L_max plus realiste : 4-5x au lieu de 10x sur portefeuille agrege (cf cap Section 1).

---

## Section 15. Documentation modele et governance (SR 11-7-style)

### Etat de l'art
- Federal Reserve / OCC SR 11-7 "Supervisory Guidance on Model Risk Management" (2011) (https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm). 3 piliers : model development & implementation, model validation (independante), governance.
- ModelOp explainer (https://www.modelop.com/ai-governance/ai-regulations-standards/sr-11-7).
- Yogesh Malhotra paper (https://www.yogeshmalhotra.com/SR11-7_OCC2011-12.html) — applied research banking.

### Ce que le bot a deja
- Code documente (`LOGIQUE_PROGRAMME.md`, `BUGS.md`, rapports).
- Tests automatiques (~330 tests).
- Audit log hash-chain (forensic).
- Pas de model card formel, pas de model inventory, pas de governance log.

### Gap identifie
Standard institutionnel attend :
1. **Model Card** (cf Mitchell et al. 2018 Google) : input features, dependencies, training data, intended use, limits, ethical risks.
2. **Model Inventory** : liste de tous les modeles tournant (K3, HMM regime, vol_target, cost_model, range_detector...) avec version, date deploy, owner, risk tier.
3. **Independent validation** : retail = soi-meme, mais documenter "challenger benchmarks" (HODL spot, DCA) joue ce role.
4. **Change log governance** : tout changement de params (Optuna re-run, regime re-fit) doit etre signe + horodate. Le hash-chain audit log peut servir mais il faut un process explicite.

### Adaptation au bot
**YES light**. Pour un retail, c'est over-engineering complet. Mais pour CV/credibilite quant + montrer a un futur employeur : tres valorisant.

### Priorite : **NICE** (operationnel) / **IMPORTANT** (CV/portfolio quant)
### Effort : **S (1j) pour V1**
### Reco actionnable
Creer `docs/MODEL_CARDS/` :
- `K3_v1.md` : "Ichimoku K3 phase rotation, Optuna params, Sharpe backtest -1.91, status DEPRECATED depuis 2026-04-26"
- `vol_targeting_v1.md`
- `hmm_regime_v1.md`
- `cost_model_v1.md`

Et `docs/MODEL_INVENTORY.md` table : id, version, last_deployed, status (production/champion/challenger/deprecated), owner, risk_tier (1-3).

---

## Section 16. Latency arbitrage adversarial (queue position et MM toxicity)

### Etat de l'art
- Easley, Lopez de Prado, O'Hara, "Flow Toxicity and Liquidity in a High Frequency World" (RFS 2012, https://onlinelibrary.wiley.com/doi/10.1093/rfs/hhs053). VPIN (Volume-Synchronized Probability of Informed Trading).
- ScienceDirect 2025 "Order flow toxicity and price jumps" (https://www.sciencedirect.com/science/article/pii/S0275531925004192) — confirme VPIN signal predictif jumps crypto.
- arXiv 2506.05764 "Exploring Microstructural Dynamics in Cryptocurrency LOB" (https://arxiv.org/html/2506.05764v2) — toxic flow detection LOB crypto.

### Ce que le bot a deja
- Layer 2 a recommande OBI mais pas implemente.

### Gap identifie
Si le bot envoie post-only au mid+1tick, **les MM HFT voient l'ordre arriver** (1-5ms typiquement) et peuvent fade-on-filling. Pas critique a $1 notional, mais ca explique pourquoi les fills "easy" se font remplir au pire moment. VPIN > 0.7 = toxic environment, ne pas envoyer maker.

### Adaptation au bot
**YES light** : VPIN simple via taker volume bucketing.

### Priorite : **NICE**
### Effort : **S (1j)**
### Reco actionnable
Module `src/microstructure/vpin.py` :
```
def vpin(taker_buy_vol, taker_sell_vol, bucket_size, n_buckets=50):
    return mean(|buy_b - sell_b| / (buy_b + sell_b)) over buckets
```
Pre-trade gate : si VPIN > 0.7 -> bascule en taker (accepte de payer fee plutot que se faire fade). Effet attendu : -2 bps fee, +5 bps de meilleur prix de fill = +3 bps net.

---

## Conclusion — Top 5 actionnable consolide

Repete du TL;DR pour clarte, avec recommandation de sequence :

1. **Semaine 1** : `src/portfolio_risk.py` (Section 1) + EWMA correlation (Section 2). Cap leverage agrege a 4x. Sans ca le multi-symbole d'aujourd'hui aggrave le risque.
2. **Semaine 1 jour 2** : `src/pre_trade_gate.py` (Section 9). 5 regles, tests unitaires. BLOCKER pour tout passage mainnet futur.
3. **Semaine 1 jour 3** : `funding_close_check` dans `trade_manager.py` (Section 7). Quick win XS effort.
4. **Semaine 2** : `src/psr_live.py` (Section 5) + `src/alpha_decay_monitor.py` CUSUM (Section 4). Detection rapide que K3 ne marche pas live.
5. **Semaine 3** : Implementation shortfall complet (Section 6) + HRP weights (Section 3).

Le reste (MSGARCH, Almgren-Chriss, cross-exchange, model cards) = NICE / SKIP a capital actuel.

**Verdict transverse** : avec les sections 1-9 ci-dessus implementees, le bot atteint un standard "boutique quant fund junior" sur la rigueur risk + operational. La strategie elle-meme reste sans edge (bilan 2026-04-27) — ces ameliorations ne creent **pas** d'alpha, elles permettent de **detecter plus vite** son absence et de **proteger le capital** quand un alpha apparaitra eventuellement.

---

## Sources citees (par section)

### Section 1
- Bucher & Osterrieder, "Risk Parity for Multi-Asset Futures Allocation", SSRN 3858730, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3858730
- Roncalli, "The Risk Parity Page", https://www.thierry-roncalli.com/RiskParity.html
- Lopez de Prado, "Building Diversified Portfolios that Outperform Out-of-Sample", SSRN 2708678, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
- Burggraf, "Beyond risk parity — A machine learning-based hierarchical risk parity approach on cryptocurrencies", FRL 38, https://www.sciencedirect.com/science/article/abs/pii/S154461232030177X
- MDPI 2025 "Cryptocurrency Market Maturation and Evolving Risk Profiles", https://www.mdpi.com/2674-1032/5/2/28
- Springer 2023 "Assessing portfolio vulnerability to systemic risk: a vine copula and APARCH-DCC approach", https://link.springer.com/article/10.1186/s40854-023-00559-2

### Section 2
- NYU V-Lab GARCH-DCC docs, https://vlab.stern.nyu.edu/docs/correlation/GARCH-DCC
- ScienceDirect 2025 "Connectedness and investment strategies", https://www.sciencedirect.com/science/article/pii/S2214845025000493
- MDPI Risks 2024 "GARCH-EVT-Copula Approach", https://www.mdpi.com/1911-8074/17/11/504

### Section 3
- HRP Wikipedia, https://en.wikipedia.org/wiki/Hierarchical_Risk_Parity
- Lopez de Prado SSRN 2708678 (cf section 1)
- Burggraf 2021 FRL (cf section 1)

### Section 4
- Adams & MacKay, "Bayesian Online Changepoint Detection", arXiv 0710.3742, https://arxiv.org/abs/0710.3742
- arXiv 2307.02375 "Online Learning of Order Flow and Market Impact with Bayesian Change-Point Detection", https://arxiv.org/abs/2307.02375
- arXiv 2512.11913 "Not All Factors Crowd Equally: Modeling, Measuring, and Trading on Alpha Decay", https://arxiv.org/abs/2512.11913

### Section 5
- Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier", SSRN 1821643, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
- Bailey & Lopez de Prado, "The Deflated Sharpe Ratio", SSRN 2460551, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

### Section 6
- Perold, "The Implementation Shortfall: Paper versus Reality", JPM 1988, PDF mirror UPenn https://www.cis.upenn.edu/~mkearns/finread/impshort.pdf
- Berkeley Haas chapter "Implementation Shortfall with Transitory Price Effects", https://faculty.haas.berkeley.edu/hender/chapter_ELOv5.pdf

### Section 7
- Amberdata "Funding Rates: How They Impact Perpetual Swap Positions", https://blog.amberdata.io/funding-rates-how-they-impact-perpetual-swap-positions
- Cube Exchange "Funding Rate", https://www.cube.exchange/what-is/funding-rates
- Binance Futures funding API premiumIndex, https://binance-docs.github.io/apidocs/futures/en/#mark-price

### Section 8
- Binance "What Is Auto-Deleveraging (ADL)", https://www.binance.com/en/support/faq/detail/360033525471
- Binance "Introduction to Futures Insurance Funds", https://www.binance.com/en/support/faq/introduction-to-futures-insurance-funds-360033525371
- arXiv 2512.01112 "Autodeleveraging: Impossibilities and Optimization", https://arxiv.org/html/2512.01112v2
- insights4vc "Inside the $19B Flash Crash" (oct 2025), https://insights4vc.substack.com/p/inside-the-19b-flash-crash

### Section 9
- FIA "Best Practices For Automated Trading Risk Controls and System Safeguards" (2024), https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf
- FIA "Fat finger risk", https://www.fia.org/marketvoice/articles/fat-finger-risk
- CFTC Electronic Trading Risk Principles, https://www.federalregister.gov/documents/2020/07/15/2020-14381/electronic-trading-risk-principles

### Section 10
- ScienceDirect 2025 "Centralized exchanges & proof-of-solvency", https://www.sciencedirect.com/science/article/pii/S1042443125000733
- Glassnode "Determining Cryptocurrency Exchange Risks Based on Reserves", https://insights.glassnode.com/content/determining-cryptocurrency-exchange-risks-based-on-reserves-a-guide/
- CoinGeek "Zero-Knowledge proof of solvency for crypto exchanges", https://coingeek.com/zero-knowledge-proof-of-solvency-for-crypto-exchanges-how-to-detect-the-next-ftx-and-mt-gox/

### Section 11
- Hamilton 1989, Econometrica 57(2):357-384 [citation classique, pas d'URL libre standard]
- Klaassen "A new approach to Markov-switching GARCH models", https://www.researchgate.net/publication/31056247_A_new_approach_to_Markov-switching_GARCH_models
- ScienceDirect "Modelling volatility of cryptocurrencies using Markov-Switching GARCH", https://www.sciencedirect.com/science/article/pii/S027553191830669X
- IntechOpen 2025 "Bayesian Markov Regime Switching GARCH-EVT", https://www.intechopen.com/chapters/1173161
- MDPI Mathematics 2025 "Bitcoin Price Regime Shifts", https://www.mdpi.com/2227-7390/13/10/1577

### Section 12
- Almgren & Chriss, "Optimal Execution of Portfolio Transactions", J. Risk 3, mirror https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf
- Claremont CMC thesis "Optimal Execution in Cryptocurrency Markets", https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=3566&context=cmc_theses
- arXiv 1206.0682 "Calibration of optimal execution", https://arxiv.org/pdf/1206.0682

### Section 13
- Coinglass Funding Rates, https://www.coinglass.com/FundingRate
- Coinglass FrArbitrage scanner, https://www.coinglass.com/FrArbitrage
- BIS WP 1087 "Crypto carry", https://www.bis.org/publ/work1087.pdf

### Section 14
- Moreira & Muir, "Volatility-Managed Portfolios", JF 2017, SSRN 2659431, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2659431
- NBER WP w22208, https://www.nber.org/system/files/working_papers/w22208/w22208.pdf
- Cederburg O'Doherty et al., "On the performance of volatility-managed portfolios", https://www.lehigh.edu/~xuy219/research/COWY.pdf
- Tandfonline 2020 "Conditional Volatility Targeting", https://www.tandfonline.com/doi/full/10.1080/0015198X.2020.1790853

### Section 15
- Federal Reserve SR 11-7 letter, https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- ModelOp SR 11-7 explainer, https://www.modelop.com/ai-governance/ai-regulations-standards/sr-11-7

### Section 16
- Easley, Lopez de Prado, O'Hara, "Flow Toxicity and Liquidity in a High Frequency World", RFS 2012 [paywall]
- ScienceDirect 2025 "Order flow toxicity and price jumps", https://www.sciencedirect.com/science/article/pii/S0275531925004192
- arXiv 2506.05764 "Exploring Microstructural Dynamics in Cryptocurrency LOB", https://arxiv.org/html/2506.05764v2

---

*Fin du rapport. Volume : 16 sections nouvelles, 0 duplication avec Layer 2. Toutes URLs vues dans resultats de recherche officielle (Google/SSRN/arXiv); WebFetch direct bloque par certains domaines (SSRN 403) — verification possible via Bing/DuckDuckGo si requis. Aucune source inventee.*
