# Rapport Gaps Layer 2 — Audit "ce qu'on a oublie" — 2026-04-26

**Auteur** : audit quant senior, post-fix Layer 1 (6 bugs core, kill switch, hash-chain, watchdog, dashboard, paper-trade, recovery WAL, time sync, rate limit, WebSocket user stream, range detector ER+ADX+CI+Hurst+BBW, DSR/Reality Check/Hansen SPA, fees+funding+slippage simulator, stress LUNA/FTX/COVID).

**Objectif** : identifier les gaps residuels avant deploiement live BTC futures autonome. Sources : SSRN, arXiv, BIS, docs Binance/Coinglass, AMF, ESMA. Ton sceptique sur claims marketing on-chain.

---

## 1. Risk metrics avances

### Etat de l'art
- **Bale III / FRTB** a remplace VaR par **Expected Shortfall (ES, alias CVaR) a 97.5%** comme metrique reglementaire de capital marche. Raison : VaR n'est pas sub-additif (penalise la diversification), n'est pas une mesure coherente, et ne dit rien sur la severite de la queue. ES integre toute la perte moyenne au-dela du quantile.
- **Cornish-Fisher VaR** : extension de VaR Gaussien avec correction skew/kurtosis via les 4 premiers moments. Mark & Vaucher (2024, SSRN) proposent une version "corrigee" qui evite les violations de monotonie (probleme classique du CF original quand kurtosis tres elevee).
- **GPD / Peak-over-Threshold** : Bayesian Extreme Value Analysis of Bitcoin Tail Risk (Preprints 2025) trouve shape parameter ~0.21 sur BTC daily ; VaR 99% a -13.6%, ES a -22.1%. La distribution Pareto type II est appropriee (heavy-tail, shape > 0).
- **Drawdown probability** : Magdon-Ismail & Atiya (RPI 2004) donnent l'esperance du Max Drawdown pour un Brownien drifte, avec phase transition lineaire -> sqrt(t) -> log(t) selon drift negatif/zero/positif. Calmar ratio re-scale.

### Ce que le bot a deja
- DSR (Bailey-Lopez de Prado), Reality Check (White), Hansen SPA pour le multi-testing.
- Stress tests LUNA/FTX/COVID (scenarios historiques).

### Gaps identifies
1. Pas de calcul ES/CVaR live ni ex-ante (par-trade) ni ex-post (PnL realise).
2. VaR Gaussien implicite dans le sizing (si on utilise sigma seul) : sous-estime la queue de BTC.
3. Pas de fit GPD/POT sur la distribution des PnL bar-par-bar -> on ignore l'incertitude reelle de la queue.
4. Pas de borne probabiliste sur le drawdown (combien de chances qu'on touche -25% sur 90 jours ?).

### Reco actionnable
- **IMPORTANT** : ajouter un module `risk_metrics.py` calculant a chaque clot
  - VaR 99% & ES 97.5% empirique sur returns 1h/4h/1d (rolling 1000 obs).
  - Cornish-Fisher VaR (corrige Mark-Vaucher) pour comparaison.
  - GPD POT fit hebdo : shape, scale, threshold = 95e percentile.
- **IMPORTANT** : alerter si max drawdown realise > E[MDD] de Magdon-Ismail (mu, sigma estimes) + 2 sigma_MDD -> signal regime change.
- **NICE** : exposer ces metriques sur le dashboard.

---

## 2. Crypto-specific edges non exploites

### Etat de l'art
- **Funding rate arb (delta-neutral spot+perp)** : 2025 academic + industry data : moyenne 19.26% annualise, max DD < 2%, Sharpe 3-6 sur backtests 3 ans. Funding overwhelmingly positive (92% du temps Q3 2025) car composante interest +0.01% ancre le neutre. Mais 95% des opportunites se ferment forcement (mean-reversion), et seulement 40% restent rentables apres frais.
- **Basis trade (cash-and-carry CME/spot ETF)** : pic 25% feb 2024, ~10% mai 2025, **~2% courant 2026** (le moins depuis approval ETF) -> edge en compression, pas une priorite.
- **BTC dominance** : avr 2026, BTC.D ~60.66%, altseason index 37/100 (Bitcoin Season). Indicateur regime : BTC.D montant -> BTC bid relatif ; BTC.D descendant -> rotation vers alts.
- **ETH/BTC** : proxy classique risk-on (haut) / risk-off (bas).

### Ce que le bot a deja
- Range detector multi-features (ADX/CI/ER/BBW/Hurst). Ne consomme **pas** funding ni basis ni dominance.

### Gaps identifies
1. Le bot ne capture **aucun** edge cash-and-carry. Sur BTC perp seul, on paye le funding quand long, on l'encaisse quand short, mais ce n'est pas un alpha, c'est un cout/revenu. Si on peut hedger spot, on a un yield 10-25% APR a faible vol.
2. Pas de feature "BTC.D" ni "ETH/BTC" comme regime indicator dans le filtre de range. C'est gratuit (CoinGecko, CMC).
3. Funding rate cross-exchange (Coinglass) pas integre comme leading indicator de stress (funding extreme = squeeze imminent).

### Reco actionnable
- **NICE** : module `funding_carry.py` simulant un sleeve delta-neutral (10-20% du capital) si compte spot Binance disponible. Backtest dedie avant.
- **IMPORTANT** : ajouter BTC.D + ETH/BTC comme features dans le range detector. Cout zero, gain attendu : reduction FP en regime alt-rotation.
- **IMPORTANT** : monitorer funding aggregé (api Coinglass ou Binance `/fapi/v1/premiumIndex`) -> si |funding 8h annualise| > 50%, baisser la taille (overcrowded).

---

## 3. On-chain features (gratuit / low-cost)

### Etat de l'art
- **MVRV Z-score** (Mahmudov & Puell) : zone rouge top, zone verte bottom. Glassnode docs ne publie pas les seuils exacts ; convention industrielle : Z > 7 = top, Z < 0.1 = bottom.
- **NVT / NVT Signal** (Woobull) : equivalent P/E. Useful long-range, peu utile intraday.
- **SOPR** (Glassnode) : >1 = vendeurs en profit (potentiel pression), <1 = en perte (capitulation). Court terme STH-SOPR plus reactif que LTH-SOPR.
- **Liquidation heatmap Coinglass** : zones de liquidations clusters -> "magnet price" (price seeks liquidity).
- APIs gratuites/low-cost confirmees : **mempool.space** (REST gratuit), bitcoin-data.com (mirror partiel Glassnode), CoinGecko, CryptoCompare.

### Ce que le bot a deja
- Rien d'on-chain. Pure price/volume/microstructure.

### Gaps identifies
1. Pas de filtre macro on-chain. En zone "MVRV Z > 7" historiquement le risque/recompense long est tres asymetrique negatif.
2. Pas de visibilite sur les pools de liquidations -> on entre potentiellement contre un magnet price.
3. Sur le marketing : **Etre sceptique** -> CryptoQuant, IntoTheBlock vendent souvent des indicators retro-fittes sans publication peer-review. MVRV, SOPR, NVT sont les seuls qui ont une base academique/methodologique solide (Puell, Woo).

### Reco actionnable
- **IMPORTANT** : pull MVRV Z-score quotidien (bitcoin-data.com gratuit, fallback CoinGecko derived). Si Z > 7 -> reduire la taille longue de 50%. Si Z < 0 -> autoriser longs avec taille +25%.
- **NICE** : SOPR daily comme filtre additionnel (court : bias short si STH-SOPR > 1.05 et momentum baisser).
- **NICE** : Coinglass liquidation heatmap (API payante 30 USD/mois) pour eviter de placer SL dans un cluster de liquidations adverses.
- **SKIP** : NVT et indicateurs CryptoQuant marketing (faible signal-to-noise documente).

---

## 4. Order management avance

### Etat de l'art
- **TWAP** : utile au-dessus de ~50k USD notional sur BTC perp (depth typique 50bps de 200-500k USD selon heure). Empirica/Talos/Bybit confirment seuils similaires.
- **Iceberg / hidden** : Binance Futures ne supporte pas hidden orders publics au sens MIT/CME. Le `reduceOnly` et `post-only` (TimeInForce = GTX) sont les leviers maker.
- **Post-only (GTX)** : sur USDT-M, maker = 0.02% (-0.5 bps possible via VIP/BNB). Taker = 0.05%. Soit 3 bps d'ecart par trade -> ~6 bps round-trip. Sur 200 trades/an, ~12% de yield drag evite.

### Ce que le bot a deja
- Inconnu cote PO/TWAP dans les fixes Layer 1. A verifier dans le code mais probable que les ordres soient market/IOC.

### Gaps identifies
1. Si market orders : on paie systematiquement taker = ~5 bps + slippage. Sur strategie ATR-based qui trade ~50-100x/an, c'est 5-10% de drag.
2. Pas de TWAP : OK tant que notional < 30k USD et que la stratgie est swing (pas HFT). Au-dela, market impact non-negligeable.

### Reco actionnable
- **IMPORTANT** : ajouter mode "post-only with timeout" -> envoie GTX au mid+1tick, attend N secondes, si pas fill -> escalade en taker. Economie ~3 bps/trade.
- **NICE** : TWAP simple (5 chunks sur 60s) au-dela de 30k USD notional.
- Verifier explicitement dans `forge.py` quel TimeInForce est utilise.

---

## 5. Market microstructure monitoring

### Etat de l'art
- **Order book imbalance** (top-of-book bid/ask size) : academic confirme effet predictif court-terme (Cont 2014 ; Binance Futures 2022-2025 : monotone, concave aux extremes ; lag 1d optimal pour crash risk regression, significatif jusqu'a 7 jours).
- **Depth at 50bps** : capacity check standard pour eviter de prendre 100% d'un side.
- **Spread monitoring** : si spread BTC-USDT futures > 2 bps c'est anormal (median ~0.5 bps).
- Source data : Binance WebSocket `@depth20@100ms` gratuit.

### Ce que le bot a deja
- WebSocket user stream (account/position). Pas mentionne de market data depth/imbalance.

### Gaps identifies
1. Pas de feature "OBI" dans le range detector -> on rate un signal microstructure documente academiquement.
2. Pas de gate "do not trade if spread > 2x median" -> on peut entrer en plein flash crash.
3. Pas de capacity check : on sizing notional sans verifier la depth disponible.

### Reco actionnable
- **IMPORTANT** : abonnement WebSocket `btcusdt@depth20@100ms`. Calcul OBI = (bid_qty - ask_qty) / (bid_qty + ask_qty) sur top 5 levels. Logger pour observabilite.
- **IMPORTANT** : pre-trade gate -> si `spread_bps > 5 * rolling_median(spread_bps, 1h)`, refuse l'entree.
- **IMPORTANT** : capacity check -> si `notional > 30% * depth_at_50bps`, scale down.

---

## 6. Backtest pitfalls additionnels

### Etat de l'art
- **Lookahead bias** : classique = utiliser close du jour pour signal du jour. Sur crypto : funding rate calcule avec data future, ATR avec close de la barre courante (pas closed-bar), volume normalisation utilisant rolling future.
- **Survivorship bias** : non-pertinent BTC seul ; redevient critique si extension multi-asset (delistings : LUNA, FTT, etc.).
- **Selection bias / overfitting** : DSR/PSR (Lopez de Prado) deja en place. PBO (Probability of Backtest Overfitting) complementaire.
- **Regime change** : Bitcoin a deja subi 4 reductions de moitie ; chaque cycle a une dynamique differente. 2020-2024 inclut ETF approval (changement structurel). Probabilite reset majeur 2026-2028 : non-nulle (MiCA, halving 2028, IA-driven flow, etc.).

### Ce que le bot a deja
- BUG-004 (closed-bar) fixed.
- DSR/Reality Check/Hansen SPA en place (multi-testing handled).

### Gaps identifies
1. **Funding rate lookahead** : si on utilise funding rate "courant" comme feature mais qu'il n'est observable qu'a la prochaine 8h boundary -> bias. A verifier.
2. **ATR** : si calcule avec barre courante (high/low pas finalises) -> bias. A verifier.
3. Pas de **walk-forward** explicite mentionne -> juste DSR. Le walk-forward expanding/rolling est complementaire et detecte le regime drift.
4. PBO non mentionne -> faisable a partir des deja-disponibles trials.

### Reco actionnable
- **BLOCKER** : audit code statique -> grep `shift()`, `iloc[-1]`, calcul `atr` et `funding_rate` ; verifier qu'aucune valeur de t+1 n'est utilisee a t.
- **IMPORTANT** : ajouter walk-forward analysis : 70% in-sample, 30% out-of-sample, 5 folds chronologiques.
- **IMPORTANT** : calculer PBO sur le grid search (Bailey-Borwein-Lopez de Prado-Zhu).
- **NICE** : monter un "regime change detector" simple (Bai-Perron sur returns, ou CUSUM sur Sharpe rolling).

---

## 7. Trade journal automation

### Etat de l'art
- **QuantStats** (Aroussi) : maintenu Python 3.10+, monte-carlo, tearsheet HTML. **pyfolio** : deprecated (post-Quantopian). **empyrical** : maintenu, no tearsheets. **jquantstats** : fork modernise.
- Schema standard trade : entry_time, exit_time, side, qty, entry_price, exit_price, fees, funding_paid, slippage_bps, signal_strength, regime_at_entry, ATR_at_entry, expected_R, actual_R, max_adverse_excursion (MAE), max_favorable_excursion (MFE).
- Reporting : Telegram bot (gratuit) -> envoi resume hebdo.

### Ce que le bot a deja
- Audit log hash-chain (forensic), paper_log. Schema interne inconnu.

### Gaps identifies
1. Si schema trade ne contient pas MAE/MFE, regime, ATR_at_entry, signal_strength -> impossible de faire post-mortem qualitatif.
2. expected_R vs actual_R non logge -> pas de Brier score / hit rate par bucket.
3. Pas de tearsheet auto -> friction pour reviewer hebdo.

### Reco actionnable
- **IMPORTANT** : etendre schema trade journal pour inclure : `regime_at_entry`, `atr_at_entry`, `signal_strength`, `mae`, `mfe`, `expected_R`, `actual_R`. Migration alembic-style si SQLite.
- **IMPORTANT** : pipeline hebdo : QuantStats `reports.html(returns)` + envoi Telegram resume (top 3 gagnants/perdants, hit rate, Sharpe rolling, drawdown).
- **NICE** : remplacer pyfolio par QuantStats explicitement si dependance presente.

---

## 8. Latency / infra monitoring

### Etat de l'art
- Binance signed API exige timestamp dans `recvWindow` (default 5000ms). Erreur `-1021` = drift > recvWindow. NTP sur VM cloud peut deriver de 50-200ms en quelques heures.
- WebSocket : Binance ferme connexion apres 24h (force re-connect). Pings server toutes les 3 minutes (pong required dans 10 min).
- Latency Binance signed REST (us-east) : median ~50ms, p99 ~200ms ; depuis EU ~80-150ms median.

### Ce que le bot a deja
- Time sync (BUG-?), rate limit middleware, WebSocket user stream, watchdog.

### Gaps identifies
1. Pas mentionne : telemetrie de latence ROUND-TRIP REST (envoi -> reponse), p50/p95/p99 logges.
2. Pas mentionne : alerte si p95 > 200ms pendant N minutes (degradation reseau / Binance side).
3. Pas mentionne : drift NTP measure explicit (host clock vs server time). Si > 500ms, kill switch.
4. WebSocket : pas mentionne de detection "spurious reconnect" (>3 reconnects/heure = anomalie).

### Reco actionnable
- **BLOCKER** : metric `api_latency_ms{endpoint, percentile}` exporte (Prometheus ou simple JSON). Alerte si p95 > 200ms 5 min consecutives.
- **BLOCKER** : drift NTP measure toutes les 60s : `local_ts - serverTime`. Si |drift| > 500ms -> resync NTP forced ; si > 2000ms -> kill switch.
- **IMPORTANT** : compteur `ws_reconnects_per_hour` -> alerte si > 5.
- **NICE** : dashboard avec heartbeat WS et latence histogram.

---

## 9. Compliance / legal France 2026

### Etat de l'art
- **MiCA** transitional period FR : DASP enregistres avant 30 dec 2024 ont jusqu'au **1er juillet 2026** pour obtenir le passport CASP AMF. **Attention** : MiCA exclut les derives crypto (laisses sous MiFID II).
- **ESMA** vise classification perpetuels BTC/ETH comme **CFD** -> leverage cap retail **2:1** sur crypto, margin close-out, negative balance protection. Mouvement amorce debut 2025, application probable 2026-2027.
- **DAC8** entree en vigueur **1er janvier 2026** : obligations declaratives renforcees pour les CASP (reporting transactionnel). Pour le particulier : pas d'obligation directe nouvelle (deja 2086 et 3916-bis).
- **Trading habituel** : critere jurisprudentiel (frequence, organisation, moyens techniques). Bot algo 24/7 avec capital significatif = **fortement** susceptible d'etre requalifie en BNC (depuis loi de finances 2022, BNC remplace BIC pour activite habituelle). Seuils micro-BNC = 77 700 EUR CA.

### Ce que le bot a deja
- Hors scope code.

### Gaps identifies
1. Si plateforme = Binance International (non CASP MiCA approved post-juillet 2026), risque de coupure d'acces depuis IP FR.
2. Si ESMA confirme classification CFD -> leverage 2:1 sur retail = changement majeur (vs 50-125x actuel sur Binance).
3. Si activite reconnue habituelle -> BNC, cotisations URSSAF, comptabilite, regime auto-entrepreneur ou EI.
4. DAC8 : Binance va reporter les transactions a l'administration FR -> verifier coherence avec declarations 2086 personnelles.

### Reco actionnable
- **BLOCKER** : se faire conseiller fiscaliste FR specialise crypto avant Q3 2026. Anticiper passage BNC + statut auto-entrepreneur si CA > 30k/an et frequence > 100 trades/mois.
- **IMPORTANT** : monitorer publication Binance EU CASP license. Plan B : Bitget / Kraken / OKX EU-licensed si Binance ferme retail FR.
- **IMPORTANT** : preparer scenario "leverage cap 2:1" : adapter sizing du bot, recalculer Sharpe attendu (probable -50% rendement absolu).

---

## 10. Backup / disaster recovery

### Etat de l'art
- **Object Lock** (S3 / Backblaze B2) : WORM compliance/governance mode, gratuit cote feature (paye juste le storage), protege contre ransomware + accident del.
- **B2 pricing** : 6 USD/TB/month storage, 10 GB/month free egress. Cle PII pour state.json + audit_log + paper_log : volume < 1 GB -> ~0.006 USD/mois.
- **RTO/RPO testing** : la majorite des operateurs (etudes Veeam) n'a JAMAIS teste une restauration end-to-end. Backup sans test = pas de backup.

### Ce que le bot a deja
- Recovery WAL (write-ahead log) -> reprise apres crash sur **meme** machine. Pas de backup off-machine.
- Audit log hash-chain : integrite mais pas redondance geo.

### Gaps identifies
1. Si VM crash hardware ou supprimee : `state.json`, `audit_log`, `paper_log` perdus.
2. Pas de backup off-host -> SPOF.
3. Pas de procedure documentee de restoration -> en cas d'incident, MTTR inconnu.
4. Test de restauration : jamais valide.

### Reco actionnable
- **BLOCKER** : `cron` toutes les heures -> tarball encrypte (age ou gpg) de `state.json`, `audit_log`, `paper_log`, `.env.enc` -> upload Backblaze B2 bucket Object Lock retention 30 jours. Cout < 1 USD/mois.
- **BLOCKER** : runbook restoration documente (markdown) + test trimestriel sur VM staging (deploiement from-scratch + restore + parite hash audit_log).
- **IMPORTANT** : second backup secondaire (Hetzner Storage Box ou rclone vers Mega/Drive) = strategie 3-2-1 (3 copies, 2 medias, 1 off-site).

---

## TLDR — Top 5 actions par criticite

1. **BLOCKER — Audit lookahead bias residuel** (Section 6). Grep funding_rate / ATR / shift dans le code. Aucun deploiement live tant que ce n'est pas verifie.
2. **BLOCKER — Backup off-host avec test de restauration** (Section 10). Backblaze B2 Object Lock + cron horaire + runbook + test trimestriel. Cout < 1 USD/mois, MTTR controle.
3. **BLOCKER — Telemetrie latence + NTP drift kill switch** (Section 8). p95 latency monitoring, drift > 500ms = resync, > 2000ms = kill. Sans ca, on trade aveugle.
4. **BLOCKER — Conseil fiscal/juridique FR avant Q3 2026** (Section 9). MiCA transition + ESMA leverage cap 2:1 + DAC8 + requalification BNC. Risque non-codable.
5. **IMPORTANT — Modules Risk + Microstructure + On-chain (top 3 quick wins)** :
   - Section 1 : ES/CVaR + GPD POT live.
   - Section 5 : OBI + spread gate + capacity check via WebSocket depth20 (gratuit).
   - Section 3 : MVRV Z-score daily filter (bitcoin-data.com gratuit).

Reste **NICE-to-have** mais pas indispensable pour MVP live : funding carry sleeve, TWAP, BTC.D feature, walk-forward formalise, PBO, QuantStats tearsheet auto.

---

## Sources citees

### Section 1
- Magdon-Ismail & Atiya, "An Analysis of the Maximum Drawdown Risk Measure", https://www.cs.rpi.edu/~magdon/ps/journal/drawdown_RISK04.pdf
- Magdon-Ismail et al., "On the Maximum Drawdown of a Brownian Motion", https://home.work.caltech.edu/pub/Magdon-Ismail2004drawdown.pdf
- Mark & Vaucher, "Cornish-Fisher Downside Risk", SSRN 4796363, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4796363
- Teng/Huang/Shih, "Tail Risk in Bitcoin under Basel Framework", SSRN 5519778, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5519778
- "Bayesian Extreme Value Analysis of Bitcoin Tail Risk", https://www.preprints.org/manuscript/202509.2093/v1/download
- "Modelling Extreme Tail Risk of Bitcoin Returns Using GPD" (IntechOpen), https://www.intechopen.com/chapters/1173468
- Borri, "Conditional Tail-Risk in Cryptocurrency Markets", SSRN 3162038, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3162038
- Risk.net "VaR vs ES", https://www.risk.net/risk-magazine/technical-paper/1506669/var-versus-expected-shortfall

### Section 2
- BIS Working Paper 1087, "Crypto carry", https://www.bis.org/publ/work1087.pdf
- "Designing funding rates for perpetual futures in cryptocurrency markets" (arXiv 2506.08573), https://arxiv.org/html/2506.08573v1
- BitMEX 2025 Q3 derivatives report, https://www.bitmex.com/blog/2025q3-derivatives-report
- CoinDesk "U.S. BTC ETF Cash-and-Carry Trade Collapses", https://www.coindesk.com/markets/2025/03/21/what-the-collapse-of-the-u-s-bitcoin-etf-cash-and-carry-trade-means-for-investors
- CFB "Revisiting the Bitcoin Basis", https://www.cfbenchmarks.com/blog/revisiting-the-bitcoin-basis-how-momentum-sentiment-impact-the-structural-drivers-of-basis-activity
- BeInCrypto "Bitcoin Dominance 60.66%", https://beincrypto.com/bitcoin-dominance-explodes-to-60-66-and-buries-altseason-hopes-for-2026/

### Section 3
- Glassnode docs MVRV-Z, https://docs.glassnode.com/guides-and-tutorials/metric-guides/mvrv/mvrv-z-score
- Woobull NVT Signal, http://charts.woobull.com/bitcoin-nvt-signal/
- Glassnode SOPR, https://studio.glassnode.com/metrics?a=BTC&m=indicators.Sopr
- Coinglass liquidation heatmap, https://www.coinglass.com/pro/futures/LiquidationHeatMap
- mempool.space REST API, https://mempool.space/docs/api/rest

### Section 4
- Binance Futures fee structure FAQ, https://www.binance.com/en/support/faq/detail/360033544231
- Empirica TWAP guide, https://empirica.io/blog/twap-strategy/
- arXiv "Deep Learning for VWAP Execution in Crypto Markets" (2502.13722), https://arxiv.org/html/2502.13722v2

### Section 5
- arXiv "Explainable Patterns in Cryptocurrency Microstructure" (2602.00776), https://arxiv.org/html/2602.00776v1
- arXiv "Exploring Microstructural Dynamics in Cryptocurrency LOB" (2506.05764), https://arxiv.org/html/2506.05764v2
- "Nowcasting bitcoin's crash risk with order imbalance" (PMC), https://pmc.ncbi.nlm.nih.gov/articles/PMC10040314/
- ScienceDirect "Order flow toxicity and price jumps", https://www.sciencedirect.com/science/article/pii/S0275531925004192

### Section 6
- Bailey & Lopez de Prado, "The Deflated Sharpe Ratio", SSRN 2460551, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- "Probability of Backtest Overfitting", https://www.researchgate.net/publication/318600389_The_probability_of_backtest_overfitting
- Blockchain Council, "Backtesting AI Crypto Trading Strategies Safely", https://www.blockchain-council.org/cryptocurrency/backtesting-ai-crypto-trading-strategies-avoiding-overfitting-lookahead-bias-data-leakage/

### Section 7
- QuantStats GitHub, https://github.com/ranaroussi/quantstats
- pyfolio status, https://tradingbrokers.com/pyfolio-alternatives/

### Section 8
- Binance dev "1021 INVALID_TIMESTAMP", https://dev.binance.vision/t/1021-invalid-timestamp-timestamp-ahead-of-server-time/4783
- python-binance FAQ, https://python-binance.readthedocs.io/en/latest/faqs.html

### Section 9
- AMF MiCA depth, https://www.amf-france.org/en/news-publications/depth/mica
- ESMA product intervention CFDs (leverage 2:1 crypto), https://www.esma.europa.eu/press-news/esma-news/esma-adopts-final-product-intervention-measures-cfds-and-binary-options
- MEXC News "ESMA Critical Move Crypto Derivatives as CFDs", https://www.mexc.com/news/791462
- Sumsub "MiCA Regulation 2026", https://sumsub.com/blog/crypto-regulations-in-the-european-union-markets-in-crypto-assets-mica/
- Cryptoast "Fiscalite activite habituelle traders", https://cryptoast.fr/fiscalite-crypto-actifs-activite-habituelle-traders-professionnels/
- AC Legal "Fiscalite cryptomonnaies France 2026", https://www.aclegal.fr/2026/04/01/fiscalite-cryptomonnaies-france-particulier/

### Section 10
- Backblaze B2 Object Lock, https://www.backblaze.com/cloud-storage/solutions/object-lock
- Backblaze "Protect Backup from Ransomware with Object Lock", https://www.backblaze.com/blog/object-lock-101-protecting-data-from-ransomware/
