Formules, définitions et exemples (pipeline_web6) — HSBC
Date: 2025-08-20

### 0) Contexte rapide
- Données: Binance (OHLCV, tf 2h), stratégie Ichimoku + ATR trailing, risques réalistes (frais, funding/rollover, slippage, latence, haltes, marges) [[memory:6051257]].
- Réglages de risque de référence: Levier 10×, Position 1%, 3 positions max/side.
- Extraits de résultats: n1 et n2 (voir `docs/HSBC_REPORT_FR.md`, `docs/TESTS_AND_RESULTS.md`).

### 1) Ichimoku — composantes clés
- **Tenkan‑sen (période N)**: \( Tenkan_N = (HH_N + LL_N) / 2 \)
  - HH_N: plus haut sur N périodes; LL_N: plus bas sur N périodes.
  - Exemple (BTC, N=34): HH_34=43 000, LL_34=40 000 → Tenkan=41 500.
- **Kijun‑sen (période M)**: \( Kijun_M = (HH_M + LL_M) / 2 \)
  - Exemple (M=82): HH_82=47 000, LL_82=35 000 → Kijun=41 000.
- **Senkou Span B (période K)**: \( SenkouB_K = (HH_K + LL_K) / 2 \), tracé en avance de `shift` périodes.
  - Exemple (K=216): HH_216=52 000, LL_216=20 000 → SenkouB=36 000.
- Utilisation: cross Tenkan/Kijun et position vs cloud (Senkou A/B) filtrent les signaux.

### 2) ATR et True Range
- **True Range**: \( TR_t = \max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|) \)
- **ATR (période n)**: EMA/SMA de TR.
  - SMA: \( ATR_t = (ATR_{t-1} \times (n-1) + TR_t) / n \)
  - Exemple: High=105, Low=100, Close_{t-1}=102 → TR=\(\max(5,3,2)=5\). Avec n=14 et ATR_{t-1}=4, ATR_t ≈ 4,07.

### 3) Trailing stop basé sur ATR
- Long: \( TS_t = \max(TS_{t-1}, Close_t - m \cdot ATR_t) \)
- Short: \( TS_t = \min(TS_{t-1}, Close_t + m \cdot ATR_t) \)
- Exemple (long): Close=2 500, ATR=100, m=2,6 → stop brut=2 500−260=2 240; si TS_{t-1}=2 300 → nouveau TS=\(\max(2 300,2 240)=2 300\).

### 4) Position sizing, levier et PnL
- Notional: \( Notional = Equity \times Position\_Size \times Leverage \)
- PnL nominal: \( PnL = Notional \times Return \)
- Exemple: Equity=10 000 €, pos=1%, levier=10 → Notional=1 000 €. Si prix +2% → PnL=20 € (impact equity ≈ +0,2%).

### 5) Multiplicateur d’équité et CAGR
- Multiplicateur: \( Mult = V_T / V_0 \)
- CAGR (n années): \( CAGR = Mult^{1/n} - 1 \)
- Exemple n1 (5 ans): V_0=1 000 €, V_T=24 536 € → Mult=24,536 → \( CAGR ≈ 24{,}536^{1/5} - 1 ≈ 0{,}896 = 89{,}6\% \).
- Exemple n2 (5 ans): V_T=59 103 € → Mult=59,103 → \( CAGR ≈ 1{,}26 = 126\% \).

### 6) Max Drawdown (MDD) et Calmar
- Courbe des plus‑hauts: \( P_t = \max(P_{t-1}, Equity_t) \)
- Drawdown: \( DD_t = (Equity_t - P_t)/P_t \le 0 \)
- MDD: \( \min_t DD_t \)
- Calmar: \( Calmar = CAGR / |MDD| \)
- Exemple MDD: equity 1 000→1 200→900→1 300 → MDD=\( (900-1 200)/1 200 = -25\% \).
- Exemple Calmar n2: \( 1{,}26 / 0{,}048 ≈ 26{,}3 \).

### 7) Sharpe (proxy) et Sortino
- Sharpe annualisé: \( Sharpe = \frac{\mu_R - r_f}{\sigma_R} \sqrt{K} \) (K=12 si mensuel, K≈365 si quotidien).
- Sortino: \( Sortino = \frac{\mu_R - r_f}{\sigma_{down}} \sqrt{K} \) (\( \sigma_{down} \) = écart‑type des rendements négatifs).
- Exemple (mensuel): \( \mu=6\%\), \( \sigma=12\%\), \( r_f≈0 \) → Sharpe ≈ \( 0{,}06/0{,}12 \times \sqrt{12} = 1{,}73 \). Si \( \sigma_{down}=8\% \) → Sortino ≈ 2{,}60.

### 8) VaR 95% (période T)
- Définition: perte seuil telle que \( P(R \le -VaR_{95}) = 5\% \) sur l’horizon considéré.
- Lien MC terminal: p5 de la distribution des multiplicateurs finaux.
- Exemple n2 (5 ans): p5=9,26× → en 5% des cas défavorables, l’issue reste > 9,26× le capital initial (lecture conservatrice terminale).

### 9) Proxy de stabilité (Lyapunov)
- Méthode (Rosenstein simplifiée): reconstruire l’espace d’états, divergence moyenne \( d(t) \) entre voisins; régresser \( \ln d(t) \) sur t → pente \( \lambda \).
- Interprétation: \( \lambda>0 \) = sensibilité accrue (instabilité). Pénalisée dans l’objectif d’optimisation.
- Exemple: pente \( \lambda=0{,}02 \) (faiblement positive) → score réduit vs \( \lambda\le 0 \).

### 10) Slippage dynamique (réaliste)
- Base: \( s_0 = 0{,}05\% = 0{,}0005 \). Ratio volume: \( q = \frac{Position\_Value}{Volume\_{USDT}} \).
- Règle (zone 0{,}1–1% du volume): \( s = s_0 (1 + 10q) \) (plancher/plafond selon q; autres zones: cf. logique).
- Exemple: Position=50 000 USDT, Volume=10 000 000 USDT → \( q=0{,}005 \) → \( s = 0{,}0005(1+0{,}05)=0{,}000525=0{,}0525\% \).

### 11) Monte Carlo (block bootstrap)
- On rééchantillonne des blocs pour préserver l’autocorrélation et calcule la distribution des multiplicateurs finaux.
- Extraits (1% pos): n1 p50=22,32×; n2 p50=50,18×; DD médiane ≈ 21–22%.
- Lecture: choisir la baseline avec meilleur p50/p5 à DD médiane comparable.

### 12) Exemple end‑to‑end (trade long simplifié)
1) Equity=10 000 €, pos=1%, levier=10 → Notional=1 000 €.
2) Entrée à 2 500; ATR=100; m=2,6 → TS brut=2 240; TS initial=2 300.
3) Slippage s≈0,0525% → prix fill ≈ 2 501,31.
4) Sortie au signal inverse à 2 575; PnL ≈ Notional × (2575−2501,31)/2501,31 ≈ 1 000 × 2,94% ≈ +29,4 € (avant frais/funding).

---

Historique
- 2025-08-20: Création (v1): formules clés, pourquoi/usage, et exemples numériques issus des analyses.


