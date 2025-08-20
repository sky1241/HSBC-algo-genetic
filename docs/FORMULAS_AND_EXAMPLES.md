Formulas, definitions and examples (pipeline_web6) — HSBC
Date: 2025-08-20

### 0) Quick context
- Data: Binance (OHLCV, 2h tf), Ichimoku + ATR trailing strategy, realistic costs/execution (fees, funding/rollover, slippage, latency, halts, margins).
- Reference risk settings: Leverage 10×, Position 1%, up to 3 positions per side.
- Results excerpts: sets n1 and n2 (see `docs/HSBC_REPORT_EN.md`, `docs/TESTS_AND_RESULTS.md`).

### 1) Ichimoku — core components
- Why: capture trend/momentum and filter with the cloud to avoid false signals.
- How: mid‑price of rolling highs/lows; compare Tenkan/Kijun and price vs cloud.
- Tenkan‑sen (period N): \( Tenkan_N = (HH_N + LL_N) / 2 \)
  - Example (BTC, N=34): HH_34=43,000; LL_34=40,000 → Tenkan=41,500.
- Kijun‑sen (period M): \( Kijun_M = (HH_M + LL_M) / 2 \)
  - Example (M=82): HH_82=47,000; LL_82=35,000 → Kijun=41,000.
- Senkou Span B (period K): \( SenkouB_K = (HH_K + LL_K) / 2 \), plotted forward by `shift` periods.

### 2) ATR and True Range
- Why: measure effective volatility to size stops and positions.
- How: EMA/SMA of True Range.
- True Range: \( TR_t = \max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|) \)
- ATR (period n): SMA form \( ATR_t = (ATR_{t-1} (n-1) + TR_t)/n \)
  - Example: High=105, Low=100, Close_{t-1}=102 → TR=\(\max(5,3,2)=5\). With n=14 and ATR_{t-1}=4 → ATR_t ≈ 4.07.

### 3) ATR trailing stop
- Why: lock‑in profits and exit when trend reverses.
- How: stop follows price at a distance of \( m\cdot ATR \).
- Long: \( TS_t = \max(TS_{t-1}, Close_t - m \cdot ATR_t) \)
- Short: \( TS_t = \min(TS_{t-1}, Close_t + m \cdot ATR_t) \)
  - Example (long): Close=2,500; ATR=100; m=2.6 → raw=2,240; if \( TS_{t-1}=2,300 \) → new TS=2,300.

### 4) Position sizing, leverage and PnL
- Why: control per‑trade risk and exposure via notional.
- How: \( Notional = Equity \times Position\_Size \times Leverage \); \( PnL = Notional \times Return \).
- Example: Equity=€10,000; size=1%; lev=10 → Notional=€1,000. If price +2% → PnL=€20 (equity +0.2%).

### 5) Equity multiple and CAGR
- Why: track long‑term compounded growth.
- How: \( Mult = V_T/V_0 \); \( CAGR = Mult^{1/n} - 1 \).
- Example n1 (5y): V_0=€1,000; V_T=€24,536 → Mult=24.536 → \( CAGR \approx 24.536^{1/5} - 1 \approx 89.6\% \).
- Example n2 (5y): V_T=€59,103 → Mult=59.103 → \( CAGR \approx 126\% \).

### 6) Max Drawdown (MDD) and Calmar
- Why: quantify downside and risk‑adjusted performance.
- How: MDD from peak equity; Calmar = CAGR/|MDD|.
- Peak curve: \( P_t = \max(P_{t-1}, Equity_t) \)
- Drawdown: \( DD_t = (Equity_t - P_t)/P_t \le 0 \)
- MDD: \( \min_t DD_t \)
- Calmar: \( Calmar = CAGR / |MDD| \)
- Example MDD: 1,000→1,200→900→1,300 → MDD=\( (900-1,200)/1,200 = -25\% \).
- Example Calmar n2: \( 1.26 / 0.048 \approx 26.3 \).

### 7) Sharpe (proxy) and Sortino
- Why: risk‑adjusted performance (total vs downside volatility).
- How: mean excess return over stdev (total/downside), annualized.
- Annualized Sharpe: \( Sharpe = \frac{\mu_R - r_f}{\sigma_R} \sqrt{K} \) (K=12 monthly, K≈365 daily).
- Sortino: \( Sortino = \frac{\mu_R - r_f}{\sigma_{down}} \sqrt{K} \).
- Example (monthly): \( \mu=6\%\), \( \sigma=12\%\), \( r_f\approx 0 \) → Sharpe ≈ \( 0.06/0.12 \times \sqrt{12} = 1.73 \). If \( \sigma_{down}=8\% \) → Sortino ≈ 2.60.

### 8) VaR 95% (horizon T)
- Why: quantify a plausible tail loss.
- How: 5% quantile of returns (or terminal p5 from MC distribution).
- Example n2 (5y): p5=9.26× → in 5% worst cases, final capital remains > 9.26× initial (terminal‑distribution reading).

### 9) Stability proxy (Lyapunov)
- Why: prefer dynamics less sensitive to initial conditions.
- How: slope of \( \ln d(t) \) vs t; \( \lambda>0 \) penalized.
- Method (Rosenstein‑style): state reconstruction, neighbor divergence \( d(t) \); regress \( \ln d(t) \) on t → slope \( \lambda \).
- Example: \( \lambda=0.02 \) (slightly positive) → score reduced vs \( \lambda\le 0 \).

### 10) Dynamic slippage (realistic)
- Why: reflect book depth and market impact.
- How: slippage grows with \( q = Position\_Value/Volume_{USDT} \), bounded.
- Base: \( s_0 = 0.05\% = 0.0005 \). Mid‑range rule: \( s = s_0 (1 + 10q) \).
- Example: Position=50,000 USDT; Volume=10,000,000 USDT → \( q=0.005 \) → \( s = 0.0005(1+0.05)=0.000525=0.0525\% \).

### 11) Monte Carlo (block bootstrap)
- Why: test path robustness without strong Gaussian assumptions.
- How: block resampling preserves autocorrelation; compute distribution of terminal multiples.
- Excerpts (1% size): n1 p50=22.32×; n2 p50=50.18×; median DD ≈ 21–22%.

### 12) End‑to‑end example (simplified long)
1) Equity=€10,000; size=1%; lev=10 → Notional=€1,000.
2) Entry 2,500; ATR=100; m=2.6 → raw TS=2,240; initial TS=2,300.
3) Slippage s≈0.0525% → fill ≈ 2,501.31.
4) Exit 2,575; PnL ≈ Notional × (2575−2501.31)/2501.31 ≈ €1,000 × 2.94% ≈ +€29.4 (pre fees/funding).

---

History
- 2025-08-20: Initial English version (v1): key formulas, why/how, and numeric examples.


