# MISSION MULTI-TF — Refonte exécution courte indexée tendance H2

> **Mission v4 proposée par Sky le 2026-04-30 après réalisation que le bot
> tourne en mono-H2 (pas la thèse multi-TF qu'il pensait avoir).**
> Document préparé en autonomie par Claude Code suite à l'audit complet.
> **STATUS : PLAN — en attente de validation Sky avant codage.**

---

## 1. Contexte & demande

### 1.1 — Citation Sky (verbatim, 2026-04-30)

> "claude je crois que y a des truc que on c est pas compris normalement
> tu était senser prendre des trade en 5 et 15 minutes mais indécé sur
> la tendence h2 enfaite"

### 1.2 — Reformulation propre

Architecture en **2 couches** :
- **Couche FILTRE (slow, H2)** : la tendance H2 dit
  - "long autorisé"
  - "short autorisé"
  - "flat" (pas de trade — marché indécis)
- **Couche EXÉCUTION (fast, 5m/15m)** : les déclencheurs d'entrée tapent
  sur les bougies courtes — mais seulement dans la direction autorisée
  par la couche filtre H2.

Pas de "deux signaux Ichimoku qui se croisent" — c'est bien un **filtre
directionnel + déclencheur d'exécution**, deux étages distincts.

### 1.3 — État actuel (ce qui ne correspond PAS à la thèse)

```yaml
# bot_settings.yaml ligne 8
timeframe: "2h"
```

Le bot tourne en **mono-H2** : Ichimoku calculé sur H2, signaux détectés
sur H2, entrées/sorties sur H2. Aucune logique 5m/15m. Aucun filtre H2
sur autre chose. C'est un bot Ichimoku H2 classique.

→ C'est pour ça qu'il génère 0 signal depuis le démarrage (4h+ sans
trade) : sur H2, les cassures de nuage nettes sont rares.

---

## 2. Architecture cible

### 2.1 — Schéma haut niveau

```
┌──────────────────────────────────────────────────────────────────┐
│                  COUCHE FILTRE (H2 — slow, 2h)                   │
│                                                                  │
│  hsbc-h2-trend.timer  ──>  h2_trend_runner.py                    │
│  (every 2h, HH:01 UTC)     │                                     │
│                            ├─ DataFetcher.get_ohlcv("2h")        │
│                            ├─ ichimoku_engine(df_h2, params_h2)  │
│                            └─ classify_h2_trend(df_h2_with_ich)  │
│                                  ↓                               │
│                            persist state.json:                   │
│                              symbols.{sym}.h2_trend = {          │
│                                direction: "long"|"short"|"flat", │
│                                cloud_top, cloud_bottom,          │
│                                last_close, computed_at_iso       │
│                              }                                   │
└──────────────────────────────────────────────────────────────────┘
                               ↓
                        (lit state.json)
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│                COUCHE EXÉCUTION (15m — fast)                     │
│                                                                  │
│  hsbc-execution-15m.timer ──>  execution_runner_15m.py           │
│  (every 15m, *:00,15,30,45)    │                                 │
│                                ├─ Read h2_trend (skip si stale)  │
│                                ├─ DataFetcher.get_ohlcv("15m")   │
│                                ├─ ichimoku_engine(df, params_15m)│
│                                ├─ Detect cassure nuage 15m       │
│                                ├─ Aiguillage: signal autorisé    │
│                                │   ssi direction == h2_trend     │
│                                ├─ SignalEngine.detect_signals()  │
│                                │   (gates VPIN/VaR/HAR/caps OK)  │
│                                └─ Execute via TradeManager       │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 — Définition de la "tendance H2"

Plusieurs conventions possibles. **Décision proposée** : utiliser la
**position du prix vs nuage Ichimoku H2** (cohérent avec le reste de la
stratégie Ichimoku, simple, déterministe) :

```python
def classify_h2_trend(close: float, cloud_top: float, cloud_bottom: float) -> str:
    if close > cloud_top:
        return "long"      # prix au-dessus du nuage → tendance haussière
    if close < cloud_bottom:
        return "short"     # prix sous le nuage → tendance baissière
    return "flat"           # prix dans le nuage → indécision, pas de trade
```

**Rationale** : c'est l'interprétation classique du nuage Ichimoku
("Kumo") comme zone de support/résistance dynamique. Si le prix le
traverse, on n'a pas de tendance claire → flat.

**Alternatives examinées et écartées** :
- ❌ `tenkan > kijun` → trop sensible, change trop souvent
- ❌ `EMA200` → étranger à la stratégie Ichimoku
- ❌ Phase K3 → c'est un label de calibration, pas une décision
  directionnelle live (cf. REPORT_CHECK section 7.6)

### 2.3 — Choix du TF d'exécution

Sky a dit "5 et 15 minutes". Deux options examinées :

| Option | Cadence | Pros | Cons |
|---|---|---|---|
| **A** : 15m seul | 4 cycles/h | Stable, peu de bruit, peu de calls API, signaux nets | Réactivité moindre |
| **B** : 5m + 15m | 12+4 cycles/h | Plus de signaux, capture mouvements rapides | 2 jobs distincts, complexité, faux positifs sur 5m |

**Décision proposée** : commencer par **15m seul** pour valider l'archi
multi-TF, puis ajouter 5m en phase 2 si validation OK. Ça réduit
considérablement la complexité initiale et le risque.

Si Sky veut le 5m dès le départ, le plan le supporte (chunk P-MTF-9
optionnel).

### 2.4 — Aiguillage : pyramidal vs parallèle

**Décision proposée** : **pyramidal** (filtre H2 strict avant exécution
15m). C'est ce que Sky décrit ("indexé sur tendance H2").

```python
# Dans signal_engine.detect_signals() ou wrapper:
if h2_trend == "flat":
    return  # rien — couche H2 dit "marché indécis"
if h2_trend == "long" and signal_short_15m:
    return  # signal 15m short rejeté car H2 dit "long autorisé"
if h2_trend == "short" and signal_long_15m:
    return  # signal 15m long rejeté car H2 dit "short autorisé"
# Sinon : signal aligné avec H2 → procéder aux gates (VPIN/VaR/HAR/caps)
```

### 2.5 — Composants existants qu'on PRÉSERVE intégralement

Tous les gates et garde-fous actuels restent en place, **appliqués au
signal 15m** :
- ✅ Daily caps (P1) : soft 2%/3%, hard 10% — comptés sur tous TF combinés
- ✅ Anti-martingale (P2) : drawdown_size_multiplier sur sizing
- ✅ Portfolio VaR95 (P3) : `can_enter_new_position` bloque
- ✅ HAR-RV regime (P4) : bloque si vol "low"
- ✅ Composite signal (P6.5) : log-only puis gate post-J+30
- ✅ VPIN gate (P7) : log-only actuellement
- ✅ Funding-aware close (P11) : à HH:55 UTC sur les 8h funding cycles
- ✅ PSR live (P5) : monitoring J+30
- ✅ Kelly portfolio-aware sizing (P0) : appliqué au sizing 15m

Tous les callbacks (`portfolio_scale_fn`, `var_gate_fn`, `regime_gate_fn`,
`drawdown_scale_fn`, `composite_log_fn`, `vpin_data_fn`) sont **réutilisés
tels quels** dans le nouveau runner 15m.

---

## 3. Découpage en chunks atomiques

Format : chaque chunk = 1 commit, 1 message clair, tests obligatoires
avant push, branche dédiée `feat/multi-tf` pour ne pas casser le soak en
cours sur main.

### **P-MTF-1 — `DataFetcher.get_ohlcv_multi_tf()` + cache**

**Goal** : permettre au runner de fetch plusieurs TF en un appel, avec
cache TTL pour éviter de re-fetch H2 toutes les 15min.

**Fichier modifié** : `binance_bot/services/data_fetcher.py`

**API à ajouter** :
```python
def get_ohlcv_multi_tf(
    self,
    timeframes: list[str],            # ex: ["2h", "15m"]
    limit: int = 300,
    use_cache: bool = True,
    cache_ttl_seconds: dict[str, int] | None = None,
    # default: {"2h": 3600, "1h": 600, "15m": 60, "5m": 30}
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV pour plusieurs TF en parallèle (séquentiel CCXT mais 1 instance).
    Cache en mémoire + fallback fichier data/ohlcv_cache_{symbol}_{tf}.parquet.
    """
```

**Tests** (`tests/test_data_fetcher_multi_tf.py`) :
- `test_get_ohlcv_multi_tf_returns_dict_per_tf` : retourne dict avec clés = tfs
- `test_cache_hit_avoids_refetch` : 2e appel dans TTL → pas de call API
- `test_cache_miss_after_ttl` : appel après expiration → re-fetch
- `test_backward_compat_single_tf` : `get_ohlcv()` existante inchangée

**Acceptance runtime** :
```bash
python -c "
from binance_bot.services.data_fetcher import DataFetcher
f = DataFetcher('BTC/USDT', '2h')
d = f.get_ohlcv_multi_tf(['2h', '15m'])
print({k: len(v) for k, v in d.items()})
"
# Attendu: {'2h': 300, '15m': 300}
```

**Effort** : ~1 jour (150 LOC + 4 tests)

---

### **P-MTF-2 — Module `trend_filter_h2`**

**Goal** : module pur stateless qui classify la tendance H2 depuis un DF
Ichimoku.

**Fichier nouveau** : `src/trend_filter_h2.py`

**API** :
```python
def classify_h2_trend(
    df_ichimoku_h2: pd.DataFrame,
    margin_pct: float = 0.0,  # marge autour du nuage (0 = strict, 0.005 = 0.5%)
) -> dict:
    """
    Returns:
        {
            "direction": "long" | "short" | "flat",
            "last_close": float,
            "cloud_top": float,
            "cloud_bottom": float,
            "distance_pct": float,  # (close - cloud_mid) / cloud_mid
            "computed_at_iso": str,
        }
    """
```

**Tests** (`tests/test_trend_filter_h2.py`) :
- `test_close_above_cloud_returns_long`
- `test_close_below_cloud_returns_short`
- `test_close_inside_cloud_returns_flat`
- `test_close_at_cloud_top_with_margin_returns_flat` (zone neutre)
- `test_empty_df_returns_flat_safe_default`
- `test_nan_close_returns_flat`

**Effort** : ~0.5 jour (50 LOC + 6 tests)

---

### **P-MTF-3 — `signal_engine.detect_signals()` accepte `trend_gate_fn`**

**Goal** : injecter un nouveau callback `trend_gate_fn` dans le SignalEngine
pour vérifier la tendance H2 avant d'autoriser un signal.

**Fichier modifié** : `binance_bot/services/signal_engine.py`

**Diff** :
```python
# Constructeur (~ligne 40-62) : ajouter param
trend_gate_fn: Optional[Callable[[str], dict]] = None,
# signature: trend_gate_fn(symbol) -> {direction, ...} (cf P-MTF-2)

# Dans detect_signals(), avant émission long/short (~ligne 644-680) :
if self.trend_gate_fn is not None:
    h2 = self.trend_gate_fn(symbol)
    direction = h2.get("direction", "flat")
    if direction == "flat":
        return signals  # H2 dit indécis → pas de trade
    if signal_long_15m and direction != "long":
        signal_long_15m = False  # H2 short → reject signal long
    if signal_short_15m and direction != "short":
        signal_short_15m = False  # H2 long → reject signal short
```

**Tests** (`tests/test_signal_engine_trend_gate.py`) :
- `test_trend_gate_long_allows_signal_long`
- `test_trend_gate_short_blocks_signal_long`
- `test_trend_gate_flat_blocks_all`
- `test_trend_gate_none_no_filter` (backward compat — pas de gate = comportement actuel)
- `test_trend_gate_callback_exception_safe` (silent fail = block by safety)

**Effort** : ~1 jour (40 LOC modif + 5 tests)

---

### **P-MTF-4 — `h2_trend_runner.py` (nouveau routine job)**

**Goal** : job systemd cyclic toutes les 2h qui calcule la tendance H2
pour les 3 symboles et persiste dans state.json.

**Fichier nouveau** : `binance_bot/routines/h2_trend_runner.py`

**Logique** :
```python
def main():
    settings = _load_settings()
    state_mgr = StateManager(settings["state_file"])
    params_h2 = state_mgr.get("params_today", DEFAULT_H2_PARAMS)

    for symbol in settings.get("symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT"]):
        fetcher = DataFetcher(symbol, "2h")
        df_h2 = fetcher.get_ohlcv(limit=300)
        df_ich = calculate_ichimoku(df_h2, **params_h2)
        trend = classify_h2_trend(df_ich)
        state_mgr.update_symbol(symbol, "h2_trend", trend)
        print(f"H2 trend [{symbol}]: {trend['direction']} (close={trend['last_close']:.2f})")

    state_mgr.persist()
```

**Tests** (`tests/test_h2_trend_runner.py`) :
- `test_runner_writes_h2_trend_per_symbol_in_state`
- `test_runner_handles_fetch_failure_gracefully` (continue avec autres symboles)
- `test_runner_uses_params_today_from_state`

**Acceptance runtime** :
```bash
python -m binance_bot.routines.h2_trend_runner
cat data/state.json | python -m json.tool | grep -A4 h2_trend
```

**Effort** : ~1.5 jour (200 LOC + 3 tests)

---

### **P-MTF-5 — `execution_runner_15m.py` (nouveau routine job)**

**Goal** : job systemd cyclic toutes les 15m qui :
1. Lit la tendance H2 depuis state (ou skip si stale > 3h)
2. Fetch bougies 15m via `get_ohlcv_multi_tf`
3. Calcule Ichimoku 15m avec params dédiés
4. Construit `trend_gate_fn` → SignalEngine
5. Exécute le cycle de signal complet (mêmes gates qu'aujourd'hui)
6. Persiste positions

**Fichier nouveau** : `binance_bot/routines/execution_runner_15m.py`

**Skeleton** (très similaire à `intraday_runner.py` mais TF=15m + trend_gate) :
```python
def main():
    # ... setup identique à intraday_runner.py (state, risk, gates) ...

    # NEW : params Ichimoku spécifiques 15m
    params_15m = settings.get("params_15m", DEFAULT_15M_PARAMS)
    # DEFAULT_15M_PARAMS = {tenkan: 9, kijun: 26, senkou_b: 52, shift: 26}

    # NEW : factory du trend_gate_fn
    def make_trend_gate_fn(state_mgr):
        def gate(symbol):
            state_mgr.reload()  # garantir frais
            t = state_mgr.get_symbol(symbol, {}).get("h2_trend", {})
            if not t:
                return {"direction": "flat", "reason": "no_h2_data"}
            # Skip si stale > 3h
            from datetime import datetime, timezone, timedelta
            computed = datetime.fromisoformat(t.get("computed_at_iso", ""))
            if datetime.now(timezone.utc) - computed > timedelta(hours=3):
                return {"direction": "flat", "reason": "h2_stale"}
            return t
        return gate

    trend_gate_fn = make_trend_gate_fn(state_mgr)

    # ... boucle par symbole ...
    for symbol in symbols:
        fetcher = DataFetcher(symbol, "15m")
        df_15m = fetcher.get_ohlcv(limit=300)
        df_ich = calculate_ichimoku(df_15m, **params_15m)

        engine = SignalEngine(
            symbol=symbol,
            settings=settings,
            trade_mgr=trade_mgr,
            state_mgr=state_mgr,
            risk_mgr=risk_mgr,
            # ... tous les gates existants ...
            trend_gate_fn=trend_gate_fn,  # ← NOUVEAU
        )
        signals = engine.detect_signals(df_ich)
        # ... execute signals ...
```

**Tests** (`tests/test_execution_runner_15m.py`) :
- `test_runner_loads_h2_trend_from_state`
- `test_runner_skips_when_h2_stale`
- `test_runner_skips_when_h2_flat`
- `test_runner_emits_long_when_aligned`
- `test_runner_blocks_short_when_h2_long`
- `test_runner_full_smoke_no_crash` (E2E mock)

**Acceptance runtime** :
```bash
# H2 trend déjà persisté par P-MTF-4
python -m binance_bot.routines.execution_runner_15m
# → log doit montrer pour chaque symbole :
#   - h2_trend lu depuis state
#   - signal calculé sur 15m
#   - gate applied (allow / blocked + reason)
```

**Effort** : ~3 jours (400 LOC + 6 tests + smoke E2E)

---

### **P-MTF-6 — Systemd units `hsbc-h2-trend.{service,timer}` + `hsbc-execution-15m.{service,timer}`**

**Goal** : 4 fichiers systemd user pour orchestrer les 2 runners.

**Fichiers nouveaux** dans `binance_bot/systemd/` (et copies dans `~/.config/systemd/user/`) :

**hsbc-h2-trend.timer** :
```ini
[Unit]
Description=HSBC H2 trend filter computation (every 2h, 1min after H2 close)

[Timer]
OnCalendar=*-*-* 00/2:01:30 UTC
Persistent=true
Unit=hsbc-h2-trend.service

[Install]
WantedBy=timers.target
```

**hsbc-h2-trend.service** :
```ini
[Unit]
Description=HSBC H2 trend filter — calcule la direction H2 par symbole
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/ludov/HSBC-algo-genetic/binance_bot
ExecStart=/home/ludov/HSBC-algo-genetic/binance_bot/systemd/hsbc-bot-runner.sh routines.h2_trend_runner
StandardOutput=append:/home/ludov/HSBC-algo-genetic/binance_bot/logs/h2_trend.log
StandardError=append:/home/ludov/HSBC-algo-genetic/binance_bot/logs/h2_trend.log
TimeoutStartSec=120

[Install]
WantedBy=default.target
```

**hsbc-execution-15m.timer** :
```ini
[Unit]
Description=HSBC execution runner 15m (every 15min, 30s after H15 close)

[Timer]
OnCalendar=*-*-* *:00/15:30 UTC
Persistent=false
Unit=hsbc-execution-15m.service

[Install]
WantedBy=timers.target
```

**hsbc-execution-15m.service** : symétrique, appel `routines.execution_runner_15m`.

**Acceptance** :
```bash
systemctl --user daemon-reload
systemctl --user enable --now hsbc-h2-trend.timer hsbc-execution-15m.timer
systemctl --user list-timers | grep hsbc-
# Doit montrer les 2 timers + les anciens (intraday, daily, etc.)
```

**Effort** : ~0.5 jour (4 fichiers + tests systemd)

---

### **P-MTF-7 — Désactiver l'ancien `hsbc-intraday.timer` (rollover)**

**Goal** : éviter qu'`intraday_runner` (mono-H2) tourne en parallèle du
nouveau setup multi-TF — risque de conflits sur les positions.

**Décision** : ne pas le supprimer (il pourrait servir de fallback), juste le **désactiver**.

```bash
systemctl --user disable --now hsbc-intraday.timer
```

Ajout d'une ligne dans `LIMITATIONS_ACTEES.md` :
```
L-009 — hsbc-intraday.timer désactivé (mission multi-TF P-MTF-7)
- Origine : refonte multi-TF — l'intraday_runner mono-H2 est remplacé
  par hsbc-h2-trend.timer (filtre) + hsbc-execution-15m.timer (exécution).
- Si rollback : systemctl --user enable --now hsbc-intraday.timer + disable
  les 2 nouveaux timers.
```

**Effort** : ~0.5h (config systemd + doc)

---

### **P-MTF-8 — Paramètres Ichimoku 15m dans `bot_settings.yaml`**

**Goal** : exposer les params Ichimoku 15m pour ne pas hardcoder dans le
runner.

**Diff `bot_settings.yaml`** :
```yaml
# Multi-TF (mission P-MTF) — params Ichimoku par TF d'exécution
# H2 = filtre directionnel (params_today depuis phase K3)
# 15m = exécution rapide (params dédiés, plus courts)
params_15m:
  tenkan: 9
  kijun: 26
  senkou_b: 52
  shift: 26
  atr_mult: 5.0    # plus serré que H2 (atr_mult=10.5 sur phase 2)
  tp_mult: 3.0     # ratio R:R 1:3 standard

# Skip exec si H2 trend > N heures (stale)
h2_trend_max_age_hours: 3.0

# Activer le multi-TF (false = comportement legacy mono-H2)
multi_tf_enabled: true
```

**Effort** : ~0.25 jour (modif yaml + doc inline)

---

### **P-MTF-9 — Flat EOD (optionnel, si Sky le confirme)**

**Goal** : forcer flat de toutes les positions 15m à HH:55 UTC en fin de
journée si exposition trop élevée.

**Décision** : à clarifier avec Sky avant codage. La thèse initiale d'Opus
mentionnait "flat à fin de journée" comme garde-fou contre overnight gap
risk. À implémenter ou pas ? Si oui :

```python
# Dans execution_runner_15m, en fin de cycle :
def maybe_flat_eod(state_mgr, hour_utc, minute_utc):
    if hour_utc == 23 and minute_utc >= 45:
        # 15min avant minuit UTC : flat tout
        for symbol in state_mgr.get("symbols", {}):
            close_all_positions(symbol)
```

**Effort** : ~0.5 jour (si Sky valide)

---

### **P-MTF-10 — Tests d'intégration end-to-end**

**Goal** : valider le pipeline complet H2 trend → execution 15m → signal
émis ou bloqué selon aiguillage.

**Fichier nouveau** : `tests/test_multi_tf_integration.py`

**Scénarios** :
1. **H2 long + signal long 15m → trade émis**
2. **H2 long + signal short 15m → bloqué**
3. **H2 short + signal long 15m → bloqué**
4. **H2 flat + n'importe quel signal → bloqué**
5. **H2 stale (> 3h) → flat par défaut → bloqué**
6. **H2 trend OK + VPIN block → bloqué (gate VPIN gagne)**
7. **H2 trend OK + daily cap atteint → flat all + block**

**Effort** : ~2 jours (E2E avec mocks pour DataFetcher et état)

---

### **P-MTF-11 — Déploiement testnet + soak 24-48h**

**Goal** : valider le multi-TF sur testnet avant de l'activer
définitivement, monitorer pendant 24-48h.

**Étapes** :
1. Push branche `feat/multi-tf` sur GitHub
2. Merge sur main après validation Sky
3. `systemctl --user daemon-reload && enable --now hsbc-h2-trend.timer hsbc-execution-15m.timer`
4. `systemctl --user disable --now hsbc-intraday.timer`
5. Observer pendant 24-48h :
   - `tail -f logs/h2_trend.log logs/execution_runner_15m.log`
   - Vérifier nombre de signaux générés vs bloqués (par cause)
   - Vérifier balance testnet, drawdown, daily caps respectés
   - Vérifier rate limits Binance (CCXT logs)
   - Vérifier psr_history.jsonl progressivement rempli

**Acceptance** :
- ≥ 5 signaux générés en 24h (preuve que le bot trade)
- 0 erreur fatale
- 0 violation des caps (loss ≤ 10% hard cap, gain auto-flat ≤ 3% soft)
- Rate limits API < 10% du seuil Binance

**Effort** : 24-48h de soak passive + monitoring actif

---

## 4. Dépendances entre chunks

```
P-MTF-1 (DataFetcher multi-TF)
    ↓
P-MTF-2 (trend_filter_h2 module)
    ↓
P-MTF-3 (SignalEngine accepte trend_gate_fn)
    ↓                       ↘
P-MTF-4 (h2_trend_runner)   P-MTF-8 (yaml params_15m)
    ↓                       ↙
P-MTF-5 (execution_runner_15m)
    ↓
P-MTF-6 (systemd units)
    ↓
P-MTF-7 (disable old intraday timer)
    ↓
P-MTF-9 (flat EOD optionnel — décision Sky)
    ↓
P-MTF-10 (tests intégration)
    ↓
P-MTF-11 (déploiement + soak)
```

**Critical path** : P-MTF-1 → P-MTF-2 → P-MTF-3 → P-MTF-4 → P-MTF-5 →
P-MTF-6 → P-MTF-11. Total minimal : ~10-12 jours.

---

## 5. Garde-fous & règles de la mission

1. **Branche dédiée** `feat/multi-tf` — pas de push direct sur main.
2. **Pas de bascule trade_mode** — reste en `live` sur testnet.
3. **Pas de modif** du pipeline historique `ichimoku_pipeline_web_v4_8_fixed.py`.
4. **Tests pytest entre chaque commit** — pas tous en fin de mission.
5. **Soak en cours préservé** : `hsbc-vpin-live` continue, le mono-H2
   intraday tourne jusqu'à P-MTF-7 où on disable proprement.
6. **Audit chain trades_audit.jsonl** intact (P9 meta-labeling).
7. **Rollback procédé documenté** dans LIMITATIONS_ACTEES.md L-009.
8. **Approbation Sky** sur le plan (ce document) avant de coder
   P-MTF-1.

---

## 6. Questions ouvertes pour Sky (à valider avant de coder)

1. **TF d'exécution initial** : 15m seul (recommandation) ou 5m + 15m
   simultanés ?
2. **Définition tendance H2** : confirmation que "prix vs nuage Ichimoku
   H2" = la bonne convention ? (vs `tenkan > kijun` ou autre)
3. **Params 15m proposés** (`tenkan=9, kijun=26, senkou_b=52, shift=26,
   atr_mult=5.0, tp_mult=3.0`) : OK pour démarrer ou tu as des valeurs
   différentes en tête ?
4. **Flat EOD** (P-MTF-9) : on l'implémente ou pas ?
5. **Phase K3** : on l'utilise UNIQUEMENT pour les params H2 (filtre) ou
   on charge aussi des params 15m par phase ? (recommandation : params
   15m fixes, indépendants de la phase, pour simplicité)
6. **Cap nombre de trades par jour** : limite stricte ou on laisse les
   gates VPIN/VaR/HAR/caps gérer ?
7. **Ordre d'exécution** : tu valides l'ordre P-MTF-1 → P-MTF-11 ou tu
   veux qu'on parallélise certains chunks (ex: P-MTF-1 + P-MTF-2 en
   parallèle puisque indépendants) ?

---

## 7. Risques & mitigations

| Risque | Probabilité | Sévérité | Mitigation |
|---|---|---|---|
| **Surapprentissage params 15m** | Moyenne | Moyenne | Params standards `9/26/52/26` à valider sur soak avant tuning |
| **Conflits positions H2 ↔ 15m** si intraday non-disabled | Élevée si oubli | Critique | P-MTF-7 obligatoire et testé avant déploiement |
| **Stale H2 trend (timer raté)** | Faible | Moyenne | `h2_trend_max_age_hours: 3.0` → flat par défaut si stale |
| **Rate limits Binance dépassés** | Faible | Moyenne | 9 calls × 4/h = 36/h ≪ limite 2400/min. CCXT backoff auto |
| **Positions overnight perdantes** | Moyenne | Élevée | Flat EOD (P-MTF-9) si Sky valide |
| **VPIN log_only baseline cassée** | Faible | Faible | log VPIN reste actif sur tous les cycles 15m |
| **PSR live calcul biaisé sur trades 15m** | Moyenne | Moyenne | Recalibrer benchmark SR à J+15 (NEXT_STEPS_30D) |
| **Rollback complexe si problème** | Faible | Élevée | Procédure documentée L-009 (3 commandes systemctl) |

---

## 8. Estimation effort total

| Chunk | Effort |
|---|---|
| P-MTF-1 DataFetcher multi-TF | 1 jour |
| P-MTF-2 trend_filter_h2 | 0.5 jour |
| P-MTF-3 SignalEngine trend_gate | 1 jour |
| P-MTF-4 h2_trend_runner | 1.5 jour |
| P-MTF-5 execution_runner_15m | 3 jours |
| P-MTF-6 Systemd units | 0.5 jour |
| P-MTF-7 Disable old timer | 0.1 jour |
| P-MTF-8 yaml params_15m | 0.25 jour |
| P-MTF-9 Flat EOD (si validé) | 0.5 jour |
| P-MTF-10 Tests intégration | 2 jours |
| P-MTF-11 Déploiement + soak | 24-48h passif |
| **Total dev** | **~10-12 jours** |
| **Total avec soak** | **~12-14 jours** |

Concentré sur ~2 semaines de travail effectif.

---

## 9. Plan de déploiement testnet (chronologie)

```
Jour 0 (Sky valide ce plan)
  └─ Créer branche feat/multi-tf
Jour 1 — P-MTF-1 (DataFetcher) + P-MTF-2 (trend_filter)
Jour 2 — P-MTF-3 (SignalEngine)
Jour 3-4 — P-MTF-4 (h2_trend_runner) + P-MTF-8 (yaml params)
Jour 5-7 — P-MTF-5 (execution_runner_15m) + P-MTF-6 (systemd)
Jour 8 — P-MTF-9 (flat EOD si Sky valide) + P-MTF-7 (disable old)
Jour 9-10 — P-MTF-10 (tests intégration)
Jour 11 — Push branche, review Sky, merge main
Jour 12-13 — P-MTF-11 (soak 24-48h testnet)
Jour 14 — Bilan soak, GO/NO-GO mainnet ou itération
```

---

## 10. Critères de succès post-soak

À J+13 (fin du soak 48h) :

| Métrique | Critère GO | Critère NO-GO |
|---|---|---|
| **Signaux générés (24h)** | ≥ 5 | < 2 (params 15m trop stricts) |
| **Trades exécutés** | ≥ 3 | 0 (problème exec) |
| **Win rate** | > 35% | < 25% (Ichimoku 15m biaisé) |
| **Caps respectés** | 100% | 1+ violation |
| **PSR (si calculable)** | ≥ 0.3 | < 0.1 (kill_system) |
| **Erreurs fatales** | 0 | ≥ 1 |
| **API rate limit** | < 30% du seuil | ≥ 80% |
| **VPIN baseline** | continue à se remplir | régression |

Si **GO** : laisser tourner 30j supplémentaires (cohérent avec
NEXT_STEPS_30D), revue à J+30.

Si **NO-GO** : itération sur params 15m ou retour mono-H2.

---

## 11. Mise à jour des memories Muninn

Une fois la mission validée et démarrée, ajouter une memory de type
`project` :

```markdown
---
name: Mission multi-TF en cours
description: Refonte exécution courte (15m) indexée tendance H2 — branche feat/multi-tf
type: project
---

Mission v4 démarrée le 2026-04-30 :
- Bot mono-H2 → archi 2 couches (filtre H2 + exécution 15m)
- Plan complet dans `MISSION_MULTI_TF.md` racine du repo
- Branche `feat/multi-tf`, mergée à J+11 attendu
- Soak testnet 48h post-merge

**Why** : Sky a réalisé le 2026-04-30 que le bot tournait en mono-H2 alors
qu'il pensait avoir une archi multi-TF. C'est un gap de la stack initiale.

**How to apply** : tant que la mission tourne, pas de modif sur le pipeline
historique ni le mono-H2 (intraday_runner reste figé). Tous les chunks
P-MTF-N respectent l'audit BUGS.md, LIMITATIONS_ACTEES.md, NEXT_STEPS_30D.md.
```

---

## 12. Ce que je NE fais PAS dans cette mission

- ❌ Implémenter le bias directionnel daily depuis phase K3 + halving (cf
  REPORT_CHECK section 7.1) — c'est une autre mission post-J+30
- ❌ Implémenter la fenêtre 60-120s post-close (cf REPORT_CHECK 7.4) —
  pareil, post-J+30
- ❌ Toucher au pipeline historique `ichimoku_pipeline_web_v4_8_fixed.py`
- ❌ Toucher au TradeManager / ordre flow
- ❌ Modifier les gates (VPIN, VaR, HAR-RV, caps, drawdown) — préservés
  intégralement
- ❌ Bascule trade_mode → mainnet — testnet uniquement
- ❌ Ajouter de nouveaux symboles — reste BTC/ETH/SOL
- ❌ MISSION_REWFA (BUG-C/D/E) — mission séparée déjà tracée

---

## 13. Demande de validation Sky

**Sky : merci de répondre par OUI / NON / MODIF sur chaque section
critique** :

- [ ] **Validation archi générale** (section 2) : 2 couches H2/15m, prix vs nuage Ichimoku H2 comme filtre directionnel — OK ?
- [ ] **TF d'exécution initial** (section 2.3) : 15m seul d'abord, puis 5m en phase 2 — OK, ou tu veux 5m+15m simultanés dès le départ ?
- [ ] **Aiguillage pyramidal strict** (section 2.4) : signal 15m rejeté si direction adverse au filtre H2 — OK ?
- [ ] **Params 15m proposés** (P-MTF-8) : `tenkan=9, kijun=26, senkou_b=52, shift=26, atr_mult=5.0, tp_mult=3.0` — OK ou modif ?
- [ ] **Flat EOD** (P-MTF-9) : on l'implémente (HH:23:45 UTC flat all) ou on saute ?
- [ ] **Disable hsbc-intraday.timer** (P-MTF-7) : OK pour rollover propre ?
- [ ] **Branche dédiée `feat/multi-tf`** (section 5.1) : OK, ou tu préfères main direct ?
- [ ] **Effort 10-14 jours** (section 8) : OK comme estimation ou tu veux compresser ?

Tu peux aussi répondre **"OK fonce sur tout"** si rien ne te choque, et
je commence P-MTF-1 immédiatement.

---

*Plan rédigé en autonomie. En attente de validation Sky.*
*Aucune ligne de code n'a été écrite à ce stade. Repo intact, soak en cours.*
