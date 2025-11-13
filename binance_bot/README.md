# Bot Binance K3 1D Stable

Bot de trading automatique implémentant la stratégie K3 1D stable validée par backtest (30 seeds, 100% survie).

## Architecture

- `configs/` : Configuration (API, paramètres par phase, settings bot)
- `data/` : Labels K3 1D stable + state.json (positions, equity)
- `services/` : Modules métier (phase labeller, Ichimoku, data fetcher)
- `bot/` : Gestion trading (state, risk, trade manager)
- `routines/` : Scripts planifiés (daily + intraday)
- `tests/` : Tests unitaires (parité avec backtest)

## Installation Express (Recommandé)

```bash
cd binance_bot
python install.py              # Setup automatique complet
# Éditer .env avec vos clés API Binance
python check_real.py           # Vérifier configuration
```

**Ou installation manuelle:**
```bash
cd binance_bot
pip install -r requirements.txt
cp configs/env.real.example .env
# Éditer .env avec vos clés Binance
```

## Configuration

1. **Générer `configs/phase_params_K3.json`** depuis vos analyses WFA (médianes par phase)
2. **Copier `data/K3_1d_stable.csv`** (labels pré-calculés)
3. **Éditer `configs/bot_settings.yaml`** (capital, levier, risque)

## Utilisation

### Mode Simulation (sans ordres réels)
```bash
python runner.py --mode simulation
```

### Mode Daily Phase Update (1×/jour à 00:05)
```bash
python routines/daily_phase_job.py
```

### Mode Intraday (toutes les 2h)
```bash
python routines/intraday_runner.py
```

### Planification automatique (cron Linux)
```cron
5 0 * * * /usr/bin/python /path/binance_bot/routines/daily_phase_job.py
5 */2 * * * /usr/bin/python /path/binance_bot/routines/intraday_runner.py
```

## Stratégie (K3 1D Stable + TP Adaptatif)

- **Phases:** 3 régimes HMM (Phase 0/1/2)
- **Labels:** 1D stable (vote majoritaire jour J pour trader J+1)
- **Trading:** Exécution H2 (12 barres/jour)
- **Paramètres:** Adaptatifs par phase (Ichimoku, ATR stop, TP)
- **Risque:** 1% capital/trade, levier max 10×, stop global 50%

## Tests

```bash
pytest tests/
```

## Sécurité

- ⚠️ **Ne jamais commit le fichier `.env`** (clés API)
- ✅ Tester sur testnet d'abord
- ✅ Limiter capital initial
- ✅ Monitoring 24/7 (logs + alertes)

## Backtest vs Live

Le bot reproduit EXACTEMENT la logique du backtest:
- Mêmes signaux Ichimoku
- Mêmes ATR stop/TP
- Mêmes règles de pyramiding
- Même stop global

Performance attendue (30 seeds K3):
- Monthly: 0.30% médian
- MDD: 12.2% médian
- Survie: 100%
