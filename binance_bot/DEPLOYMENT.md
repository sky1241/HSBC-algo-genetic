# Guide de Déploiement Bot Binance K3

## Pré-requis

1. **Python 3.10+** installé
2. **Clés API Binance** (testnet ou live)
3. **Labels K3 1D stable** (`K3_1d_stable.csv`)
4. **Paramètres optimisés** par phase (médianes WFA K3)

---

## Installation

```bash
cd binance_bot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Configuration

### 1. Clés API

```bash
cp configs/env.example .env
# Éditer .env avec vos clés
```

```env
BINANCE_API_KEY=votre_cle_api
BINANCE_API_SECRET=votre_secret
BINANCE_TESTNET=true  # false pour live
```

### 2. Copier données

```bash
cp ../outputs/fourier/labels_frozen/BTC_FUSED_2h/K3_1d_stable.csv data/
```

### 3. Générer paramètres par phase

Depuis vos analyses K3 (médianes), créer `configs/phase_params_K3.json`:

```json
{
  "0": {"tenkan": 27, "kijun": 100, "senkou_b": 180, "shift": 93, "atr_mult": 11.8, "tp_mult": 20.0},
  "1": {"tenkan": 29, "kijun": 52, "senkou_b": 210, "shift": 96, "atr_mult": 19.3, "tp_mult": 30.0},
  "2": {"tenkan": 21, "kijun": 35, "senkou_b": 90, "shift": 44, "atr_mult": 10.5, "tp_mult": 8.0}
}
```

---

## Tests

### Test unitaires
```bash
python tests/test_signal_engine.py
```

### Test daily job (manuel)
```bash
python routines/daily_phase_job.py
```

### Test intraday (simulation)
```bash
python routines/intraday_runner.py
```

---

## Déploiement Production

### Option 1: Cron (Linux/Mac)

```bash
crontab -e
```

Ajouter:
```cron
# Daily phase update à 00:05 UTC
5 0 * * * cd /path/binance_bot && /path/.venv/bin/python routines/daily_phase_job.py >> logs/daily.log 2>&1

# Intraday runner toutes les 2h
5 */2 * * * cd /path/binance_bot && /path/.venv/bin/python routines/intraday_runner.py >> logs/intraday.log 2>&1
```

### Option 2: Systemd (Linux)

Créer `/etc/systemd/system/binance-bot-intraday.timer`:

```ini
[Unit]
Description=Binance Bot Intraday (toutes les 2h)

[Timer]
OnCalendar=*:05:00  # Toutes les 2h à :05
Persistent=true

[Install]
WantedBy=timers.target
```

### Option 3: Docker (recommandé)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "runner.py", "--mode", "intraday"]
```

Orchestrer avec `docker-compose` + cron externe ou k8s CronJob.

---

## Monitoring

### Logs
- `logs/daily.log` : mise à jour phase quotidienne
- `logs/intraday.log` : signaux + trades
- `data/state.json` : état actuel (positions, equity)

### Alertes (TODO)
- Intégrer Telegram/Slack pour notifications (stop global, erreurs API)

### Dashboard (TODO)
- Streamer `state.json` vers Grafana/Google Sheets

---

## Sécurité

- ⚠️ **Testnet d'abord** : valider pendant 1-2 semaines
- ⚠️ **Capital limité** : démarrer avec petit capital (100-500 USDT)
- ⚠️ **Monitoring 24/7** : configurer alertes
- ⚠️ **Backup état** : sauvegarder `state.json` régulièrement
- ⚠️ **Permissions API** : trading uniquement (pas de withdraw)

---

## Troubleshooting

### Bot ne détecte pas de signaux
- Vérifier que `daily_phase_job` a tourné aujourd'hui
- Vérifier `state.json` : `phase_today` et `params_today` doivent être définis

### Erreur "Labels introuvables"
- Copier `K3_1d_stable.csv` dans `data/`

### Erreur API Binance
- Vérifier clés API dans `.env`
- Vérifier rate limits (CCXT `enableRateLimit=True`)

### Stop global atteint
- Bot s'arrête automatiquement
- Vérifier `state.json` : `equity` et `max_drawdown`
- Décider si relance avec nouveau capital ou analyse post-mortem

