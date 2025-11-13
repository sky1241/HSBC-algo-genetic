# Guide Configuration COMPTE RÉEL Binance

## ⚠️ SÉCURITÉ AVANT TOUT

1. **Capital limité**: Commencer avec une petite somme (100-500 USDT max)
2. **Clés API sécurisées**: Permissions TRADING uniquement (pas de withdraw)
3. **Monitoring 24/7**: Surveiller logs et equity
4. **Stop global**: Configuré à 50% (bot s'arrête automatiquement)

---

## Configuration Compte Réel

### 1. Créer clés API Binance

1. Aller sur Binance → API Management
2. Créer nouvelle clé API
3. **Permissions**: ✅ Enable Reading + ✅ Enable Futures (trading uniquement)
4. ❌ **NE PAS** activer Withdraw
5. Copier API Key et Secret

### 2. Configurer .env

```bash
cd binance_bot
cp configs/env.real.example .env
```

Éditer `.env`:
```env
BINANCE_API_KEY=votre_cle_api_reelle
BINANCE_API_SECRET=votre_secret_api_reel
BINANCE_TESTNET=false
```

### 3. Vérifier paramètres K3

Le fichier `configs/phase_params_K3.json` doit contenir les médianes de tes 30 seeds K3.

**Si tu veux régénérer depuis tes résultats réels:**
```bash
python scripts/extract_k3_params_for_bot.py
```

### 4. Copier labels K3 1D stable

```bash
# Copier depuis ton dossier de résultats
cp outputs/fourier/labels_frozen/BTC_FUSED_2h/K3_1d_stable.csv binance_bot/data/
```

### 5. Activer mode LIVE

Éditer `routines/intraday_runner.py` ligne ~50:
```python
trade_mode = "live"  # Au lieu de "simulation"
```

---

## Tests Avant Live

### Test 1: Daily Phase Job
```bash
python routines/daily_phase_job.py
```
Vérifier que `data/state.json` est créé avec phase et params.

### Test 2: Intraday (simulation d'abord)
```bash
python routines/intraday_runner.py
```
Vérifier logs: doit afficher `[SIMULATION]` si mode simulation.

### Test 3: Vérifier connexion Binance
```python
python -c "from services.data_fetcher import DataFetcher; d=DataFetcher(); print(d.get_current_price())"
```

---

## Déploiement Production

### Option 1: Cron Windows (Task Scheduler)

1. Ouvrir Task Scheduler
2. Créer tâche "Binance Bot Daily" (déclencheur: tous les jours 00:05)
3. Action: `python.exe` → `C:\path\binance_bot\routines\daily_phase_job.py`
4. Créer tâche "Binance Bot Intraday" (déclencheur: toutes les 2h)
5. Action: `python.exe` → `C:\path\binance_bot\routines\intraday_runner.py`

### Option 2: Service Windows (NSSM)

```bash
# Installer NSSM
nssm install BinanceBotDaily "C:\Python\python.exe" "C:\path\binance_bot\routines\daily_phase_job.py"
nssm start BinanceBotDaily
```

### Option 3: Linux/Mac (Cron)

```cron
# Daily phase update (00:05 UTC)
5 0 * * * cd /path/binance_bot && /path/.venv/bin/python routines/daily_phase_job.py >> logs/daily.log 2>&1

# Intraday runner (toutes les 2h)
5 */2 * * * cd /path/binance_bot && /path/.venv/bin/python routines/intraday_runner.py >> logs/intraday.log 2>&1
```

---

## Monitoring

### Logs
- `logs/daily.log` : Mise à jour phase quotidienne
- `logs/intraday.log` : Signaux + trades exécutés
- `data/state.json` : État actuel (positions, equity, drawdown)

### Vérifier état
```bash
# Lire state.json
python -c "import json; print(json.dumps(json.load(open('binance_bot/data/state.json')), indent=2))"
```

### Alertes (TODO)
- Configurer Telegram/Slack pour notifications:
  - Stop global atteint
  - Erreur API
  - Trade exécuté

---

## Checklist Avant Live

- [ ] Clés API créées (permissions trading uniquement)
- [ ] `.env` configuré avec clés réelles
- [ ] `BINANCE_TESTNET=false` dans .env
- [ ] `trade_mode = "live"` dans intraday_runner.py
- [ ] `phase_params_K3.json` généré depuis analyses réelles
- [ ] `K3_1d_stable.csv` copié dans `data/`
- [ ] Capital limité (100-500 USDT)
- [ ] Test daily_phase_job OK
- [ ] Test intraday (simulation) OK
- [ ] Connexion Binance testée
- [ ] Planification automatique configurée (cron/service)
- [ ] Monitoring logs activé

---

## En Cas de Problème

### Bot ne trade pas
- Vérifier `state.json`: `phase_today` et `params_today` doivent être définis
- Vérifier logs: erreurs API?
- Vérifier que daily_phase_job a tourné aujourd'hui

### Stop global atteint
- Bot s'arrête automatiquement
- Vérifier `state.json`: `equity` et `max_drawdown`
- Décider: relancer avec nouveau capital ou analyse post-mortem

### Erreur API Binance
- Vérifier clés API dans `.env`
- Vérifier permissions API (trading activé?)
- Vérifier rate limits (CCXT gère automatiquement)

---

## Performance Attendue (K3 1D Stable)

Basé sur tes 30 seeds:
- **Monthly médian**: ~0.30% (0.14% - 0.65% IQR)
- **MDD médian**: ~12.2% (8% - 13% IQR)
- **Survie**: 100% (toutes seeds > 0)

⚠️ **Performance live peut différer** (slippage, latence, spread).

---

## Support

En cas de bug ou question:
1. Vérifier logs (`logs/intraday.log`)
2. Vérifier `state.json`
3. Tester connexion Binance manuellement
4. Relancer en mode simulation pour debug

