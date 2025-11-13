# Guide Rapide - Bot Binance K3

## 🚀 Installation Express (3 commandes)

```bash
cd binance_bot
python install.py              # Setup automatique
# Éditer .env avec vos clés API
python check_real.py           # Vérifier config
```

## 📋 Configuration Minimale

1. **Clés API Binance** → Éditer `.env`
2. **Labels K3** → Copier `K3_1d_stable.csv` dans `data/`
3. **Paramètres phases** → Déjà dans `configs/phase_params_K3.json`

## ⚙️ Utilisation

### Daily (1×/jour à 00:05)
```bash
python routines/daily_phase_job.py
```

### Intraday (toutes les 2h)
```bash
python routines/intraday_runner.py
```

## 🔒 Compte Réel

1. `.env`: `BINANCE_TESTNET=false`
2. `intraday_runner.py` ligne 62: `trade_mode = "live"`
3. Capital limité (100-500 USDT)

## 📚 Documentation

- `README.md` - Architecture complète
- `CONFIG_REAL.md` - Guide compte réel détaillé
- `API_LIBRARY.md` - Bibliothèques API (CCXT utilisé)

## ⚠️ Sécurité

- Clés API: permissions trading uniquement (pas withdraw)
- Stop global: 50% equity automatique
- Monitoring: logs dans `logs/`

---

**Bibliothèque utilisée**: CCXT (simple, fiable, multi-exchange)

