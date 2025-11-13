# 🚀 GUIDE RAPIDE: Configuration Clés API Binance

## ✅ Quelle clé choisir sur Binance?

**👉 CHOISIS LA PREMIÈRE: "Auto-générée (HMAC)"**

- ✅ Compatible avec CCXT (bibliothèque utilisée par le bot)
- ✅ Simple: API Key + Secret (comme un login/mot de passe)
- ✅ Standard pour trading automatique

❌ **NE PRENDS PAS** Ed25519/RSA (c'est pour trading haute fréquence, plus complexe)

---

## 📝 Étapes de Configuration

### 1. Sur Binance (création clé)

1. Va dans **API Management** → **Create API**
2. Choisis **"Auto-générée (HMAC)"**
3. **IMPORTANT**: Active uniquement les permissions:
   - ✅ **Enable Reading** (lire données)
   - ✅ **Enable Futures** (si tu veux du levier)
   - ✅ **Enable Spot & Margin Trading** (pour trading)
   - ❌ **NE PAS activer** "Enable Withdrawals" (sécurité)
4. Copie **API Key** et **Secret Key** (tu ne verras le secret qu'une fois!)

### 2. Dans le Bot (configuration locale)

```powershell
cd binance_bot
python configure_api_keys.py
```

Tu entres:
- API Key (copié depuis Binance)
- Secret (copié depuis Binance)
- Mode testnet? (o/n) → **Commence par "o" pour tester!**

✅ Les clés sont sauvegardées dans `.env` (local uniquement, pas sur GitHub)

### 3. Vérifier la configuration

```powershell
python check_real.py
```

Ça va tester la connexion à Binance et afficher ton solde.

---

## 👀 Comment voir ton algo tourner sur Binance?

### Option 1: Interface Binance (recommandé)

1. **Binance Web/App** → **Futures** (ou Spot)
2. Tu verras:
   - Positions ouvertes en temps réel
   - Ordres (entry, stop loss, take profit)
   - Historique des trades
   - P&L (profit/perte)

### Option 2: Logs du Bot

Le bot écrit dans `binance_bot/logs/`:
- `bot.log` → toutes les actions (signaux, trades, erreurs)
- `trades.json` → historique des trades

### Option 3: Dashboard (à venir)

On peut créer un petit dashboard web pour visualiser en temps réel.

---

## 🔄 Fonctionnement du Bot

### Daily (1x par jour, 00:00 UTC)
- Met à jour la phase HMM (K3) du marché
- Charge les paramètres Ichimoku optimisés pour cette phase

### Intraday (toutes les 2h)
- Vérifie signaux Ichimoku
- Ouvre/ferme positions selon la logique du backtest
- Place stop loss + take profit automatiquement

### Tu verras sur Binance:
- ✅ Positions ouvertes (LONG ou SHORT)
- ✅ Ordres stop loss actifs
- ✅ Ordres take profit actifs
- ✅ Trades exécutés dans l'historique

---

## ⚠️ Sécurité

- ✅ `.env` est dans `.gitignore` → **jamais sur GitHub**
- ✅ Clés stockées localement uniquement
- ✅ Permissions limitées (pas de withdrawal)
- ✅ Mode testnet pour tester avant le réel

---

## 🎯 Prochaines Étapes

1. Créer clé API sur Binance (HMAC)
2. `python configure_api_keys.py` → entrer clés
3. `python check_real.py` → vérifier connexion
4. `python routines/daily_phase_job.py` → test daily
5. `python routines/intraday_runner.py --mode simulation` → test intraday
6. `python routines/intraday_runner.py --mode live` → **GO LIVE!** 🚀

---

## 📞 Support

Si erreur de connexion:
- Vérifie que les clés sont correctes dans `.env`
- Vérifie permissions API sur Binance
- Vérifie que tu es sur testnet ou live selon config


