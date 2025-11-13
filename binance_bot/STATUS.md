# ✅ CONFIGURATION COMPLÈTE - Bot Binance Prêt

## 🔐 Clés API Configurées

✅ **Clés API sauvegardées dans `.env`** (local uniquement, PAS sur GitHub)
- Fichier: `binance_bot/.env`
- Protégé par `.gitignore` → **jamais commité sur GitHub**
- Mode: **LIVE (compte réel)** - `BINANCE_TESTNET=false`

---

## 📁 Fichiers Configurés

✅ **Labels K3 1D stable**
- Source: `outputs/fourier/labels_frozen/BTC_FUSED_2h/K3_1d_stable.csv`
- Copié dans: `binance_bot/data/K3_1d_stable.csv`

✅ **Paramètres phases K3**
- Fichier: `binance_bot/configs/phase_params_K3.json`
- Contient médianes optimisées pour phases 0, 1, 2

✅ **Configuration bot**
- Fichier: `binance_bot/configs/bot_settings.yaml`
- Symbol: BTC/USDT
- Timeframe: 2h
- Risk management configuré

---

## 🧪 Tests à Effectuer

### 1. Test Connexion Binance

```powershell
cd binance_bot
python test_connection.py
```

**Attendu:**
- ✅ Connexion réussie
- ✅ Solde USDT affiché
- ✅ Prix BTC/USDT récupéré

### 2. Test Daily Phase Job

```powershell
python routines/daily_phase_job.py
```

**Attendu:**
- ✅ Phase du jour déterminée
- ✅ Paramètres chargés
- ✅ `data/state.json` créé/mis à jour

### 3. Test Intraday Runner (SIMULATION)

```powershell
python routines/intraday_runner.py
```

**Attendu:**
- ✅ Bougies récupérées depuis Binance
- ✅ Ichimoku calculé
- ✅ Signaux détectés (si conditions remplies)
- ⚠️ Mode SIMULATION → pas d'ordres réels

---

## 🚀 Passage en LIVE

### ⚠️ AVANT DE PASSER EN LIVE:

1. **Tester en simulation** plusieurs jours
2. **Vérifier les logs** (`binance_bot/logs/bot.log`)
3. **Vérifier le solde** sur Binance
4. **Commencer avec petit capital** (test)

### Pour activer le mode LIVE:

1. **Modifier `routines/intraday_runner.py` ligne 62:**
   ```python
   trade_mode = "live"  # Au lieu de "simulation"
   ```

2. **Vérifier `.env`:**
   ```
   BINANCE_TESTNET=false
   ```

3. **Lancer le bot:**
   ```powershell
   # Daily (1×/jour à 00:05 UTC)
   python routines/daily_phase_job.py
   
   # Intraday (toutes les 2h)
   python routines/intraday_runner.py
   ```

---

## 📊 Monitoring

### Sur Binance:
- **Futures** → Positions ouvertes
- **Orders** → Stop loss / Take profit actifs
- **Trade History** → Historique des trades

### Logs Locaux:
- `binance_bot/logs/bot.log` → Toutes les actions
- `binance_bot/data/state.json` → État actuel (phase, positions, equity)

---

## 🔒 Sécurité

✅ **Clés API protégées:**
- `.env` dans `.gitignore`
- Jamais commité sur GitHub
- Stockage local uniquement

✅ **Permissions Binance:**
- ✅ Reading (lecture données)
- ✅ Futures Trading
- ❌ Withdrawals (désactivé pour sécurité)

---

## 📞 Support

Si erreur de connexion:
1. Vérifier `.env` (clés correctes)
2. Vérifier permissions API sur Binance
3. Vérifier connexion internet
4. Exécuter `python test_connection.py` pour diagnostic

---

## ✅ Checklist Avant Live

- [ ] Test connexion réussi (`test_connection.py`)
- [ ] Daily phase job fonctionne
- [ ] Intraday runner fonctionne en simulation
- [ ] Logs vérifiés (pas d'erreurs)
- [ ] Solde Binance vérifié
- [ ] Mode LIVE activé dans `intraday_runner.py`
- [ ] `BINANCE_TESTNET=false` dans `.env`
- [ ] Capital de départ défini (petit montant pour test)

---

**🎯 Le bot est prêt! Commence par tester la connexion.**

