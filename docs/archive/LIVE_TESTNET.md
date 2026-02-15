Live (Binance Testnet)
======================

Pré‑requis
----------
- Créer des clés API Futures Testnet (lecture/écriture)
- Remplir outputs/TESTNET_ENV.txt: BINANCE_API_KEY, BINANCE_API_SECRET, POSITION_SIZE, LEVERAGE, MAX_POS_PER_SIDE, garde‑fous
- Vérifier BEST_BASELINE.json dans outputs/

Lancement
---------

pwsh
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -u .\outputs\live_trader_testnet.py

Fonctionnalités
---------------
- Lecture BEST_BASELINE.json et signaux Ichimoku (cassure Kumo)
- Ordres MARKET + Trailing Stop serveur Binance (TRAILING_STOP_MARKET) avec callbackRate dérivé de ATR×mult/prix
- Fallback STOP_MARKET si le trailing n’est pas accepté
- Garde‑fous: stop global (ex: −50%), daily loss cap (ex: 8%)

Notes
-----
- Le script n’utilise pas de proxy par défaut. Si votre réseau bloque Binance, définir proxies dans ccxt.
- Pour déploiement 24/7, voir la roadmap (Docker/VPS).


