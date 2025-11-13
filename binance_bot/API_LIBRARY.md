# Bibliothèque API Binance

## Deux Options Disponibles

### Option 1: CCXT (actuellement utilisé) ✅
- **Avantage**: Multi-exchange (si tu veux changer d'exchange plus tard)
- **Avantage**: Interface unifiée pour tous les exchanges
- **Inconvénient**: Pas la bibliothèque officielle Binance

**Installation:**
```bash
pip install ccxt
```

**Utilisé dans:** `services/data_fetcher.py`

---

### Option 2: Binance Connector Python (officiel) 🆕
- **Avantage**: Bibliothèque officielle Binance (recommandée par Binance)
- **Avantage**: Meilleure intégration avec spécificités Binance
- **Inconvénient**: Spécifique à Binance uniquement

**Installation:**
```bash
pip install binance-connector
```

**Utilisé dans:** `services/data_fetcher_binance_official.py` (version alternative)

---

## Recommandation

**Pour compte réel Binance**: Utiliser **Binance Connector** (officiel) est recommandé car:
- Support officiel Binance
- Potentiellement plus à jour avec nouvelles fonctionnalités
- Meilleure gestion des erreurs spécifiques Binance

**Pour flexibilité multi-exchange**: Garder **CCXT** si tu veux tester sur d'autres exchanges.

---

## Changer de Bibliothèque

Si tu veux utiliser la bibliothèque officielle Binance:

1. **Modifier `routines/intraday_runner.py`:**
```python
# Remplacer
from services.data_fetcher import DataFetcher

# Par
from services.data_fetcher_binance_official import DataFetcherBinanceOfficial as DataFetcher
```

2. **Modifier `bot/trade_manager.py`:**
```python
# Adapter les appels API selon la nouvelle bibliothèque
# (voir data_fetcher_binance_official.py pour exemples)
```

3. **Installer:**
```bash
pip install binance-connector
```

---

## Note

Les deux bibliothèques fonctionnent très bien. CCXT est déjà implémenté et testé. La bibliothèque officielle Binance est disponible en alternative si tu préfères.

