# Data Sources

This directory contains cached OHLCV price histories used in experiments.
Each CSV follows the header `timestamp,open,high,low,close,volume`.

## Files

- **BTC_USDT_1d.csv** – daily BTC/USDT data from Binance, covering 2017-08-17 to 2025-08-26.
- **BTC_USDT_2h.csv** – 2-hour BTC/USDT data from Binance, covering 2017-08-17 04:00 to 2025-08-26 10:00.
- **BTC_USD_1d.csv** – daily BTC/USD data from Bitstamp, covering 2021-12-06 to 2024-08-30.
- **BTC_USD_1h.csv** – hourly BTC/USD data from Bitstamp, covering 2011-08-18 12:00 to 2025-08-26 12:00.
- **BTC_USD_2h.csv** – 2-hour BTC/USD data from Bitstamp, covering 2011-08-18 12:00 to 2025-08-26 12:00.
- **ETH_USDT_2h.csv** – 2-hour ETH/USDT data from Binance, covering 2020-08-13 08:00 to 2025-08-13 08:00.
- **DOGE_USDT_2h.csv** – 2-hour DOGE/USDT data from Binance, covering 2020-08-13 08:00 to 2025-08-13 08:00.

## Source & Usage Rights

Data is downloaded via [CCXT](https://github.com/ccxt/ccxt) from public exchange APIs.
Usage is subject to the providers' terms of service:

- Binance: <https://www.binance.com/en/terms>
- Bitstamp: <https://www.bitstamp.net/legal/api/>

Data is provided for research and educational purposes. Redistribution may be restricted by the exchanges.

## Updating the Cache

Use [`scripts/fetch_btc_history.py`](../scripts/fetch_btc_history.py) to fetch or refresh histories. Example:

```bash
py -3 scripts/fetch_btc_history.py --symbol BTC/USDT --timeframe 2h --exchange binance --years-back 8
```

Adjust the symbol, timeframe, or exchange as needed.

