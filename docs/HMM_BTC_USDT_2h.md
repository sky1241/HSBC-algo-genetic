### HMM K‑selection — BTC_USDT 2h (provisoire)

Note: la sélection ci‑dessous doit être ré‑estimée sur le dataset H2 fusionné (Bitstamp+Binance) avec grille K ∈ [3..10] et validation out‑of‑sample. Les labels de production seront figés pour K=3 et K=5, conformément au protocole.

|   K |    AIC |    BIC |   LL_OOS |
|----:|-------:|-------:|---------:|
|   6 | 830065 | 831095 |  -244648 |
|   5 | 860087 | 860901 |  -274154 |
|   4 | 911554 | 912168 |  -280856 |
|   3 | 970660 | 971093 |  -263212 |

Recommandation provisoire: comparer K=3 et K=5 (labels figés) sur l’horizon OOS; ne retenir K plus élevé (K≥6) que si le BIC baisse nettement et si le LL_OOS par observation s’améliore de façon robuste.