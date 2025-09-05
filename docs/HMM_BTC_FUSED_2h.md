### HMM K‑selection (seeds=60) — BTC_FUSED 2h

|   K |         AIC_mean |       AIC_median |         BIC_mean |       BIC_median |           LL_mean |   LL_median |   count |
|----:|-----------------:|-----------------:|-----------------:|-----------------:|------------------:|------------:|--------:|
|  10 | 763049           | 757524           | 765118           | 759593           | -791747           |     -241134 |      60 |
|   9 | 778565           | 771512           | 780348           | 773295           | -620558           |     -242613 |      60 |
|   8 | 805041           | 805259           | 806556           | 806774           |      -1.09892e+06 |     -244210 |      60 |
|   7 | 834172           | 822046           | 835436           | 823310           |      -1.05259e+06 |     -247284 |      60 |
|   6 | 846977           | 837859           | 848007           | 838889           | -251521           |     -245566 |      60 |
|   5 | 894325           | 884526           | 895138           | 885339           | -255159           |     -248568 |      60 |
|   4 | 936024           | 944672           | 936638           | 945286           | -260336           |     -257642 |      60 |
|   3 | 970406           | 968145           | 970839           | 968578           | -255837           |     -254696 |      60 |
|   2 |      1.05216e+06 |      1.05226e+06 |      1.05243e+06 |      1.05253e+06 | -264792           |     -264795 |      60 |

Recommandation (BIC_median): K=10

Conclusions

- Sur BTC_FUSED 2h (2011→2025) avec features spectre Welch + Ichimoku, la sélection par BIC_median et LL_OOS_median s'améliore en montant K, avec un optimum à K=10.
- Pour la comparaison thèse et l'analyse pratique, on fige néanmoins K=3 (Up/Down/Range) et K=5 (Accumulation, Expansion, Euphorie, Distribution, Bear) afin de comparer « fixe vs par phase » côté Ichimoku.
- Étape suivante: geler les labels K=3 et K=5, puis lancer l'optimisation Ichimoku (30×2 seeds) « fixe vs par phase » et comparer equity/CAGR/Calmar/Sharpe/MDD par fenêtres halving.