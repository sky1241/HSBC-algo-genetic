# Méthode de Welch et analyse alignée sur le halving

Ce document poursuit le rappel théorique en détaillant l'estimation de la densité spectrale via la méthode de Welch et son application aux données BTC H2 depuis le halving.

## 1. Principe de la méthode de Welch
1. **Segmentation** : découper la série en \(K\) segments de longueur \(L\) avec chevauchement.
2. **Fenêtrage** : multiplier chaque segment par une fenêtre (Hanning dans le dépôt).
3. **FFT** : calculer la transformée de Fourier de chaque segment.
4. **Périodogramme segmentaire** : normaliser le carré du module.
5. **Moyenne** : la PSD finale est la moyenne des périodogrammes segmentaires.

Cette procédure réduit la variance de l'estimation par rapport au périodogramme simple.

Les paramètres usuels pour `scipy.signal.welch` sont `nperseg ≈ 256` et `noverlap ≈ 128` avec une fenêtre de Hanning. Un `nperseg` plus long améliore la résolution fréquentielle mais diminue le nombre de segments disponibles, ce qui accroît la variance de l'estimation; augmenter `noverlap` réduit la variance au prix d'un surcoût de calcul.

## 2. Pipeline depuis le halving
1. **Alignement temporel** : fixer \(t=0\) au halving (ex. 20 avril 2024) et charger les chandeliers H2.
2. **Log‑rendements** : calculer \(r_t = \ln P_t - \ln P_{t-1}\) puis \(|r_t|\) ou \(r_t^2\) pour la volatilité.
3. **Fenêtre mensuelle** : recalculer la PSD chaque mois sur ~360 barres H2 (≈30 jours) pour capter les changements rapides.
4. **Pente log–log et LFP** : estimer l'exposant \(\alpha\) du bruit \(1/f^\alpha\) et le ratio d'énergie basse fréquence pour guider les paramètres Ichimoku.

## 3. Exemple de code et graphique
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scripts.fourier_utils import (
    compute_welch_psd,
    dominant_period,
    low_freq_power_ratio,
)

# 1. Charger les chandeliers H2 et se placer après le halving
csv = Path("data/BTC_USDT_2h.csv")
df = (pd.read_csv(csv, parse_dates=['timestamp'])
        .set_index('timestamp')
        .sort_index())
df = df[df.index >= "2024-04-20"]            # t = 0 au halving
df['log_close'] = np.log(df['close'])
df['ret'] = df['log_close'].diff().dropna()

# 2. Découper en fenêtres mensuelles
bars_per_month = 30 * 12                     # ~360 barres H2
starts = range(0, len(df['ret']) - bars_per_month + 1, bars_per_month)

spectra = []
for s in starts:
    seg = df['ret'].iloc[s:s + bars_per_month].values
    freqs, psd = compute_welch_psd(seg, fs=1.0)
    spectra.append((freqs[1:], psd[1:]))      # ignorer la composante DC

# 3. Tracé du dernier mois
freqs, psd = spectra[-1]
plt.figure(figsize=(6,4))
plt.loglog(freqs, psd, label="PSD ret H2 (dernier mois)")
plt.xlabel("Fréquence (cycles/barre)")
plt.ylabel("Puissance")
plt.title("Méthode de Welch – BTC ret depuis halving")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.show()

# 4. Métriques supplémentaires
P = dominant_period(freqs, psd)
lfp = low_freq_power_ratio(freqs, psd, f0=1/(5 * 12))
print(f"Période dominante ≈ {P:.1f} barres")
print(f"LFP ≈ {lfp:.2%}")

# 5. Pente log–log (optionnel)
log_f, log_psd = np.log(freqs), np.log(psd)
alpha, _ = np.polyfit(log_f, log_psd, 1)
print("Alpha ≈", -alpha)
```

Ce script illustre le calcul de la PSD mensuelle, l'extraction de la période dominante et du LFP, ainsi que l'estimation de la pente \(\alpha\). Il produit un graphique log–log pour la fenêtre la plus récente. Une valeur de LFP élevée (> 60 %) traduirait un régime tendanciel dominé par les basses fréquences.

## 4. Suivi de l'évolution mensuelle
L'exemple suivant calcule la pente \(\alpha\) pour chaque mois depuis le halving et affiche son évolution temporelle.

```python
alphas = []
for freqs, psd in spectra:
    log_f, log_psd = np.log(freqs), np.log(psd)
    a, _ = np.polyfit(log_f, log_psd, 1)
    alphas.append(-a)

plt.figure(figsize=(6,3))
plt.plot(range(len(alphas)), alphas, marker="o")
plt.xlabel("Mois depuis le halving")
plt.ylabel(r"pente $\alpha$")
plt.title("Évolution mensuelle de la pente 1/f^\alpha")
plt.grid(True, ls=":")
plt.show()
```

Un \(\alpha\) croissant indique un marché dominé par les basses fréquences (tendance), alors qu'un \(\alpha\) proche de 0 signale un régime plus bruité.

---
**Fin :** cette série de documents couvre la discussion complète, du rappel théorique jusqu'à l'exemple de graphique.
