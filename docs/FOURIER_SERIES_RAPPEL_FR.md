# Séries de Fourier — Rappel théorique

Ce document synthétise les échanges autour de l'analyse fréquentielle des prix du BTC. Il sert de point de départ avant de passer à l'étude de Welch alignée sur le halving.

## 1. Transformée de Fourier discrète
Pour une suite de données discrètes \(x_n\) (ex. rendements), la série de Fourier discrète est
\[
X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i kn/N}, \qquad x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{2\pi i kn/N}.
\]
L'algorithme FFT calcule efficacement ces coefficients en \(O(N\log N)\).

## 2. Utilité pour les séries financières
La transformée de Fourier permet de passer du temps aux fréquences pour repérer les périodicités dominantes, filtrer le bruit et construire des indicateurs. Les outils principaux sont la densité spectrale de puissance (PSD), la période dominante \(P = 1/f_*\) et le ratio d'énergie basse fréquence (LFP) qui mesure la part de puissance sous un seuil \(f_0\).

La PSD peut s'estimer par le **périodogramme**
\[
S_{xx}(f_k) = \frac{1}{N f_s}\left|\sum_{n=0}^{N-1} x_n e^{-i 2\pi k n/N}\right|^2,
\qquad f_k = k\,\frac{f_s}{N},
\]
où \(f_s = 1/\Delta t\) est la fréquence d'échantillonnage (1 barre\(^{-1}\) en H2). L'unité de \(S_{xx}\) est une puissance par unité de fréquence.

Le ratio d'énergie basse fréquence se définit alors par
\[
\mathrm{LFP} = \frac{\sum_{f < f_0} \mathrm{PSD}(f)}{\sum_f \mathrm{PSD}(f)}.
\]
Par exemple, choisir \(f_0 = 1/(5 \times 12)\) cycles/barre revient à mesurer la part de puissance associée aux périodes supérieures à 5 jours. Une valeur de LFP > 0,6 suggère un régime dominé par les basses fréquences.

## 3. Interprétation du spectre des prix BTC
Des analyses externes (Medium, Wikipedia, Signal Processing SE) montrent que le spectre FFT des **niveaux de prix** BTC suit \(1/f^2\), typique d'un bruit brun : le prix s'apparente à l'intégrale d'un bruit blanc. On obtient alors une pente \(-2\) en log–log, donc peu d'information exploitable.

## 4. Rendements et volatilité
On préfère étudier les **log‑rendements** \(r_t = \ln P_t - \ln P_{t-1}\). Leur PSD est presque plate (bruit blanc) tandis que la volatilité \(|r_t|\) ou \(r_t^2\) exhibe souvent un spectre \(1/f^\alpha\) (« bruit rose ») révélant une mémoire longue.

## 5. Recette d'analyse
- Fenêtre roulante de 180–360 jours (≈2160–4320 barres H2).
- Calcul de la PSD via Welch pour \(r_t\) et pour \(|r_t|\) ou \(r_t^2\).
- Extraction de la période dominante \(P\) et du LFP pour orienter les paramètres Ichimoku.
- En cas de non‑stationnarité, passer à des méthodes temps–fréquence (STFT, ondelettes).

## 6. Implémentation dans le dépôt
Le module `scripts/fourier_utils.py` contient `compute_welch_psd` et la fonction utilitaire `analyze_csv` qui lit un fichier OHLCV, calcule \(P\), LFP et la flatness spectrale puis retourne des réglages suggérés.

## 7. Alignement sur le halving
`src/phase_aware_module.py` recense les dates de halving et sert à initialiser la phase. Les log‑rendements \(\ln P_t - \ln P_{t-1}\) sont calculés pour plusieurs métriques dont la volatilité annualisée.

## 8. Données H2
Le fichier `data/BTC_USDT_2h.csv` contient des chandeliers de 2 heures, point de départ idéal pour fixer \(t=0\) au halving et manipuler les rendements \(r_t\) à cette granularité.

## 9. Approche recommandée (depuis le halving)
1. Définir \(t=0\) au dernier halving (ex. 20 avril 2024) et extraire les séries H2.
2. Calculer \(r_t\), \(|r_t|\) ou \(r_t^2\).
3. Estimer la PSD (Welch) sur une fenêtre glissante mensuelle ou longue.
4. Analyser les pentes log–log: ~0 pour \(r_t\), \(1/f^\alpha\) pour la volatilité.
5. Extraire \(P\) et LFP pour suggérer des réglages Ichimoku adaptatifs.
6. Utiliser STFT/ondelettes si le contenu fréquentiel évolue rapidement.
7. Comparer les résultats entre phases de halving pour détecter des changements structurels.

## Références
[^1]: Telmo Subira Rodriguez, "BTC case study: applying basic Digital Signal Processing into financial data", *Medium*, 10 juin 2018. https://medium.com/drill/btc-case-study-applying-basic-digital-signal-processing-into-financial-data-ec34cd47c77b (consulté le 26 août 2025).
[^2]: "Brownian noise", *Wikipedia*. https://en.wikipedia.org/wiki/Brownian_noise (consulté le 26 août 2025).
[^3]: Peter K., « Why does the power spectral density of a random walk fall off as 1/f^2? », *Signal Processing Stack Exchange*. https://dsp.stackexchange.com/q/27566 (consulté le 26 août 2025).

---
**Suite :** l'étape suivante décrit la méthode de Welch et l'alignement sur le halving dans `FOURIER_SERIES_WELCH_HALVING_FR.md`.
