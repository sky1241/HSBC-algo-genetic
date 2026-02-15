# PROMPT - Recherche Implémentation NHHM (Non-Homogeneous HMM)

**Objectif:** Trouver du code, tutoriels, et implémentations pour transformer un HMM classique en HMM non-homogène avec prédiction directionnelle.

---

## CONTEXTE DE MON SYSTÈME

### Ce que j'ai actuellement

```python
# HMM classique avec hmmlearn
from hmmlearn.hmm import GaussianHMM

# Features d'entrée (par bougie H2)
features = [
    'P1', 'P2', 'P3', 'P4', 'P5', 'P6',  # Périodes Fourier dominantes
    'LFP',                                 # Low Frequency Power
    'volatility_fourier'                   # Volatilité spectrale
]

# Entraînement
model = GaussianHMM(n_components=3, covariance_type="full")
model.fit(X_train)

# Prédiction = label dur (0, 1, ou 2)
labels = model.predict(X_test)
```

### Mon problème

Le HMM actuel donne un **label de régime** (Phase 0, 1, 2) mais PAS de **direction** (bull vs bear).

Exemple:
- Label = 1 (Momentum) → Je sais que c'est du momentum
- Mais je NE sais PAS si c'est momentum HAUSSIER ou BAISSIER

### Ce que je veux

Un **Non-Homogeneous HMM (NHHM)** où:
1. Les probabilités de transition dépendent de covariables exogènes (pas fixes)
2. Le modèle prédit la DIRECTION (probabilité bull vs bear)
3. Output = P(bull | features, état actuel) au lieu d'un label dur

---

## CE QUE JE CHERCHE

### 1. Implémentations Python

Cherche spécifiquement:

- **PyMC / PyMC3 / PyMC5** : Bayesian HMM avec transitions time-varying
- **NumPyro / Pyro** : HMM avec covariables dans les transitions
- **statsmodels MarkovRegression / MarkovAutoregression** : Markov-switching models
- **pomegranate** : HMM avec transitions dépendant de features
- **hmmlearn extensions** : Peut-on modifier hmmlearn pour NHHM ?

### 2. Papers avec code

Le papier de référence:
- **"Exploring the predictability of cryptocurrencies via Bayesian hidden Markov models"**
- Auteurs: Koki, Leonardos, Piliouras (2022)
- Journal: Research in International Business and Finance
- arXiv: https://arxiv.org/abs/2011.03741

Questions:
- Y a-t-il du code associé à ce papier ?
- Y a-t-il des implémentations similaires sur GitHub ?

### 3. Tutoriels / Exemples

Cherche:
- "Non-homogeneous HMM Python tutorial"
- "Time-varying transition probabilities HMM"
- "Markov switching model with exogenous variables Python"
- "Bayesian HMM PyMC cryptocurrency"
- "Hidden Markov Model covariate-dependent transitions"

### 4. GitHub Repos

Critères:
- Stars > 50 (qualité minimum)
- Python
- Finance / Trading / Time series
- Implémente transitions non-homogènes

---

## FORMAT DE RÉPONSE SOUHAITÉ

Pour chaque ressource trouvée:

```
### [Nom de la ressource]

**Type:** [Library / Paper+Code / Tutorial / GitHub Repo]
**Lien:** [URL]
**Pertinence:** [Haute/Moyenne/Faible]

**Ce que ça fait:**
- Point 1
- Point 2

**Code exemple (si dispo):**
```python
# Snippet minimal
```

**Limitations:**
- Ce qui manque ou ce qu'il faudra adapter
```

---

## CONTRAINTES TECHNIQUES

- Python 3.10+
- Doit tourner sur CPU (pas de GPU)
- Données: 60,000 bougies H2 BTC (14 ans)
- Features: ~10 colonnes numériques
- États: 3-5 régimes
- Objectif: prédire direction (bull/bear) sur horizon 12-24 bougies

---

## CE QUE JE NE VEUX PAS

- Théorie sans code
- Implémentations en R ou Matlab uniquement
- Solutions nécessitant TensorFlow/PyTorch lourd (préférer léger)
- Papers sans implémentation accessible

---

## QUESTION BONUS

Si tu ne trouves pas d'implémentation NHHM directe, propose:

1. Comment modifier `hmmlearn.GaussianHMM` pour avoir des transitions dépendant de covariables ?
2. Comment utiliser `statsmodels.MarkovRegression` pour obtenir des probabilités directionnelles ?
3. Quelle est l'approche la plus SIMPLE pour passer de "label régime" à "proba direction" ?

---

FIN DU PROMPT
