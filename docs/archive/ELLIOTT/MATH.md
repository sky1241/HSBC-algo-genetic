### Formalisation mathématique (succincte)

1) Détection de swings (pivot highs/lows)
- Soit \(P_t\) le prix; pivots via fenêtre \(k\):
  - sommet si \(P_t = \max\{P_{t-k..t+k}\}\); creux si \(P_t = \min\{P_{t-k..t+k}\}\).

2) Séquencement 5–3 (heuristique quant)
- Construire la suite des pivots \((S_i)\), i croissant.
- Vagues impulsives: alternance directionnelle cohérente (5 segments), amplitude croissante/non décroissante; vague 3 non la plus courte.
- Correctives: 3 segments opposés; retracements \(\approx\) 0.382–0.618 de l’impulsion précédente.

3) Ratios de Fibonacci & tests
- Extensions: \(|v_3| / |v_1| \in [1.0, 2.618]\) typique; retracement \(|A|/|1\text{–}5|\) dans [0.382, 0.618].
- Tests quantifiés (tolérances ±10–15%).

4) Lien temps–fréquence (Fourier)
- Régimes impulsifs: LFP élevé, pentes nuage (SSA/SSB) positives/négatives; P1/P2/P3 stables.
- Régimes correctifs/range: LFP plus faible, flatness plus élevée, P-k instables.

