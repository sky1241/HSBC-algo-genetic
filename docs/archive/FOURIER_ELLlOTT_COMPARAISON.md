### Comparaison Fourier ↔ Vagues d’Elliott (style thèse)

But: relier régimes Fourier (P1–P3, LFP, volume) aux motifs Elliott (impulsifs/correctifs) et aux phases halving, pour un usage opérationnel (gating, ranges Ichimoku).

1) Cartographie conceptuelle
- Impulsif (Elliott 1–3–5) ≈ régime up/down (LFP haut, P1/P2 stables), souvent phases expansion/euphorie.
- Correctif (A–B–C) ≈ range (LFP bas à moyen), souvent accumulation/distribution.
- Capitulation: extrême du bear (M très négatif, V_ann très haut, DD profond) — signal de rupture.

2) Métriques à comparer
- Par phase/régime: médianes P1–P3, moyennes LFP (prix & volume), vol, retours.
- Pentes nuage (SSA/SSB) et ADX pour renforcer la lecture impulsif/correctif.

3) Méthode
- Labels: `docs/PHASE_LABELS/` + `outputs/fourier/phase_labels/`.
- Tests quantifiés Elliott: pivots k-bars + tolérances Fibonacci (voir `docs/ELLIOTT/MATH.md`).
- Matrices de confusion et scores (F1/kappa) pour évaluer correspondance.

4) Implications Ichimoku
- Mapping P1→(tenkan,kijun,senkou_b,shift) par phase/régime confirmé par l’impulsivité.
- Gating (intensité pools) modulé par LFP et par détection impulsif/correctif.

5) Prochaines étapes
- Implémenter un détecteur de motifs 5–3 quantifié et appliquer aux segments up/down/range.
- Rapporter tables et graphiques overlay (prix + labellisation) par phase.
