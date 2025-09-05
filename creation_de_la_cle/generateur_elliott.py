"""Génère des séries de prix synthétiques respectant la structure de base des vagues d'Elliott."""
from __future__ import annotations

import numpy as np


def generate_elliott_wave_series(start: float = 100.0, scale: float = 10.0, cycles: int = 1, seed: int | None = None) -> np.ndarray:
    """Génère une série de prix suivant un motif 1-2-3-4-5-A-B-C.

    Parameters
    ----------
    start: float
        Prix initial.
    scale: float
        Amplitude de base utilisée pour dimensionner les vagues.
    cycles: int
        Nombre de cycles complets à générer.
    seed: int | None
        Graine pour la reproductibilité.
    """
    rng = np.random.default_rng(seed)
    series = []
    price = start

    for _ in range(cycles):
        w1 = scale * rng.uniform(0.5, 1.0)
        w3 = scale * rng.uniform(max(w1 / scale, 1.0), 2.0)  # vague 3 >= vague 1
        w5 = scale * rng.uniform(0.5, w3 / scale * 0.9)

        w2 = -scale * rng.uniform(0.3, min(w1 / scale * 0.9, 0.8))
        if price + w1 + w2 <= price:
            w2 = -(w1 * 0.8)  # assure que la vague 2 ne dépasse pas le départ

        w4 = -scale * rng.uniform(0.3, min(w3 / scale * 0.6, 0.8))
        if price + w1 + w2 + w3 + w4 <= price + w1:
            w4 = -( (price + w1 + w2 + w3) - (price + w1) ) * 0.9  # pas d'empiètement sur vague 1

        wA = -scale * rng.uniform(0.4, w5 / scale + 0.3)
        wB = scale * rng.uniform(0.2, 0.6)
        wC = -scale * rng.uniform(max(0.6, abs(wA) / scale), abs(wA) / scale + 0.5)

        increments = [w1, w2, w3, w4, w5, wA, wB, wC]
        for inc in increments:
            price += inc
            series.append(price)

    return np.array(series)


if __name__ == "__main__":
    data = generate_elliott_wave_series(cycles=2, seed=42)
    print(data)
