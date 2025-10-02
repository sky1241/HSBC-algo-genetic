import numpy as np

from scripts.fourier_core import spectral_flatness
from src.features_fourier import _spectral_flatness as sf_local


def test_flatness_zero_and_one_returns_zero():
    s = np.array([0.0, 1.0], dtype=float)
    assert spectral_flatness(s) == 0.0
    assert sf_local(s) == 0.0


def test_flatness_all_zeros_returns_zero():
    s = np.array([0.0, 0.0, 0.0], dtype=float)
    assert spectral_flatness(s) == 0.0
    assert sf_local(s) == 0.0


def test_flatness_positive_uniform_is_one():
    s = np.array([5.0, 5.0, 5.0], dtype=float)
    assert np.isclose(spectral_flatness(s), 1.0)
    assert np.isclose(sf_local(s), 1.0)


def test_flatness_strictly_positive_non_uniform_between_0_and_1():
    s = np.array([1.0, 2.0], dtype=float)
    val1 = spectral_flatness(s)
    val2 = sf_local(s)
    assert 0.0 < val1 <= 1.0
    assert 0.0 < val2 <= 1.0

