from __future__ import annotations

from importlib import import_module


def import_heat():
    return import_module('scripts.plot_heatmaps_plotly_live')


def import_trials3d():
    return import_module('scripts.plot_trials_3d_live')


