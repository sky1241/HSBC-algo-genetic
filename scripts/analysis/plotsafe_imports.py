from __future__ import annotations

from importlib import import_module
import sys
from pathlib import Path

# Ensure repository root and scripts dir are on sys.path for direct execution
_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def import_heat():
    # Prefer local module name; fallback to package-style if available
    try:
        return import_module('plot_heatmaps_plotly_live')
    except ModuleNotFoundError:
        return import_module('scripts.plot_heatmaps_plotly_live')


def import_trials3d():
    # Prefer local module name; fallback to package-style if available
    try:
        return import_module('plot_trials_3d_live')
    except ModuleNotFoundError:
        return import_module('scripts.plot_trials_3d_live')


