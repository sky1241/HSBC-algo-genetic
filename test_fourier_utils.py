import pandas as pd
import pytest
from pathlib import Path

from scripts.fourier_utils import analyze_csv


def _make_csv(df: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_analyze_csv_uses_close_column(tmp_path: Path) -> None:
    """analyze_csv should locate the close column without assuming its position."""
    data = {
        "timestamp": range(300),
        "open": [1.0] * 300,
        "high": [1.0] * 300,
        "low": [1.0] * 300,
        "Close": [1.0] * 300,  # Mixed case to test case-insensitivity
        "volume": [1.0] * 300,
    }
    csv_path = _make_csv(pd.DataFrame(data), tmp_path)
    stats = analyze_csv(csv_path, timeframe_hours=2.0, window_days=1)
    assert "P_bars" in stats


def test_analyze_csv_raises_without_close(tmp_path: Path) -> None:
    df = pd.DataFrame({"open": [1, 2], "volume": [10, 20]})
    csv_path = _make_csv(df, tmp_path)
    with pytest.raises(ValueError, match="No close column"):
        analyze_csv(csv_path)

