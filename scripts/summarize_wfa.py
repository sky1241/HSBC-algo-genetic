#!/usr/bin/env python3

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class Summary:
    count: int
    equity_med: Optional[float]
    equity_q25: Optional[float]
    equity_q75: Optional[float]
    sharpe_med: Optional[float]
    sharpe_q25: Optional[float]
    sharpe_q75: Optional[float]
    mdd_med: Optional[float]
    mdd_q25: Optional[float]
    mdd_q75: Optional[float]


def _percentile(values: List[float], p: float) -> Optional[float]:
    arr = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not arr:
        return None
    arr.sort()
    if len(arr) == 1:
        return float(arr[0])
    r = (p / 100.0) * (len(arr) - 1)
    lo = int(math.floor(r))
    hi = int(math.ceil(r))
    t = r - lo
    if hi == lo:
        return float(arr[lo])
    return float(arr[lo] + (arr[hi] - arr[lo]) * t)


def _load_json_safely(path: Path) -> Optional[dict]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        # Replace non-JSON tokens like Infinity by null
        raw = re.sub(r"(:\s*)Infinity(\s*)([,}])", r"\1null\2\3", raw)
        return json.loads(raw)
    except Exception:
        return None


def _gather_metrics(files: Iterable[Path]) -> Tuple[List[float], List[float], List[float]]:
    eq_list: List[float] = []
    sh_list: List[float] = []
    mdd_list: List[float] = []
    for fp in files:
        data = _load_json_safely(fp)
        if not data:
            continue
        overall = data.get("overall") or {}
        eq = overall.get("equity_mult")
        sh = overall.get("sharpe_proxy_mean")
        if isinstance(eq, (int, float)):
            eq_list.append(float(eq))
        if isinstance(sh, (int, float)):
            sh_list.append(float(sh))
        # Approx MDD from min equity across folds if present
        min_fold_vals: List[float] = []
        for fd in data.get("folds", []) or []:
            metrics = fd.get("metrics") or {}
            me = metrics.get("min_equity")
            if isinstance(me, (int, float)):
                min_fold_vals.append(float(me))
        if min_fold_vals:
            min_fold = min(min_fold_vals)
            mdd_pct = (1.0 - min_fold) * 100.0
            mdd_list.append(mdd_pct)
    return eq_list, sh_list, mdd_list


def summarize_dir(dir_paths: List[Path]) -> Summary:
    files: List[Path] = []
    for d in dir_paths:
        if d.exists():
            files.extend(list(d.rglob("*.json")))
    eq, sh, mdd = _gather_metrics(files)
    return Summary(
        count=len(files),
        equity_med=_percentile(eq, 50),
        equity_q25=_percentile(eq, 25),
        equity_q75=_percentile(eq, 75),
        sharpe_med=_percentile(sh, 50),
        sharpe_q25=_percentile(sh, 25),
        sharpe_q75=_percentile(sh, 75),
        mdd_med=_percentile(mdd, 50),
        mdd_q25=_percentile(mdd, 25),
        mdd_q75=_percentile(mdd, 75),
    )


def _fmt(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.3f}"


def write_summary(out_path: Path, annual: Optional[Summary], monthly: Optional[Summary]) -> None:
    lines: List[str] = []
    if annual is not None:
        lines.append("=== ANNUAL ===")
        lines.append(f"n={annual.count}")
        lines.append(f"equity_med={_fmt(annual.equity_med)}  IQR=[{_fmt(annual.equity_q25)}, {_fmt(annual.equity_q75)}]")
        lines.append(f"sharpe_med={_fmt(annual.sharpe_med)}  IQR=[{_fmt(annual.sharpe_q25)}, {_fmt(annual.sharpe_q75)}]")
        lines.append(f"MDD%_approx_med={_fmt(annual.mdd_med)}  IQR=[{_fmt(annual.mdd_q25)}, {_fmt(annual.mdd_q75)}]")
    if monthly is not None:
        lines.append("=== MONTHLY ===")
        lines.append(f"n={monthly.count}")
        lines.append(f"equity_med={_fmt(monthly.equity_med)}  IQR=[{_fmt(monthly.equity_q25)}, {_fmt(monthly.equity_q75)}]")
        lines.append(f"sharpe_med={_fmt(monthly.sharpe_med)}  IQR=[{_fmt(monthly.sharpe_q25)}, {_fmt(monthly.sharpe_q75)}]")
        lines.append(f"MDD%_approx_med={_fmt(monthly.mdd_med)}  IQR=[{_fmt(monthly.mdd_q25)}, {_fmt(monthly.mdd_q75)}]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annual", type=str, help="Path to annual WFA outputs root", default="outputs/scheduler_annual_btc")
    ap.add_argument("--monthly", type=str, help="Path to monthly WFA outputs root", default="outputs/scheduler_monthly_btc")
    ap.add_argument("--out", type=str, help="Output summary path", default=None)
    args = ap.parse_args()

    # Only pass the root directories; summarize_dir will recurse.
    # Passing both root and subdirs would double-count files.
    annual_dirs = [Path(args.annual)]
    monthly_dirs = [Path(args.monthly)]

    annual_s = summarize_dir(annual_dirs)
    monthly_s = summarize_dir(monthly_dirs)

    out = Path(args.out) if args.out else Path("docs") / ("WFA_SUMMARY_" + Path.cwd().name + ".txt")
    write_summary(out, annual_s, monthly_s)
    print(out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
