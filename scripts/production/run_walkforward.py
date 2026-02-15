#!/usr/bin/env python3
"""Run walk-forward analysis over monthly segments.

This script splits a user-specified period into monthly segments and runs the
pipeline's main `run_profile` function on each segment. Metrics of the shared
portfolio produced by the pipeline are aggregated into a comparative report.
"""

from __future__ import annotations

import argparse
import os
import glob
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd

# Allow import of pipeline from repository root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ichimoku_pipeline_web_v4_8_fixed as pipeline


@dataclass
class SegmentResult:
    start: datetime
    end: datetime
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "start": self.start.strftime("%Y-%m-%d"),
            "end": self.end.strftime("%Y-%m-%d"),
        }
        out.update(self.metrics)
        return out


def monthly_segments(start: datetime, end: datetime) -> List[tuple[datetime, datetime]]:
    """Return list of (segment_start, segment_end) for each calendar month."""
    segments = []
    cur = start.replace(day=1)
    # adjust start if not first of month
    if start > cur:
        cur = start
    while cur <= end:
        # Compute first day of next month
        if cur.month == 12:
            next_month = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            next_month = cur.replace(month=cur.month + 1, day=1)
        seg_end = min(next_month - timedelta(days=1), end)
        segments.append((cur, seg_end))
        cur = next_month
    return segments


def run_segment(profile: str, seg_start: datetime, seg_end: datetime, out_dir: str, trials: int = 0, seed: int | None = None) -> Dict[str, Any]:
    """Run pipeline for a single segment and return shared portfolio metrics."""
    # Compute years_back from segment length
    years_back = (seg_end - seg_start).days / 365.25

    # Patch profile configuration
    original_years_back = pipeline.PROFILES[profile]["years_back"]
    pipeline.PROFILES[profile]["years_back"] = years_back

    # Patch datetime.utcnow to return segment end
    class FixedDatetime(datetime):
        @classmethod
        def utcnow(cls):
            return seg_end

    original_datetime = pipeline.datetime
    pipeline.datetime = FixedDatetime

    segment_dir = os.path.join(out_dir, f"{seg_start:%Y%m%d}_{seg_end:%Y%m%d}")
    os.makedirs(segment_dir, exist_ok=True)

    try:
        pipeline.run_profile(profile, trials=trials, seed=seed, out_dir=segment_dir)
    finally:
        pipeline.PROFILES[profile]["years_back"] = original_years_back
        pipeline.datetime = original_datetime

    # Locate latest shared portfolio file in segment directory
    pattern = os.path.join(segment_dir, f"shared_portfolio_{profile}_*.json")
    files = glob.glob(pattern)
    if not files:
        return {}
    latest = max(files, key=os.path.getmtime)
    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("shared_metrics", data)
    return metrics


def aggregate_results(results: List[SegmentResult], out_dir: str, profile: str, start: datetime, end: datetime) -> None:
    rows = [r.to_dict() for r in results]
    if not rows:
        print("No results to aggregate.")
        return
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"walkforward_{profile}_{start:%Y%m%d}_{end:%Y%m%d}.csv")
    df.to_csv(csv_path, index=False)
    md_path = os.path.join(out_dir, f"walkforward_{profile}_{start:%Y%m%d}_{end:%Y%m%d}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    print(f"Aggregated CSV saved to {csv_path}")
    print(f"Markdown report saved to {md_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run walk-forward backtests over monthly segments")
    p.add_argument("profile", help="Profile name defined in pipeline")
    p.add_argument("start", help="Start date YYYY-MM-DD")
    p.add_argument("end", help="End date YYYY-MM-DD")
    p.add_argument("--out-dir", default="walkforward_outputs", help="Base output directory")
    p.add_argument("--trials", type=int, default=0, help="Number of Optuna trials per segment")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    profile = args.profile
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    if start > end:
        raise ValueError("Start date must be before end date")

    os.makedirs(args.out_dir, exist_ok=True)

    segments = monthly_segments(start, end)
    results: List[SegmentResult] = []
    for seg_start, seg_end in segments:
        print(f"Running segment {seg_start:%Y-%m-%d} to {seg_end:%Y-%m-%d}…")
        metrics = run_segment(profile, seg_start, seg_end, args.out_dir, trials=args.trials, seed=args.seed)
        results.append(SegmentResult(seg_start, seg_end, metrics))

    aggregate_results(results, args.out_dir, profile, start, end)


if __name__ == "__main__":
    main()
