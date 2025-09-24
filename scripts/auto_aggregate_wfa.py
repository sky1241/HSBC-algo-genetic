#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import time
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_DIRS: List[Path] = [
    Path('outputs') / 'wfa_phase_k3' / 'seed_123',
    Path('outputs') / 'wfa_phase_k3' / 'seed_321',
    Path('outputs') / 'wfa_phase_k3' / 'seed_777',
    Path('outputs') / 'wfa_phase_k3' / 'seed_999',
    Path('outputs') / 'wfa_phase_k5' / 'seed_123',
    Path('outputs') / 'wfa_phase_k5' / 'seed_321',
    Path('outputs') / 'wfa_phase_k5' / 'seed_777',
    Path('outputs') / 'wfa_phase_k5' / 'seed_999',
]


def has_result_json(d: Path) -> bool:
    if not d.exists():
        return False
    for fp in d.rglob('*.json'):
        # Prefer phase WFA jsons
        if fp.name.startswith('WFA_phase_') and fp.is_file():
            return True
    # Fallback: any json
    return any(fp.is_file() for fp in d.rglob('*.json'))


def wait_until_done(dirs: List[Path], poll_s: int = 60) -> None:
    while True:
        missing = [str(d) for d in dirs if not has_result_json(d)]
        if not missing:
            print('[OK] All result JSONs detected.')
            return
        print(f'[WAIT] Missing results in {len(missing)} dirs. Next check in {poll_s}s...')
        time.sleep(poll_s)


def run(cmd: List[str]) -> int:
    print('[RUN]', ' '.join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description='Watch WFA phase runs and auto-aggregate summaries')
    ap.add_argument('--dirs', nargs='*', default=[str(p) for p in DEFAULT_DIRS], help='List of seed output directories to wait for')
    ap.add_argument('--poll', type=int, default=60, help='Polling interval in seconds')
    args = ap.parse_args()

    dirs = [Path(p) for p in args.dirs]
    (Path('outputs') / 'wfa_phase_agg').mkdir(parents=True, exist_ok=True)

    print('[INFO] Waiting for phase WFA completion in:')
    for d in dirs:
        print(' -', d.as_posix())
    wait_until_done(dirs, poll_s=int(args.poll))

    # Generate summaries
    py = sys.executable or 'python'
    # Annual (reference)
    run([py, 'scripts/summarize_wfa.py', '--annual', 'outputs/scheduler_annual_btc', '--monthly', 'outputs/none', '--out', 'docs/WFA_SUMMARY_ANNUAL.txt'])
    # Phase K3
    run([py, 'scripts/summarize_wfa.py', '--annual', 'outputs/wfa_phase_k3', '--monthly', 'outputs/none', '--out', 'docs/WFA_SUMMARY_PHASE_K3.txt'])
    # Phase K5
    run([py, 'scripts/summarize_wfa.py', '--annual', 'outputs/wfa_phase_k5', '--monthly', 'outputs/none', '--out', 'docs/WFA_SUMMARY_PHASE_K5.txt'])

    print('[DONE] Summaries written:')
    print(' - docs/WFA_SUMMARY_ANNUAL.txt')
    print(' - docs/WFA_SUMMARY_PHASE_K3.txt')
    print(' - docs/WFA_SUMMARY_PHASE_K5.txt')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


