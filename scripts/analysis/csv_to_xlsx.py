#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--xlsx', required=False)
    args = p.parse_args()

    csv_path = Path(args.csv)
    xlsx_path = Path(args.xlsx) if args.xlsx else csv_path.with_suffix('.xlsx')

    df = pd.read_csv(csv_path)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='FOURIER_ALL', index=False)
    print(f"Wrote {xlsx_path}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
