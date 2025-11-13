#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner principal: point d'entrée CLI pour le bot.

Usage:
    python runner.py --mode daily      # Met à jour phase (1×/jour)
    python runner.py --mode intraday   # Exécute signaux (toutes les 2h)
    python runner.py --mode simulation # Mode simulation continu
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routines import daily_phase_job, intraday_runner


def main():
    parser = argparse.ArgumentParser(description="Bot Binance K3 1D Stable")
    parser.add_argument(
        "--mode",
        choices=["daily", "intraday", "simulation"],
        required=True,
        help="Mode d'exécution"
    )
    args = parser.parse_args()
    
    if args.mode == "daily":
        return daily_phase_job.main()
    elif args.mode == "intraday":
        return intraday_runner.main()
    elif args.mode == "simulation":
        print("Mode simulation continu non implémenté (utiliser intraday en mode simulation)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

