#!/usr/bin/env bash
# Wrapper pour lancer un script du bot via le venv du repo.
# Usage: hsbc-bot-runner.sh <module>
#   ex: hsbc-bot-runner.sh routines.intraday_runner
set -euo pipefail

REPO="/home/ludov/HSBC-algo-genetic"
VENV="${REPO}/.venv/bin/python"
LOG_DIR="${REPO}/binance_bot/logs"
mkdir -p "${LOG_DIR}"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <module> (ex: routines.intraday_runner)" >&2
    exit 2
fi
MODULE="$1"

cd "${REPO}/binance_bot"
exec "${VENV}" -m "${MODULE}"
