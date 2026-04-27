#!/usr/bin/env bash
# Lance le dashboard Flask local sur http://localhost:8080.
# Usage: ./binance_bot/dashboard/run.sh
set -euo pipefail

REPO="/home/ludov/HSBC-algo-genetic"
cd "${REPO}"

export HSBC_DASHBOARD_HOST="${HSBC_DASHBOARD_HOST:-127.0.0.1}"
export HSBC_DASHBOARD_PORT="${HSBC_DASHBOARD_PORT:-8080}"

exec "${REPO}/.venv/bin/python" -m binance_bot.dashboard.app
