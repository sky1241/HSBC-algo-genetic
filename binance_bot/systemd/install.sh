#!/usr/bin/env bash
# Installe les unités systemd USER-level (pas besoin de sudo).
# Lance:
#   ./install.sh         — copy + enable + start timers
#   ./install.sh status  — status des unités
#   ./install.sh stop    — disable + stop timers
#   ./install.sh logs    — tail des logs intraday
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
USER_UNITS="${HOME}/.config/systemd/user"
UNITS=(hsbc-intraday.service hsbc-intraday.timer hsbc-daily.service hsbc-daily.timer hsbc-dashboard.service hsbc-watchdog.service hsbc-watchdog.timer hsbc-backup.service hsbc-backup.timer)
TIMERS=(hsbc-intraday.timer hsbc-daily.timer hsbc-watchdog.timer hsbc-backup.timer)
LONG_RUNNING=(hsbc-dashboard.service)

cmd="${1:-install}"

case "${cmd}" in
  install)
    mkdir -p "${USER_UNITS}"
    # Le dossier logs/ doit exister AVANT le 1er run — systemd ouvre le fichier
    # de log dans append: avant de spawn le wrapper.
    mkdir -p "/home/ludov/HSBC-algo-genetic/binance_bot/logs"
    for u in "${UNITS[@]}"; do
      cp -v "${THIS_DIR}/${u}" "${USER_UNITS}/${u}"
    done
    chmod +x "${THIS_DIR}/hsbc-bot-runner.sh"
    chmod +x "/home/ludov/HSBC-algo-genetic/binance_bot/dashboard/run.sh" || true
    systemctl --user daemon-reload
    for t in "${TIMERS[@]}"; do
      systemctl --user enable --now "${t}"
    done
    for s in "${LONG_RUNNING[@]}"; do
      systemctl --user enable --now "${s}"
    done
    echo
    echo "→ Pour que les timers et le dashboard tournent sans login actif:"
    echo "    sudo loginctl enable-linger ${USER}"
    echo "→ Dashboard: http://localhost:8080"
    echo
    systemctl --user list-timers --all | grep -E 'hsbc|NEXT' || true
    ;;
  status)
    for t in "${TIMERS[@]}"; do
      echo "=== ${t} ==="
      systemctl --user status "${t}" --no-pager || true
    done
    for s in "${LONG_RUNNING[@]}"; do
      echo "=== ${s} ==="
      systemctl --user status "${s}" --no-pager || true
    done
    systemctl --user list-timers --all | grep -E 'hsbc|NEXT' || true
    ;;
  stop)
    for t in "${TIMERS[@]}"; do
      systemctl --user disable --now "${t}" || true
    done
    for s in "${LONG_RUNNING[@]}"; do
      systemctl --user disable --now "${s}" || true
    done
    ;;
  logs)
    LOG_FILE="/home/ludov/HSBC-algo-genetic/binance_bot/logs/intraday.log"
    if [ -f "${LOG_FILE}" ]; then
      tail -n 100 -f "${LOG_FILE}"
    else
      echo "Pas encore de log — le timer n'a pas encore tourné. Voir: journalctl --user -u hsbc-intraday.service"
    fi
    ;;
  *)
    echo "Usage: $0 {install|status|stop|logs}"
    exit 2
    ;;
esac
