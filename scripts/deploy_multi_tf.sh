#!/usr/bin/env bash
# P-MTF-11 — Script de déploiement multi-TF (Sky décide quand l'exécuter).
#
# Bascule l'archi du bot du mode mono-H2 (hsbc-intraday) au mode multi-TF
# (hsbc-h2-trend + hsbc-execution-15m). Voir MISSION_MULTI_TF.md.
#
# Usage:
#   bash scripts/deploy_multi_tf.sh         # mode interactif (confirmation)
#   bash scripts/deploy_multi_tf.sh --yes   # mode auto (pas de prompt)
#   bash scripts/deploy_multi_tf.sh --rollback  # retour à mono-H2

set -euo pipefail

REPO="/home/ludov/HSBC-algo-genetic"
cd "${REPO}"

ROLLBACK=false
AUTO=false
for arg in "$@"; do
    case "$arg" in
        --rollback) ROLLBACK=true ;;
        --yes) AUTO=true ;;
        *) echo "Unknown arg: $arg"; exit 2 ;;
    esac
done

confirm() {
    local prompt="$1"
    if $AUTO; then return 0; fi
    read -p "$prompt [y/N] " -n 1 -r; echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

print_state() {
    echo "=== Timers HSBC actuels ==="
    systemctl --user list-timers --all --no-pager 2>&1 | grep -E "UNIT|hsbc-" || true
    echo ""
    echo "=== Services HSBC actifs ==="
    systemctl --user list-units --type=service --no-pager 2>&1 | grep hsbc || true
}

if $ROLLBACK; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ROLLBACK : retour mono-H2 (désactive multi-TF, réactive mono)"
    echo "═══════════════════════════════════════════════════════════════"
    print_state
    if confirm "Procéder au ROLLBACK ?"; then
        systemctl --user disable --now hsbc-h2-trend.timer hsbc-execution-15m.timer
        systemctl --user enable --now hsbc-intraday.timer
        echo "✓ Rollback fait."
        print_state
    else
        echo "Annulé."; exit 0
    fi
    exit 0
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  DÉPLOIEMENT MULTI-TF (P-MTF-11)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Bascule prévue :"
echo "  → DISABLE  hsbc-intraday.timer       (mono-H2)"
echo "  → ENABLE   hsbc-h2-trend.timer       (toutes les 2h)"
echo "  → ENABLE   hsbc-execution-15m.timer  (toutes les 15min)"
echo ""
print_state
echo ""
if ! confirm "Procéder au déploiement multi-TF ?"; then
    echo "Annulé."; exit 0
fi

# 1. Validation pré-déploiement : units chargés
echo ""
echo "── 1. Vérif units chargés ─────────────────────────────────────"
for u in hsbc-h2-trend.service hsbc-h2-trend.timer hsbc-execution-15m.service hsbc-execution-15m.timer; do
    if systemctl --user cat "$u" >/dev/null 2>&1; then
        echo "  ✓ $u"
    else
        echo "  ✗ $u INTROUVABLE — abort"
        exit 1
    fi
done

# 2. Disable l'ancien
echo ""
echo "── 2. Disable hsbc-intraday.timer ─────────────────────────────"
systemctl --user disable --now hsbc-intraday.timer
echo "  ✓ hsbc-intraday.timer désactivé"

# 3. Enable les nouveaux
echo ""
echo "── 3. Enable hsbc-h2-trend + hsbc-execution-15m ──────────────"
systemctl --user enable --now hsbc-h2-trend.timer
systemctl --user enable --now hsbc-execution-15m.timer
echo "  ✓ Timers multi-TF actifs"

# 4. Trigger immédiat de h2-trend pour avoir un snapshot frais
echo ""
echo "── 4. Trigger immédiat h2-trend (snapshot initial) ────────────"
systemctl --user start hsbc-h2-trend.service
sleep 5
echo "  ✓ Premier run h2_trend lancé"

# 5. Vérification finale
echo ""
echo "── 5. État final ──────────────────────────────────────────────"
print_state
echo ""
echo "── 6. Logs récents h2_trend ──────────────────────────────────"
tail -20 binance_bot/logs/h2_trend.log 2>&1 || echo "(log pas encore créé)"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  DÉPLOIEMENT TERMINÉ — soak monitoring 24-48h"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Surveillance recommandée :"
echo "  - tail -f binance_bot/logs/h2_trend.log"
echo "  - tail -f binance_bot/logs/execution_15m.log"
echo "  - watch -n 30 'systemctl --user list-timers | grep hsbc'"
echo "  - cat binance_bot/data/state.json | python3 -m json.tool | grep -A5 h2_trend"
echo ""
echo "Si problème : bash scripts/deploy_multi_tf.sh --rollback"
