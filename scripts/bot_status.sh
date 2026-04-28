#!/usr/bin/env bash
# F-status — Outil monitoring du soak HSBC-algo-genetic.
# Usage : scripts/bot_status.sh
# Resume l'etat du bot en 1 commande, < 1 seconde.
# 8 sections : header / git / trade mode / timers / data / trades /
# daily PnL / health.
set -uo pipefail

REPO="/home/ludov/HSBC-algo-genetic"
DATA="${REPO}/binance_bot/data"
LOGS="${REPO}/binance_bot/logs"
CFG="${REPO}/binance_bot/configs/bot_settings.yaml"

# Colors only if stdout is a tty
if [ -t 1 ]; then
    G="\033[32m"; Y="\033[33m"; R="\033[31m"; C="\033[36m"; B="\033[1m"; N="\033[0m"
else
    G=""; Y=""; R=""; C=""; B=""; N=""
fi

_sep() { printf "%b────────────────────────────────────────────────────────────────────%b\n" "$C" "$N"; }
_hdr() { printf "%b%s%b\n" "$B" "$1" "$N"; }
_ok() { printf "  %b✅%b %s\n" "$G" "$N" "$1"; }
_warn() { printf "  %b⚠️%b  %s\n" "$Y" "$N" "$1"; }
_kill() { printf "  %b❌%b %s\n" "$R" "$N" "$1"; }

_lines() {
    # _lines <file> -> int (0 if absent)
    if [ -f "$1" ]; then wc -l < "$1" | tr -d ' '; else echo 0; fi
}

_jsonkey() {
    # _jsonkey <file> <key> -> value or "null"
    python3 -c "
import json, sys
try:
    print(json.load(open('$1')).get('$2', 'null'))
except Exception:
    print('null')
" 2>/dev/null
}

_now_utc() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# ─── Section 1 : Header + soak J+N ─────────────────────────────────
_sep
_hdr "HSBC bot status — $(_now_utc)"
SOAK_FILE="${DATA}/soak_start.txt"
if [ -f "$SOAK_FILE" ]; then
    SOAK_TS=$(head -1 "$SOAK_FILE")
    SOAK_EPOCH=$(date -u -d "$SOAK_TS" +%s 2>/dev/null || echo 0)
    NOW_EPOCH=$(date -u +%s)
    if [ "$SOAK_EPOCH" -gt 0 ]; then
        DAYS=$(( (NOW_EPOCH - SOAK_EPOCH) / 86400 ))
        HOURS=$(( ((NOW_EPOCH - SOAK_EPOCH) % 86400) / 3600 ))
        printf "  Soak start : %s (J+%d %dh)\n" "$SOAK_TS" "$DAYS" "$HOURS"
    fi
else
    _warn "soak_start.txt absent (run scripts/bot_status.sh apres demarrage soak)"
fi

# ─── Section 2 : Git ───────────────────────────────────────────────
_sep
_hdr "Git"
cd "$REPO" 2>/dev/null && {
    HEAD_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
    BRANCH=$(git branch --show-current 2>/dev/null || echo "?")
    DIRTY=$(git status --porcelain 2>/dev/null | wc -l)
    printf "  HEAD     : %s\n" "$HEAD_HASH"
    printf "  Branche  : %s\n" "$BRANCH"
    if [ "$DIRTY" -gt 0 ]; then
        _warn "$DIRTY fichier(s) uncommitted"
    else
        _ok "tree propre"
    fi
}

# ─── Section 3 : Trade mode ────────────────────────────────────────
_sep
_hdr "Trade mode"
if [ -f "$CFG" ]; then
    TRADE_MODE=$(grep -E "^trade_mode:" "$CFG" | head -1 | awk -F'"' '{print $2}')
    TRADE_MODE=${TRADE_MODE:-"?"}
    if [ "$TRADE_MODE" = "live" ]; then
        _warn "trade_mode=live (testnet=$(grep BINANCE_TESTNET "${REPO}/binance_bot/.env" 2>/dev/null | head -1 | cut -d= -f2))"
    else
        _ok "trade_mode=$TRADE_MODE"
    fi
else
    _kill "bot_settings.yaml absent"
fi

# ─── Section 4 : Timers systemd ────────────────────────────────────
_sep
_hdr "Timers systemd actifs (--user)"
TIMERS_OUTPUT=$(systemctl --user list-timers --all 2>/dev/null | grep "hsbc-" || true)
if [ -z "$TIMERS_OUTPUT" ]; then
    _warn "aucun timer hsbc-* trouve (systemctl --user list-timers)"
else
    echo "$TIMERS_OUTPUT" | awk '{
        # Format: NEXT LEFT LAST PASSED UNIT ACTIVATES
        printf "  %-30s last: %-12s next: %s\n", $NF, $4 " " $5, $1 " " $2
    }' | head -10
fi

# ─── Section 5 : Données collectées ────────────────────────────────
_sep
_hdr "Donnees collectees (jsonl lines)"
for sym in BTCUSDT ETHUSDT SOLUSDT; do
    N_TLS=$(_lines "${DATA}/flow_top_ls_${sym}.jsonl")
    N_TKR=$(_lines "${DATA}/flow_taker_${sym}.jsonl")
    N_OI=$(_lines "${DATA}/flow_oi_${sym}.jsonl")
    printf "  flow_top_ls/taker/oi %s : %s / %s / %s\n" "$sym" "$N_TLS" "$N_TKR" "$N_OI"
done
LIQ=$(_lines "${DATA}/flow_liq_buckets.jsonl")
COMP=$(_lines "${DATA}/flow_composite_log.jsonl")
META=$(_lines "${DATA}/trades_meta.jsonl")
PSR=$(_lines "${DATA}/psr_history.jsonl")
VPIN_EVT=$(_lines "${DATA}/vpin_events.jsonl")
VPIN_LIVE=$(_lines "${DATA}/vpin_live.jsonl")
printf "  flow_liq_buckets    : %s (WS-001 disabled)\n" "$LIQ"
printf "  flow_composite_log  : %s (target ~720 = J+30 a 5min)\n" "$COMP"
printf "  trades_meta         : %s\n" "$META"
printf "  psr_history         : %s\n" "$PSR"
printf "  vpin_events         : %s (post P7-bis)\n" "$VPIN_EVT"
printf "  vpin_live           : %s (post P7-bis)\n" "$VPIN_LIVE"

# ─── Section 6 : Trades ────────────────────────────────────────────
_sep
_hdr "Trades (testnet)"
if [ -f "${DATA}/state.json" ]; then
    python3 -c "
import json
try:
    s = json.load(open('${DATA}/state.json'))
    syms = s.get('symbols') or {}
    for sym, blob in syms.items():
        nl = len(blob.get('positions_long', []) or [])
        ns = len(blob.get('positions_short', []) or [])
        print(f'  {sym:12s} long={nl}  short={ns}')
except Exception as e:
    print(f'  ERR reading state.json: {e}')
" 2>/dev/null
else
    _warn "state.json absent"
fi

# ─── Section 7 : Daily PnL ─────────────────────────────────────────
_sep
_hdr "Daily PnL (fraction capital)"
DAILY_PNL=$(_jsonkey "${DATA}/state.json" "daily_pnl_pct")
DAILY_LOSS=$(_jsonkey "${DATA}/state.json" "daily_loss")
HARD_CAP=$(grep -E "^daily_loss_hard_cap_pct:" "$CFG" 2>/dev/null | awk '{print $2}' || echo "?")
SOFT_LOSS=$(grep -E "^daily_loss_soft_cap_pct:" "$CFG" 2>/dev/null | awk '{print $2}' || echo "?")
SOFT_GAIN=$(grep -E "^daily_gain_soft_cap_pct:" "$CFG" 2>/dev/null | awk '{print $2}' || echo "?")
printf "  daily_pnl_pct  : %s   (caps loss=%s soft=%s gain=%s soft)\n" "$DAILY_PNL" "$HARD_CAP" "$SOFT_LOSS" "$SOFT_GAIN"
printf "  daily_loss     : %s\n" "$DAILY_LOSS"

# ─── Section 8 : Health ────────────────────────────────────────────
_sep
_hdr "Health"
if [ -f "${DATA}/.killed" ]; then
    _kill "kill_switch ACTIF (data/.killed present) — investiguer urgemment"
else
    _ok "kill_switch OK (no .killed file)"
fi

# Drawdown (equity vs rolling_equity_high)
python3 -c "
import json
try:
    s = json.load(open('${DATA}/state.json'))
    eq = float(s.get('equity', 1.0)) * float(s.get('initial_capital_usdt', 1.0))
    high = float(s.get('rolling_equity_high_usdt', eq))
    if high > 0:
        dd = (eq - high) / high
        if dd <= -0.15:
            print(f'  ❌ DD = {dd*100:+.2f}% (high=\${high:.2f}, eq=\${eq:.2f}) — KILL trigger imminent')
        elif dd <= -0.05:
            print(f'  ⚠️  DD = {dd*100:+.2f}% (high=\${high:.2f}, eq=\${eq:.2f})')
        else:
            print(f'  ✅ DD = {dd*100:+.2f}% (high=\${high:.2f}, eq=\${eq:.2f})')
except Exception as e:
    print(f'  ⚠️  DD calc failed: {e}')
" 2>/dev/null

# Audit log integrity verify
python3 -c "
import sys
sys.path.insert(0, '${REPO}/binance_bot')
try:
    from bot.audit_log import AuditLog
    a = AuditLog('${DATA}/trades_audit.jsonl')
    st = a.verify_chain()
    if st.valid:
        print(f'  ✅ audit_log integrity OK ({st.n_entries} entries)')
    else:
        print(f'  ❌ audit_log TAMPERED at seq={st.error_seq} — {st.error}')
except Exception as e:
    print(f'  ⚠️  audit verify failed: {e}')
" 2>/dev/null

# Last PSR
LAST_PSR=$(tail -1 "${DATA}/psr_history.jsonl" 2>/dev/null)
if [ -n "$LAST_PSR" ]; then
    PSR_VAL=$(echo "$LAST_PSR" | python3 -c "import json, sys; d=json.loads(sys.stdin.read()); print(d.get('psr','null'))" 2>/dev/null)
    PSR_ALERT=$(echo "$LAST_PSR" | python3 -c "import json, sys; d=json.loads(sys.stdin.read()); print(d.get('alert','?'))" 2>/dev/null)
    if [ "$PSR_VAL" = "null" ] || [ "$PSR_ALERT" = "insufficient_data" ]; then
        printf "  ⚠️  PSR = null (alert=%s)\n" "$PSR_ALERT"
    else
        printf "  PSR last = %s (alert=%s)\n" "$PSR_VAL" "$PSR_ALERT"
    fi
fi

_sep
echo "Done. Run again anytime: scripts/bot_status.sh"
