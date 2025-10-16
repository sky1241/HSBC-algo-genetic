#!/usr/bin/env python3
import json
from pathlib import Path

files = list(Path("outputs/wfa_fixed_k3").rglob("WFA_annual_*.json"))
print(f"Fixed K3: {len(files)} fichiers")
for f in files:
    data = json.loads(f.read_text(encoding="utf-8"))
    folds = data.get("folds", [])
    if not folds:
        print(f"{f.parent.name}: NO FOLDS")
        continue
    eq_final = 1.0
    for fold in folds:
        eq_final *= fold["metrics"]["equity_mult"]
    mdd_max = max([fold["metrics"]["max_drawdown"] for fold in folds])
    trades_total = sum([fold["metrics"]["trades"] for fold in folds])
    monthly_geo = (eq_final ** (1.0 / (14 * 12))) - 1.0
    status = "✅ PASS" if mdd_max <= 0.50 and trades_total >= 280 else "❌ FAIL"
    print(f"{f.parent.name}: {status} | eq={eq_final:.2f} (+{(eq_final-1)*100:.0f}%) | MDD={mdd_max*100:.1f}% | trades={trades_total} | monthly={monthly_geo*100:.2f}%")

