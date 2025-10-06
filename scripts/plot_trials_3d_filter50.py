import os, json
from pathlib import Path
import pandas as pd, numpy as np
import plotly.express as px

root = Path("outputs/trial_logs/phase")
jsonls = list(root.rglob("trials_from_wfa.jsonl"))
rows=[]
for p in jsonls:
    K = p.parent.name
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try: d=json.loads(line)
                except: continue
                params=d.get("params") or {}
                tenkan=params.get("tenkan"); kijun=params.get("kijun"); sb=params.get("senkou_b")
                r_kj=params.get("r_kijun"); r_sb=params.get("r_senkou")
                try:
                    if kijun is None and tenkan is not None and r_kj is not None:
                        kijun=float(r_kj)*float(tenkan)
                    if sb is None:
                        base_kj = kijun if kijun is not None else (float(r_kj)*float(tenkan) if (tenkan is not None and r_kj is not None) else None)
                        if base_kj is not None and tenkan is not None and r_sb is not None:
                            sb=max(float(base_kj), float(r_sb)*float(tenkan))
                except: pass
                mt=d.get("metrics_train") or {}
                eq_ret=mt.get("eq_ret")
                if eq_ret is None:
                    eq_mult=mt.get("equity_mult")
                    if isinstance(eq_mult,(int,float)) and eq_mult>0: eq_ret=float(eq_mult)-1.0
                min_equity=mt.get("min_equity")
                if isinstance(min_equity,(int,float)):
                    dd = 1.0 - float(min_equity)
                else:
                    dd = mt.get("max_drawdown")
                try:
                    rows.append(dict(
                        K=K, tenkan=float(tenkan) if tenkan is not None else np.nan,
                        kijun=float(kijun) if kijun is not None else np.nan,
                        senkou_b=float(sb) if sb is not None else np.nan,
                        shift=float(params.get("shift", np.nan)),
                        atr_mult=float(params.get("atr_mult", np.nan)),
                        eq_ret=float(eq_ret) if eq_ret is not None else np.nan,
                        mdd=float(dd) if dd is not None else np.nan,
                        trades=int(mt.get("trades") or 0),
                        fold=str(d.get("run_context",{}).get("fold","")),
                        phase=str(d.get("run_context",{}).get("phase_label","")),
                        trial=int(d.get("trial_number", -1)),
                    ))
                except: continue
    except Exception:
        continue

if not rows:
    print("No trials JSONL found"); raise SystemExit(0)

df=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan).dropna(subset=["tenkan","kijun","atr_mult","eq_ret"])
# Gating: rendement >= 50% (eq_ret>=0.5) et MDD<=0.5 (si disponible)
df=df[(df["eq_ret"]>=0.5) & (df["mdd"].isna() | (df["mdd"]<=0.5))].copy()
if df.empty:
    print("No points match eq_ret>=0.5 and mdd<=0.5"); raise SystemExit(0)

fig=px.scatter_3d(
    df, x="tenkan", y="kijun", z="atr_mult",
    color=(df["eq_ret"]*100.0), color_continuous_scale="RdYlGn",
    size="trades",
    hover_data=["K","fold","phase","trial","senkou_b","shift","eq_ret","mdd","trades"],
    title="Top trials 3D — eq_ret ≥ 50% & MDD ≤ 50%"
)
fig.update_layout(scene=dict(zaxis_title="ATR×"), coloraxis_colorbar=dict(title="eq_ret (%)"))

aout=Path("docs/IMAGES/top_trials_live_50p.html"); aout.parent.mkdir(parents=True, exist_ok=True); fig.write_html(str(aout), include_plotlyjs="cdn"); print(aout)
