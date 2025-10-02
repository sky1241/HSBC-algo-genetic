import os
import glob
import json
import math
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None  # graceful fallback


def main() -> int:
    here = pathlib.Path(__file__).resolve().parents[1]
    json_glob = str(here / "outputs" / "wfa_phase_k*" / "seed_*" / "WFA_phase_*.json")
    jsons = glob.glob(json_glob)

    rows = []
    for fp in jsons:
        try:
            base = os.path.basename(fp)
            if "WFA_phase_K" in base:
                k_str = base.split("WFA_phase_K")[1].split("_")[0]
            else:
                low = fp.lower()
                k_str = low.split("wfa_phase_k")[1].split(os.sep)[0]
            seed = int([p for p in fp.split(os.sep) if p.startswith("seed_")][0].split("_")[1])

            with open(fp, "r", encoding="utf-8") as f:
                j = json.load(f)
            ov = j.get("overall", {})
            eq = float(ov.get("equity_mult", float("nan")))
            folds = int(ov.get("folds", len(j.get("folds", [])) or 1))
            # Equity-based metrics
            cagr_y = (eq ** (1.0 / max(1, folds)) - 1.0) if (eq > 0 and folds > 0) else float("nan")
            cagr_m = ((1.0 + cagr_y) ** (1.0 / 12.0) - 1.0) if not math.isnan(cagr_y) else float("nan")
            eq_ret = (eq - 1.0) if (eq > 0) else float("nan")  # total return (relative), e.g., 0.59 for Eqx 1.59
            mdd = float(ov.get("max_drawdown", float("nan")))
            sh = float(ov.get("sharpe_proxy_mean", float("nan")))

            for fr in j.get("folds", []):
                pbs = fr.get("params_by_state", {}) or {}
                for phase, pm in pbs.items():
                    rows.append(
                        dict(
                            K=f"K{k_str}",
                            seed=seed,
                            phase=str(phase),
                            tenkan=float(pm.get("tenkan", np.nan)),
                            kijun=float(pm.get("kijun", np.nan)),
                            senkou_b=float(pm.get("senkou_b", np.nan)),
                            shift=float(pm.get("shift", np.nan)),
                            atr_mult=float(pm.get("atr_mult", np.nan)),
                            cagr_m=cagr_m,
                            eq_ret=eq_ret,
                            mdd=mdd,
                            sharpe=sh,
                        )
                    )
        except Exception:
            # Ignore malformed/partial files gracefully
            continue

    if not rows:
        print("No phase JSONs found.")
        return 0

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["tenkan", "kijun", "senkou_b", "shift", "atr_mult", "eq_ret", "mdd"]
    )
    df = df[df["mdd"] <= 0.50]

    out_dir = here / "docs" / "IMAGES" / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build all pairwise combinations of Ichimoku params
    from itertools import combinations
    ichimoku_params = ["tenkan", "kijun", "senkou_b", "shift", "atr_mult"]
    pairs = list(combinations(ichimoku_params, 2))

    def bin_series(s: pd.Series, nb: int = 15) -> pd.Series:
        try:
            return pd.cut(s, bins=nb)
        except Exception:
            return s

    for K, gk in df.groupby("K"):
        for phase, gp in gk.groupby("phase"):
            for a, b in pairs:
                sub = gp[[a, b, "eq_ret"]].dropna()
                if len(sub) < 20:
                    continue
                A = bin_series(sub[a])
                B = bin_series(sub[b])
                # Median of equity total return (relative). We multiply by 100 at plot time for %.
                piv = sub.assign(A=A, B=B).groupby(["A", "B"])["eq_ret"].median().unstack("B")
                if piv is None or piv.shape[0] == 0:
                    continue
                plt.figure(figsize=(8, 6))
                if sns is not None:
                    # Green = good, Red = bad
                    sns.heatmap(
                        piv * 100.0,
                        cmap="RdYlGn",
                        center=0.0,
                        cbar_kws={"label": "Rendement total médian (%)"},
                    )
                else:
                    plt.imshow(piv * 100.0, aspect="auto", cmap="RdYlGn", origin="lower")
                    plt.colorbar(label="Rendement total médian (%)")
                plt.title(f"{K} — phase={phase} — {a}×{b} (MDD≤50%) — couleur: Eqx-1 (%)")
                plt.xlabel(b)
                plt.ylabel(a)
                fn = out_dir / f"{K}_phase_{phase}_{a}_x_{b}.png"
                plt.tight_layout()
                plt.savefig(fn, dpi=160)
                plt.close()

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import os, glob, json, math, pathlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt; import seaborn as sns

here = pathlib.Path(__file__).resolve().parents[1]
jsons = glob.glob(str(here / "outputs" / "wfa_phase_k*" / "seed_*" / "WFA_phase_*.json"))
rows = []
for fp in jsons:
    try:
        K = None
        base = os.path.basename(fp)
        if "WFA_phase_K" in base:
            K = base.split("WFA_phase_K")[1].split("_")[0]
        else:
            K = fp.lower().split("wfa_phase_k")[1].split(os.sep)[0]
        seed = int([p for p in fp.split(os.sep) if p.startswith("seed_")][0].split("_")[1])
        with open(fp, "r", encoding="utf-8") as f:
            j = json.load(f)
        ov = j.get("overall", {})
        eq = float(ov.get("equity_mult", float("nan")))
        folds = int(ov.get("folds", len(j.get("folds", [])) or 1))
        cagr_y = (eq**(1.0/max(1, folds)) - 1.0) if (eq>0 and folds>0) else float("nan")
        cagr_m = ((1.0 + cagr_y)**(1.0/12.0) - 1.0) if (not math.isnan(cagr_y)) else float("nan")
        mdd = float(ov.get("max_drawdown", float("nan")))
        sharpe = float(ov.get("sharpe_proxy_mean", float("nan")))
        # paramètres par phase (prennent ceux optimisés sur train pour chaque fold; approximation: on stocke tous)
        for fr in j.get("folds", []):
            pbs = fr.get("params_by_state", {}) or {}
            for phase, pm in pbs.items():
                rows.append(dict(
                    K=f"K{K}", seed=seed, phase=str(phase),
                    tenkan=float(pm.get("tenkan", float("nan"))),
                    kijun=float(pm.get("kijun", float("nan"))),
                    senkou_b=float(pm.get("senkou_b", float("nan"))),
                    shift=float(pm.get("shift", float("nan"))),
                    atr_mult=float(pm.get("atr_mult", float("nan"))),
                    cagr_m=cagr_m, mdd=mdd, sharpe=sharpe
                ))
    except Exception:
        pass

df = pd.DataFrame(rows)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["tenkan","kijun","senkou_b","shift","atr_mult","cagr_m","mdd"])
df = df[df["mdd"] <= 0.50]  # contrainte MDD
out_dir = here / "docs" / "IMAGES" / "heatmaps"
out_dir.mkdir(parents=True, exist_ok=True)

pairs = [("tenkan","kijun"),("senkou_b","shift"),("kijun","senkou_b"),("atr_mult","shift")]
bins = dict(tenkan=15, kijun=15, senkou_b=15, shift=15, atr_mult=15)

def bin_series(s, nb):
    try:
        return pd.cut(s, bins=nb)
    except Exception:
        return s

for K, gk in df.groupby("K"):
    for phase, gp in gk.groupby("phase"):
        for a,b in pairs:
            sub = gp[[a,b,"cagr_m"]].dropna()
            if len(sub) < 20: 
                continue
            A = bin_series(sub[a], bins.get(a, 15))
            B = bin_series(sub[b], bins.get(b, 15))
            piv = sub.assign(A=A, B=B).groupby(["A","B"])["cagr_m"].median().unstack("B")
            if piv is None or piv.shape[0]==0:
                continue
            plt.figure(figsize=(8,6))
            sns.heatmap(piv*100.0, cmap="viridis", cbar_kws={"label":"Rendement mensuel médian (%)"})
            plt.title(f"{K} — phase={phase} — {a}×{b} (MDD≤50%)")
            plt.xlabel(b); plt.ylabel(a)
            fn = out_dir / f"{K}_phase_{phase}_{a}_x_{b}.png"
            plt.tight_layout(); plt.savefig(fn, dpi=160); plt.close()
print(str(out_dir))
