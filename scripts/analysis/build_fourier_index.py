#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construit un index Markdown des rapports Fourier mensuels (H2).
Parcourt outputs/fourier/monthly/YYYY/MM/ et écrit docs/FOURIER_RAPPORTS_BTC_2h.md

Usage:
  py -3 scripts/build_fourier_index.py --root outputs/fourier/monthly --symbol BTC/USDT --timeframe 2h
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=str(Path('outputs') / 'fourier' / 'monthly'))
    p.add_argument('--symbol', default='BTC/USDT')
    p.add_argument('--timeframe', default='2h')
    p.add_argument('--out', default=str(Path('docs') / 'FOURIER_RAPPORTS_BTC_2h.md'))
    args = p.parse_args()

    root = Path(args.root)
    entries: list[tuple[int,int,Path]] = []
    if not root.exists():
        print(f"No directory: {root}")
        return 1

    for ydir in sorted([d for d in root.iterdir() if d.is_dir()]):
        try:
            y = int(ydir.name)
        except Exception:
            continue
        for mdir in sorted([d for d in ydir.iterdir() if d.is_dir()]):
            try:
                m = int(mdir.name)
            except Exception:
                continue
            entries.append((y, m, mdir))

    lines: list[str] = []
    lines.append(f"Fourier — rapports mensuels ({args.symbol} {args.timeframe})\n")
    lines.append("Classés du plus ancien au plus récent. Chaque section embarque les graphiques (une seule page).\n")
    # Section comparaison 2h vs 1d (affichée principalement sur la page 2h)
    try:
        if args.timeframe == '2h':
            base_plots = Path('outputs') / 'fourier' / 'plots'
            sym = args.symbol.replace('/', '_')
            h2_p_ann = (Path('..') / base_plots / f"{sym}_2h_P_annual.png").as_posix()
            h2_l_ann = (Path('..') / base_plots / f"{sym}_2h_LFP_annual.png").as_posix()
            d1_p_ann = (Path('..') / base_plots / f"{sym}_1d_P_annual.png").as_posix()
            d1_l_ann = (Path('..') / base_plots / f"{sym}_1d_LFP_annual.png").as_posix()
            lines.append("\n### Comparaison 2h vs 1d — P1/P2/P3 et LFP (rolling annual)\n")
            lines.append(f"- 2h P: {h2_p_ann}\n- 2h LFP: {h2_l_ann}\n- 1d P: {d1_p_ann}\n- 1d LFP: {d1_l_ann}\n")
            lines.append(f"\n![P 2h annual]({h2_p_ann})\n\n![P 1d annual]({d1_p_ann})\n")
            lines.append(f"\n![LFP 2h annual]({h2_l_ann})\n\n![LFP 1d annual]({d1_l_ann})\n")
    except Exception:
        pass

    # Section Prix vs Volume pour le timeframe courant (annual + monthly)
    try:
        base_plots = Path('outputs') / 'fourier' / 'plots'
        sym = args.symbol.replace('/', '_')
        pvol_ann = (Path('..') / base_plots / f"{sym}_{args.timeframe}_PVOL_annual.png").as_posix()
        lfpv_ann = (Path('..') / base_plots / f"{sym}_{args.timeframe}_LFPv_annual.png").as_posix()
        pvol_mon = (Path('..') / base_plots / f"{sym}_{args.timeframe}_PVOL_monthly.png").as_posix()
        lfpv_mon = (Path('..') / base_plots / f"{sym}_{args.timeframe}_LFPv_monthly.png").as_posix()
        lines.append("\n### Prix vs Volume — cycles et LFP (rolling annual / monthly)\n")
        lines.append(f"- PVOL annual: {pvol_ann}\n- LFPv annual: {lfpv_ann}\n- PVOL monthly: {pvol_mon}\n- LFPv monthly: {lfpv_mon}\n")
        lines.append(f"\n![PVOL annual]({pvol_ann})\n\n![LFPv annual]({lfpv_ann})\n")
        lines.append(f"\n![PVOL monthly]({pvol_mon})\n\n![LFPv monthly]({lfpv_mon})\n")
    except Exception:
        pass
    for (y, m, mdir) in entries:
        sym = args.symbol.replace('/', '_')
        csv = mdir / f"FREQ_{sym}_{args.timeframe}_{y:04d}-{m:02d}.csv"
        p_png = mdir / f"P_{sym}_{args.timeframe}_{y:04d}-{m:02d}.png"
        lfp_png = mdir / f"LFP_{sym}_{args.timeframe}_{y:04d}-{m:02d}.png"
        lines.append(f"\n### {y:04d}-{m:02d}")
        # Résumé P/LFP (lecture première valeur non nulle si dispo)
        if csv.exists():
            try:
                import pandas as _pd
                tmp = _pd.read_csv(csv)
                def _first(col: str):
                    return tmp[col].dropna().iloc[0] if col in tmp.columns and tmp[col].dropna().shape[0] > 0 else ''
                valP1 = _first('P1_bars') or _first('P_bars')
                valP2 = _first('P2_bars')
                valP3 = _first('P3_bars')
                valL = _first('LFP')
                if valP2 != '' or valP3 != '':
                    lines.append(f"- Résumé: P1≈ {valP1} | P2≈ {valP2} | P3≈ {valP3} | LFP≈ {valL}")
                else:
                    lines.append(f"- Résumé: P≈ {valP1} | LFP≈ {valL}")
            except Exception:
                lines.append(f"- Fichier CSV: {csv.as_posix()}")
        # Images embarquées
        # Chemins relatifs depuis docs/ vers outputs/ (../outputs/...)
        def _doc_rel(p: Path) -> str:
            return (Path('..') / p).as_posix()

        if p_png.exists():
            lines.append(f"\n![P {y:04d}-{m:02d}]({_doc_rel(p_png)})\n")
        if lfp_png.exists():
            lines.append(f"\n![LFP {y:04d}-{m:02d}]({_doc_rel(lfp_png)})\n")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Wrote {outp}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


