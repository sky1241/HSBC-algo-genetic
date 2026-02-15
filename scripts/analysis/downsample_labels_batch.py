#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Générer labels 1D stable pour K4, K6, K7, K8 en batch."""
import subprocess
import sys

print("="*70)
print("📊 Génération labels 1D stable pour K4/K6/K7/K8")
print("="*70)

for k in [4, 6, 7, 8]:
    print(f"\n🔄 Génération K{k}...")
    cmd = [sys.executable, "scripts/downsample_labels_2h_to_1d.py", "--k", str(k)]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        print(f"✅ K{k} OK")
    else:
        print(f"❌ K{k} FAIL (code {result.returncode})")

print("\n" + "="*70)
print("✅ Génération terminée")
print("="*70)

