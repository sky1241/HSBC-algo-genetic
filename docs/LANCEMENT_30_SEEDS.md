# Guide de Lancement - 30 Seeds K5 en Parallele

**Date**: 2025-02-03
**Version**: v5.0 (P2+P3 Complete)

---

## Pre-requis

### 1. Verifier l'environnement Python
```powershell
python --version  # Python 3.10+ requis
pip install pandas numpy optuna ccxt
```

### 2. Verifier les labels K5
```powershell
# Les labels doivent exister
Test-Path "outputs/fourier/labels_frozen/BTC_FUSED_2h/K5.csv"
```

### 3. Verifier les donnees BTC_FUSED_2h
```powershell
# Le fichier de donnees doit exister
Test-Path "data/BTC_FUSED_2h.csv"
```

---

## Commande Principale - Lancement 30 Seeds

### Option A: Script PowerShell (Recommande)

Creer et executer ce script `launch_30_seeds_k5.ps1`:

```powershell
# launch_30_seeds_k5.ps1
# Lancement parallele de 30 seeds K5 (300 trials chacun)

$ErrorActionPreference = "Continue"
$ROOT = $PSScriptRoot
if (-not $ROOT) { $ROOT = Get-Location }

# Configuration
$TRIALS = 300
$LABELS_CSV = "$ROOT\outputs\fourier\labels_frozen\BTC_FUSED_2h\K5.csv"
$OUT_BASE = "$ROOT\outputs\wfa_phase_k5"

# Liste des 30 seeds
$SEEDS = @(
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
    301, 302, 303, 304, 305, 306, 307, 308, 309, 310
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "LANCEMENT 30 SEEDS K5" -ForegroundColor Cyan
Write-Host "Trials: $TRIALS | Seeds: $($SEEDS.Count)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Creer dossier de sortie
New-Item -ItemType Directory -Force -Path $OUT_BASE | Out-Null

# Lancer chaque seed en background
foreach ($seed in $SEEDS) {
    $outDir = "$OUT_BASE\seed_$seed"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $logFile = "$outDir\run.log"

    Write-Host "Lancement seed $seed..." -ForegroundColor Yellow

    Start-Job -Name "K5_seed_$seed" -ScriptBlock {
        param($root, $labels, $trials, $seed, $outDir, $logFile)
        Set-Location $root
        $env:USE_FUSED_H2 = "1"
        & python scripts/run_scheduler_wfa_phase.py `
            --labels-csv $labels `
            --trials $trials `
            --seed $seed `
            --use-fused `
            --out-dir $outDir 2>&1 | Tee-Object -FilePath $logFile
    } -ArgumentList $ROOT, $LABELS_CSV, $TRIALS, $seed, $outDir, $logFile

    # Pause entre les lancements pour eviter surcharge
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "$($SEEDS.Count) jobs lances en background!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Commandes utiles:" -ForegroundColor Cyan
Write-Host "  Get-Job                    # Voir tous les jobs"
Write-Host "  Get-Job | Where State -eq Running  # Jobs en cours"
Write-Host "  Receive-Job -Name K5_seed_101      # Voir output d'un job"
Write-Host "  Get-Job | Remove-Job -Force        # Supprimer tous les jobs"
```

**Execution:**
```powershell
cd "C:\Users\ludov\OneDrive\Bureau\HSBC-algo-genetic-main"
.\launch_30_seeds_k5.ps1
```

---

### Option B: Commande Bash/WSL (Alternative)

```bash
cd "/mnt/c/Users/ludov/OneDrive/Bureau/HSBC-algo-genetic-main"

# Lancer 30 seeds en parallele (max 6 simultanes pour eviter surcharge CPU)
for seed in 101 102 103 104 105 106 107 108 109 110 \
             201 202 203 204 205 206 207 208 209 210 \
             301 302 303 304 305 306 307 308 309 310; do

    mkdir -p "outputs/wfa_phase_k5/seed_$seed"

    USE_FUSED_H2=1 python scripts/run_scheduler_wfa_phase.py \
        --labels-csv outputs/fourier/labels_frozen/BTC_FUSED_2h/K5.csv \
        --trials 300 \
        --seed $seed \
        --use-fused \
        --out-dir "outputs/wfa_phase_k5/seed_$seed" \
        > "outputs/wfa_phase_k5/seed_$seed/run.log" 2>&1 &

    # Limiter a 6 jobs paralleles
    while [ $(jobs -r | wc -l) -ge 6 ]; do
        sleep 60
    done
done

echo "30 seeds lances! Utilisez 'jobs' pour voir le statut."
```

---

### Option C: Commande Unique (Un seed a la fois)

Pour tester un seul seed:
```powershell
cd "C:\Users\ludov\OneDrive\Bureau\HSBC-algo-genetic-main"

$env:USE_FUSED_H2 = "1"
python scripts/run_scheduler_wfa_phase.py `
    --labels-csv outputs/fourier/labels_frozen/BTC_FUSED_2h/K5.csv `
    --trials 300 `
    --seed 42 `
    --use-fused `
    --out-dir outputs/wfa_phase_k5/seed_42
```

---

## Monitoring

### Voir l'avancement de tous les jobs
```powershell
Get-Job | Format-Table Name, State, HasMoreData
```

### Voir l'avancement detaille
```powershell
Get-ChildItem "outputs\wfa_phase_k5" -Recurse -Filter "PROGRESS.json" | ForEach-Object {
    $content = Get-Content $_.FullName | ConvertFrom-Json
    Write-Host "$($_.Directory.Name): $($content.percent)% (fold $($content.folds_done)/$($content.folds_total))"
}
```

### Voir les logs en temps reel
```powershell
Get-Content -Wait "outputs\wfa_phase_k5\seed_101\run.log" -Tail 50
```

### Verifier les erreurs
```powershell
Get-ChildItem "outputs\wfa_phase_k5" -Recurse -Filter "*.log" | ForEach-Object {
    $errors = Select-String -Path $_.FullName -Pattern "ERROR|Exception|Traceback" -SimpleMatch
    if ($errors) {
        Write-Host "Erreurs dans $($_.FullName):" -ForegroundColor Red
        $errors | Select-Object -First 5
    }
}
```

---

## Duree Estimee

| Configuration | Duree par seed | Duree totale (30 seeds) |
|---------------|----------------|-------------------------|
| 300 trials, 14 folds | ~4-6 heures | ~5-7 jours (sequentiel) |
| 300 trials, 14 folds (6 paralleles) | ~4-6 heures | ~20-28 heures |
| 100 trials, 14 folds | ~1-2 heures | ~6-10 heures |

**Note:** La duree depend de votre CPU. Avec 6 jobs paralleles sur un bon PC, comptez ~24h.

---

## Apres Completion

### 1. Verifier que tous les seeds sont termines
```powershell
Get-ChildItem "outputs\wfa_phase_k5" -Directory | ForEach-Object {
    $json = Get-ChildItem $_.FullName -Filter "WFA_phase_K5_*.json" | Select-Object -First 1
    if ($json) {
        Write-Host "$($_.Name): OK" -ForegroundColor Green
    } else {
        Write-Host "$($_.Name): INCOMPLET" -ForegroundColor Red
    }
}
```

### 2. Agreger les resultats
```powershell
python scripts/aggregate_k5_results.py --input-dir outputs/wfa_phase_k5
```

### 3. Comparer K3 vs K5
```powershell
python scripts/compare_k3_k5.py
```

---

## Troubleshooting

### Job bloque ou plante
```powershell
# Arreter un job specifique
Stop-Job -Name "K5_seed_101"
Remove-Job -Name "K5_seed_101"

# Relancer ce seed
python scripts/run_scheduler_wfa_phase.py --labels-csv ... --seed 101 ...
```

### Memoire insuffisante
- Reduire le nombre de jobs paralleles (4 au lieu de 6)
- Fermer les applications en arriere-plan

### Erreur "Fused CSV not configured"
```powershell
$env:USE_FUSED_H2 = "1"
# Ou verifier que data/BTC_FUSED_2h.csv existe
```

---

## Fichiers de Sortie

Apres completion, vous aurez:
```
outputs/wfa_phase_k5/
├── seed_101/
│   ├── WFA_phase_K5_BTC_fused_YYYYMMDD_HHMMSS.json
│   ├── PROGRESS.json
│   └── run.log
├── seed_102/
│   └── ...
└── seed_310/
    └── ...
```

Chaque JSON contient:
- `overall`: metriques agregees (equity_mult, max_drawdown, trades)
- `folds`: details par annee avec params_by_state

---

**Bonne optimisation!**
