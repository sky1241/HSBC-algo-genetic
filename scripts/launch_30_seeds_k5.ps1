# launch_30_seeds_k5.ps1
# Lancement parallele de 30 seeds K5 (300 trials chacun)
# Usage: .\scripts\launch_30_seeds_k5.ps1

$ErrorActionPreference = "Continue"
$ROOT = Split-Path -Parent $PSScriptRoot
if (-not $ROOT -or $ROOT -eq "") {
    $ROOT = Get-Location
}

# Configuration
$TRIALS = 300
$LABELS_CSV = Join-Path $ROOT "outputs\fourier\labels_frozen\BTC_FUSED_2h\K5.csv"
$OUT_BASE = Join-Path $ROOT "outputs\wfa_phase_k5"
$MAX_PARALLEL = 6  # Nombre max de jobs paralleles

# Liste des 30 seeds
$SEEDS = @(
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
    301, 302, 303, 304, 305, 306, 307, 308, 309, 310
)

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  LANCEMENT 30 SEEDS K5 - WFA Phase-Adapte" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Trials par seed:     $TRIALS"
Write-Host "  Nombre de seeds:     $($SEEDS.Count)"
Write-Host "  Jobs paralleles max: $MAX_PARALLEL"
Write-Host "  Labels CSV:          $LABELS_CSV"
Write-Host "  Output:              $OUT_BASE"
Write-Host ""

# Verifier que les labels existent
if (-not (Test-Path $LABELS_CSV)) {
    Write-Host "ERREUR: Labels K5 non trouves: $LABELS_CSV" -ForegroundColor Red
    Write-Host "Executez d'abord: python scripts/freeze_hmm_labels.py" -ForegroundColor Yellow
    exit 1
}

# Creer dossier de sortie
New-Item -ItemType Directory -Force -Path $OUT_BASE | Out-Null

Write-Host "Lancement des jobs..." -ForegroundColor Green
Write-Host ""

$jobCount = 0
foreach ($seed in $SEEDS) {
    # Attendre si trop de jobs en cours
    while ((Get-Job -State Running).Count -ge $MAX_PARALLEL) {
        Write-Host "  Attente (max $MAX_PARALLEL jobs)..." -ForegroundColor Gray
        Start-Sleep -Seconds 30
    }

    $outDir = Join-Path $OUT_BASE "seed_$seed"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $logFile = Join-Path $outDir "run.log"

    Write-Host "  [$($jobCount + 1)/$($SEEDS.Count)] Seed $seed -> $outDir" -ForegroundColor Yellow

    Start-Job -Name "K5_seed_$seed" -ScriptBlock {
        param($root, $labels, $trials, $seed, $outDir, $logFile)

        Set-Location $root
        $env:USE_FUSED_H2 = "1"
        $env:PYTHONIOENCODING = "utf-8"

        $startTime = Get-Date
        Write-Output "=== Seed $seed - Debut: $startTime ===" | Out-File $logFile

        & python scripts/run_scheduler_wfa_phase.py `
            --labels-csv $labels `
            --trials $trials `
            --seed $seed `
            --use-fused `
            --out-dir $outDir 2>&1 | Tee-Object -FilePath $logFile -Append

        $endTime = Get-Date
        $duration = $endTime - $startTime
        Write-Output "=== Seed $seed - Fin: $endTime (Duree: $duration) ===" | Out-File $logFile -Append

    } -ArgumentList $ROOT, $LABELS_CSV, $TRIALS, $seed, $outDir, $logFile | Out-Null

    $jobCount++
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  $($SEEDS.Count) JOBS LANCES EN BACKGROUND!" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Commandes utiles:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  # Voir tous les jobs"
Write-Host "  Get-Job | Format-Table Name, State, HasMoreData"
Write-Host ""
Write-Host "  # Voir jobs en cours"
Write-Host "  Get-Job -State Running"
Write-Host ""
Write-Host "  # Voir avancement"
Write-Host "  Get-ChildItem '$OUT_BASE' -Recurse -Filter 'PROGRESS.json' | ForEach-Object {"
Write-Host "    `$j = Get-Content `$_.FullName | ConvertFrom-Json"
Write-Host "    Write-Host `"`$(`$_.Directory.Name): `$(`$j.percent)%`""
Write-Host "  }"
Write-Host ""
Write-Host "  # Logs en temps reel (seed 101)"
Write-Host "  Get-Content -Wait '$OUT_BASE\seed_101\run.log' -Tail 20"
Write-Host ""
Write-Host "  # Attendre tous les jobs"
Write-Host "  Get-Job | Wait-Job"
Write-Host ""
Write-Host "  # Nettoyer apres completion"
Write-Host "  Get-Job | Remove-Job"
Write-Host ""
Write-Host "Duree estimee: ~20-30 heures avec $MAX_PARALLEL jobs paralleles" -ForegroundColor Yellow
Write-Host ""
