# launch_cycle_test.ps1
# Test WFA avec labels CYCLE_cash_bear (2-3 seeds)
# Usage: .\scripts\launch_cycle_test.ps1

$ErrorActionPreference = "Continue"
$ROOT = Split-Path -Parent $PSScriptRoot
if (-not $ROOT -or $ROOT -eq "") {
    $ROOT = Get-Location
}

# Configuration
$TRIALS = 100  # Moins de trials pour test rapide
$LABELS_CSV = Join-Path $ROOT "data\CYCLE_cash_bear.csv"
$OUT_BASE = Join-Path $ROOT "outputs\wfa_phase_cycle_test"
$MAX_PARALLEL = 3

# Seeds de test
$SEEDS = @(101, 102, 103)

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  TEST WFA CYCLE - 3 seeds (rapide)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Trials par seed:     $TRIALS"
Write-Host "  Nombre de seeds:     $($SEEDS.Count)"
Write-Host "  Labels CSV:          $LABELS_CSV"
Write-Host "  Output:              $OUT_BASE"
Write-Host ""

# Verifier que les labels existent
if (-not (Test-Path $LABELS_CSV)) {
    Write-Host "ERREUR: Labels CYCLE non trouves: $LABELS_CSV" -ForegroundColor Red
    exit 1
}

# Creer dossier de sortie
New-Item -ItemType Directory -Force -Path $OUT_BASE | Out-Null

Write-Host "Lancement des jobs..." -ForegroundColor Green
Write-Host ""

$jobCount = 0
foreach ($seed in $SEEDS) {
    $outDir = Join-Path $OUT_BASE "seed_$seed"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $logFile = Join-Path $outDir "run.log"

    Write-Host "  [$($jobCount + 1)/$($SEEDS.Count)] Seed $seed -> $outDir" -ForegroundColor Yellow

    Start-Job -Name "CYCLE_seed_$seed" -ScriptBlock {
        param($root, $labels, $trials, $seed, $outDir, $logFile)

        Set-Location $root
        $env:USE_FUSED_H2 = "1"
        $env:PYTHONIOENCODING = "utf-8"

        $startTime = Get-Date
        Write-Output "=== Seed $seed - Debut: $startTime ===" | Out-File $logFile

        & py scripts/run_scheduler_wfa_phase.py `
            --labels-csv $labels `
            --trials $trials `
            --seed $seed `
            --use-fused `
            --out-dir $outDir `
            --no-checkpoint 2>&1 | Tee-Object -FilePath $logFile -Append

        $endTime = Get-Date
        $duration = $endTime - $startTime
        Write-Output "=== Seed $seed - Fin: $endTime (Duree: $duration) ===" | Out-File $logFile -Append

    } -ArgumentList $ROOT, $LABELS_CSV, $TRIALS, $seed, $outDir, $logFile | Out-Null

    $jobCount++
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  $($SEEDS.Count) JOBS LANCES!" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Commandes:" -ForegroundColor Cyan
Write-Host "  Get-Job | Format-Table Name, State"
Write-Host "  Get-Job | Wait-Job"
Write-Host "  Get-Content '$OUT_BASE\seed_101\run.log' -Tail 30"
Write-Host ""
Write-Host "Duree estimee: ~30-60 min avec 100 trials" -ForegroundColor Yellow
Write-Host ""
