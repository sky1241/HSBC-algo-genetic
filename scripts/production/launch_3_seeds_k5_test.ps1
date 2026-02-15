# LANCEMENT 3 SEEDS K5 EN PARALLELE
# Test rapide avant de lancer les 30 seeds

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " LANCEMENT 3 SEEDS K5 (TEST PARALLELE)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Duree estimee: 15-20 heures (3 seeds x 5-6h)"
Write-Host ""

$ROOT = Split-Path -Parent $PSScriptRoot
$LABELS = "$ROOT\outputs\fourier\labels_frozen\BTC_FUSED_2h\K5.csv"
$OUT_DIR = "$ROOT\outputs\wfa_phase_k5"
$TRIALS = 300
$SEEDS = @(2001, 2002, 2003)

# Verifier que les labels existent
if (-not (Test-Path $LABELS)) {
    Write-Host "[ERREUR] Labels K5 non trouves: $LABELS" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Labels K5 trouves" -ForegroundColor Green
Write-Host ""

# Lancer les 3 seeds en parallele
$jobs = @()

foreach ($seed in $SEEDS) {
    $seedDir = "$OUT_DIR\seed_$seed"

    Write-Host "Lancement seed $seed..." -ForegroundColor Yellow

    $job = Start-Job -ScriptBlock {
        param($root, $labels, $outDir, $trials, $seed)

        Set-Location $root

        $env:PYTHONIOENCODING = "utf-8"

        py scripts/run_scheduler_wfa_phase.py `
            --labels-csv $labels `
            --trials $trials `
            --seed $seed `
            --use-fused `
            --out-dir $outDir `
            --checkpoint-interval 5

    } -ArgumentList $ROOT, $LABELS, "$OUT_DIR\seed_$seed", $TRIALS, $seed

    $jobs += $job
    Write-Host "  Job ID: $($job.Id)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " $($jobs.Count) JOBS LANCES EN PARALLELE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Pour suivre la progression:"
Write-Host "  - PROGRESS.json dans chaque dossier seed_*"
Write-Host "  - Get-Job | Format-Table"
Write-Host "  - Receive-Job -Id <ID>"
Write-Host ""
Write-Host "Pour arreter:"
Write-Host "  - Stop-Job -Id <ID>"
Write-Host "  - Les checkpoints permettent de reprendre"
Write-Host ""

# Attendre que tous les jobs finissent
Write-Host "Attente de la fin des jobs..." -ForegroundColor Yellow
Write-Host "(Ctrl+C pour arreter - les checkpoints sauvegarderont la progression)"
Write-Host ""

$completed = 0
while ($completed -lt $jobs.Count) {
    $completed = ($jobs | Where-Object { $_.State -eq "Completed" -or $_.State -eq "Failed" }).Count

    # Afficher progression
    foreach ($seed in $SEEDS) {
        $progressFile = "$OUT_DIR\seed_$seed\PROGRESS.json"
        if (Test-Path $progressFile) {
            $progress = Get-Content $progressFile | ConvertFrom-Json
            Write-Host "  Seed $seed : $([math]::Round($progress.percent, 1))% (fold $($progress.folds_done)/$($progress.folds_total))" -ForegroundColor Gray
        }
    }

    Start-Sleep -Seconds 60
    Write-Host "--- $(Get-Date -Format 'HH:mm:ss') - $completed/$($jobs.Count) jobs termines ---"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " TOUS LES JOBS TERMINES" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Afficher les resultats
foreach ($job in $jobs) {
    Write-Host ""
    Write-Host "Job $($job.Id) - Status: $($job.State)" -ForegroundColor $(if ($job.State -eq "Completed") { "Green" } else { "Red" })
    Receive-Job -Job $job
}

Write-Host ""
Write-Host "Resultats dans: $OUT_DIR"
