# ============================================================================
# Lancement test 5 seeds K3 avec labels 1D stables (phases daily + trading H2)
# ============================================================================
#
# Objectif:
#   Tester stratégie hybride:
#   - Phases détectées sur base 1D (label majoritaire jour J pour jour J+1)
#   - Trading exécuté sur H2 (12 opportunités/jour)
#   - Attendu: + de trades, meilleure stabilité, + de rendement
#
# Seeds: 1001-1005 (5 seeds test)
# Trials: 300 par fold
# Durée estimée: 24-48h pour les 5 seeds
#
# Usage:
#   .\scripts\launch_k3_1d_stable_test.ps1
#
# ============================================================================

$ErrorActionPreference = 'Stop'

# Configuration
$LABELS_CSV = "outputs\fourier\labels_frozen\BTC_FUSED_2h\K3_1d_stable.csv"
$OUT_ROOT = "E:\ichimoku_runs\wfa_phase_k3_1d_stable"
$SEEDS = @(1001, 1002, 1003, 1004, 1005)
$TRIALS = 300
$PYTHON = ".venv\Scripts\python.exe"
$SCRIPT = "scripts\run_scheduler_wfa_phase.py"

# Vérifications préliminaires
Write-Host "🔍 Vérifications préliminaires..." -ForegroundColor Cyan

if (-not (Test-Path $LABELS_CSV)) {
    Write-Host "❌ Labels introuvables: $LABELS_CSV" -ForegroundColor Red
    Write-Host "   Exécuter d'abord:" -ForegroundColor Yellow
    Write-Host "   python scripts\downsample_labels_2h_to_1d.py --k 3" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $PYTHON)) {
    Write-Host "❌ Python venv introuvable: $PYTHON" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $SCRIPT)) {
    Write-Host "❌ Script WFA introuvable: $SCRIPT" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Fichiers OK" -ForegroundColor Green
Write-Host ""

# Affichage config
Write-Host "📊 Configuration Test K3 1D Stable:" -ForegroundColor Cyan
Write-Host "   Labels:  $LABELS_CSV"
Write-Host "   Seeds:   $($SEEDS -join ', ')"
Write-Host "   Trials:  $TRIALS par fold"
Write-Host "   Output:  $OUT_ROOT\seed_XXXX"
Write-Host ""

# Confirmation
$confirm = Read-Host "Lancer les 5 seeds en parallèle? (y/N)"
if ($confirm -ne 'y' -and $confirm -ne 'Y') {
    Write-Host "❌ Annulé par l'utilisateur" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "🚀 Lancement des 5 seeds K3 1D stable..." -ForegroundColor Green
Write-Host ""

# Lancement parallèle des 5 seeds
$jobs = @()

foreach ($seed in $SEEDS) {
    $outDir = Join-Path $OUT_ROOT "seed_$seed"
    
    # Créer répertoire output
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    
    # Fichiers logs
    $logFile = Join-Path $outDir "run.log"
    $errFile = Join-Path $outDir "run.err"
    
    Write-Host "   Seed $seed → $outDir" -ForegroundColor Cyan
    
    # Lancer en background
    $job = Start-Job -ScriptBlock {
        param($python, $script, $labels, $outDir, $seed, $trials, $logFile, $errFile)
        
        & $python $script `
            --labels-csv $labels `
            --out-dir $outDir `
            --seed $seed `
            --trials $trials `
            --use-fused `
            *> $logFile 2> $errFile
            
    } -ArgumentList $PYTHON, $SCRIPT, $LABELS_CSV, $outDir, $seed, $TRIALS, $logFile, $errFile
    
    $jobs += $job
}

Write-Host ""
Write-Host "✅ 5 seeds lancés en parallèle (Job IDs: $($jobs.Id -join ', '))" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Monitoring:" -ForegroundColor Cyan
Write-Host "   Get-Job                     → voir statut jobs"
Write-Host "   Get-Job | Receive-Job       → voir output"
Write-Host "   Get-Job | Stop-Job          → arrêter tous"
Write-Host ""
Write-Host "📂 Logs temps réel:"
foreach ($seed in $SEEDS) {
    $logFile = Join-Path $OUT_ROOT "seed_$seed\run.log"
    Write-Host "   Seed $seed : Get-Content -Wait $logFile"
}

Write-Host ""
Write-Host "⏱️  Durée estimée: 24-48h pour les 5 seeds" -ForegroundColor Yellow
Write-Host ""
Write-Host "📈 Suivi avancement:" -ForegroundColor Cyan
Write-Host "   Commande à copier/coller toutes les 5-10 min:" -ForegroundColor Yellow
Write-Host ""
$monitorCmd = "Get-ChildItem 'E:\ichimoku_runs\wfa_phase_k3_1d_stable' -Recurse -Filter 'PROGRESS.json' -ErrorAction SilentlyContinue | ForEach-Object { try { " + '$j = Get-Content $_.FullName -Raw | ConvertFrom-Json; $seed = Split-Path $_.Directory -Leaf; "$seed : $([math]::Round($j.percent,1))%"' + " } catch {} }"
Write-Host "   $monitorCmd" -ForegroundColor White
Write-Host ""
Write-Host ""
Write-Host "🎯 Une fois terminé, comparer avec:" -ForegroundColor Green
Write-Host "   python scripts\compare_h2_vs_1d_stable.py" -ForegroundColor White
Write-Host ""

# Afficher job status initial
Start-Sleep -Seconds 5
Write-Host "📊 Status initial des jobs:" -ForegroundColor Cyan
Get-Job | Format-Table -AutoSize

Write-Host ""
Write-Host "✅ Setup terminé! Les seeds tournent en background." -ForegroundColor Green
Write-Host "   Utilise Get-Job pour monitorer." -ForegroundColor Cyan

