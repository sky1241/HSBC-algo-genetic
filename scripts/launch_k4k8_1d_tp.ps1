# ============================================================================
# Lancement K4 et K8 avec TP adaptatif (30 seeds, labels 1D stable)
# ============================================================================
# Utilise mêmes seeds que K3/K5 + TP optimisé par Optuna

$repo = "C:\Users\ludov\OneDrive\Bureau\teste algo\ichimoku_pipeline_web_v4_8"
Set-Location -LiteralPath $repo

# Récupérer seeds du batch
$seedsDoc = Get-ChildItem (Join-Path $repo "docs\seeds\SEEDS_BATCH_K3K4K5K8_*.json") | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1 -ExpandProperty FullName

if (-not (Test-Path $seedsDoc)) {
    Write-Host "❌ Fichier seeds introuvable" -ForegroundColor Red
    exit 1
}

$seeds = (Get-Content $seedsDoc -Raw | ConvertFrom-Json).seeds
$PYTHON = Join-Path $repo ".venv\Scripts\python.exe"
$SCRIPT = Join-Path $repo "scripts\run_scheduler_wfa_phase.py"
$TRIALS = 300

Write-Host "🚀 Lancement K4 & K8 avec TP adaptatif (30 seeds chacun)" -ForegroundColor Green
Write-Host ""

# ===== K4 =====
Write-Host "📊 K=4 (30 seeds séquentiels)..." -ForegroundColor Cyan
$LABELS_K4 = Join-Path $repo "outputs\fourier\labels_frozen\BTC_FUSED_2h\K4_1d_stable.csv"
$OUT_K4 = "E:\ichimoku_runs\wfa_phase_k4_1d_tp"

if (-not (Test-Path $LABELS_K4)) {
    Write-Host "❌ Labels K4 introuvables: $LABELS_K4" -ForegroundColor Red
    exit 1
}

Get-ChildItem $OUT_K4 -Recurse -File -Filter ".lock" -ErrorAction SilentlyContinue | Remove-Item -Force
$i = 0
foreach ($s in $seeds) {
    $i++
    $out = Join-Path $OUT_K4 ("seed_{0}" -f $s)
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $log = Join-Path $out "run.log"
    $err = Join-Path $out "run.err"
    Remove-Item $log,$err -ErrorAction SilentlyContinue
    Write-Host "  [$i/30] Seed $s..." -ForegroundColor Yellow
    & $PYTHON $SCRIPT --labels-csv $LABELS_K4 --granularity annual --trials $TRIALS --seed $s --jobs 1 --loss-mult 3.0 --use-fused --out-dir $out *> $log 2> $err
    if (Test-Path (Join-Path $out "WFA_phase_*.json")) {
        Write-Host "    ✅ OK" -ForegroundColor Green
    } else {
        Write-Host "    ❌ FAIL" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "✅ K4 terminé (30 seeds)" -ForegroundColor Green
Write-Host ""

# ===== K8 =====
Write-Host "📊 K=8 (30 seeds séquentiels)..." -ForegroundColor Cyan
$LABELS_K8 = Join-Path $repo "outputs\fourier\labels_frozen\BTC_FUSED_2h\K8_1d_stable.csv"
$OUT_K8 = "E:\ichimoku_runs\wfa_phase_k8_1d_tp"

if (-not (Test-Path $LABELS_K8)) {
    Write-Host "❌ Labels K8 introuvables: $LABELS_K8" -ForegroundColor Red
    exit 1
}

Get-ChildItem $OUT_K8 -Recurse -File -Filter ".lock" -ErrorAction SilentlyContinue | Remove-Item -Force
$i = 0
foreach ($s in $seeds) {
    $i++
    $out = Join-Path $OUT_K8 ("seed_{0}" -f $s)
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $log = Join-Path $out "run.log"
    $err = Join-Path $out "run.err"
    Remove-Item $log,$err -ErrorAction SilentlyContinue
    Write-Host "  [$i/30] Seed $s..." -ForegroundColor Yellow
    & $PYTHON $SCRIPT --labels-csv $LABELS_K8 --granularity annual --trials $TRIALS --seed $s --jobs 1 --loss-mult 3.0 --use-fused --out-dir $out *> $log 2> $err
    if (Test-Path (Join-Path $out "WFA_phase_*.json")) {
        Write-Host "    ✅ OK" -ForegroundColor Green
    } else {
        Write-Host "    ❌ FAIL" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "✅✅ K4 & K8 terminés (60 seeds total)" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Analyse après complétion:" -ForegroundColor Cyan
Write-Host "   .venv\Scripts\python.exe scripts\analyze_k4_1d_stable.py" -ForegroundColor White
Write-Host "   .venv\Scripts\python.exe scripts\analyze_k8_1d_stable.py" -ForegroundColor White

