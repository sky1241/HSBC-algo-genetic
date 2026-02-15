# ============================================================================
# Lancement 30 seeds K5 avec labels 1D stables (phases daily + trading H2)
# ============================================================================
# Utilise les mêmes 30 seeds que K3 pour comparaison directe

$repo = "C:\Users\ludov\OneDrive\Bureau\teste algo\ichimoku_pipeline_web_v4_8"
Set-Location -LiteralPath $repo

# Récupérer les 30 seeds du batch K3/K5/K8
$seedsDoc = Get-ChildItem (Join-Path $repo "docs\seeds\SEEDS_BATCH_K3K4K5K8_*.json") | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1 -ExpandProperty FullName

if (-not (Test-Path $seedsDoc)) {
    Write-Host "❌ Fichier seeds introuvable: $seedsDoc" -ForegroundColor Red
    exit 1
}

$seeds = (Get-Content $seedsDoc -Raw | ConvertFrom-Json).seeds

# Configuration
$LABELS_CSV = Join-Path $repo "outputs\fourier\labels_frozen\BTC_FUSED_2h\K5_1d_stable.csv"
$OUT_ROOT = "E:\ichimoku_runs\wfa_phase_k5_1d_stable"
$TRIALS = 300

# Vérifications
if (-not (Test-Path $LABELS_CSV)) {
    Write-Host "❌ Labels K5 1D stable introuvables: $LABELS_CSV" -ForegroundColor Red
    Write-Host "   Exécuter d'abord: python scripts\downsample_labels_2h_to_1d.py --k 5" -ForegroundColor Yellow
    exit 1
}

Write-Host "🚀 Lancement K5 1D Stable: 30 seeds séquentiels" -ForegroundColor Green
Write-Host "   Seeds: $($seeds.Count)"
Write-Host "   Labels: $LABELS_CSV"
Write-Host "   Output: $OUT_ROOT"
Write-Host ""

# Nettoyage locks
Get-ChildItem $OUT_ROOT -Recurse -File -Filter ".lock" -ErrorAction SilentlyContinue | Remove-Item -Force

# Lancement séquentiel
$i = 0
foreach ($s in $seeds) {
    $i++
    $out = Join-Path $OUT_ROOT ("seed_{0}" -f $s)
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Write-Host "[$i/30] Seed $s..." -ForegroundColor Cyan
    .venv\Scripts\python.exe scripts\run_scheduler_wfa_phase.py `
        --labels-csv $LABELS_CSV `
        --granularity annual `
        --trials $TRIALS `
        --seed $s `
        --jobs 1 `
        --loss-mult 3.0 `
        --use-fused `
        --out-dir $out | Out-Null
    if (Test-Path "$out\WFA_phase_*.json") {
        Write-Host "  ✅ OK" -ForegroundColor Green
    } else {
        Write-Host "  ❌ FAIL" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "✅ Terminé: 30 seeds K5 1D stable" -ForegroundColor Green

