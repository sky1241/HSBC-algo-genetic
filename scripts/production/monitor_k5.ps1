# monitor_k5.ps1 - Monitoring rapide des 30 seeds K5
# Usage: .\scripts\monitor_k5.ps1

$OUT_BASE = "outputs\wfa_phase_k5"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MONITORING K5 1D STABLE - 30 SEEDS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Compter les seeds
$total = 0
$complete = 0
$running = 0
$pending = 0

$progressFiles = Get-ChildItem "$OUT_BASE\seed_*\PROGRESS.json" -ErrorAction SilentlyContinue

foreach ($f in $progressFiles) {
    $total++
    $j = Get-Content $f.FullName | ConvertFrom-Json
    if ($j.phase -eq "complete") {
        $complete++
    } elseif ($j.percent -gt 0) {
        $running++
    } else {
        $pending++
    }
}

Write-Host "Etat global:" -ForegroundColor Yellow
Write-Host "  Total:    $total / 30 seeds"
Write-Host "  Termines: $complete"
Write-Host "  En cours: $running"
Write-Host "  Pending:  $pending"
Write-Host ""

# Processus Python
$pyProcs = Get-Process python* -ErrorAction SilentlyContinue
Write-Host "Processus Python: $($pyProcs.Count)" -ForegroundColor Yellow
Write-Host ""

# Top 10 progression
Write-Host "Progression par seed:" -ForegroundColor Yellow
$progressFiles | ForEach-Object {
    $j = Get-Content $_.FullName | ConvertFrom-Json
    $seed = $_.Directory.Name
    $pct = [math]::Round($j.percent, 1)
    $phase = $j.phase
    $bar = "=" * [math]::Floor($pct / 5)
    Write-Host ("  {0,-10} [{1,-20}] {2,5}% ({3})" -f $seed, $bar, $pct, $phase)
} | Sort-Object

Write-Host ""
Write-Host "Rafraichir: .\scripts\monitor_k5.ps1" -ForegroundColor Gray
Write-Host ""
