Param(
  [int]$IntervalSec = 120,
  [string[]]$Ks = @('K2','K3','K5','K8')
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$WS   = Split-Path -Parent $here

function Combine-WFAJsonToJsonl {
  param(
    [string]$K
  )
  $outRoot = Join-Path $WS 'outputs\trial_logs\phase'
  $outK    = Join-Path $outRoot $K
  if (-not (Test-Path $outK)) { New-Item -ItemType Directory -Path $outK -Force | Out-Null }
  $logPath = Join-Path $outK 'trials_from_wfa.jsonl'
  # Rebuild fresh each pass (idempotent)
  Remove-Item $logPath -Force -ErrorAction SilentlyContinue

  $src = Join-Path $WS ("outputs\wfa_phase_" + ($K.ToLower()) )
  if (-not (Test-Path $src)) { return $null }
  foreach ($sd in (Get-ChildItem $src -Directory -Filter 'seed_*' -ErrorAction SilentlyContinue)) {
    foreach ($fp in (Get-ChildItem $sd.FullName -File -Filter 'WFA_phase_*.json' -ErrorAction SilentlyContinue)) {
      try {
        $json = Get-Content $fp.FullName -Raw | ConvertFrom-Json
        foreach ($fr in $json.folds) {
          $eq = [double]$fr.metrics.equity_mult
          $dd = [double]$fr.metrics.max_drawdown
          $ert = $eq - 1.0
          foreach ($kv in $fr.params_by_state.PSObject.Properties) {
            $pm = $kv.Value
            $rec = [pscustomobject]@{
              params = @{ tenkan=[int]$pm.tenkan; kijun=[int]$pm.kijun; senkou_b=[int]$pm.senkou_b; shift=[int]$pm.shift; atr_mult=[double]$pm.atr_mult }
              metrics_train = @{ eq_ret=[double]$ert; max_drawdown=[double]$dd }
            } | ConvertTo-Json -Compress -Depth 6
            Add-Content -Path $logPath -Value $rec
          }
        }
      } catch { }
    }
  }
  return $logPath
}

function Get-PythonExe {
  $cands = @(
    (Join-Path $WS '.venv\Scripts\python.exe'),
    (Join-Path $WS 'venv\Scripts\python.exe')
  ) + @(
    (where.exe python 2>$null | Select-Object -First 1)
  )
  foreach ($c in $cands) { if ($c -and (Test-Path $c)) { return $c } }
  throw 'Python not found.'
}

$py = Get-PythonExe
$plotScript = Join-Path $WS 'scripts\plot_top_trials.py'

while ($true) {
  foreach ($K in $Ks) {
    try {
      $jsonl = Combine-WFAJsonToJsonl -K $K
      if ($jsonl -and (Test-Path $jsonl)) {
        $outPng = Join-Path $WS ("docs\IMAGES\top_trials_" + $K + ".png")
        & $py $plotScript --input $jsonl --out $outPng --top 0.10 --mdd-max 0.50 | Out-Null
      }
    } catch { }
  }
  Start-Sleep -Seconds $IntervalSec
}


