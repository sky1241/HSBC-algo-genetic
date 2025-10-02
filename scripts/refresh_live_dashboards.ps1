Param(
  [int]$IntervalSec = 60,
  [string]$KArg = 'ALL',
  [int]$Bins = 20,
  [switch]$Open
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$WS   = Split-Path -Parent $here

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

$py   = Get-PythonExe
$heat = Join-Path $WS 'scripts\plot_heatmaps_plotly_live.py'
$scat = Join-Path $WS 'scripts\plot_trials_3d_live.py'

$tag  = ($KArg.ToUpper() -replace ',', '_')
$hp   = Join-Path $WS ("docs\IMAGES\heatmaps_live_{0}.html" -f $tag)
$tp   = Join-Path $WS ("docs\IMAGES\top_trials_live_{0}.html" -f $tag)

while ($true) {
  & $py $heat --k $KArg --bins $Bins | Out-Null
  & $py $scat --k $KArg | Out-Null
  if ($Open) {
    if (Test-Path $hp) { Start-Process $hp }
    if (Test-Path $tp) { Start-Process $tp }
    $Open = $false
  }
  Start-Sleep -Seconds $IntervalSec
}


