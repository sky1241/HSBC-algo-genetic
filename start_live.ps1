$ErrorActionPreference = 'SilentlyContinue'
param(
  [switch]$NewRun,
  [switch]$NoTail
)

# 1) Ouvrir la page LIVE en premier
if (Test-Path -Path (Join-Path $PSScriptRoot 'outputs\LIVE_REPORT.html')) {
  Start-Process (Join-Path $PSScriptRoot 'outputs\LIVE_REPORT.html')
}
if (Test-Path env:ICHIMOKU_LIVE_DIR) {
  $alt = Join-Path $env:ICHIMOKU_LIVE_DIR 'LIVE_REPORT.html'
  if (Test-Path $alt) { Start-Process $alt }
}

# 2) Trouver le dernier log, sinon démarrer une nouvelle run en arrière-plan
$logs = Get-ChildItem -Path (Join-Path $PSScriptRoot 'outputs') -Filter 'optuna_pipeline_web6_*.log' | Sort-Object LastWriteTime -Descending
$log = $logs | Select-Object -First 1

if ($NewRun -or $null -eq $log) {
  $ts = Get-Date -Format yyyyMMdd_HHmmss
  $outDir = Join-Path $PSScriptRoot 'outputs'
  if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
  $logPath = Join-Path $outDir ("optuna_pipeline_web6_" + $ts + ".log")

  if (Test-Path (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')) {
    $py = (Resolve-Path (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')).Path
  } else {
    $py = 'python'
  }

  Start-Job -Name ("optuna_web6_" + $ts) -ScriptBlock {
    & $using:py -u (Join-Path $using:PSScriptRoot 'ichimoku_pipeline_web_v4_8_fixed.py') 'pipeline_web6' '--trials' '5000' '--seed' '42' '--baseline-json' (Join-Path $using:PSScriptRoot 'baselines.json') '--out-dir' (Join-Path $using:PSScriptRoot 'outputs') *> $using:logPath
  } | Out-Null

  Start-Sleep -Seconds 2
  $log = Get-Item $logPath
}

# Démarrer en parallèle une tâche qui force une trace de "dernière maj" côté HTML
Start-Job -Name live_touch -ScriptBlock {
  Param($root)
  $out = Join-Path $root 'outputs'
  $html = Join-Path $out 'LIVE_REPORT.html'
  while ($true) {
    try {
      if (Test-Path $html) {
        $ts = (Get-Date).ToString('s')
        $content = Get-Content $html -Raw
        if ($content -notmatch 'Derniere maj') {
          $content = "<!doctype html><html><head><meta charset=utf-8><meta http-equiv=refresh content=5><title>LIVE REPORT</title></head><body><h1>LIVE REPORT</h1><p>Optimisation en cours… Cette page se met à jour automatiquement.</p><p style='color:#555'>Derniere maj: $ts</p></body></html>"
        } else {
          $content = [regex]::Replace($content, 'Derniere maj: .*?<', "Derniere maj: $ts<")
        }
        $tmp = $html + '.tmp'
        Set-Content -Path $tmp -Value $content -Encoding UTF8
        Move-Item -Force -Path $tmp -Destination $html
      }
    } catch {}
    Start-Sleep -Seconds 10
  }
} -ArgumentList $PSScriptRoot | Out-Null

Write-Output ("Suivi en direct: " + $log.FullName)
if (-not $NoTail) {
  Get-Content $log.FullName -Wait
}

 


