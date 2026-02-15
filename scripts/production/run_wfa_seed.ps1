param(
  [Parameter(Mandatory=$true)]
  [int]$Seed,
  [ValidateSet('annual','monthly')]
  [string]$Granularity = 'annual',
  [int]$Trials = 300,
  [int]$Jobs = 1,
  [double]$MddMax = 0.20
)

$ErrorActionPreference = 'Stop'
try {
  $repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
  Set-Location $repo
  $env:USE_FUSED_H2 = '1'

  $outdir = Join-Path $repo ("outputs\scheduler_{0}_btc\seed_{1}" -f $Granularity, $Seed)
  New-Item -ItemType Directory -Force -Path $outdir | Out-Null
  $log = Join-Path $outdir ("RUN_{0}.txt" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

  Write-Host ("Running WFA {0} seed {1} → {2}" -f $Granularity, $Seed, $outdir)
  py -3 scripts\run_scheduler_wfa.py --use-fused --granularity $Granularity --trials $Trials --jobs $Jobs --seed $Seed --atr-sweep --atr-sweep-span 1.0 --atr-sweep-step 0.2 --mdd-max $MddMax --out-dir $outdir 2>&1 | Tee-Object -FilePath $log
}
catch {
  Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
  Write-Host ("Log: {0}" -f $log)
  Read-Host "Terminé. Appuyez Entrée pour fermer"
}
