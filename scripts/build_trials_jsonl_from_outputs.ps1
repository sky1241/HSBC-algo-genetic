Param(
  [string]$OutputsRoot = 'outputs'
)

$ErrorActionPreference = 'Stop'

function Get-KTagFromFileName {
  param([string]$FileName, [string]$Fallback)
  $m = [regex]::Match($FileName, 'WFA_phase_(K\d+)_')
  if ($m.Success) { return $m.Groups[1].Value.ToUpper() }
  if ($Fallback) { return $Fallback.ToUpper() }
  return 'K'
}

function Add-TrialsFromWFAJson {
  param(
    [string]$JsonPath,
    [string]$OutJsonl
  )
  try {
    $j = Get-Content $JsonPath -Raw | ConvertFrom-Json
    foreach ($fr in $j.folds) {
      $eq  = [double]$fr.metrics.equity_mult
      $dd  = [double]$fr.metrics.max_drawdown
      $ert = $eq - 1.0
      foreach ($kv in $fr.params_by_state.PSObject.Properties) {
        $pm  = $kv.Value
        $rec = @{
          params        = @{ tenkan=[int]$pm.tenkan; kijun=[int]$pm.kijun; senkou_b=[int]$pm.senkou_b; shift=[int]$pm.shift; atr_mult=[double]$pm.atr_mult }
          metrics_train = @{ eq_ret=[double]$ert; max_drawdown=[double]$dd }
          run_context   = @{ phase_label=$kv.Name; fold=$fr.period }
        } | ConvertTo-Json -Compress -Depth 6
        Add-Content -Path $OutJsonl -Value $rec
      }
    }
  } catch {
    Write-Warning "Failed to parse: $JsonPath"
  }
}

if (-not (Test-Path $OutputsRoot)) {
  throw "Outputs root not found: $OutputsRoot"
}

$files = Get-ChildItem -Path $OutputsRoot -Recurse -File -Filter 'WFA_phase_*.json' -ErrorAction SilentlyContinue
if (-not $files) {
  Write-Warning 'No WFA_phase_*.json found. Nothing to aggregate.'
  exit 0
}

foreach ($f in $files) {
  $parent = Split-Path -Leaf $f.DirectoryName
  $fallbackK = if ($parent -match 'wfa_phase_k\d+') { $parent.ToUpper().Replace('WFA_PHASE_', '') } else { '' }
  $K = Get-KTagFromFileName -FileName $f.Name -Fallback $fallbackK
  $outDir = Join-Path $OutputsRoot ("trial_logs/phase/$K")
  New-Item -ItemType Directory -Path $outDir -Force | Out-Null
  $jsonl = Join-Path $outDir 'trials_from_wfa.jsonl'
  Add-TrialsFromWFAJson -JsonPath $f.FullName -OutJsonl $jsonl
}

Get-ChildItem -Path (Join-Path $OutputsRoot 'trial_logs/phase') -Recurse -Filter 'trials_from_wfa.jsonl' -ErrorAction SilentlyContinue |
  Select-Object FullName, Length, LastWriteTime |
  Sort-Object FullName


