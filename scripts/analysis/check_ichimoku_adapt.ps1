using namespace System.Globalization
$ErrorActionPreference = "SilentlyContinue"

function Parse-Double {
  param([Parameter(Mandatory=$false)]$s)
  if($null -eq $s){ return $null }
  $t = (""+$s).Trim().Replace('%','').Replace(',','.')
  $out = 0.0
  if([double]::TryParse($t, [NumberStyles]::Float, [CultureInfo]::InvariantCulture, [ref]$out)){
    return $out
  } else { return $null }
}

param([string]$K='K3',[double]$mddMax=50,[int]$trMin=280)
$target = ([math]::Pow(1.10,12) - 1.0) * 100.0
$p = Join-Path (Join-Path 'outputs\trial_logs\phase' $K) 'trials_from_wfa.jsonl'
if(-not (Test-Path $p)){ "JSONL missing: $p"; exit 0 }
$pts = @()
foreach($line in (Get-Content $p -ErrorAction SilentlyContinue)){
  if([string]::IsNullOrWhiteSpace($line)){ continue }
  try{ $rec = $line | ConvertFrom-Json } catch { continue }
  if(-not $rec){ continue }
  $mt = $null
  if($rec.PSObject.Properties.Name -contains 'metrics_test'){ $mt = $rec.metrics_test }
  elseif($rec.PSObject.Properties.Name -contains 'metrics_train'){ $mt = $rec.metrics_train }
  if(-not $mt){ continue }
  $er = $null
  if($mt.PSObject.Properties.Name -contains 'eq_ret'){
    $er = [double]$mt.eq_ret * 100.0
  } elseif($mt.PSObject.Properties.Name -contains 'equity_mult'){
    $em = [double]$mt.equity_mult
    $er = ($em - 1.0) * 100.0
  }
  if($null -eq $er){ continue }
  $mdd = $null
  if($mt.PSObject.Properties.Name -contains 'max_drawdown'){
    $mdd = [double]$mt.max_drawdown
    if($mdd -le 1){ $mdd = $mdd * 100 }
  }
  if($null -eq $mdd){ continue }
  $tr = $null
  if($mt.PSObject.Properties.Name -contains 'trades'){
    try{ $tr = [int]$mt.trades } catch { $tr = $null }
  }
  if($null -eq $tr){ continue }
  if($mdd -le $mddMax -and $tr -ge $trMin){
    $pts += [pscustomobject]@{ er=$er; mdd=$mdd; trades=$tr; trial=([int]$rec.trial_number) }
  }
}

$n = ($pts|Measure-Object).Count
if($n -eq 0){ "No valid points (K=$K, MDD<=$mddMax, trades>=$trMin)."; exit 0 }
$succ = $pts | Where-Object { $_.er -ge $target }
$ns = ($succ|Measure-Object).Count
$rate = [math]::Round(100.0 * $ns / $n, 2)
$ers = $pts | Select-Object -Expand er | Sort-Object
$med = [math]::Round($ers[[math]::Floor(($n-1)/2)],2)
$p90 = [math]::Round($ers[[math]::Floor(0.9*($n-1))],2)
$ts = Get-Date -Format yyyyMMdd_HHmmss
$out = "docs/ICHIMOKU_ADAPT_${K}_${ts}.md"
$lines = @()
$lines += "### Ichimoku adaptatif ($K)"
$lines += "- File: $p"
$lines += "- Filters: MDD<=$mddMax, trades>=$trMin, prefer metrics_test"
$lines += ("- Target 10%/mois => annual ~ {0}%" -f [math]::Round($target,2))
$lines += ("- Valid points: {0}" -f $n)
$lines += ("- Success (>= target): {0} ({1}%)" -f $ns,$rate)
$lines += ("- eq_ret% median={0} | p90={1}" -f $med,$p90)
$verdict = 'NO'
if($ns -gt 0){ $verdict = 'YES' }
$lines += ("- Verdict: {0}" -f $verdict)
Set-Content -Path $out -Value ($lines -join "`n") -Encoding utf8
"VERDICT: $verdict | PASS $ns/$n ($rate%) | REPORT: $out"
