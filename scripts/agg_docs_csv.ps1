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

$csvs = Get-ChildItem 'docs' -Recurse -File -Filter '*.csv' -ErrorAction SilentlyContinue
$rows = @()
foreach($f in $csvs){
  try{ $data = Import-Csv -Path $f.FullName -ErrorAction Stop } catch { continue }
  foreach($r in $data){
    $K = if($r.PSObject.Properties.Name -contains 'K'){ $r.K } elseif($r.PSObject.Properties.Name -contains 'k'){ $r.k } else { 'UNK' }
    $eq = $null
    if($r.PSObject.Properties.Name -contains 'eq_ret'){
      $eq = Parse-Double $r.eq_ret
    } elseif($r.PSObject.Properties.Name -contains 'equity_mult'){
      $em = Parse-Double $r.equity_mult
      if($em -ne $null){ $eq = ($em - 1.0) * 100.0 }
    }
    if($eq -ne $null -and [math]::Abs($eq) -lt 3){ $eq = $eq * 100 }
    $mdd = $null
    if($r.PSObject.Properties.Name -contains 'mdd'){
      $mdd = Parse-Double $r.mdd
    } elseif($r.PSObject.Properties.Name -contains 'max_drawdown'){
      $mdd = Parse-Double $r.max_drawdown
    }
    if($mdd -ne $null -and $mdd -le 1){ $mdd = $mdd * 100 }
    $tr = $null
    if($r.PSObject.Properties.Name -contains 'trades'){
      try{ $tr = [int]$r.trades } catch { $tr = $null }
    }
    if($eq -ne $null -and $mdd -ne $null -and $tr -ne $null){
      $rows += [pscustomobject]@{ file=$f.Name; K=$K; eq_ret=$eq; mdd=$mdd; trades=$tr }
    }
  }
}

$filtered = $rows | Where-Object { $_.mdd -le 50 -and $_.trades -ge 280 }
$groups = $filtered | Group-Object K
$agg = @()
foreach($g in $groups){
  $items = $g.Group
  $n = ($items|Measure-Object).Count
  $eqs = $items | Select-Object -ExpandProperty eq_ret | Sort-Object
  $mdds = $items | Select-Object -ExpandProperty mdd | Sort-Object
  $trAvg = ($items | Select-Object -ExpandProperty trades | Measure-Object -Average).Average
  if($n -gt 0){
    $medEq = [math]::Round($eqs[[math]::Floor(($n-1)/2)],2)
    $p90Eq = [math]::Round($eqs[[math]::Floor(0.9*($n-1))],2)
    $medMdd = [math]::Round($mdds[[math]::Floor(($n-1)/2)],2)
  } else { $medEq=$null; $p90Eq=$null; $medMdd=$null }
  $agg += [pscustomobject]@{ K=$g.Name; n=$n; eq_med=$medEq; eq_p90=$p90Eq; mdd_med=$medMdd; trades_avg=[math]::Round($trAvg,2) }
}

$verdict = 'inconclusif'
if(($agg | Where-Object { $_.n -gt 0 -and $_.eq_p90 -ge 500 }).Count -gt 0){
  $verdict = 'signal_haute_ATR_OUI'
} elseif(($agg | Where-Object { $_.n -gt 0 -and $_.eq_med -ge 200 }).Count -gt 0){
  $verdict = 'signal_moderé'
} else {
  $verdict = 'signal_non'
}

$ts2 = Get-Date -Format yyyyMMdd_HHmmss
$out = "docs/CSV_AGG_REPORT_${ts2}.md"
$lines = @()
$lines += "### Synthèse CSV ($ts2)"
$lines += ""
$lines += "- Filtres: MDD<=50%, trades>=280"
$lines += ("- Fichiers scannés: {0}" -f $csvs.Count)
$lines += ("- Verdict: {0}" -f $verdict)
$lines += ""
foreach($a in ($agg | Sort-Object K)){
  $lines += ("- K={0}: n={1}, eq_med={2}%, eq_p90={3}%, mdd_med={4}%, trades_avg={5}" -f $a.K,$a.n,$a.eq_med,$a.eq_p90,$a.mdd_med,$a.trades_avg)
}
Set-Content -Path $out -Value ($lines -join "`n") -Encoding utf8
"REPORT: $out | VERDICT: $verdict"
