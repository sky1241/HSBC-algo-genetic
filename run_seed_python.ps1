Param(
    [Parameter(Mandatory=$true)][int]$Seed,
    [string]$Label = "",
    [int]$Trials = 1000,
    [string]$ProfileName = "pipeline_web6",
    [switch]$DisablePlateau,
    [string]$OutDir = "outputs"
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

$py = (Test-Path .\.venv\Scripts\python.exe) ? (Resolve-Path .\.venv\Scripts\python.exe).Path : 'python'
$baseline = (Get-ChildItem .\$OutDir -Filter 'best_params_per_symbol_pipeline_web6_*.json' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Desc | Select-Object -ExpandProperty FullName -First 1)
if (-not $baseline) { $baseline = (Resolve-Path .\baselines.json).Path }

$ts = Get-Date -Format yyyyMMdd_HHmmss
$labelSafe = if ([string]::IsNullOrWhiteSpace($Label)) { "seed$Seed" } else { $Label }
$log = Join-Path $OutDir ("optuna_"+$ProfileName+"_"+$ts+".out.log")
$status = Join-Path $OutDir ("TRIALS_"+$labelSafe+"_"+$ts+".txt")
"["+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')+"] pruned=0 finished=0 lastIndex=-1" | Set-Content -Path $status -Encoding UTF8

if ($DisablePlateau.IsPresent) { $env:PLATEAU_DISABLED = '1' } else { Remove-Item Env:PLATEAU_DISABLED -ErrorAction SilentlyContinue }
$env:PYTHONIOENCODING = 'utf-8'

$cmd = "& '"+$py+"' -u .\ichimoku_pipeline_web_v4_8_fixed.py "+
       $ProfileName+" --trials "+$Trials+" --seed "+$Seed+
       " --baseline-json '"+$baseline+"' --out-dir "+$OutDir+
       " 2>&1 | Tee-Object -FilePath '"+$log+"'"

Start-Process powershell.exe -WorkingDirectory $root -WindowStyle Normal -ArgumentList @('-NoExit','-ExecutionPolicy','Bypass','-NoProfile','-Command', $cmd) | Out-Null

$watch = {
    param($log,$statusPath)
    while ($true) {
        try { $pruned = (Select-String -Path $log -Pattern '\bTrial \d+ pruned\b' -AllMatches -Encoding UTF8 -ErrorAction SilentlyContinue).Matches.Count } catch { $pruned = 0 }
        try { $finished = (Select-String -Path $log -Pattern '\bTrial \d+ finished\b' -AllMatches -Encoding UTF8 -ErrorAction SilentlyContinue).Matches.Count } catch { $finished = 0 }
        try { $lastIndex = (Select-String -Path $log -Pattern 'Trial (\d+)' -AllMatches -Encoding UTF8 -ErrorAction SilentlyContinue | ForEach-Object { $_.Matches } | ForEach-Object { $_.Groups[1].Value } | ForEach-Object { [int]$_ } | Sort-Object -Desc | Select-Object -First 1) } catch { $lastIndex = $null }
        $tsNow = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        "[${tsNow}] pruned=${pruned} finished=${finished} lastIndex=${lastIndex}" | Set-Content -Path $statusPath -Encoding UTF8
        Start-Sleep -Seconds 5
    }
}

Start-Job -Name ('watch_trials_'+$labelSafe+'_'+$ts) -ScriptBlock $watch -ArgumentList $log,$status | Out-Null
Start-Process notepad $status
Write-Output ("Started run: profile="+$ProfileName+" seed="+$Seed+" trials="+$Trials+" label="+$labelSafe)
Write-Output ("Baseline: " + $baseline)
if (Test-Path $log) { Write-Output ("Log: " + (Resolve-Path $log).Path) } else { Write-Output ("Log: " + $log) }
Write-Output ("Status: " + (Resolve-Path $status).Path)


