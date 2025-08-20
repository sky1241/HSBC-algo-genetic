Param(
    [string]$ProfileName = "pipeline_web6",
    [string]$Label = "",
    [int]$Trials = 5000,
    [int]$Seed = 42,
    [string]$BaselineJson = ".\baselines.json",
    [string]$OutDir = "outputs",
    [bool]$OpenLogs = $false,
    [switch]$OpenStatus,
    [switch]$OpenReport,
    [bool]$PreArchive = $true,
    [switch]$Foreground
)

$ErrorActionPreference = 'Stop'

# Go to repo root
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Ensure venv
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Host "Creating virtual env..."
    py -3 -m venv .venv
}
$py = '.\\.venv\\Scripts\\python.exe'

# Force UTF-8 for all Python child processes (prevents UnicodeEncodeError on Windows consoles)
$env:PYTHONIOENCODING = 'utf-8'

# Install deps
if (Test-Path .\requirements.txt) {
    & $py -m pip install --upgrade pip | Out-Null
    & $py -m pip install -r .\requirements.txt | Out-Null
}

# Ensure outputs folder
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

$ts = Get-Date -Format yyyyMMdd_HHmmss
$logBase = Join-Path $OutDir ("optuna_" + $ProfileName + "_" + $ts)

# Prepare live status file (pruned trials)
if ($OpenStatus.IsPresent) {
    # Close previous Notepads on STATUS files and remove old status files
    try {
        $np = Get-CimInstance Win32_Process -Filter "Name='notepad.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine -match "\\outputs\\STATUS.*\.txt" }
        if ($np) { $np | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch {} } }
    } catch {}
    try {
        Get-ChildItem (Join-Path $OutDir 'STATUS_*.txt') -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    } catch {}
}
$statusPath = Join-Path $OutDir ("STATUS_PRUNED_" + $ts + ".txt")
"Pruned trials: 0" | Set-Content -Path $statusPath -Encoding UTF8
if ($OpenStatus.IsPresent) { try { Start-Process notepad $statusPath } catch {}; $OpenLogs = $false }

Write-Host "Running pipeline: profile=$ProfileName trials=$Trials seed=$Seed"

# Optionally pre-archive current outputs for cleanliness
if ($PreArchive -and (Test-Path $OutDir)) {
    try {
        $prevZip = Join-Path ".\archives" ("prev_before_" + $ts + ".zip")
        if (-not (Test-Path ".\archives")) { New-Item -ItemType Directory -Path ".\archives" | Out-Null }
        if (Test-Path $prevZip) { Remove-Item $prevZip -Force }
        Compress-Archive -Path (Join-Path $OutDir '*') -DestinationPath $prevZip -CompressionLevel Optimal -Force
        Write-Host "Archived previous outputs to: $prevZip"
    } catch { Write-Warning "Pre-archive failed: $($_.Exception.Message)" }
}

# Run analysis (foreground or new visible console)
if ($Foreground.IsPresent) {
    Write-Host "Running in foreground. Output will stream here and also be saved to $logBase.out.log"
    # Start background monitor for pruned count
    $monitorJob = Start-Job -ScriptBlock {
        param($logPath, $statusPath)
        $lastCount = -1
        while ($true) {
            try { $count = (Select-String -Path $logPath -Pattern 'pruned' -SimpleMatch -ErrorAction SilentlyContinue).Count } catch { $count = 0 }
            if ($count -ne $lastCount) { "Pruned trials: $count" | Set-Content -Path $statusPath -Encoding UTF8; $lastCount = $count }
            Start-Sleep -Seconds 2
        }
    } -ArgumentList ("$logBase.out.log"), $statusPath
    try {
        # Build explicit arg list to preserve spaces in paths
        $argsPy = @(
            '.\\ichimoku_pipeline_web_v4_8_fixed.py',
            $ProfileName,
            '--trials', $Trials,
            '--seed', $Seed,
            '--baseline-json', $BaselineJson,
            '--out-dir', $OutDir
        )
        # Merge stderr into stdout, tee to file and console
        & $py @argsPy 2>&1 | Tee-Object -FilePath "$logBase.out.log"
    } catch {
        Write-Warning "Pipeline failed: $($_.Exception.Message)" | Tee-Object -FilePath "$logBase.err.log"
    }
    if ($monitorJob) { try { Stop-Job $monitorJob -Force } catch {}; try { Remove-Job $monitorJob -Force } catch {} }
} else {
    # New PowerShell window, visible, with redirected logs
    try {
        # Resolve baseline path if possible (safeguard for spaces in path)
        try { $BaselineJson = (Resolve-Path $BaselineJson -ErrorAction Stop).Path } catch {}
        # Build a single, properly quoted argument string for Start-Process
        $argList = ".\\ichimoku_pipeline_web_v4_8_fixed.py `"$ProfileName`" --trials $Trials --seed $Seed --baseline-json `"$BaselineJson`" --out-dir `"$OutDir`""
        $proc = Start-Process -FilePath $py -ArgumentList $argList -RedirectStandardOutput ("$logBase.out.log") -RedirectStandardError ("$logBase.err.log") -PassThru -WindowStyle Normal
        Write-Host "Started pipeline PID=$($proc.Id). Logs: $logBase.out.log | $logBase.err.log"
        # Start monitor tied to PID
        $monitorJob = Start-Job -ScriptBlock {
            param($logPath, $statusPath, $pidWatch)
            $lastCount = -1
            while ($true) {
                $running = $false
                try { $p=[System.Diagnostics.Process]::GetProcessById($pidWatch); if ($p) { $running = -not $p.HasExited } } catch { $running = $false }
                try { $count = (Select-String -Path $logPath -Pattern 'pruned' -SimpleMatch -ErrorAction SilentlyContinue).Count } catch { $count = 0 }
                if ($count -ne $lastCount) { "Pruned trials: $count" | Set-Content -Path $statusPath -Encoding UTF8; $lastCount = $count }
                if (-not $running) { break }
                Start-Sleep -Seconds 2
            }
        } -ArgumentList ("$logBase.out.log"), $statusPath, $proc.Id
    } catch {
        Write-Warning "Failed to start pipeline process: $($_.Exception.Message)"
    }
    # Open logs in Notepad for the CURRENT run
    if ($OpenLogs) {
        try { Start-Process notepad ("$logBase.out.log") } catch {}
        try { Start-Process notepad ("$logBase.err.log") } catch {}
    }
    # Wait for completion
    if ($proc) { Wait-Process -Id $proc.Id }
    if ($monitorJob) { try { Stop-Job $monitorJob -Force } catch {}; try { Remove-Job $monitorJob -Force } catch {} }
}

# Post-processing: graphs and master report
try { & $py .\outputs\build_graphs_from_snapshots.py | Out-Null } catch { Write-Warning "Graph build failed: $($_.Exception.Message)" }
# Pass run label to master report for summary CSV
$env:RUN_LABEL = if ([string]::IsNullOrWhiteSpace($Label)) { $ProfileName } else { $Label }
try { & $py .\outputs\generate_master_report.py $ProfileName | Out-Null } catch { Write-Warning "Master report generation failed: $($_.Exception.Message)" }

# Copy to Temp for easy viewing
$tempDir = "C:\\Temp\\ichimoku_master_report"
if (-not (Test-Path $tempDir)) { New-Item -ItemType Directory -Path $tempDir | Out-Null }
Copy-Item .\outputs\* $tempDir -Recurse -Force

# Open report (optional, no auto-open of internet page)
if ($OpenReport.IsPresent) {
    if (Test-Path "$tempDir\\MASTER_REPORT.html") { Start-Process "$tempDir\\MASTER_REPORT.html" }
}

# Archive under archives/ with label
$archivesDir = ".\archives"
if (-not (Test-Path $archivesDir)) { New-Item -ItemType Directory -Path $archivesDir | Out-Null }
$labelSafe = if ([string]::IsNullOrWhiteSpace($Label)) { $ProfileName } else { $Label }
$zipPath = Join-Path $archivesDir ("$labelSafe" + "_" + $ts + "_report.zip")
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path .\outputs\* -DestinationPath $zipPath -CompressionLevel Optimal -Force

# Quick access copy of MASTER_REPORT.html
if (Test-Path .\outputs\MASTER_REPORT.html) {
    Copy-Item .\outputs\MASTER_REPORT.html (Join-Path $archivesDir ("$labelSafe" + "_MASTER_REPORT.html")) -Force
}

# Show results
Write-Host "\nSaved:"
Get-Item $zipPath | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize | Out-String -Width 200 | Write-Output
if (Test-Path (Join-Path $archivesDir ("$labelSafe" + "_MASTER_REPORT.html"))) {
    Get-Item (Join-Path $archivesDir ("$labelSafe" + "_MASTER_REPORT.html")) | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize | Out-String -Width 200 | Write-Output
}
Start-Process explorer.exe "/select, $((Resolve-Path $zipPath).Path)"

Write-Host "\nDone."


