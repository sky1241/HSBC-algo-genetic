# launch_k5_k8_parallel.ps1
# Lance K5 ET K8 en parallèle avec système de reprise après crash
# Usage: .\scripts\launch_k5_k8_parallel.ps1

$ErrorActionPreference = "Continue"
$ROOT = Split-Path -Parent $PSScriptRoot
if (-not $ROOT -or $ROOT -eq "") {
    $ROOT = Get-Location
}

# Configuration
$TRIALS = 300
$LABELS_DIR = Join-Path $ROOT "outputs\fourier\labels_frozen\BTC_FUSED_2h"
$MAX_PARALLEL = 6  # Total jobs parallèles (3 K5 + 3 K8)

# Seeds (mêmes pour K5 et K8)
$SEEDS = @(
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
    301, 302, 303, 304, 305, 306, 307, 308, 309, 310
)

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  LANCEMENT K5 + K8 EN PARALLELE" -ForegroundColor Cyan
Write-Host "  Avec système de reprise après crash" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Trials par seed:     $TRIALS"
Write-Host "  Nombre de seeds:     $($SEEDS.Count) x 2 (K5 et K8)"
Write-Host "  Jobs paralleles max: $MAX_PARALLEL"
Write-Host "  Labels K5:           $LABELS_DIR\K5.csv"
Write-Host "  Labels K8:           $LABELS_DIR\K8.csv"
Write-Host ""

# Vérifier labels K5
$K5_CSV = Join-Path $LABELS_DIR "K5.csv"
if (-not (Test-Path $K5_CSV)) {
    Write-Host "ERREUR: Labels K5 non trouves: $K5_CSV" -ForegroundColor Red
    Write-Host "Generation des labels K8..." -ForegroundColor Yellow
}

# Vérifier/Générer labels K8
$K8_CSV = Join-Path $LABELS_DIR "K8.csv"
if (-not (Test-Path $K8_CSV)) {
    Write-Host "Labels K8 non trouves, generation..." -ForegroundColor Yellow
    # Chercher dans les archives
    $K8_ZIP = Join-Path $ROOT "outputs\fourier\hmm\BTC_FUSED_2h\by_K\K8\PRED_K8.zip"
    if (Test-Path $K8_ZIP) {
        Expand-Archive -Path $K8_ZIP -DestinationPath (Split-Path $K8_ZIP) -Force
        $K8_SRC = Join-Path (Split-Path $K8_ZIP) "PRED_K8.csv"
        if (Test-Path $K8_SRC) {
            # Convertir au format labels
            $k8Data = Import-Csv $K8_SRC
            $k8Labels = $k8Data | Select-Object @{N='timestamp';E={$_.timestamp}}, @{N='label';E={$_.state}}
            $k8Labels | Export-Csv -Path $K8_CSV -NoTypeInformation
            Write-Host "  Labels K8 generes: $K8_CSV" -ForegroundColor Green
        }
    } else {
        Write-Host "ERREUR: Impossible de generer K8 - archive non trouvee" -ForegroundColor Red
        Write-Host "Continuons avec K5 seulement..." -ForegroundColor Yellow
    }
}

# Créer dossiers de sortie
$OUT_K5 = Join-Path $ROOT "outputs\wfa_phase_k5"
$OUT_K8 = Join-Path $ROOT "outputs\wfa_phase_k8"
New-Item -ItemType Directory -Force -Path $OUT_K5 | Out-Null
New-Item -ItemType Directory -Force -Path $OUT_K8 | Out-Null

# Fonction pour vérifier si un seed est déjà complété
function Test-SeedComplete {
    param($OutDir, $Seed)
    $resultsFile = Join-Path $OutDir "seed_$Seed\RESULTS_FINAL.json"
    if (Test-Path $resultsFile) {
        try {
            $data = Get-Content $resultsFile | ConvertFrom-Json
            return $data.status -eq "completed"
        } catch {
            return $false
        }
    }
    return $false
}

# Fonction pour lancer un job
function Start-OptimizationJob {
    param($K, $Seed, $LabelsCSV, $OutDir, $Trials, $RootDir)

    $seedOutDir = Join-Path $OutDir "seed_$Seed"
    New-Item -ItemType Directory -Force -Path $seedOutDir | Out-Null
    $logFile = Join-Path $seedOutDir "run.log"

    Start-Job -Name "K${K}_seed_$Seed" -ScriptBlock {
        param($root, $labels, $trials, $seed, $outDir, $logFile, $k)

        Set-Location $root
        $env:USE_FUSED_H2 = "1"
        $env:PYTHONIOENCODING = "utf-8"
        $env:CHECKPOINT_ENABLED = "1"

        $startTime = Get-Date
        Write-Output "=== K$k Seed $seed - Debut: $startTime ===" | Out-File $logFile

        # Lancer avec checkpoints
        & python scripts/run_scheduler_wfa_phase.py `
            --labels-csv $labels `
            --trials $trials `
            --seed $seed `
            --use-fused `
            --out-dir $outDir `
            --enable-checkpoints 2>&1 | Tee-Object -FilePath $logFile -Append

        $endTime = Get-Date
        $duration = $endTime - $startTime
        Write-Output "=== K$k Seed $seed - Fin: $endTime (Duree: $duration) ===" | Out-File $logFile -Append

        # Marquer comme complet
        $resultsFile = Join-Path $outDir "RESULTS_FINAL.json"
        @{
            status = "completed"
            seed = $seed
            k = $k
            duration_seconds = $duration.TotalSeconds
            completion_time = $endTime.ToString("o")
        } | ConvertTo-Json | Out-File $resultsFile

    } -ArgumentList $RootDir, $LabelsCSV, $Trials, $Seed, $seedOutDir, $logFile, $K
}

Write-Host "Verification des seeds deja completes..." -ForegroundColor Green

# Filtrer les seeds déjà complétés
$pendingK5 = @()
$pendingK8 = @()

foreach ($seed in $SEEDS) {
    if (-not (Test-SeedComplete $OUT_K5 $seed)) {
        $pendingK5 += $seed
    }
    if ((Test-Path $K8_CSV) -and -not (Test-SeedComplete $OUT_K8 $seed)) {
        $pendingK8 += $seed
    }
}

Write-Host ""
Write-Host "Seeds K5: $($SEEDS.Count - $pendingK5.Count) completes, $($pendingK5.Count) restants" -ForegroundColor Yellow
if (Test-Path $K8_CSV) {
    Write-Host "Seeds K8: $($SEEDS.Count - $pendingK8.Count) completes, $($pendingK8.Count) restants" -ForegroundColor Yellow
}
Write-Host ""

# Lancer les jobs en alternant K5 et K8
Write-Host "Lancement des jobs..." -ForegroundColor Green
Write-Host ""

$totalJobs = 0
$k5Index = 0
$k8Index = 0

while ($k5Index -lt $pendingK5.Count -or $k8Index -lt $pendingK8.Count) {
    # Attendre si trop de jobs en cours
    while ((Get-Job -State Running).Count -ge $MAX_PARALLEL) {
        # Afficher progression
        $running = Get-Job -State Running
        $completed = Get-Job -State Completed
        Write-Host "  [$($completed.Count) termines, $($running.Count) en cours] Attente..." -ForegroundColor Gray
        Start-Sleep -Seconds 60

        # Nettoyer les jobs terminés
        Get-Job -State Completed | ForEach-Object {
            Receive-Job $_ | Out-Null
            Remove-Job $_
        }
    }

    # Lancer un job K5 si disponible
    if ($k5Index -lt $pendingK5.Count) {
        $seed = $pendingK5[$k5Index]
        Write-Host "  [K5] Seed $seed -> $OUT_K5\seed_$seed" -ForegroundColor Cyan
        Start-OptimizationJob -K 5 -Seed $seed -LabelsCSV $K5_CSV -OutDir $OUT_K5 -Trials $TRIALS -RootDir $ROOT | Out-Null
        $k5Index++
        $totalJobs++
        Start-Sleep -Milliseconds 500
    }

    # Lancer un job K8 si disponible
    if ($k8Index -lt $pendingK8.Count -and (Test-Path $K8_CSV)) {
        # Vérifier qu'on ne dépasse pas le max
        if ((Get-Job -State Running).Count -lt $MAX_PARALLEL) {
            $seed = $pendingK8[$k8Index]
            Write-Host "  [K8] Seed $seed -> $OUT_K8\seed_$seed" -ForegroundColor Magenta
            Start-OptimizationJob -K 8 -Seed $seed -LabelsCSV $K8_CSV -OutDir $OUT_K8 -Trials $TRIALS -RootDir $ROOT | Out-Null
            $k8Index++
            $totalJobs++
            Start-Sleep -Milliseconds 500
        }
    }
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  $totalJobs JOBS LANCES!" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Le systeme de CHECKPOINT sauvegarde toutes les 10 min." -ForegroundColor Yellow
Write-Host "En cas de crash, relancez ce script - il reprendra automatiquement." -ForegroundColor Yellow
Write-Host ""
Write-Host "Commandes utiles:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  # Voir tous les jobs"
Write-Host "  Get-Job | Format-Table Name, State, HasMoreData"
Write-Host ""
Write-Host "  # Voir progression K5"
Write-Host "  (Get-ChildItem '$OUT_K5' -Directory).Count"
Write-Host ""
Write-Host "  # Voir progression K8"
Write-Host "  (Get-ChildItem '$OUT_K8' -Directory).Count"
Write-Host ""
Write-Host "  # Logs en temps reel"
Write-Host "  Get-Content -Wait '$OUT_K5\seed_101\run.log' -Tail 20"
Write-Host ""
Write-Host "  # Attendre tous les jobs"
Write-Host "  Get-Job | Wait-Job"
Write-Host ""
Write-Host "  # Nettoyer apres completion"
Write-Host "  Get-Job | Remove-Job"
Write-Host ""
Write-Host "  # REPRENDRE APRES CRASH (relancer ce script)"
Write-Host "  .\scripts\launch_k5_k8_parallel.ps1"
Write-Host ""
Write-Host "Duree estimee: ~4-6 semaines avec $MAX_PARALLEL jobs" -ForegroundColor Yellow
Write-Host "(K5: ~3 semaines, K8: ~4 semaines, en parallele)" -ForegroundColor Yellow
Write-Host ""
