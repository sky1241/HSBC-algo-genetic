# ============================================================================
# LANCEUR BOT BINANCE LIVE - Mode continu avec monitoring
# ============================================================================
# 
# Ce script lance le bot en mode LIVE et le fait tourner en continu:
# - Daily phase job (1×/jour à 00:05 UTC)
# - Intraday runner (toutes les 2h)
# - Affiche les logs en temps réel
#
# ⚠️ ATTENTION: Mode LIVE = Ordres réels sur Binance!
# ============================================================================

$ErrorActionPreference = 'Stop'

$repo = "C:\Users\ludov\OneDrive\Bureau\teste algo\ichimoku_pipeline_web_v4_8"
$python = Join-Path $repo ".venv\Scripts\python.exe"
$dailyScript = Join-Path $repo "binance_bot\routines\daily_phase_job.py"
$intradayScript = Join-Path $repo "binance_bot\routines\intraday_runner.py"

# Vérifications
if (-not (Test-Path $python)) {
    Write-Host "❌ Python venv introuvable: $python" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $dailyScript)) {
    Write-Host "❌ Script daily introuvable: $dailyScript" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $intradayScript)) {
    Write-Host "❌ Script intraday introuvable: $intradayScript" -ForegroundColor Red
    exit 1
}

Set-Location -LiteralPath $repo

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "🚀 LANCEMENT BOT BINANCE LIVE" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  MODE LIVE ACTIVÉ - Ordres réels sur Binance!" -ForegroundColor Yellow
Write-Host ""
Write-Host "📊 Le bot va:" -ForegroundColor Cyan
Write-Host "   1. Mettre à jour la phase du jour (1×/jour)"
Write-Host "   2. Vérifier les signaux toutes les 2h"
Write-Host "   3. Passer des ordres réels sur Binance"
Write-Host ""
Write-Host "📈 Tu peux voir les trades sur:" -ForegroundColor Cyan
Write-Host "   - Binance Web/App → Futures → Positions"
Write-Host "   - Binance Web/App → Orders (Stop Loss / Take Profit)"
Write-Host ""
Write-Host "🛑 Pour arrêter: Ctrl+C" -ForegroundColor Yellow
Write-Host ""
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""

# 1. Lancer daily phase job une fois
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 📅 Lancement Daily Phase Job..." -ForegroundColor Cyan
& $python $dailyScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur daily phase job!" -ForegroundColor Red
    exit 1
}
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ✅ Daily Phase Job terminé" -ForegroundColor Green
Write-Host ""

# 2. Boucle intraday (synchronisé avec fermeture bougies H2)
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 🔄 Démarrage boucle intraday (synchronisé H2)..." -ForegroundColor Cyan
Write-Host ""

function Get-NextH2Close {
    # Bougies H2 se ferment à: 00:00, 02:00, 04:00, 06:00, 08:00, 10:00, 12:00, 14:00, 16:00, 18:00, 20:00, 22:00 UTC
    $now = [DateTimeOffset]::UtcNow
    $currentHour = $now.Hour
    $currentMinute = $now.Minute
    $currentSecond = $now.Second
    
    # Trouver prochaine heure paire (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
    $nextHour = [Math]::Floor($currentHour / 2) * 2 + 2
    if ($nextHour -ge 24) {
        $nextHour = 0
        $nextDay = $now.AddDays(1)
    } else {
        $nextDay = $now
    }
    
    # Si on est déjà dans une heure paire et qu'il reste moins de 1 minute, prendre la suivante
    if ($currentHour % 2 -eq 0 -and $currentMinute -ge 59) {
        $nextHour = ($currentHour + 2) % 24
        if ($nextHour -lt $currentHour) {
            $nextDay = $now.AddDays(1)
        }
    }
    
    $nextClose = New-Object DateTimeOffset($nextDay.Year, $nextDay.Month, $nextDay.Day, $nextHour, 0, 0, [TimeSpan]::Zero)
    
    # Si la prochaine fermeture est dans moins de 30 secondes, prendre la suivante
    if (($nextClose - $now).TotalSeconds -lt 30) {
        $nextClose = $nextClose.AddHours(2)
    }
    
    return $nextClose
}

# Synchroniser avec prochaine fermeture bougie H2
$nextRun = Get-NextH2Close
$now = [DateTimeOffset]::UtcNow
$waitSeconds = [Math]::Max(0, [Math]::Floor(($nextRun - $now).TotalSeconds))

if ($waitSeconds -gt 0) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ⏰ Synchronisation avec fermeture bougie H2..." -ForegroundColor Cyan
    Write-Host "   Prochaine fermeture: $($nextRun.ToString('yyyy-MM-dd HH:mm:ss')) UTC" -ForegroundColor Yellow
    Write-Host "   Attente: $([Math]::Floor($waitSeconds / 60)) min $($waitSeconds % 60) sec" -ForegroundColor Yellow
    Write-Host ""
    Start-Sleep -Seconds $waitSeconds
}

while ($true) {
    $now = [DateTimeOffset]::UtcNow
    Write-Host "="*70 -ForegroundColor Gray
    Write-Host "[$($now.ToString('yyyy-MM-dd HH:mm:ss')) UTC] 🔄 RUN INTRADAY (après fermeture bougie H2)" -ForegroundColor Yellow
    Write-Host "="*70 -ForegroundColor Gray
    
    try {
        & $python $intradayScript
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ✅ Run terminé avec succès" -ForegroundColor Green
        } elseif ($exitCode -eq 2) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ⚠️  Daily phase job non exécuté aujourd'hui" -ForegroundColor Yellow
            Write-Host "   Relancement du daily phase job..." -ForegroundColor Yellow
            & $python $dailyScript
        } elseif ($exitCode -eq 3) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 🛑 STOP GLOBAL ATTEINT - Arrêt du bot" -ForegroundColor Red
            break
        } else {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ⚠️  Erreur (code: $exitCode)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ❌ Exception: $_" -ForegroundColor Red
    }
    
    # Calculer prochaine fermeture bougie H2 (dans exactement 2h)
    $nextRun = Get-NextH2Close
    $now = [DateTimeOffset]::UtcNow
    $waitSeconds = [Math]::Max(0, [Math]::Floor(($nextRun - $now).TotalSeconds))
    
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ⏰ Prochaine fermeture bougie H2: $($nextRun.ToString('HH:mm:ss')) UTC" -ForegroundColor Cyan
    Write-Host "   Attente: $([Math]::Floor($waitSeconds / 60)) min $($waitSeconds % 60) sec" -ForegroundColor Cyan
    Write-Host ""
    
    # Attendre jusqu'à prochaine fermeture bougie H2
    Start-Sleep -Seconds $waitSeconds
}

Write-Host ""
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "🛑 BOT ARRÊTÉ" -ForegroundColor Red
Write-Host "="*70 -ForegroundColor Cyan

