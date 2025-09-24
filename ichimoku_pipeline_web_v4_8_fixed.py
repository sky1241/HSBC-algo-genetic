#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ichimoku Pipeline - Version 4.8
Backtesting automatique avec optimisation des paramètres
"""

import ccxt
import pandas as pd
import numpy as np
import random
import os
import sys
from datetime import datetime, timedelta, timezone
import time
import argparse
try:
    import optuna
    from optuna.pruners import SuccessiveHalvingPruner
except Exception:
    optuna = None

# Allow importing modules from the local `src` directory
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from ichimoku.risk import daily_loss_threshold

def log(msg):
    """Log avec timestamp UTC (ASCII-only pour compatibilité Windows)

    Si la variable d'environnement ICHIMOKU_QUIET=1 est definie, on limite l'affichage
    aux messages de progression essentiels pour garder la barre lisible.
    """
    try:
        # Filtrer en mode silencieux (comparaison sur version ASCII pour ignorer les accents)
        if os.environ.get("ICHIMOKU_QUIET") == "1":
            allow_keys = (
                "portfolio partage",
                "optuna",
                "fixed run",
                "run parametres fixes",
                "telechargement",
                "live",  # for LIVE report writes
            )
            m = str(msg)
            m_safe = m.encode('ascii', 'ignore').decode('ascii', 'ignore').lower()
            if not any(k in m_safe for k in allow_keys):
                return
        # Utiliser datetime timezone-aware pour éviter DeprecationWarning
        now_utc = datetime.now(timezone.utc)
        text = f"[{now_utc.replace(tzinfo=None).isoformat(timespec='seconds')}Z] {msg}"
        safe = text.encode('ascii', 'ignore').decode('ascii', 'ignore')
        print(safe, flush=True)
    except Exception:
        try:
            now_utc = datetime.now(timezone.utc)
            print(f"[{now_utc.replace(tzinfo=None).isoformat(timespec='seconds')}Z] ", flush=True)
        except Exception:
            pass

# -------- Binance fidelity helpers --------
def _read_json_safe(path: str):
    try:
        import json as _json
        with open(path, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None

def _symbol_to_binance(sym: str) -> str:
    try:
        base, quote = sym.split("/")
        return f"{base}{quote}"
    except Exception:
        return sym.replace("/", "")

def _round_to_step(value: float, step: float) -> float:
    if step is None or step <= 0:
        return float(value)
    return float(np.floor((value + 1e-15) / step) * step)

def _load_binance_filters(outputs_dir: str) -> dict:
    """Load exchangeInfo from outputs folder. Returns map symbol-> {tickSize, stepSize, minNotional}."""
    path = os.path.join(outputs_dir, "binance_usdm_exchangeInfo.json")
    data = _read_json_safe(path)
    result: dict[str, dict] = {}
    if isinstance(data, dict) and isinstance(data.get("symbols"), list):
        for s in data["symbols"]:
            sym = s.get("symbol")
            if not sym:
                continue
            tick = None
            step = None
            min_notional = None
            for f in s.get("filters", []):
                ftype = f.get("filterType")
                if ftype == "PRICE_FILTER":
                    try:
                        tick = float(f.get("tickSize"))
                    except Exception:
                        tick = None
                elif ftype in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                    try:
                        step = float(f.get("stepSize"))
                    except Exception:
                        step = step or None
                elif ftype == "MIN_NOTIONAL":
                    try:
                        min_notional = float(f.get("notional"))
                    except Exception:
                        min_notional = None
            result[sym] = {
                "tickSize": tick if (tick and tick > 0) else 0.01,
                "stepSize": step if (step and step > 0) else 0.001,
                "minNotional": min_notional if (min_notional and min_notional > 0) else 5.0,
            }
    return result

def utc_ms(dt):
    """Convertit datetime en millisecondes UTC"""
    return int(dt.timestamp() * 1000)

def fetch_ohlcv_range(exchange, symbol, timeframe, since_ms, until_ms, cache_dir="data", use_cache=True):
    """Récupère les données OHLCV avec cache"""
    # Nom de fichier FIXE basé seulement sur symbole et timeframe (pas de timestamps)
    cache_file = os.path.join(cache_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")
    
    if use_cache and os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Vérifier si les données existantes couvrent la période demandée
            start_date = pd.to_datetime(since_ms, unit='ms')
            end_date = pd.to_datetime(until_ms, unit='ms')
            
            # Vérifier si on a suffisamment de données (au moins 80% de la période demandée)
            required_days = (end_date - start_date).days
            available_days = (df.index.max() - df.index.min()).days
            
            if len(df) > 0 and available_days >= required_days * 0.8:
                # Filtrer pour la période demandée (avec une marge de tolérance)
                margin_days = max(30, required_days * 0.1)  # Marge de 10% ou 30 jours minimum
                start_with_margin = start_date - timedelta(days=margin_days)
                end_with_margin = end_date + timedelta(days=margin_days)
                
                df_filtered = df[(df.index >= start_with_margin) & (df.index <= end_with_margin)]
                
                if len(df_filtered) > 0:
                    log(f"📁 Données chargées depuis le cache: {len(df_filtered)} bougies (période demandée + marge)")
                    # Retourner seulement la période demandée
                    return df[(df.index >= start_date) & (df.index <= end_date)]
                else:
                    log(f"⚠️ Cache existant mais données insuffisantes pour la période demandée")
            else:
                log(f"⚠️ Cache existant mais période insuffisante (disponible: {available_days}j, requis: {required_days}j)")
        except Exception as e:
            log(f"⚠️ Erreur lors du chargement du cache: {e}")
            pass
    
    # Téléchargement
    all_data = []
    current_ms = since_ms
    
    while current_ms < until_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_ms, limit=1000)
            if not ohlcv:
                break
            
            all_data.extend(ohlcv)
            current_ms = ohlcv[-1][0] + 1
            
            if len(ohlcv) < 1000:
                break
                
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            log(f"⚠️  Erreur lors du téléchargement: {e}")
            break
    
    if not all_data:
        return pd.DataFrame()
    
    # Conversion en DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Sauvegarder TOUTES les données téléchargées (pas seulement la période demandée)
    df_all = df.copy()
    
    # Filtrage par période pour le retour
    df = df[(df.index >= pd.to_datetime(since_ms, unit='ms')) & 
            (df.index <= pd.to_datetime(until_ms, unit='ms'))]
    
    # Cache - sauvegarder toutes les données
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        df_all.to_csv(cache_file)
        log(f"💾 Données sauvegardées en cache: {len(df_all)} bougies (total téléchargé)")
    
    return df

def calculate_lyapunov_exponent(returns, window=50):
    """Calcule l'exposant de Lyapunov pour mesurer la stabilité"""
    if len(returns) < window:
        return 0.0
    
    try:
        # Prendre les derniers returns
        recent_returns = returns[-window:]
        
        # Calculer la divergence exponentielle
        if len(recent_returns) > 1:
            # Méthode simplifiée : variance des returns
            variance = np.var(recent_returns)
            lyapunov = np.log(1 + variance) / window
            return float(lyapunov)
        else:
            return 0.0
    except:
        return 0.0

def calculate_true_atr(df, period=14):
    """Calcule le vrai ATR (Average True Range)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR = moyenne mobile simple du True Range
    atr = true_range.rolling(window=period).mean()
    
    return atr

def calculate_dynamic_slippage(volume_usdt, position_value, base_slippage=0.0005):
    """Calcule le slippage dynamique basé sur le volume et la taille de position"""
    if volume_usdt <= 0 or position_value <= 0:
        return base_slippage
    
    # Ratio position/volume (plus c'est élevé, plus le slippage est important)
    volume_ratio = position_value / volume_usdt
    
    # Slippage exponentiel : 0.05% de base, peut monter à 5% pour gros ordres
    if volume_ratio <= 0.001:  # Position < 0.1% du volume
        return base_slippage
    elif volume_ratio <= 0.01:  # Position < 1% du volume
        return base_slippage * (1 + volume_ratio * 10)
    elif volume_ratio <= 0.1:   # Position < 10% du volume
        return base_slippage * (1 + volume_ratio * 50)
    else:  # Position > 10% du volume
        return min(0.05, base_slippage * (1 + volume_ratio * 100))  # Max 5%

def validate_market_data(df, symbol):
    """Vérifie la qualité des données en temps réel (CRITIQUE !)"""
    errors = []
    
    # Vérifier les données manquantes CRITIQUES (prix et volume)
    critical_columns = ['open', 'high', 'low', 'close', 'volume']
    critical_missing = df[critical_columns].isnull().sum()
    if critical_missing.sum() > 0:
        errors.append(f"Données critiques manquantes: {critical_missing.to_dict()}")
    
    # Vérifier les données manquantes des indicateurs (IGNORER les NaN initiaux réels)
    indicator_columns = ['tenkan', 'kijun', 'senkou_a', 'senkou_b', 'cloud_top', 'cloud_bottom', 'ATR']
    # Déterminer dynamiquement le vrai warmup: position du premier non-NaN le plus tardif
    dynamic_start = 0
    for col in indicator_columns:
        try:
            first_valid_ts = df[col].first_valid_index()
            if first_valid_ts is not None:
                pos = df.index.get_loc(first_valid_ts)
                dynamic_start = max(dynamic_start, pos)
        except Exception:
            continue
    # Ajouter une petite marge de sécurité
    dynamic_start = min(len(df), dynamic_start + 1)
    if len(df) > dynamic_start:
        recent_data = df.iloc[dynamic_start:]
        indicator_missing = recent_data[indicator_columns].isnull().sum()
        # Tolérer jusqu'à 5% de NaN résiduels après warmup
        for col in indicator_columns:
            if indicator_missing[col] > 0:
                missing_ratio = indicator_missing[col] / max(1, len(recent_data))
                if missing_ratio > 0.05:
                    errors.append(f"Indicateur {col}: {missing_ratio*100:.1f}% de données manquantes")
    
    # Vérifier les prix négatifs ou nuls
    negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).sum().sum()
    if negative_prices > 0:
        errors.append(f"Prix négatifs/nuls détectés: {negative_prices}")
    
    # Vérifier les volumes nuls
    zero_volumes = (df['volume'] <= 0).sum()
    if zero_volumes > 0:
        errors.append(f"Volumes nuls détectés: {zero_volumes}")
    
    # Vérifier la cohérence high >= low
    invalid_hl = (df['high'] < df['low']).sum()
    if invalid_hl > 0:
        errors.append(f"High < Low détectés: {invalid_hl}")
    
    # Vérifier les gaps de prix extrêmes (>20% en 2h)
    price_changes = abs(df['close'].pct_change())
    extreme_gaps = (price_changes > 0.20).sum()
    if extreme_gaps > 0:
        errors.append(f"Gaps extrêmes détectés: {extreme_gaps}")
    
    # Vérifier les timestamps
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dt.total_seconds() / 3600  # en heures
        expected_interval = 2  # 2h timeframe
        
        # Tolérance plus large : 1.5h à 2.5h (au lieu de 2h ± 0.1h)
        tolerance = 0.5  # ±30 minutes
        invalid_intervals = (abs(time_diffs - expected_interval) > tolerance).sum()
        
        # Ignorer les premiers intervalles qui peuvent être irréguliers
        if len(df) > 10:  # Vérifier seulement après 10 données
            recent_intervals = time_diffs.iloc[10:]  # Ignorer les 10 premiers
            invalid_intervals = (abs(recent_intervals - expected_interval) > tolerance).sum()
            
            if invalid_intervals > 0:
                # Tolérer jusqu'à 10% d'intervalles irréguliers
                irregular_ratio = invalid_intervals / len(recent_intervals)
                if irregular_ratio > 0.10:  # Plus de 10% d'intervalles irréguliers
                    errors.append(f"Intervalles de temps invalides: {invalid_intervals} ({irregular_ratio*100:.1f}%)")
    
    return errors

def simulate_execution_latency():
    """Latence désactivée pour backtests H2 (ne change pas les signaux)."""
    return 0.0

# ========== ALGORITHME GÉNÉTIQUE ICHIMOKU ========== #

class IchimokuTrader:
    """Un trader Ichimoku avec son ADN (paramètres)"""
    
    def __init__(self, tenkan, kijun, senkou_b, shift, atr_mult):
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b
        self.shift = shift
        self.atr_mult = atr_mult
        self.fitness = 0.0
        self.generation = 0
        self.performance_history = []
    
    def get_params(self):
        """Retourne les paramètres du trader"""
        return {
            "tenkan": self.tenkan,
            "kijun": self.kijun,
            "senkou_b": self.senkou_b,
            "shift": self.shift,
            "atr_mult": self.atr_mult
        }
    
    def mutate(self, mutation_rate=0.15, mutation_strength=0.2):
        """Mutation génétique pour explorer de nouveaux paramètres"""
        if random.random() < mutation_rate:
            # Mutation du Tenkan
            if random.random() < 0.2:
                self.tenkan = max(1, min(70, self.tenkan + random.randint(-15, 15)))
            
            # Mutation du Kijun
            if random.random() < 0.2:
                self.kijun = max(1, min(70, self.kijun + random.randint(-15, 15)))
            
            # Mutation du Senkou B
            if random.random() < 0.2:
                self.senkou_b = max(1, min(70, self.senkou_b + random.randint(-15, 15)))
            
            # Mutation du Shift
            if random.random() < 0.2:
                self.shift = max(1, min(99, self.shift + random.randint(-20, 20)))
            
            # Mutation de l'ATR
            if random.random() < 0.2:
                self.atr_mult = max(1.0, min(14.0, self.atr_mult + random.uniform(-3, 3)))
                self.atr_mult = round(self.atr_mult, 1)
    
    def crossover(self, other_trader):
        """Croisement avec un autre trader pour créer un enfant"""
        child = IchimokuTrader(
            tenkan=random.choice([self.tenkan, other_trader.tenkan]),
            kijun=random.choice([self.kijun, other_trader.kijun]),
            senkou_b=random.choice([self.senkou_b, other_trader.senkou_b]),
            shift=random.choice([self.shift, other_trader.shift]),
            atr_mult=round((self.atr_mult * 0.6 + other_trader.atr_mult * 0.4), 1)
        )
        return child

def create_initial_population(population_size=100, ranges=None):
    """Crée une population initiale de traders Ichimoku"""
    if ranges is None:
        ranges = {
            "tenkan": (1, 70),
            "kijun": (1, 70),
            "senkou_b": (1, 70),
            "shift": (1, 99),
            "atr_mult": (1.0, 14.0)
        }
    
    population = []
    
    # Créer des traders avec des stratégies variées
    for i in range(population_size):
        if i < 20:  # 20% de traders conservateurs
            trader = IchimokuTrader(
                tenkan=random.randint(5, 15),
                kijun=random.randint(20, 40),
                senkou_b=random.randint(40, 60),
                shift=random.randint(20, 30),
                atr_mult=round(random.uniform(2.0, 6.0), 1)
            )
        elif i < 40:  # 20% de traders agressifs
            trader = IchimokuTrader(
                tenkan=random.randint(15, 25),
                kijun=random.randint(10, 20),
                senkou_b=random.randint(25, 40),
                shift=random.randint(15, 25),
                atr_mult=round(random.uniform(6.0, 12.0), 1)
            )
        elif i < 60:  # 20% de traders équilibrés
            trader = IchimokuTrader(
                tenkan=random.randint(10, 20),
                kijun=random.randint(15, 30),
                senkou_b=random.randint(30, 50),
                shift=random.randint(18, 28),
                atr_mult=round(random.uniform(4.0, 8.0), 1)
            )
        else:  # 40% de traders aléatoires
            trader = IchimokuTrader(
                tenkan=random.randint(ranges["tenkan"][0], ranges["tenkan"][1]),
                kijun=random.randint(ranges["kijun"][0], ranges["kijun"][1]),
                senkou_b=random.randint(ranges["senkou_b"][0], ranges["senkou_b"][1]),
                shift=random.randint(ranges["shift"][0], ranges["shift"][1]),
                atr_mult=round(random.uniform(ranges["atr_mult"][0], ranges["atr_mult"][1]), 1)
            )
        
        population.append(trader)
    
    return population

def calculate_fitness_score(results):
    """Calcule le score de fitness d'un trader basé sur ses résultats"""
    if not results or "equity_mult" not in results:
        return 0.0
    
    # Score composite : Performance + Stabilité + Qualité
    performance_score = min(2.0, results["equity_mult"]) / 2.0  # 0 à 1
    
    # Stabilité (inversé car Lyapunov bas = stable)
    stability_score = max(0, 1 - results.get("lyapunov_exponent", 0))  # 0 à 1
    
    # Qualité des trades
    quality_score = results.get("win_rate", 0.5)  # 0 à 1
    
    # Calmar Ratio (performance/risque)
    calmar_score = min(1.0, max(0, results.get("calmar_ratio", 0)))  # 0 à 1
    
    # Score final pondéré
    fitness = (
        performance_score * 0.35 +    # Performance (35%)
        stability_score * 0.25 +      # Stabilité (25%)
        quality_score * 0.20 +        # Qualité des trades (20%)
        calmar_score * 0.20           # Ratio performance/risque (20%)
    )
    
    return fitness

def evolve_population(population, elite_size=20, mutation_rate=0.15):
    """Fait évoluer la population vers la génération suivante"""
    # Trier par fitness
    population.sort(key=lambda x: x.fitness, reverse=True)
    
    # Garder les élites (meilleurs 20%)
    elite = population[:elite_size]
    
    # Créer la nouvelle génération
    new_population = elite.copy()
    
    # Croisement des élites pour créer des enfants
    while len(new_population) < len(population):
        parent1 = random.choice(elite)
        parent2 = random.choice(elite)
        child = parent1.crossover(parent2)
        child.generation = max(parent1.generation, parent2.generation) + 1
        new_population.append(child)
    
    # Mutation pour maintenir la diversité
    for trader in new_population[elite_size:]:  # Pas de mutation sur les élites
        trader.mutate(mutation_rate)
    
    # Ajouter quelques immigrants pour éviter la consanguinité
    immigrant_count = max(5, len(population) // 20)
    for _ in range(immigrant_count):
        immigrant = IchimokuTrader(
            tenkan=random.randint(1, 70),
            kijun=random.randint(1, 70),
            senkou_b=random.randint(1, 70),
            shift=random.randint(1, 99),
            atr_mult=round(random.uniform(1.0, 14.0), 1)
        )
        immigrant.generation = max(t.generation for t in new_population) + 1
        new_population.append(immigrant)
    
    # Garder la taille de population constante
    return new_population[:len(population)]

def calculate_ichimoku(df, tenkan, kijun, senkou_b, shift):
    """Calcule les indicateurs Ichimoku"""
    # Tenkan-sen (Conversion Line)
    tenkan_high = df['high'].rolling(window=tenkan).max()
    tenkan_low = df['low'].rolling(window=tenkan).min()
    df['tenkan'] = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_high = df['high'].rolling(window=kijun).max()
    kijun_low = df['low'].rolling(window=kijun).min()
    df['kijun'] = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A)
    df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(shift)
    # Versions non décalées pour la logique de signal (alignées sur la bougie courante)
    df['senkou_a_sig'] = ((df['tenkan'] + df['kijun']) / 2)
    
    # Senkou Span B (Leading Span B)
    senkou_b_high = df['high'].rolling(window=senkou_b).max()
    senkou_b_low = df['low'].rolling(window=senkou_b).min()
    df['senkou_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(shift)
    df['senkou_b_sig'] = ((senkou_b_high + senkou_b_low) / 2)
    
    # Cloud (pour affichage/diagnostic)
    df['cloud_top'] = df[['senkou_a', 'senkou_b']].max(axis=1)
    df['cloud_bottom'] = df[['senkou_a', 'senkou_b']].min(axis=1)
    # Cloud pour signaux (non décalé)
    df['cloud_top_sig'] = df[['senkou_a_sig', 'senkou_b_sig']].max(axis=1)
    df['cloud_bottom_sig'] = df[['senkou_a_sig', 'senkou_b_sig']].min(axis=1)
    
    # Position relative au nuage (basé sur le nuage non décalé pour générer des signaux suffisants)
    df['above_cloud'] = df['close'] > df['cloud_top_sig']
    df['below_cloud'] = df['close'] < df['cloud_bottom_sig']
    
    # Direction/Snapping des signaux façon Pine: utiliser les lignes "lead" décalées de (shift-1) vers le passé
    df['lead1_sig'] = df['senkou_a_sig'].shift(shift - 1)  # équiv. leadLine1[displacement-1]
    df['lead2_sig'] = df['senkou_b_sig'].shift(shift - 1)  # équiv. leadLine2[displacement-1]
    df['cloud_bullish'] = df['lead1_sig'] > df['lead2_sig']  # lead2 < lead1
    df['cloud_bearish'] = df['lead2_sig'] > df['lead1_sig']  # lead2 > lead1
    
    # Détection des signaux (Pine):
    # crossup = crossover(close, lead2_sig), crossdn = crossunder(close, lead2_sig)
    prev_close = df['close'].shift(1)
    prev_lead2 = df['lead2_sig'].shift(1)
    crossup = (df['close'] > df['lead2_sig']) & (prev_close <= prev_lead2)
    crossdn = (df['close'] < df['lead2_sig']) & (prev_close >= prev_lead2)
    green_candle = df['close'] > df['open']
    red_candle = df['close'] < df['open']
    # BUY si lead2 > lead1 (nuage vendeur), bougie verte et cross au-dessus de lead2
    df['signal_long'] = crossup & df['cloud_bearish'] & green_candle
    # SELL si lead2 < lead1 (nuage acheteur), bougie rouge et cross sous lead2
    df['signal_short'] = crossdn & df['cloud_bullish'] & red_candle
    
    # ATR pour trailing stop (vrai calcul !)
    df['ATR'] = calculate_true_atr(df, period=14)
    
    return df

def backtest_long_short(df, tenkan, kijun, senkou_b, shift, atr_mult, loss_mult=3.0, symbol=None, timeframe="2h"):
    """Backtest avec stratégie long + short"""
    data = calculate_ichimoku(df.copy(), tenkan, kijun, senkou_b, shift)
    
    # Positions long et short (jusqu'à 3 par côté, jamais les deux en même temps sur un symbole)
    # Chaque position: {"entry": float, "trailing": float, "position_value": float}
    positions_long = []
    positions_short = []

    equity = 1.0
    trades_long = []
    trades_short = []
    wins_long = 0
    wins_short = 0
    
    # Configuration des paramètres de trading (overrides via env: POSITION_SIZE, LEVERAGE)
    initial_capital = 1000  # Capital initial en euros
    try:
        _env_pos = os.getenv("POSITION_SIZE")
        _env_lev = os.getenv("LEVERAGE")
    except Exception:
        _env_pos = None; _env_lev = None
    position_size = float(_env_pos) if _env_pos else 0.01  # part du capital par trade
    leverage = float(_env_lev) if _env_lev else 10  # levier par défaut = 10
    # Cap dur sécurité demandé: 1% et x10 max
    position_size = min(max(0.0, position_size), 0.01)
    leverage = min(max(1.0, leverage), 10.0)
    
    # Gestion du risque (AUGMENTÉE POUR PLUS DE SIGNALS !)
    max_trades_per_symbol = 50  # Maximum 50 trades par type par paire (pour 200 signaux/an)
    max_total_trades = 300  # Maximum 300 positions totales (50 par type × 3 paires)
    
    # STOP LOSSES OPTIMISÉS AVEC ATR (RÉDUCTION DES LIQUIDATIONS !)
    liquidation_threshold = 0.05  # 5% liquidation threshold (augmenté de 2% à 5%)
    stop_global = 0.80  # Stop global à 80% du capital (plus conservateur)
    
    # Paramètres de volume et liquidité (OPTIMISÉS POUR PLUS DE SIGNALS !)
    min_volume_usdt = 100000  # Volume minimum 100K USDT (réduit de 1M à 100K)
    max_volume_impact = 0.01  # Impact max 1% sur le marché (augmenté de 0.1% à 1%)
    volume_liquidity_threshold = 0.01  # 1% du volume disponible (réduit de 10% à 1%)
    min_order_size_usdt = 5  # Ordre minimum 5 USDT (Binance)
    max_order_size_usdt = 1000000  # Ordre maximum 1M USDT
    max_orders_per_second = 10  # Limite Binance
    
    # Paramètres de slippage et latence (RÉALISTES !)
    base_slippage = 0.0005  # 0.05% slippage de base
    max_slippage = 0.002  # 0.2% slippage maximum
    execution_latency_ms = 200  # Latence moyenne 200ms
    
    # Paramètres de financement futures
    funding_rate = 0.0001  # 0.01% par 8h (réaliste)
    funding_interval = 8  # Toutes les 8h
    rollover_cost = 0.0005  # 0.05% par rollover
    rollover_interval = 7  # Tous les 7 jours
    
    # Paramètres de validation des données (95% FIABILITÉ !)
    max_consecutive_errors = 5  # Maximum 5 erreurs consécutives
    data_validation_interval = 50  # Validation toutes les 50 bougies
    
    # Frais Binance (ULTRA-RÉALISTES !)
    commission_rate = 0.001  # 0.1% par trade (spot)
    funding_rate = 0.0001   # 0.01% toutes les 8h (futures)
    funding_interval = 8    # Heures entre paiements de financement
    
    # SLIPPAGE DYNAMIQUE basé sur le volume (CRITIQUE !)
    base_slippage = 0.0005  # Slippage de base 0.05%
    
    # Gestion des volumes (ULTRA-RÉALISTE !)
    min_volume_usdt = 10.0      # Volume minimum en USDT (Binance)
    max_volume_impact = 0.02    # Impact maximum sur le prix (2%)
    volume_liquidity_threshold = 0.001  # Seuil de liquidité (0.1% du volume)
    
    # LIMITES BINANCE (CRITIQUE !)
    min_order_size_usdt = 10.0  # Taille minimale d'ordre Binance
    max_order_size_usdt = 1000000.0  # Taille maximale d'ordre (1M USDT)
    max_orders_per_second = 10  # Limite de fréquence Binance
    
    # Gestion des margin calls (ULTRA-RÉALISTE Binance !)
    initial_margin = 0.10        # 10% margin initial (Binance futures)
    maintenance_margin = 0.025   # 2.5% maintenance margin (Binance futures)
    liquidation_threshold = 0.02 # 2% liquidation threshold (Binance futures)
    margin_call_buffer = 0.005   # 0.5% buffer avant margin call
    
    # Gestion des rollovers futures (CRITIQUE pour 5 ans !)
    rollover_cost = 0.0005      # 0.05% coût de rollover (réaliste)
    rollover_interval = 30      # Rollover tous les 30 jours (futures quarterly)
    days_since_rollover = 0     # Compteur de jours depuis le dernier rollover
    
    # Gestion des haltes de trading (CRITIQUE !)
    trading_halt_threshold = 0.30  # Si gap > 30%, arrêter le trading
    max_consecutive_errors = 5      # Max 5 erreurs consécutives avant arrêt
    
    # Compteurs pour la gestion des erreurs
    consecutive_errors = 0
    total_execution_latency = 0.0
    execution_count = 0

    # Daily loss tracking
    current_day = None
    daily_loss = 0.0
    daily_threshold = float("inf")

    for ts, row in data.iterrows():
        close = row["close"]

        day = ts.date()
        if day != current_day:
            current_day = day
            daily_loss = 0.0
            daily_threshold = daily_loss_threshold(row.get("ATR", np.nan), loss_mult) / close if close else float("inf")

        # VÉRIFICATION DES DONNÉES OPTIMISÉE (95% FIABILITÉ - VALIDATION TOUTES LES 50 BOUGIES)
        try:
            # Validation partielle pour optimiser performance/fiabilité
            current_index = data.index.get_loc(ts)
            if current_index % 50 == 0:  # Vérifier toutes les 50 bougies
                data_errors = validate_market_data(data.loc[:ts], symbol)
                if data_errors:
                    log(f"🔍 Validation {symbol} (bougie {current_index}): {len(data_errors)} erreurs détectées")
            else:
                data_errors = []  # Pas de validation sur cette bougie
            
            if data_errors:
                consecutive_errors += 1
                log(f"⚠️ Erreurs de données {symbol}: {data_errors}")
                
                if consecutive_errors >= max_consecutive_errors:
                    log(f"🚨 ARRÊT DU TRADING {symbol} - Trop d'erreurs de données !")
                    break
            else:
                consecutive_errors = 0  # Reset si pas d'erreur
        except Exception as e:
            consecutive_errors += 1
            log(f"⚠️ Erreur validation données {symbol}: {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                log(f"🚨 ARRÊT DU TRADING {symbol} - Erreurs critiques !")
                break

        # Update trailing pour toutes les positions LONG
        if len(positions_long) > 0:
            atr_stop_mult = atr_mult * 1.5
            updated_positions = []
            for pos in positions_long:
                current_trailing = pos["trailing"]
                new_trailing = close - (row["ATR"] * atr_stop_mult) if pd.notna(row["ATR"]) else current_trailing
                if pd.notna(new_trailing):
                    if np.isnan(current_trailing):
                        pos["trailing"] = new_trailing
                    else:
                        pos["trailing"] = max(current_trailing, new_trailing)
                updated_positions.append(pos)
            positions_long = updated_positions

        # Update trailing pour toutes les positions SHORT
        if len(positions_short) > 0:
            atr_stop_mult = atr_mult * 1.5
            updated_positions = []
            for pos in positions_short:
                current_trailing = pos["trailing"]
                new_trailing = close + (row["ATR"] * atr_stop_mult) if pd.notna(row["ATR"]) else current_trailing
                if pd.notna(new_trailing):
                    if np.isnan(current_trailing):
                        pos["trailing"] = new_trailing
                    else:
                        pos["trailing"] = min(current_trailing, new_trailing)
                updated_positions.append(pos)
            positions_short = updated_positions

        # Frais de financement toutes les 8h (futures)
        current_hour = ts.hour
        if current_hour % funding_interval == 0:  # Toutes les 8h
            # Appliquer le funding au prorata du notional ouvert
            open_notional = sum(p["position_value"] for p in positions_long) + sum(p["position_value"] for p in positions_short)
            if open_notional > 0:
                current_capital = max(1e-12, equity * 1000.0)
                equity -= funding_rate * (open_notional / current_capital)
        
        # Gestion des rollovers futures (CRITIQUE !)
        days_since_rollover += 1
        if days_since_rollover >= rollover_interval:
            # Rollover nécessaire - coût appliqué sur le notional ouvert
            open_notional = sum(p["position_value"] for p in positions_long) + sum(p["position_value"] for p in positions_short)
            if open_notional > 0:
                current_capital = max(1e-12, equity * 1000.0)
                rollover_equity_cost = rollover_cost * (open_notional / current_capital)
                equity -= rollover_equity_cost
                log(f"🔄 Rollover futures - Notional: {open_notional:.2f}€ - Coût_eq: {rollover_equity_cost:.6f} - Nouvelle equity: {equity:.4f}")
            days_since_rollover = 0  # Reset compteur

        # STOP LOSS DYNAMIQUE BASÉ SUR LA VOLATILITÉ (NOUVEAU !)
        # Ajuster les stop losses en fonction de la volatilité du marché
        if len(positions_long) > 0 and pd.notna(row["ATR"]):
            # Si l'ATR augmente (volatilité croissante), élargir les stop losses
            atr_ratio = row["ATR"] / (data["ATR"].rolling(20).mean().iloc[-1] if len(data) >= 20 else row["ATR"])
            if atr_ratio > 1.5:  # Volatilité 50% au-dessus de la moyenne
                # Élargir le stop loss pour éviter les sorties prématurées
                volatility_adjustment = 1.5
                updated_positions = []
                for pos in positions_long:
                    current_trailing = pos["trailing"]
                    new_trailing = close - (row["ATR"] * atr_mult * volatility_adjustment)
                    if pd.notna(new_trailing):
                        if np.isnan(current_trailing):
                            pos["trailing"] = new_trailing
                        else:
                            pos["trailing"] = max(current_trailing, new_trailing)
                    updated_positions.append(pos)
                positions_long = updated_positions
        
        if len(positions_short) > 0 and pd.notna(row["ATR"]):
            # Même logique pour les positions short
            atr_ratio = row["ATR"] / (data["ATR"].rolling(20).mean().iloc[-1] if len(data) >= 20 else row["ATR"])
            if atr_ratio > 1.5:  # Volatilité 50% au-dessus de la moyenne
                volatility_adjustment = 1.5
                updated_positions = []
                for pos in positions_short:
                    current_trailing = pos["trailing"]
                    new_trailing = close + (row["ATR"] * atr_mult * volatility_adjustment)
                    if pd.notna(new_trailing):
                        if np.isnan(current_trailing):
                            pos["trailing"] = new_trailing
                        else:
                            pos["trailing"] = min(current_trailing, new_trailing)
                    updated_positions.append(pos)
                positions_short = updated_positions

        # VÉRIFICATION DES MARGIN CALLS (CRITIQUE !)
        if equity <= liquidation_threshold:
            # Fermer toutes les positions ouvertes immédiatement (liquidation)
            if len(positions_long) > 0:
                for pos in positions_long:
                    exit_price = close
                    ret = ((exit_price / pos["entry"]) - 1.0) * leverage
                    commission_cost = pos["position_value"] * commission_rate * 2
                    ret -= commission_cost / pos["position_value"]
                    current_capital = max(1e-12, equity * 1000.0)
                    equity *= (1.0 + ret * (pos["position_value"] / current_capital))
                    wins_long += 1 if ret > 0 else 0
                    trades_long.append({
                        "timestamp": ts,
                        "entry": pos["entry"],
                        "exit": exit_price,
                        "ret": ret,
                        "position_value": pos["position_value"],
                        "exit_reason": "liquidation_forced",
                        "symbol": symbol,
                        "type": "long"
                    })
                positions_long = []
            if len(positions_short) > 0:
                for pos in positions_short:
                    exit_price = close
                    ret = ((pos["entry"] / exit_price) - 1.0) * leverage
                    commission_cost = pos["position_value"] * commission_rate * 2
                    ret -= commission_cost / pos["position_value"]
                    current_capital = max(1e-12, equity * 1000.0)
                    equity *= (1.0 + ret * (pos["position_value"] / current_capital))
                    wins_short += 1 if ret > 0 else 0
                    trades_short.append({
                        "timestamp": ts,
                        "entry": pos["entry"],
                        "exit": exit_price,
                        "ret": ret,
                        "position_value": pos["position_value"],
                        "exit_reason": "liquidation_forced",
                        "symbol": symbol,
                        "type": "short"
                    })
                positions_short = []
            # Protection contre boucle infinie
            if equity <= liquidation_threshold * 0.3:
                log(f"STOP TRADING SAFETY {symbol} - Equity trop bas: {equity:.4f}")
                # Ne pas break: continuer l'analyse comme demandé
                continue
            continue
        
        elif equity <= maintenance_margin:
            # MARGIN CALL - Fermer toutes les positions
            log(f"⚠️ MARGIN CALL ! Equity: {equity:.4f} < {maintenance_margin:.4f}")
            # Fermer toutes les positions ouvertes
            if len(positions_long) > 0:
                for pos in positions_long:
                    ret = ((close / pos["entry"]) - 1.0) * leverage
                    commission_cost = pos["position_value"] * commission_rate * 2
                    ret -= commission_cost / pos["position_value"]
                    current_capital = max(1e-12, equity * 1000.0)
                    equity *= (1.0 + ret * (pos["position_value"] / current_capital))
                    trades_long.append({
                        "timestamp": ts,
                        "entry": pos["entry"],
                        "exit": close,
                        "ret": ret,
                        "position_value": pos["position_value"],
                        "exit_reason": "margin_call",
                        "symbol": symbol,
                        "type": "long"
                    })
                positions_long = []
            if len(positions_short) > 0:
                for pos in positions_short:
                    ret = ((pos["entry"] / close) - 1.0) * leverage
                    commission_cost = pos["position_value"] * commission_rate * 2
                    ret -= commission_cost / pos["position_value"]
                    current_capital = max(1e-12, equity * 1000.0)
                    equity *= (1.0 + ret * (pos["position_value"] / current_capital))
                    trades_short.append({
                        "timestamp": ts,
                        "entry": pos["entry"],
                        "exit": close,
                        "ret": ret,
                        "position_value": pos["position_value"],
                        "exit_reason": "margin_call",
                        "symbol": symbol,
                        "type": "short"
                    })
                positions_short = []
            
            # Margin call - réinitialiser et continuer le trading
            log(f"⚠️ MARGIN CALL {symbol} - Réinitialisation pour continuer le trading !")
            # Réinitialiser les positions pour permettre de nouveaux trades
            position_long = 0
            position_short = 0
            entry_price_long = 0.0
            entry_price_short = 0.0
            trailing_long = np.nan
            trailing_short = np.nan
            # Continuer le trading au lieu d'arrêter
            # MAIS avec une protection contre la boucle infinie
            if equity <= liquidation_threshold * 0.3:  # Si vraiment trop bas, arrêter
                log(f"STOP TRADING SAFETY {symbol} - Equity trop bas: {equity:.4f}")
                # Ne pas break: continuer l'analyse
                continue
            continue

        # Si signal LONG et positions SHORT ouvertes -> fermer SHORT d'abord (opposite signal)
        if row["signal_long"] and len(positions_short) > 0:
            to_close = positions_short
            positions_short = []
            for pos in to_close:
                ret = ((pos["entry"] / close) - 1.0) * leverage
                commission_cost = pos["position_value"] * commission_rate * 2
                ret -= commission_cost / pos["position_value"]
                if ret < 0:
                    daily_loss += -ret
                current_capital = max(1e-12, equity * 1000.0)
                equity *= (1.0 + ret * (pos["position_value"] / current_capital))
                trades_short.append({
                    "timestamp": ts,
                    "entry": pos["entry"],
                    "exit": close,
                    "ret": ret,
                    "position_value": pos["position_value"],
                    "exit_reason": "opposite_signal",
                    "symbol": symbol,
                    "type": "short",
                })
        # Entrée LONG: seulement si pas de SHORT ouverts (jamais long et short en même temps sur un symbole)
        if len(positions_short) == 0 and row["signal_long"] and daily_loss < daily_threshold:
            current_capital = equity * 1000
            position_size_euros = current_capital * position_size  # 1% du capital seulement

            if len(positions_long) >= 3:
                # Cap de 3 positions atteint
                pass
            else:
                if current_capital < position_size_euros:
                    pass
                else:
                    if position_size_euros < 5 or position_size_euros > 1000000:
                        pass
                    else:
                        # Signal sur close, entrée sur open suivant (réaliste !)
                        next_open = data.loc[data.index > ts, 'open'].iloc[0] if len(data.loc[data.index > ts]) > 0 else close
                        volume_usdt = row["volume"] * close
                        position_value = position_size_euros
                        dynamic_slippage = calculate_dynamic_slippage(volume_usdt, position_value, base_slippage)
                        entry_price = next_open * (1 + dynamic_slippage)
                        atr_stop_mult = atr_mult * 2.0
                        trailing = entry_price - (row["ATR"] * atr_stop_mult) if pd.notna(row["ATR"]) else np.nan
                        positions_long.append({
                            "entry": entry_price,
                            "trailing": trailing,
                            "position_value": position_value
                        })
                        execution_latency = simulate_execution_latency()
                        total_execution_latency += execution_latency
                        execution_count += 1
                        log(f"📈 Entrée LONG {symbol} - Prix: {entry_price:.4f}, Slippage: {dynamic_slippage*100:.3f}%, Latence: {execution_latency*1000:.1f}ms")
            

        # Si signal SHORT et positions LONG ouvertes -> fermer LONG d'abord (opposite signal)
        if row["signal_short"] and len(positions_long) > 0:
            to_close = positions_long
            positions_long = []
            for pos in to_close:
                ret = ((close / pos["entry"]) - 1.0) * leverage
                commission_cost = pos["position_value"] * commission_rate * 2
                ret -= commission_cost / pos["position_value"]
                if ret < 0:
                    daily_loss += -ret
                current_capital = max(1e-12, equity * 1000.0)
                equity *= (1.0 + ret * (pos["position_value"] / current_capital))
                trades_long.append({
                    "timestamp": ts,
                    "entry": pos["entry"],
                    "exit": close,
                    "ret": ret,
                    "position_value": pos["position_value"],
                    "exit_reason": "opposite_signal",
                    "symbol": symbol,
                    "type": "long"
                })
        # Entrée SHORT: seulement si pas de LONG ouverts
        if len(positions_long) == 0 and row["signal_short"] and daily_loss < daily_threshold:
            current_capital = equity * 1000
            position_size_euros = current_capital * position_size  # 1% du capital seulement
            if len(positions_short) >= 3:
                pass
            else:
                if current_capital < position_size_euros:
                    pass
                else:
                    if position_size_euros < 5 or position_size_euros > 1000000:
                        pass
                    else:
                        next_open = data.loc[data.index > ts, 'open'].iloc[0] if len(data.loc[data.index > ts]) > 0 else close
                        volume_usdt = row["volume"] * close
                        position_value = position_size_euros
                        dynamic_slippage = calculate_dynamic_slippage(volume_usdt, position_value, base_slippage)
                        entry_price = next_open * (1 - dynamic_slippage)
                        atr_stop_mult = atr_mult * 2.0
                        trailing = entry_price + (row["ATR"] * atr_stop_mult) if pd.notna(row["ATR"]) else np.nan
                        positions_short.append({
                            "entry": entry_price,
                            "trailing": trailing,
                            "position_value": position_value
                        })
                        execution_latency = simulate_execution_latency()
                        total_execution_latency += execution_latency
                        execution_count += 1
                        log(f"📉 Entrée SHORT {symbol} - Prix: {entry_price:.4f}, Slippage: {dynamic_slippage*100:.3f}%, Latence: {execution_latency*1000:.1f}ms")

        # Sorties LONG: trailing stop ou stop global
        if len(positions_long) > 0:
            # Stop global dynamique
            current_capital = equity * 1000
            stop_global_euros = current_capital * stop_global
            
            # Vérifier le stop global dynamique
            if current_capital <= stop_global_euros:
                # Stop global touché, fermer toutes les positions
                for pos in positions_long:
                    exit_price = close
                    ret = ((exit_price / pos["entry"]) - 1.0) * leverage
                    commission_cost = pos["position_value"] * commission_rate * 2
                    ret -= commission_cost / pos["position_value"]
                    exit_latency = simulate_execution_latency()
                    total_execution_latency += exit_latency
                    execution_count += 1
                    current_capital = max(1e-12, equity * 1000.0)
                    equity *= (1.0 + ret * (pos["position_value"] / current_capital))
                    if ret < 0:
                        daily_loss += -ret
                    wins_long += 1 if ret > 0 else 0
                    trades_long.append({
                        "timestamp": ts,
                        "entry": pos["entry"],
                        "exit": exit_price,
                        "ret": ret,
                        "position_value": pos["position_value"],
                        "exit_reason": "stop_global",
                        "symbol": symbol,
                        "type": "long"
                    })
                positions_long = []
                continue
            
            # Sorties par trailing stop pour chaque position LONG
            remaining = []
            for pos in positions_long:
                if (not np.isnan(pos["trailing"])) and (close <= pos["trailing"]):
                    ret = ((close / pos["entry"]) - 1.0) * leverage
                    commission_cost = pos["position_value"] * commission_rate * 2
                    ret -= commission_cost / pos["position_value"]
                    if ret < 0:
                        daily_loss += -ret
                    current_capital = max(1e-12, equity * 1000.0)
                    equity *= (1.0 + ret * (pos["position_value"] / current_capital))
                    wins_long += 1 if ret > 0 else 0
                    trades_long.append({
                        "timestamp": ts,
                        "entry": pos["entry"],
                        "exit": close,
                        "ret": ret,
                        "position_value": pos["position_value"],
                        "exit_reason": "trailing_stop",
                        "symbol": symbol,
                        "type": "long"
                    })
                else:
                    remaining.append(pos)
            positions_long = remaining
        
        # Sorties SHORT
        if len(positions_short) > 0:
            # Stop global dynamique
            current_capital = equity * 1000
            stop_global_euros = current_capital * stop_global
            
            # Vérifier le stop global dynamique
            if current_capital <= stop_global_euros:
                # Stop global touché, fermer toutes les positions
                for pos in positions_short:
                    ret = ((pos["entry"] / close) - 1.0) * leverage
                    commission_cost = pos["position_value"] * commission_rate * 2
                    ret -= commission_cost / pos["position_value"]
                    if ret < 0:
                        daily_loss += -ret
                    current_capital = max(1e-12, equity * 1000.0)
                    equity *= (1.0 + ret * (pos["position_value"] / current_capital))
                    wins_short += 1 if ret > 0 else 0
                    trades_short.append({
                        "timestamp": ts,
                        "entry": pos["entry"],
                        "exit": close,
                        "ret": ret,
                        "position_value": pos["position_value"],
                        "exit_reason": "stop_global",
                        "symbol": symbol,
                        "type": "short"
                    })
                positions_short = []
                continue
            
            remaining = []
            for pos in positions_short:
                if (not np.isnan(pos["trailing"])) and (close >= pos["trailing"]):
                    ret = ((pos["entry"] / close) - 1.0) * leverage
                    commission_cost = pos["position_value"] * commission_rate * 2
                    ret -= commission_cost / pos["position_value"]
                    if ret < 0:
                        daily_loss += -ret
                    current_capital = max(1e-12, equity * 1000.0)
                    equity *= (1.0 + ret * (pos["position_value"] / current_capital))
                    wins_short += 1 if ret > 0 else 0
                    trades_short.append({
                        "timestamp": ts,
                        "entry": pos["entry"],
                        "exit": close,
                        "ret": ret,
                        "position_value": pos["position_value"],
                        "exit_reason": "trailing_stop",
                        "symbol": symbol,
                        "type": "short"
                    })
                else:
                    remaining.append(pos)
            positions_short = remaining

    # Fin de période: clôturer toutes les positions restantes
    if len(positions_long) > 0:
        close = data["close"].iloc[-1]
        for pos in positions_long:
            ret = ((close / pos["entry"]) - 1.0) * leverage
            commission_cost = pos["position_value"] * commission_rate * 2
            ret -= commission_cost / pos["position_value"]
            if ret < 0:
                daily_loss += -ret
            current_capital = max(1e-12, equity * 1000.0)
            equity *= (1.0 + ret * (pos["position_value"] / current_capital))
            wins_long += 1 if ret > 0 else 0
            # Pondération du retour par la taille de position (ret_w)
            ret_w = ret * (pos["position_value"] / current_capital)
            trades_long.append({
                "timestamp": data.index[-1],
                "entry": pos["entry"],
                "exit": close,
                "ret": ret,
                "ret_w": ret_w,
                "position_value": pos["position_value"],
                "exit_reason": "eod",
                "symbol": symbol,
                "type": "long"
            })

    if len(positions_short) > 0:
        close = data["close"].iloc[-1]
        for pos in positions_short:
            ret = ((pos["entry"] / close) - 1.0) * leverage
            commission_cost = pos["position_value"] * commission_rate * 2
            ret -= commission_cost / pos["position_value"]
            if ret < 0:
                daily_loss += -ret
            current_capital = max(1e-12, equity * 1000.0)
            equity *= (1.0 + ret * (pos["position_value"] / current_capital))
            wins_short += 1 if ret > 0 else 0
            # Pondération du retour par la taille de position (ret_w)
            ret_w = ret * (pos["position_value"] / current_capital)
            trades_short.append({
                "timestamp": data.index[-1],
                "entry": pos["entry"],
                "exit": close,
                "ret": ret,
                "ret_w": ret_w,
                "position_value": pos["position_value"],
                "exit_reason": "eod",
                "symbol": symbol,
                "type": "short"
            })

    # metrics - combiner long et short
    all_trades = trades_long + trades_short
    returns = pd.Series([t["ret"] for t in all_trades], dtype=float)
    n_trades = len(all_trades)
    total_wins = wins_long + wins_short
    win_rate = (total_wins / n_trades) if n_trades > 0 else 0.0
    
    # Calculer le temps moyen en position
    def calculate_avg_time_in_position(trades_list):
        if not trades_list:
            return 0.0
        total_time = 0
        for trade in trades_list:
            if "timestamp" in trade and "entry" in trade:
                # Calculer la durée en heures (timeframe dynamique)
                if timeframe == "1h":
                    duration = 1
                elif timeframe == "2h":
                    duration = 2
                elif timeframe == "4h":
                    duration = 4
                elif timeframe == "1d":
                    duration = 24
                else:
                    duration = 2  # défaut 2h
                total_time += duration
        return total_time / len(trades_list) if trades_list else 0.0
    
    avg_time_long = calculate_avg_time_in_position(trades_long)
    avg_time_short = calculate_avg_time_in_position(trades_short)

    if n_trades > 0:
        # Utiliser le retour pondéré si disponible pour approcher l'équité
        returns_w = pd.Series([t.get("ret_w", t["ret"]) for t in all_trades], dtype=float)
        eq_curve = (1.0 + returns_w).cumprod()
        max_dd = (eq_curve.cummax() - eq_curve).max() / max(eq_curve.cummax().max(), 1e-12)
        # Borne de sécurité (≤ 100%)
        if not np.isfinite(max_dd):
            max_dd = 1.0
        else:
            max_dd = float(min(max_dd, 1.0))
        
        # Calculer l'exposant de Lyapunov
        lyapunov_exp = calculate_lyapunov_exponent(returns)
    else:
        max_dd = 0.0
        lyapunov_exp = 0.0

    start_ts = df.index[0] if len(df.index) else datetime.utcnow().astimezone(timezone.utc)
    end_ts = df.index[-1] if len(df.index) else start_ts
    years = (end_ts - start_ts).days / 365.25 if end_ts > start_ts else 0.0
    # Accepter les pertes ! Si equity négatif, CAGR = -100% (perte totale)
    if years > 0:
        if equity > 0:
            cagr = (equity ** (1/years) - 1.0)
        else:
            cagr = -1.0  # Perte de 100% = -100%
    else:
        cagr = 0.0

    if n_trades > 1 and returns.std(ddof=1) > 0:
        sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    else:
        sharpe = 0.0

    profit_factor = (returns[returns > 0].sum() / abs(returns[returns < 0].sum())) if (returns[returns < 0].sum() != 0) else float("inf")
    expectancy = returns.mean() if n_trades > 0 else 0.0

    # MÉTRIQUES AVANCÉES (Calmar, Sortino, VaR, Recovery Factor)
    if n_trades > 0 and max_dd > 0:
        # Calmar Ratio = CAGR / Max Drawdown
        calmar_ratio = cagr / max_dd if max_dd > 0 else 0.0
        
        # Sortino Ratio = Performance / Downside Deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std(ddof=1)
            sortino_ratio = returns.mean() / downside_deviation if downside_deviation > 0 else 0.0
        else:
            sortino_ratio = float("inf")  # Aucune perte = ratio infini
        
        # VaR (Value at Risk) - 95% confidence
        var_95 = np.percentile(returns, 5)  # 5ème percentile = 95% confidence
        
        # Recovery Factor = Profit Total / Max Drawdown
        total_profit = equity - 1.0
        recovery_factor = total_profit / max_dd if max_dd > 0 else 0.0
        
        # Temps de récupération estimé (en jours)
        if cagr > 0 and max_dd > 0:
            recovery_days = (max_dd / cagr) * 365.25
        else:
            recovery_days = float("inf")
    else:
        calmar_ratio = 0.0
        sortino_ratio = 0.0
        var_95 = 0.0
        recovery_factor = 0.0
        recovery_days = 0.0

    # Calcul des scores de volume et liquidité
    if n_trades > 0:
        # Analyser la qualité des volumes des trades
        volume_scores = []
        liquidity_scores = []
        
        for trade in all_trades:
            if "timestamp" in trade:
                # Trouver le volume au moment du trade
                trade_volume = data.loc[trade["timestamp"], "volume"] if trade["timestamp"] in data.index else 0
                trade_price = trade["entry"]
                volume_usdt = trade_volume * trade_price
                
                # Score de volume (0-1, 1 = excellent)
                if volume_usdt >= 1000000:  # > 1M USDT
                    volume_score = 1.0
                elif volume_usdt >= 100000:  # > 100k USDT
                    volume_score = 0.8
                elif volume_usdt >= 10000:  # > 10k USDT
                    volume_score = 0.6
                elif volume_usdt >= 1000:  # > 1k USDT
                    volume_score = 0.4
                else:
                    volume_score = 0.2
                
                volume_scores.append(volume_score)
                
                # Score de liquidité (0-1, 1 = excellent)
                position_value = 1000 * position_size  # Valeur de la position
                liquidity_ratio = position_value / volume_usdt if volume_usdt > 0 else 1.0
                liquidity_score = max(0, 1 - liquidity_ratio)  # Plus le ratio est bas, meilleur c'est
                liquidity_scores.append(liquidity_score)
        
        volume_quality_score = np.mean(volume_scores) if volume_scores else 0.0
        liquidity_score = np.mean(liquidity_scores) if liquidity_scores else 0.0
    else:
        volume_quality_score = 0.0
        liquidity_score = 0.0

    # Calcul des métriques de margin et liquidation
    margin_call_count = 0
    liquidation_count = 0
    min_equity_reached = 1.0
    
    # Calcul des métriques de latence et qualité d'exécution
    avg_execution_latency = total_execution_latency / execution_count if execution_count > 0 else 0.0
    execution_success_rate = (execution_count / (execution_count + consecutive_errors)) if (execution_count + consecutive_errors) > 0 else 0.0
    
    # Analyser tous les trades pour compter les margin calls et liquidations
    for trade in all_trades:
        if trade.get("exit_reason") == "margin_call":
            margin_call_count += 1
        elif trade.get("exit_reason") == "liquidation_forced":
            liquidation_count += 1
    
    # Trouver l'équité minimum atteinte
    if len(all_trades) > 0:
        # Simuler l'équité au fil du temps pour trouver le minimum
        current_equity = 1.0
        min_equity_reached = 1.0
        
        for trade in all_trades:
            # Simuler l'effet pondéré par la taille de position utilisée lors du trade
            pv = float(trade.get("position_value", 0.0))
            # Approche: pondérer par la taille de position relative à un capital de 1000€
            current_equity *= (1.0 + trade["ret"] * (pv / 1000.0))
            min_equity_reached = min(min_equity_reached, current_equity)

    metrics = {
        "equity_mult": float(equity),
        "CAGR": float(cagr),
        "sharpe_proxy": float(sharpe),
        "max_drawdown": float(max_dd),
        "trades": int(n_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else float("inf"),
        "expectancy": float(expectancy),
        "avg_time_long_hours": float(avg_time_long),
        "avg_time_short_hours": float(avg_time_short),
        "nb_trades_long": len(trades_long),
        "nb_trades_short": len(trades_short),
        "lyapunov_exponent": float(lyapunov_exp),  # Exposant de Lyapunov !
        "volume_quality": float(volume_quality_score) if 'volume_quality_score' in locals() else 0.0,  # Qualité des volumes
        "liquidity_score": float(liquidity_score) if 'liquidity_score' in locals() else 0.0,  # Score de liquidité
        "calmar_ratio": float(calmar_ratio),        # Calmar Ratio !
        "sortino_ratio": float(sortino_ratio) if np.isfinite(sortino_ratio) else float("inf"),  # Sortino Ratio !
        "var_95": float(var_95),                    # VaR 95% !
        "recovery_factor": float(recovery_factor),  # Recovery Factor !
        "recovery_days": float(recovery_days) if np.isfinite(recovery_days) else float("inf"),  # Temps de récupération !
        "margin_calls": int(margin_call_count) if 'margin_call_count' in locals() else 0,  # Nombre de margin calls !
        "liquidations": int(liquidation_count) if 'liquidation_count' in locals() else 0,  # Nombre de liquidations !
        "min_equity": float(min_equity_reached) if 'min_equity_reached' in locals() else 1.0,  # Équité minimum atteinte !
        "avg_execution_latency": float(avg_execution_latency) if 'avg_execution_latency' in locals() else 0.0,  # Latence moyenne d'exécution !
        "execution_success_rate": float(execution_success_rate) if 'execution_success_rate' in locals() else 0.0,  # Taux de succès d'exécution !
        "consecutive_errors": int(consecutive_errors) if 'consecutive_errors' in locals() else 0  # Erreurs consécutives !
    }
    return metrics

# ------------- Pipeline ------------- #

PROFILES = {
    "pipeline_web6": {
        "symbols": ["BTC/USDT", "ETH/USDT", "DOGE/USDT"],
        "timeframe": "2h",
        "years_back": 5,
        "loss_mult": 3.0,
        "ranges": {
            "tenkan": (1, 70),
            "kijun": (1, 70),
            "senkou_b": (1, 70),
            "shift": (1, 99),
            "atr_mult": (1.0, 14.0)  # ATR optimisé 1-14 !
        },
        "default_trials": 5000  # Recommandé pour robustesse
    }
}

def _load_local_csv_if_configured(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Optionally load a local CSV instead of remote fetch when env flag is set.

    USE_FUSED_H2=1 enables using data/BTC_FUSED_2h(_clean).csv for symbol BTC/USDT at 2h.
    Robustly parses timestamp and coerces OHLCV to numeric to avoid 100% NaN.
    """
    try:
        use_fused = os.environ.get("USE_FUSED_H2", "0") == "1"
    except Exception:
        use_fused = False
    if not use_fused:
        return None
    if symbol != "BTC/USDT" or timeframe != "2h":
        return None

    from pathlib import Path
    p_clean = Path("data/BTC_FUSED_2h_clean.csv")
    p_raw = Path("data/BTC_FUSED_2h.csv")
    p = p_clean if p_clean.exists() else p_raw
    if not p.exists():
        return None

    try:
        # Read without parse_dates first, normalize columns
        df = pd.read_csv(p)
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Detect timestamp column
        ts_candidates = [
            "timestamp", "date", "datetime", "time", "open_time", "ts"
        ]
        ts_col = next((c for c in ts_candidates if c in df.columns), None)
        if ts_col is None:
            return None

        # Parse timestamp
        ts_series = df[ts_col]
        if pd.api.types.is_numeric_dtype(ts_series):
            # Epoch seconds or ms
            # Heuristic: values > 10^11 -> ms
            multiplier = 1000 if ts_series.astype("int64").gt(10**11).any() else 1
            df[ts_col] = pd.to_datetime(df[ts_col] / multiplier, unit="s", utc=True)
        else:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

        df = df.set_index(ts_col).sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Standardize OHLCV columns to numeric
        name_map = {
            "open": "open", "high": "high", "low": "low", "close": "close",
            "volume": "volume", "vol": "volume", "qty": "volume", "quote_volume": "volume"
        }
        # Create expected columns if alternative names exist
        for alt, canonical in list(name_map.items()):
            if alt in df.columns and canonical not in df.columns:
                df[canonical] = df[alt]
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                # Missing required columns -> cannot proceed
                return None

        # Drop rows where all OHLC are NaN
        ohlc = df[["open", "high", "low", "close"]]
        df = df.loc[~ohlc.isna().all(axis=1)]

        return df
    except Exception:
        return None

def backtest_shared_portfolio(market_data: dict[str, pd.DataFrame], params_by_symbol: dict[str, dict], timeframe: str = "2h", record_curve: bool = False) -> dict:
    """Backtest multi-paires en parallèle avec capital commun et contrainte de fonds disponibles.

    - Chaque entrée consomme 1% du capital courant (equity * 1000 * 1%).
    - Pas de long et short simultanés sur un même symbole ; max 3 entrées par côté par symbole.
    - Levier, commissions, funding et rollover cohérents avec backtest_long_short.
    - La barre de progression est affichée sur la timeline concaténée de toutes les paires.
    """
    # Paramètres globaux (alignés sur backtest_long_short) — overrides via env
    try:
        _env_pos = os.getenv("POSITION_SIZE")
        _env_lev = os.getenv("LEVERAGE")
        _env_maxpos = os.getenv("MAX_POS_PER_SIDE")
    except Exception:
        _env_pos = None; _env_lev = None; _env_maxpos = None
    position_size = float(_env_pos) if _env_pos else 0.01
    leverage = float(_env_lev) if _env_lev else 10
    # Cap dur sécurité demandé: 1% et x10 max
    position_size = min(max(0.0, position_size), 0.01)
    leverage = min(max(1.0, leverage), 10.0)
    try:
        max_positions_per_side_cap = int(_env_maxpos) if _env_maxpos else 3
    except Exception:
        max_positions_per_side_cap = 3
    commission_rate = 0.001
    funding_rate = 0.0001
    funding_interval = 8
    rollover_cost = 0.0005
    rollover_interval_days = 30
    base_slippage = 0.0005
    stop_global = 0.80
    liquidation_threshold = 0.02
    maintenance_margin = 0.025

    # Préparer les données/indicateurs par symbole
    processed: dict[str, pd.DataFrame] = {}
    for sym, df in market_data.items():
        p = params_by_symbol.get(sym, {})
        tenkan = int(p.get("tenkan", 9))
        kijun = int(p.get("kijun", 26))
        senkou_b = int(p.get("senkou_b", 52))
        shift = int(p.get("shift", 26))
        df_sig = calculate_ichimoku(df.copy(), tenkan, kijun, senkou_b, shift)
        processed[sym] = df_sig

    # Timeline commune
    all_indexes = sorted(set().union(*[df.index for df in processed.values()]))
    total_steps = len(all_indexes)
    if total_steps == 0:
        return {"equity_mult": 1.0, "trades": 0}

    # Positions par symbole
    positions_long: dict[str, list] = {sym: [] for sym in processed}
    positions_short: dict[str, list] = {sym: [] for sym in processed}

    # Equity global
    equity = 1.0
    min_equity = equity
    min_equity_ts = None
    equity_points = []  # [(timestamp, equity)] si record_curve
    days_since_rollover = 0
    last_day = None

    # Journal des trades
    all_trades = []

    # Helpers
    def total_open_notional() -> float:
        total = 0.0
        for s in processed:
            total += sum(p["position_value"] for p in positions_long[s])
            total += sum(p["position_value"] for p in positions_short[s])
        return total

    def available_cash_eur() -> float:
        current_capital = equity * 1000.0
        return max(0.0, current_capital - total_open_notional())

    # Préparer pointeurs par symbole pour accéder au "next open"
    next_open_cache: dict[str, dict[pd.Timestamp, float]] = {}
    for sym, df in processed.items():
        # Map timestamp -> next open
        nxt = {}
        idx = df.index
        for i in range(len(idx)):
            ts = idx[i]
            if i + 1 < len(idx):
                nxt[ts] = float(df.iloc[i + 1]["open"]) if "open" in df.columns else float(df.iloc[i + 1]["close"])
            else:
                nxt[ts] = float(df.iloc[i]["close"])  # fallback: dernier close
        next_open_cache[sym] = nxt

    # Charger filtres Binance si dispo
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    binance_filters = _load_binance_filters(outputs_dir)
    
    def _binance_constraints_ok(symbol_ccxt: str, price: float, qty_notional: float) -> bool:
        sym_b = _symbol_to_binance(symbol_ccxt)
        f = binance_filters.get(sym_b)
        if not f:
            return qty_notional >= 5.0
        return (qty_notional >= float(f.get("minNotional", 5.0)))
    
    def _apply_binance_rounding(symbol_ccxt: str, price: float, qty: float) -> tuple[float, float]:
        sym_b = _symbol_to_binance(symbol_ccxt)
        f = binance_filters.get(sym_b)
        if not f:
            return float(price), float(qty)
        price_rounded = _round_to_step(price, float(f.get("tickSize", 0.01)))
        qty_rounded = _round_to_step(qty, float(f.get("stepSize", 0.001)))
        return price_rounded, qty_rounded

    # Boucle principale sur la timeline globale
    for step_idx, ts in enumerate(all_indexes, start=1):
        # Progression
        if step_idx % max(1, total_steps // 100) == 0 or step_idx == total_steps:
            progress = (step_idx / total_steps) * 100
            bar_len = 30
            filled = int(bar_len * step_idx // total_steps)
            bar = '█' * filled + '░' * (bar_len - filled)
            log(f"Portfolio partagé {step_idx:,}/{total_steps:,} ({progress:.1f}%) [{bar}] equity≈{equity:.3f}")

        # Rollover quotidien (mesuré par changement de jour)
        if last_day is None or ts.date() != last_day:
            last_day = ts.date()
            days_since_rollover += 1
            if days_since_rollover >= rollover_interval_days:
                open_notional = total_open_notional()
                if open_notional > 0:
                    current_capital = max(1e-12, equity * 1000.0)
                    cost = rollover_cost * (open_notional / current_capital)
                    equity -= cost
                    log(f"🔄 Rollover futures - Notional: {open_notional:.2f}€ - Coût_eq: {cost:.6f} - Nouvelle equity: {equity:.4f}")
                days_since_rollover = 0

        # Funding toutes les 8h (si un symbole possède cette bougie à cette heure)
        if ts.hour % funding_interval == 0:
            open_notional = total_open_notional()
            if open_notional > 0:
                current_capital = max(1e-12, equity * 1000.0)
                equity -= funding_rate * (open_notional / current_capital)
                if equity < min_equity:
                    min_equity = equity
                    min_equity_ts = ts

        # Sécurité liquidation/margin call sur equity global
        if equity <= liquidation_threshold:
            # Fermer tout au prix courant (close) pour tous les symboles ayant une bougie à ts
            for sym, df in processed.items():
                if ts not in df.index:
                    continue
                close = float(df.loc[ts, "close"]) if "close" in df.columns else float(df.loc[ts, "open"]) 
                # Longs
                if positions_long[sym]:
                    for pos in positions_long[sym]:
                        ret = ((close / pos["entry"]) - 1.0) * leverage
                        commission_cost = pos["position_value"] * commission_rate * 2
                        ret -= commission_cost / pos["position_value"]
                        current_capital = max(1e-12, equity * 1000.0)
                        weight = pos["position_value"] / current_capital
                        ret_w = ret * weight
                        pnl_eur = ret * pos["position_value"]
                        equity *= (1.0 + ret_w)
                        all_trades.append({"timestamp": ts, "entry_ts": pos.get("entry_ts", ts), "entry": pos["entry"], "exit": close, "ret": ret, "ret_w": ret_w, "pnl_eur": pnl_eur, "symbol": sym, "type": "long", "exit_reason": "liquidation_forced"})
                    positions_long[sym] = []
                # Shorts
                if positions_short[sym]:
                    for pos in positions_short[sym]:
                        ret = ((pos["entry"] / close) - 1.0) * leverage
                        commission_cost = pos["position_value"] * commission_rate * 2
                        ret -= commission_cost / pos["position_value"]
                        current_capital = max(1e-12, equity * 1000.0)
                        weight = pos["position_value"] / current_capital
                        ret_w = ret * weight
                        pnl_eur = ret * pos["position_value"]
                        equity *= (1.0 + ret_w)
                        all_trades.append({"timestamp": ts, "entry_ts": pos.get("entry_ts", ts), "entry": pos["entry"], "exit": close, "ret": ret, "ret_w": ret_w, "pnl_eur": pnl_eur, "symbol": sym, "type": "short", "exit_reason": "liquidation_forced"})
                    positions_short[sym] = []
            # Continuer l'analyse même après liquidation forcée
            if equity < min_equity:
                min_equity = equity
                min_equity_ts = ts
            continue
        elif equity <= maintenance_margin:
            log(f"⚠️ MARGIN CALL PORTFOLIO ! Equity: {equity:.4f} < {maintenance_margin:.4f}")
            for sym, df in processed.items():
                if ts not in df.index:
                    continue
                close = float(df.loc[ts, "close"]) if "close" in df.columns else float(df.loc[ts, "open"]) 
                if positions_long[sym]:
                    for pos in positions_long[sym]:
                        ret = ((close / pos["entry"]) - 1.0) * leverage
                        commission_cost = pos["position_value"] * commission_rate * 2
                        ret -= commission_cost / pos["position_value"]
                        current_capital = max(1e-12, equity * 1000.0)
                        weight = pos["position_value"] / current_capital
                        ret_w = ret * weight
                        pnl_eur = ret * pos["position_value"]
                        equity *= (1.0 + ret_w)
                        all_trades.append({"timestamp": ts, "entry_ts": pos.get("entry_ts", ts), "entry": pos["entry"], "exit": close, "ret": ret, "ret_w": ret_w, "pnl_eur": pnl_eur, "symbol": sym, "type": "long", "exit_reason": "margin_call"})
                    positions_long[sym] = []
                if positions_short[sym]:
                    for pos in positions_short[sym]:
                        ret = ((pos["entry"] / close) - 1.0) * leverage
                        commission_cost = pos["position_value"] * commission_rate * 2
                        ret -= commission_cost / pos["position_value"]
                        current_capital = max(1e-12, equity * 1000.0)
                        weight = pos["position_value"] / current_capital
                        ret_w = ret * weight
                        pnl_eur = ret * pos["position_value"]
                        equity *= (1.0 + ret_w)
                        all_trades.append({"timestamp": ts, "entry_ts": pos.get("entry_ts", ts), "entry": pos["entry"], "exit": close, "ret": ret, "ret_w": ret_w, "pnl_eur": pnl_eur, "symbol": sym, "type": "short", "exit_reason": "margin_call"})
                    positions_short[sym] = []
            # Continuer après margin call
            if equity < min_equity:
                min_equity = equity
                min_equity_ts = ts

        # Parcourir les symboles qui ont une bougie à ts
        for sym, df in processed.items():
            if ts not in df.index:
                continue
            row = df.loc[ts]
            close = float(row["close"]) if "close" in df.columns else float(row["open"]) 
            volume_usdt = float(row["volume"] * close) if ("volume" in df.columns and "close" in df.columns) else 0.0
            atr_val = float(row["ATR"]) if "ATR" in df.columns and pd.notna(row["ATR"]) else float("nan")
            p = params_by_symbol.get(sym, {})
            atr_mult = float(p.get("atr_mult", 3.0))

            # Trailing updates
            if positions_long[sym]:
                atr_stop_mult = atr_mult * 1.5
                for pos in positions_long[sym]:
                    new_trailing = close - (atr_val * atr_stop_mult) if not np.isnan(atr_val) else pos["trailing"]
                    if not np.isnan(new_trailing):
                        pos["trailing"] = max(pos["trailing"], new_trailing) if not np.isnan(pos["trailing"]) else new_trailing
            if positions_short[sym]:
                atr_stop_mult = atr_mult * 1.5
                for pos in positions_short[sym]:
                    new_trailing = close + (atr_val * atr_stop_mult) if not np.isnan(atr_val) else pos["trailing"]
                    if not np.isnan(new_trailing):
                        pos["trailing"] = min(pos["trailing"], new_trailing) if not np.isnan(pos["trailing"]) else new_trailing

            # Sorties LONG (trailing / stop global)
            if positions_long[sym]:
                current_capital = equity * 1000.0
                stop_global_eur = current_capital * stop_global
                if current_capital <= stop_global_eur:
                    # Stop global du portefeuille -> fermer
                    for pos in positions_long[sym]:
                        ret = ((close / pos["entry"]) - 1.0) * leverage
                        commission_cost = pos["position_value"] * commission_rate * 2
                        ret -= commission_cost / pos["position_value"]
                        current_capital = max(1e-12, equity * 1000.0)
                        equity *= (1.0 + ret * (pos["position_value"] / current_capital))
                        all_trades.append({"timestamp": ts, "entry_ts": pos.get("entry_ts", ts), "entry": pos["entry"], "exit": close, "ret": ret, "symbol": sym, "type": "long", "exit_reason": "stop_global"})
                    positions_long[sym] = []
                else:
                    remaining = []
                    for pos in positions_long[sym]:
                        if (not np.isnan(pos["trailing"])) and (close <= pos["trailing"]):
                            ret = ((close / pos["entry"]) - 1.0) * leverage
                            commission_cost = pos["position_value"] * commission_rate * 2
                            ret -= commission_cost / pos["position_value"]
                            current_capital = max(1e-12, equity * 1000.0)
                            weight = pos["position_value"] / current_capital
                            ret_w = ret * weight
                            pnl_eur = ret * pos["position_value"]
                            equity *= (1.0 + ret_w)
                            all_trades.append({"timestamp": ts, "entry_ts": pos.get("entry_ts", ts), "entry": pos["entry"], "exit": close, "ret": ret, "ret_w": ret_w, "pnl_eur": pnl_eur, "symbol": sym, "type": "long", "exit_reason": "trailing_stop"})
                        else:
                            remaining.append(pos)
                    positions_long[sym] = remaining

            # Sorties SHORT
            if positions_short[sym]:
                current_capital = equity * 1000.0
                stop_global_eur = current_capital * stop_global
                if current_capital <= stop_global_eur:
                    for pos in positions_short[sym]:
                        ret = ((pos["entry"] / close) - 1.0) * leverage
                        commission_cost = pos["position_value"] * commission_rate * 2
                        ret -= commission_cost / pos["position_value"]
                        current_capital = max(1e-12, equity * 1000.0)
                        equity *= (1.0 + ret * (pos["position_value"] / current_capital))
                        all_trades.append({"timestamp": ts, "entry_ts": pos.get("entry_ts", ts), "entry": pos["entry"], "exit": close, "ret": ret, "symbol": sym, "type": "short", "exit_reason": "stop_global"})
                    positions_short[sym] = []
                else:
                    remaining = []
                    for pos in positions_short[sym]:
                        if (not np.isnan(pos["trailing"])) and (close >= pos["trailing"]):
                            ret = ((pos["entry"] / close) - 1.0) * leverage
                            commission_cost = pos["position_value"] * commission_rate * 2
                            ret -= commission_cost / pos["position_value"]
                            current_capital = max(1e-12, equity * 1000.0)
                            weight = pos["position_value"] / current_capital
                            ret_w = ret * weight
                            pnl_eur = ret * pos["position_value"]
                            equity *= (1.0 + ret_w)
                            all_trades.append({"timestamp": ts, "entry_ts": pos.get("entry_ts", ts), "entry": pos["entry"], "exit": close, "ret": ret, "ret_w": ret_w, "pnl_eur": pnl_eur, "symbol": sym, "type": "short", "exit_reason": "trailing_stop"})
                        else:
                            remaining.append(pos)
                    positions_short[sym] = remaining

            # Opposite signal closures
            if bool(row.get("signal_long", False)) and positions_short[sym]:
                to_close = positions_short[sym]; positions_short[sym] = []
                for pos in to_close:
                    ret = ((pos["entry"] / close) - 1.0) * leverage
                    commission_cost = pos["position_value"] * commission_rate * 2
                    ret -= commission_cost / pos["position_value"]
                    current_capital = max(1e-12, equity * 1000.0)
                    weight = pos["position_value"] / current_capital
                    ret_w = ret * weight
                    pnl_eur = ret * pos["position_value"]
                    equity *= (1.0 + ret_w)
                    all_trades.append({"timestamp": ts, "entry_ts": pos.get("entry_ts", ts), "entry": pos["entry"], "exit": close, "ret": ret, "ret_w": ret_w, "pnl_eur": pnl_eur, "symbol": sym, "type": "short", "exit_reason": "opposite_signal"})

            if bool(row.get("signal_short", False)) and positions_long[sym]:
                to_close = positions_long[sym]; positions_long[sym] = []
                for pos in to_close:
                    ret = ((close / pos["entry"]) - 1.0) * leverage
                    commission_cost = pos["position_value"] * commission_rate * 2
                    ret -= commission_cost / pos["position_value"]
                    current_capital = max(1e-12, equity * 1000.0)
                    weight = pos["position_value"] / current_capital
                    ret_w = ret * weight
                    pnl_eur = ret * pos["position_value"]
                    equity *= (1.0 + ret_w)
                    all_trades.append({"timestamp": ts, "entry_ts": pos.get("entry_ts", ts), "entry": pos["entry"], "exit": close, "ret": ret, "ret_w": ret_w, "pnl_eur": pnl_eur, "symbol": sym, "type": "long", "exit_reason": "opposite_signal"})

            # Entrées LONG
            if bool(row.get("signal_long", False)) and not positions_short[sym]:
                # Respecter le cap configurable de positions par côté
                if len(positions_long[sym]) < max_positions_per_side_cap:
                    cash = available_cash_eur()
                    current_capital = equity * 1000.0
                    position_value = current_capital * position_size
                    if cash >= position_value and 5 <= position_value <= 1_000_000:
                        dynamic_slippage = calculate_dynamic_slippage(volume_usdt, position_value, base_slippage)
                        raw_entry = next_open_cache[sym].get(ts, close) * (1 + dynamic_slippage)
                        # Convertir notional en qty, puis appliquer arrondis Binance
                        qty = position_value / max(raw_entry, 1e-12)
                        entry_price, qty = _apply_binance_rounding(sym, raw_entry, qty)
                        # Recalcule notional et valider minNotional
                        notional = entry_price * qty
                        if not _binance_constraints_ok(sym, entry_price, notional):
                            # impossible de respecter filtres → ignorer l'entrée
                            pass
                        else:
                            atr_stop_mult = atr_mult * 2.0
                            trailing = entry_price - (atr_val * atr_stop_mult) if not np.isnan(atr_val) else np.nan
                            positions_long[sym].append({"entry": entry_price, "trailing": trailing, "position_value": notional, "entry_ts": ts})
                            log(f"📈 Entrée LONG {sym} - Prix: {entry_price:.4f}, Notional: {notional:.2f}")
                        atr_stop_mult = atr_mult * 2.0
                        trailing = entry_price - (atr_val * atr_stop_mult) if not np.isnan(atr_val) else np.nan
                        positions_long[sym].append({"entry": entry_price, "trailing": trailing, "position_value": position_value, "entry_ts": ts})
                        log(f"📈 Entrée LONG {sym} - Prix: {entry_price:.4f}, Slippage: {dynamic_slippage*100:.3f}%")

            # Entrées SHORT
            if bool(row.get("signal_short", False)) and not positions_long[sym]:
                if len(positions_short[sym]) < max_positions_per_side_cap:
                    cash = available_cash_eur()
                    current_capital = equity * 1000.0
                    position_value = current_capital * position_size
                    if cash >= position_value and 5 <= position_value <= 1_000_000:
                        dynamic_slippage = calculate_dynamic_slippage(volume_usdt, position_value, base_slippage)
                        raw_entry = next_open_cache[sym].get(ts, close) * (1 - dynamic_slippage)
                        qty = position_value / max(raw_entry, 1e-12)
                        entry_price, qty = _apply_binance_rounding(sym, raw_entry, qty)
                        notional = entry_price * qty
                        if not _binance_constraints_ok(sym, entry_price, notional):
                            pass
                        else:
                            atr_stop_mult = atr_mult * 2.0
                            trailing = entry_price + (atr_val * atr_stop_mult) if not np.isnan(atr_val) else np.nan
                            positions_short[sym].append({"entry": entry_price, "trailing": trailing, "position_value": notional, "entry_ts": ts})
                            log(f"📉 Entrée SHORT {sym} - Prix: {entry_price:.4f}, Notional: {notional:.2f}")

        # Enregistrement courbe + mise à jour du minimum d'equity en fin d'itération
        if record_curve:
            equity_points.append((ts, float(equity)))
        if equity < min_equity:
            min_equity = equity
            min_equity_ts = ts

    # Clôture finale de toutes les positions au dernier prix disponible de chaque symbole
    for sym, df in processed.items():
        if len(df.index) == 0:
            continue
        close = float(df["close"].iloc[-1]) if "close" in df.columns else float(df["open"].iloc[-1])
        if positions_long[sym]:
            for pos in positions_long[sym]:
                ret = ((close / pos["entry"]) - 1.0) * leverage
                commission_cost = pos["position_value"] * commission_rate * 2
                ret -= commission_cost / pos["position_value"]
                current_capital = max(1e-12, equity * 1000.0)
                weight = pos["position_value"] / current_capital
                ret_w = ret * weight
                pnl_eur = ret * pos["position_value"]
                equity *= (1.0 + ret_w)
                all_trades.append({"timestamp": df.index[-1], "entry_ts": pos.get("entry_ts", df.index[-1]), "entry": pos["entry"], "exit": close, "ret": ret, "ret_w": ret_w, "pnl_eur": pnl_eur, "symbol": sym, "type": "long", "exit_reason": "eod"})
        if positions_short[sym]:
            for pos in positions_short[sym]:
                ret = ((pos["entry"] / close) - 1.0) * leverage
                commission_cost = pos["position_value"] * commission_rate * 2
                ret -= commission_cost / pos["position_value"]
                current_capital = max(1e-12, equity * 1000.0)
                weight = pos["position_value"] / current_capital
                ret_w = ret * weight
                pnl_eur = ret * pos["position_value"]
                equity *= (1.0 + ret_w)
                all_trades.append({"timestamp": df.index[-1], "entry_ts": pos.get("entry_ts", df.index[-1]), "entry": pos["entry"], "exit": close, "ret": ret, "ret_w": ret_w, "pnl_eur": pnl_eur, "symbol": sym, "type": "short", "exit_reason": "eod"})

    # Mise à jour finale du minimum d'equity après clôture
    if record_curve and len(equity_points) == 0 and total_steps > 0:
        # Si rien n'a été enregistré (cas extrême), enregistrer dernier point
        equity_points.append((all_indexes[-1], float(equity)))
    if equity < min_equity:
        min_equity = equity
        min_equity_ts = all_indexes[-1] if total_steps > 0 else None

    # Calcul des métriques portefeuille (retours pondérés si dispo)
    returns = pd.Series([t.get("ret_w", t["ret"]) for t in all_trades], dtype=float)
    n_trades = int(len(all_trades))
    start_ts = min(df.index[0] for df in processed.values() if len(df.index)) if processed else datetime.utcnow().astimezone(timezone.utc)
    end_ts = max(df.index[-1] for df in processed.values() if len(df.index)) if processed else start_ts
    years = (end_ts - start_ts).days / 365.25 if end_ts > start_ts else 0.0
    if years > 0:
        cagr = (equity ** (1/years) - 1.0) if equity > 0 else -1.0
    else:
        cagr = 0.0
    if n_trades > 1 and returns.std(ddof=1) > 0:
        sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    else:
        sharpe = 0.0
    if n_trades > 0:
        eq_curve = (1.0 + returns).cumprod()
        max_dd = (eq_curve.cummax() - eq_curve).max() / max(eq_curve.cummax().max(), 1e-12)
        # Borne de sécurité (≤100%)
        if not np.isfinite(max_dd):
            max_dd = 1.0
        else:
            max_dd = float(min(max_dd, 1.0))
    else:
        max_dd = 0.0

    out = {
        "equity_mult": float(equity),
        "CAGR": float(cagr),
        "sharpe_proxy": float(sharpe),
        "max_drawdown": float(max_dd),
        "trades": n_trades,
        "min_equity": float(min_equity),
        # Réglages de backtest (pour reporting)
        "position_size": float(position_size),
        "leverage": float(leverage),
        "max_positions_per_side": int(max_positions_per_side_cap),
    }
    # Per-symbol contributions (P&L in EUR and max DD indexed to portfolio equity)
    try:
        df_tr = pd.DataFrame(all_trades) if len(all_trades) > 0 else pd.DataFrame()
        if not df_tr.empty and 'symbol' in df_tr.columns:
            per_symbol: dict[str, dict] = {}
            for sym in sorted(df_tr['symbol'].dropna().unique()):
                sub = df_tr[df_tr['symbol'] == sym]
                pnl_eur = float(sub['pnl_eur'].sum()) if 'pnl_eur' in sub.columns else 0.0
                if 'ret_w' in sub.columns and len(sub) > 0:
                    eq_contrib = (1.0 + sub['ret_w'].astype(float)).cumprod()
                    max_dd_sym = float(((eq_contrib.cummax() - eq_contrib).max()) / max(eq_contrib.cummax().max(), 1e-12))
                else:
                    max_dd_sym = 0.0
                per_symbol[sym] = {"pnl_eur": pnl_eur, "max_dd_indexed": max_dd_sym}
            out["per_symbol"] = per_symbol
    except Exception:
        pass
    if min_equity_ts is not None:
        try:
            out["min_equity_ts"] = pd.Timestamp(min_equity_ts).isoformat()
        except Exception:
            pass
    if record_curve:
        try:
            out["equity_curve"] = [
                {"timestamp": pd.Timestamp(ts).isoformat(), "equity_mult": float(eq)} for ts, eq in equity_points
            ]
        except Exception:
            # fallback simple
            out["equity_curve"] = [(str(ts), float(eq)) for ts, eq in equity_points]
        # Include trades detail for plotting
        try:
            trades_serialized = []
            for t in all_trades:
                trades_serialized.append({
                    "timestamp": pd.Timestamp(t["timestamp"]).isoformat(),
                    "entry_ts": pd.Timestamp(t.get("entry_ts", t["timestamp"])).isoformat(),
                    "entry": float(t.get("entry", float("nan"))),
                    "exit": float(t.get("exit", float("nan"))),
                    "ret": float(t.get("ret", float("nan"))),
                    "symbol": str(t.get("symbol", "")),
                    "type": str(t.get("type", "")),
                    "exit_reason": str(t.get("exit_reason", "")),
                })
            out["trades_detail"] = trades_serialized
        except Exception:
            pass
    return out

def run_profile(profile_name, trials=0, seed=None, out_dir="outputs", use_cache=True, use_genetic=True, fixed_params=None, baseline_map=None, loss_mult=None):
    if profile_name not in PROFILES:
        raise ValueError(f"Profil inconnu: {profile_name}")

    cfg = PROFILES[profile_name]
    os.makedirs(out_dir, exist_ok=True)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    symbols = cfg["symbols"]
    timeframe = cfg["timeframe"]
    years_back = cfg["years_back"]
    r = cfg["ranges"]
    trials = trials or cfg["default_trials"]
    loss_mult = loss_mult if loss_mult is not None else cfg.get("loss_mult", 3.0)

    # période
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=int(365.25 * years_back))
    since_ms = utc_ms(start_dt)
    until_ms = utc_ms(end_dt)

    # exchange
    ex = ccxt.binance({"enableRateLimit": True})

    # fetch data pour chaque symbole
    market_data = {}
    for sym in symbols:
        local_df = _load_local_csv_if_configured(sym, timeframe)
        if local_df is not None:
            log(f"Utilisation CSV local (fused) pour {sym} {timeframe} — historique complet")
            df = local_df
        else:
            log(f"Téléchargement {sym} {timeframe} sur ~{years_back} ans…")
            df = fetch_ohlcv_range(ex, sym, timeframe, since_ms, until_ms, cache_dir="data", use_cache=use_cache)
        if df.empty:
            log(f"⚠️  Pas de données pour {sym}.")
            continue
        market_data[sym] = df

    if not market_data:
        raise RuntimeError("Aucune donnée chargée.")

    # Charger un baseline plus récent si présent dans outputs (fallback: baselines.json via CLI)
    if baseline_map is None:
        try:
            # Cherche un best_params_per_symbol_* récent
            outs = sorted([p for p in os.listdir(out_dir) if p.startswith('best_params_per_symbol_') and p.endswith('.json')], reverse=True)
            if outs:
                with open(os.path.join(out_dir, outs[0]), 'r', encoding='utf-8') as f:
                    import json as _json
                    baseline_map = _json.load(f)
        except Exception:
            pass

    # Si des trials sont demandés, utiliser l'optimisation Optuna par symbole (met à jour LIVE_REPORT)
    if trials and trials > 0:
        optuna_optimize_profile_per_symbol(
            profile_name=profile_name,
            n_trials=int(trials),
            seed=seed,
            out_dir=out_dir,
            use_cache=use_cache,
            jobs=1,
            fast_ratio=1.0,
            baseline_map=baseline_map,
        )
        return

    all_rows = []
    
    # Mode paramètres fixes: exécuter un seul run avec les paramètres fournis (pas de génétique, pas d'aléatoire)
    if fixed_params is not None:
        log(f"Run paramètres fixes: {fixed_params}")
        total = len(market_data)
        done = 0
        for sym, df in market_data.items():
            m = backtest_long_short(
                df,
                int(fixed_params.get("tenkan", 9)),
                int(fixed_params.get("kijun", 26)),
                int(fixed_params.get("senkou_b", 52)),
                int(fixed_params.get("shift", 26)),
                float(fixed_params.get("atr_mult", 3.0)),
                loss_mult=loss_mult,
                symbol=sym,
                timeframe=timeframe,
            )
            m.update({
                "trial": 1,
                "generation": 0,
                "trader_id": 0,
                "tenkan": int(fixed_params.get("tenkan", 9)),
                "kijun": int(fixed_params.get("kijun", 26)),
                "senkou_b": int(fixed_params.get("senkou_b", 52)),
                "shift": int(fixed_params.get("shift", 26)),
                "atr_mult": float(fixed_params.get("atr_mult", 3.0)),
                "symbol": sym,
            })
            all_rows.append(m)
            done += 1
            progress = (done / total) * 100
            bar_length = 30
            filled_length = int(bar_length * done // total)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            log(f"Fixed run {done}/{total} ({progress:.1f}%) [{bar}] {sym} equity≈{m.get('equity_mult', float('nan')):.3f}")
        # Skip the rest; export after this block
        
        if not all_rows:
            log("No results generated in fixed-params mode.")
            return
        
        df_runs = pd.DataFrame(all_rows)
        ts_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        runs_path = os.path.join(out_dir, f"runs_{profile_name}_{ts_label}.csv")
        df_runs.to_csv(runs_path, index=False)
        log(f"Export: {runs_path}")
        
        # Agrégation simple comme recherche aléatoire
        agg = df_runs.groupby(["trial", "tenkan", "kijun", "senkou_b", "shift", "atr_mult"]).agg({
            "equity_mult": "mean",
            "CAGR": "mean",
            "sharpe_proxy": "mean",
            "max_drawdown": "mean",
            "trades": "mean",
            "win_rate": "mean",
            "avg_time_long_hours": "mean",
            "avg_time_short_hours": "mean",
            "nb_trades_long": "mean",
            "nb_trades_short": "mean",
            "lyapunov_exponent": "mean",
            "volume_quality": "mean",
            "liquidity_score": "mean",
            "calmar_ratio": "mean",
            "sortino_ratio": "mean",
            "var_95": "mean",
            "recovery_factor": "mean",
            "recovery_days": "mean",
            "margin_calls": "sum",
            "liquidations": "sum",
            "min_equity": "min",
            "avg_execution_latency": "mean",
            "execution_success_rate": "mean",
            "consecutive_errors": "max",
        }).round(4)
        agg.columns = ["avg_equity", "avg_CAGR", "avg_sharpe", "avg_mdd", "avg_trades", "avg_winrate",
                       "avg_time_long", "avg_time_short", "nb_trades_long", "nb_trades_short", "avg_lyapunov",
                       "avg_volume_quality", "avg_liquidity_score", "avg_calmar", "avg_sortino", "avg_var_95",
                       "avg_recovery_factor", "avg_recovery_days", "total_margin_calls", "total_liquidations", "min_equity_reached",
                       "avg_execution_latency", "avg_execution_success_rate", "max_consecutive_errors"]
        agg["total_profit_euros"] = (agg["avg_equity"] - 1.0) * 1000.0
        top_path = os.path.join(out_dir, f"top_{profile_name}_{ts_label}.csv")
        agg.sort_values(["avg_lyapunov", "avg_equity"], ascending=[True, False]).head(30).to_csv(top_path, index=False)
        log(f"Export: {top_path}")
        
        symbol_stats = df_runs.groupby("symbol").agg({
            "equity_mult": ["count", "mean", "sum"],
            "trades": "sum",
            "win_rate": "mean",
        }).round(4)
        symbol_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in symbol_stats.columns.values]
        symbol_stats = symbol_stats.rename(columns={
            "equity_mult_count": "nb_backtest_runs",
            "equity_mult_mean": "avg_equity_mult_per_run",
            "equity_mult_sum": "total_equity_mult_across_runs",
            "trades_sum": "total_trades_across_runs",
            "win_rate_mean": "avg_winrate_per_run",
        })
        symbol_stats["total_profit_euros_across_runs"] = (symbol_stats["total_equity_mult_across_runs"] - symbol_stats["nb_backtest_runs"]) * 1000
        symbol_stats["avg_trades_per_run"] = (symbol_stats["total_trades_across_runs"] / symbol_stats["nb_backtest_runs"]).round(2)
        symbol_path = os.path.join(out_dir, f"symbol_stats_{profile_name}_{ts_label}.csv")
        symbol_stats.to_csv(symbol_path)
        log("Statistiques par paire:")
        log(f"   - Chaque paire a été testée sur {symbol_stats['nb_backtest_runs'].iloc[0]} backtests individuels")
        log(f"   - Résultats détaillés: {symbol_stats}")
        return
    
    if use_genetic:
        # ========== ALGORITHME GÉNÉTIQUE ICHIMOKU ========== #
        log(f"🧬 DÉMARRAGE DE L'ÉVOLUTION GÉNÉTIQUE ICHIMOKU !")
        log(f"🎯 Population: 100 traders, Générations: {trials//100}")
        
        # Créer la population initiale
        population = create_initial_population(population_size=100, ranges=r)
        log(f"👥 Population initiale créée: {len(population)} traders")
        
        # Configuration de l'évolution
        generations = max(1, trials // 100)  # Minimum 1 génération, 5000 trials = 50 générations
        elite_size = max(2, len(population) // 5)  # Garder les 20% meilleurs, minimum 2
        
        log(f"🚀 Évolution sur {generations} générations...")
        
        for generation in range(generations):
            log(f"🧬 GÉNÉRATION {generation + 1}/{generations}")
            
            # Évaluer chaque trader de la population
            for i, trader in enumerate(population):
                try:
                    # Backtest sur tous les symboles
                    sym_metrics = []
                    for sym, df in market_data.items():
                        params = trader.get_params()
                        m = backtest_long_short(df,
                            params["tenkan"],
                            params["kijun"],
                            params["senkou_b"],
                            params["shift"],
                            params["atr_mult"],
                            loss_mult=loss_mult,
                            symbol=sym,
                            timeframe=timeframe
                        )
                        
                        # Ajouter les métriques du symbole
                        m.update({
                            "trial": generation * 100 + i + 1,
                            "generation": generation + 1,
                            "trader_id": i + 1,
                            "tenkan": params["tenkan"],
                            "kijun": params["kijun"],
                            "senkou_b": params["senkou_b"],
                            "shift": params["shift"],
                            "atr_mult": params["atr_mult"],
                            "symbol": sym
                        })
                        
                        sym_metrics.append(m["equity_mult"])
                        all_rows.append(m)
                    
                    # Calculer le fitness du trader
                    if sym_metrics:
                        avg_equity = np.mean(sym_metrics)
                        # Créer un résultat synthétique pour le fitness
                        synthetic_results = {
                            "equity_mult": avg_equity,
                            "lyapunov_exponent": np.mean([r.get("lyapunov_exponent", 0) for r in all_rows[-len(sym_metrics):]]),
                            "win_rate": np.mean([r.get("win_rate", 0.5) for r in all_rows[-len(sym_metrics):]]),
                            "calmar_ratio": np.mean([r.get("calmar_ratio", 0) for r in all_rows[-len(sym_metrics):]])
                        }
                        trader.fitness = calculate_fitness_score(synthetic_results)
                        trader.performance_history.append(avg_equity)
                    
                except Exception as e:
                    log(f"⚠️  Erreur trader {i+1} génération {generation+1}: {e}")
                    trader.fitness = 0.0
                    continue
            
            # Afficher les meilleurs de cette génération
            population.sort(key=lambda x: x.fitness, reverse=True)
            best_trader = population[0]
            log(f"🏆 Meilleur trader G{generation+1}: Fitness={best_trader.fitness:.4f}, "
                f"Params={best_trader.get_params()}, "
                f"Equity≈{best_trader.performance_history[-1]:.3f}")
            
            # Évolution vers la génération suivante (sauf la dernière)
            if generation < generations - 1:
                population = evolve_population(population, elite_size=elite_size)
                log(f"🔄 Population évoluée vers la génération {generation + 2}")
            
            # Progress bar
            progress = ((generation + 1) / generations) * 100
            bar_length = 30
            filled_length = int(bar_length * (generation + 1) // generations)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            avg_fitness = np.mean([t.fitness for t in population])
            log(f"Génération {generation+1:,}/{generations:,} ({progress:.1f}%) [{bar}] avg_fitness≈{avg_fitness:.4f}")
        
        log(f"🎉 ÉVOLUTION GÉNÉTIQUE TERMINÉE ! {generations} générations évoluées !")
        
    else:
        # ========== RECHERCHE ALÉATOIRE CLASSIQUE ========== #
        log(f"🚀 Démarrage de {trials:,} essais aléatoires sur {len(market_data)} symboles...")
        log(f"📊 Période: {years_back} ans, Timeframe: {timeframe}")
        
        for trial in range(1, trials + 1):
            # Paramètres aléatoires
            tenkan = random.randint(r["tenkan"][0], r["tenkan"][1])
            kijun = random.randint(r["kijun"][0], r["kijun"][1])
            sen_b = random.randint(r["senkou_b"][0], r["senkou_b"][1])
            shift = random.randint(r["shift"][0], r["shift"][1])
            atr_mult = round(random.uniform(r["atr_mult"][0], r["atr_mult"][1]), 1)
            
            sym_metrics = []
            
            # Backtest sur chaque symbole
            for sym, df in market_data.items():
                try:
                    m = backtest_long_short(df, tenkan, kijun, sen_b, shift, atr_mult, loss_mult=loss_mult, symbol=sym, timeframe=timeframe)
                    m.update({
                        "trial": trial,
                        "generation": 0,  # Pas de génération pour la recherche aléatoire
                        "trader_id": 0,
                        "tenkan": tenkan,
                        "kijun": kijun,
                        "senkou_b": sen_b,
                        "shift": shift,
                        "atr_mult": atr_mult,
                        "symbol": sym
                    })
                    sym_metrics.append(m["equity_mult"])
                    all_rows.append(m)
                except Exception as e:
                    log(f"⚠️  Erreur trial {trial} {sym}: {e}")
                    continue
            
            # Progress bar
            if trial % max(1, trials // 100) == 0:
                progress = (trial / trials) * 100
                bar_length = 30
                filled_length = int(bar_length * trial // trials)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                avg_equity = float(np.mean(sym_metrics)) if sym_metrics else float("nan")
                log(f"Trial {trial:,}/{trials:,} ({progress:.1f}%) [{bar}] avg_equity≈{avg_equity:.3f}")

    if not all_rows:
        log("❌ Aucun résultat généré.")
        return

    # DataFrame des résultats
    df_runs = pd.DataFrame(all_rows)
    
    # Export complet
    ts_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    runs_path = os.path.join(out_dir, f"runs_{profile_name}_{ts_label}.csv")
    df_runs.to_csv(runs_path, index=False)
    log(f"✅ Export: {runs_path}")

    # Top 30 par equity moyen
    if use_genetic:
        # Agrégation pour l'algorithme génétique
        agg = df_runs.groupby(["generation", "trader_id", "tenkan", "kijun", "senkou_b", "shift", "atr_mult"]).agg({
            "equity_mult": "mean",
            "CAGR": "mean",
            "sharpe_proxy": "mean",
            "max_drawdown": "mean",
            "trades": "mean",
            "win_rate": "mean",
            "avg_time_long_hours": "mean",
            "avg_time_short_hours": "mean",
            "nb_trades_long": "mean",
            "nb_trades_short": "mean",
            "lyapunov_exponent": "mean",
            "volume_quality": "mean",
            "liquidity_score": "mean",
            "calmar_ratio": "mean",
            "sortino_ratio": "mean",
            "var_95": "mean",
            "recovery_factor": "mean",
            "recovery_days": "mean",
            "margin_calls": "sum",
            "liquidations": "sum",
            "min_equity": "min",
            "avg_execution_latency": "mean",
            "execution_success_rate": "mean",
            "consecutive_errors": "max"
        }).round(4)
    else:
        # Agrégation pour la recherche aléatoire classique
        agg = df_runs.groupby(["trial", "tenkan", "kijun", "senkou_b", "shift", "atr_mult"]).agg({
        "equity_mult": "mean",
        "CAGR": "mean",
        "sharpe_proxy": "mean",
        "max_drawdown": "mean",
        "trades": "mean",
        "win_rate": "mean",
        "avg_time_long_hours": "mean",
        "avg_time_short_hours": "mean",
        "nb_trades_long": "mean",
        "nb_trades_short": "mean",
        "lyapunov_exponent": "mean",  # Lyapunov ajouté !
        "volume_quality": "mean",      # Qualité des volumes !
        "liquidity_score": "mean",    # Score de liquidité !
        "calmar_ratio": "mean",       # Calmar Ratio !
        "sortino_ratio": "mean",      # Sortino Ratio !
        "var_95": "mean",             # VaR 95% !
        "recovery_factor": "mean",    # Recovery Factor !
        "recovery_days": "mean",      # Temps de récupération !
        "margin_calls": "sum",        # Total des margin calls !
        "liquidations": "sum",        # Total des liquidations !
        "min_equity": "min",          # Équité minimum atteinte !
        "avg_execution_latency": "mean",  # Latence moyenne d'exécution !
        "execution_success_rate": "mean",  # Taux de succès d'exécution !
        "consecutive_errors": "max"        # Erreurs consécutives max !
    }).round(4)
    
    agg.columns = ["avg_equity", "avg_CAGR", "avg_sharpe", "avg_mdd", "avg_trades", "avg_winrate", 
                   "avg_time_long", "avg_time_short", "nb_trades_long", "nb_trades_short", "avg_lyapunov",
                   "avg_volume_quality", "avg_liquidity_score", "avg_calmar", "avg_sortino", "avg_var_95",
                   "avg_recovery_factor", "avg_recovery_days", "total_margin_calls", "total_liquidations", "min_equity_reached",
                   "avg_execution_latency", "avg_execution_success_rate", "max_consecutive_errors"]
    
    # Ajouter le profit en euros (capital initial = 1000€)
    agg["total_profit_euros"] = (agg["avg_equity"] - 1.0) * 1000.0
    
    # Ajouter l'ATR moyen pour chaque combinaison
    agg["avg_atr_mult"] = df_runs.groupby(["trial", "tenkan", "kijun", "senkou_b", "shift", "atr_mult"])["atr_mult"].mean()
    top_path = os.path.join(out_dir, f"top_{profile_name}_{ts_label}.csv")
    
    # Tri par stabilité (Lyapunov bas) PUIS performance (equity haut)
    agg_sorted = agg.sort_values(["avg_lyapunov", "avg_equity"], ascending=[True, False])
    agg_sorted.head(30).to_csv(top_path, index=False)
    
    log(f"🎯 Top 30 trié par stabilité (Lyapunov) + performance !")
    
    # Tri alternatif par Calmar Ratio (performance/risque)
    calmar_sorted = agg.sort_values("avg_calmar", ascending=False)
    calmar_path = os.path.join(out_dir, f"top_calmar_{profile_name}_{ts_label}.csv")
    calmar_sorted.head(30).to_csv(calmar_path, index=False)
    
    log(f"📊 Top 30 par Calmar Ratio (performance/risque) : {calmar_path}")
    
    # Tri par Recovery Factor (récupération rapide)
    recovery_sorted = agg.sort_values("avg_recovery_factor", ascending=False)
    recovery_path = os.path.join(out_dir, f"top_recovery_{profile_name}_{ts_label}.csv")
    recovery_sorted.head(30).to_csv(recovery_path, index=False)
    
    log(f"🔄 Top 30 par Recovery Factor (récupération) : {recovery_path}")
    
    log(f"✅ Export: {top_path}")

    # Statistiques détaillées par paire
    symbol_stats = df_runs.groupby("symbol").agg({
        "equity_mult": ["count", "mean", "sum"], # count is nb_runs, mean is avg_equity_mult, sum is total_equity_mult
        "trades": "sum", # This will now sum all trades from all runs for that symbol
        "win_rate": "mean"
    }).round(4)

    # Flatten multi-index columns
    symbol_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in symbol_stats.columns.values]

    # Rename columns for clarity
    symbol_stats = symbol_stats.rename(columns={
        "equity_mult_count": "nb_backtest_runs",
        "equity_mult_mean": "avg_equity_mult_per_run",
        "equity_mult_sum": "total_equity_mult_across_runs",
        "trades_sum": "total_trades_across_runs",
        "win_rate_mean": "avg_winrate_per_run"
    })

    # Calculate total profit in euros across all runs for that symbol
    symbol_stats["total_profit_euros_across_runs"] = (symbol_stats["total_equity_mult_across_runs"] - symbol_stats["nb_backtest_runs"]) * 1000
    
    # Calculate average trades per run for clarity
    symbol_stats["avg_trades_per_run"] = (symbol_stats["total_trades_across_runs"] / symbol_stats["nb_backtest_runs"]).round(2)
    
    symbol_path = os.path.join(out_dir, f"symbol_stats_{profile_name}_{ts_label}.csv")
    symbol_stats.to_csv(symbol_path)
    
    log(f"📊 Statistiques par paire:")
    log(f"   - Chaque paire a été testée sur {symbol_stats['nb_backtest_runs'].iloc[0]} backtests individuels")
    log(f"   - Résultats détaillés: {symbol_stats}")

def ensure_dir(path):
    """Crée le répertoire s'il n'existe pas"""
    os.makedirs(path, exist_ok=True)

def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame index is DatetimeIndex in UTC but tz-naive (UTC assumed).

    - If tz-aware → convert to UTC then drop timezone (tz_localize(None)).
    - If tz-naive → leave as is.
    """
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
    except Exception:
        pass
    return df

def make_annual_folds(df: pd.DataFrame, start_year: int | None = None, end_year: int | None = None):
    """Retourne une liste de (YYYY-MM-DD, YYYY-MM-DD) par année civile disponible dans df.index."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df doit avoir un index DatetimeIndex")
    years = pd.Index(sorted(df.index.year.unique()))
    if start_year is not None:
        years = years[years >= start_year]
    if end_year is not None:
        years = years[years <= end_year]
    folds = []
    for y in years:
        # Utiliser des timestamps tz-naive (UTC implicite) pour matcher l'index normalisé
        start = pd.Timestamp(f"{y}-01-01")
        end = pd.Timestamp(f"{y}-12-31")
        if ((df.index >= start) & (df.index <= end)).any():
            folds.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    if not folds:
        raise ValueError("Aucun fold annuel trouvé pour la plage demandée")
    return folds

def sample_params_optuna(trial):
    """Params Ichimoku + ATR_mult sous contraintes: Tenkan ≤ Kijun ≤ SenkouB"""
    tenkan = trial.suggest_int("tenkan", 5, 30)
    r_kijun = trial.suggest_int("r_kijun", 1, 5)
    r_senkou = trial.suggest_int("r_senkou", 1, 9)
    kijun = max(tenkan, r_kijun * tenkan)
    senkou_b = max(kijun, r_senkou * tenkan)
    # Étend le domaine de recherche pour shift (1 à 100)
    shift = trial.suggest_int("shift", 1, 100)
    # Étend l'amplitude ATR pour permettre des stops plus larges (~3000 pts si ATR H2 élevé)
    atr_mult = trial.suggest_float("atr_mult", 0.5, 15.0, step=0.1)
    return {"tenkan": int(tenkan), "kijun": int(kijun), "senkou_b": int(senkou_b), "shift": int(shift), "atr_mult": float(atr_mult)}

def compute_score_optuna(cagr_list, sharpe_list, dd_list, trades_list):
    mean_sharpe = float(np.mean(sharpe_list)) if sharpe_list else 0.0
    mean_cagr = float(np.mean(cagr_list)) if cagr_list else 0.0
    mean_dd = float(np.mean(dd_list)) if dd_list else 0.0
    stab_penalty = float(np.std(sharpe_list) + 0.5 * np.std(cagr_list)) if sharpe_list else 0.0
    trade_penalty = 0.0 if (float(np.mean(trades_list)) if trades_list else 0.0) >= 30.0 else 0.5
    return 0.6 * mean_sharpe + 0.3 * mean_cagr - 0.3 * mean_dd - 0.5 * stab_penalty - trade_penalty

def optuna_objective(trial, market_data: dict, timeframe: str, start_year: int | None, end_year: int | None, fast_ratio: float, loss_mult: float | None = None):
    params = sample_params_optuna(trial)
    cagr_list = []
    sharpe_list = []
    dd_list = []
    trades_list = []
    fold_idx = 0
    total_folds = 0
    # Compter total folds pour reporting
    for _, df in market_data.items():
        total_folds += len(make_annual_folds(df, start_year, end_year))
    total_folds = max(1, total_folds)
    for sym, df in market_data.items():
        df = ensure_utc_index(df)
        folds = make_annual_folds(df, start_year, end_year)
        for (start, end) in folds:
            # Timestamps tz-naive (UTC implicite) alignés avec l'index
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            if 0.0 < fast_ratio < 1.0:
                horizon = end_ts - start_ts
                cut_ts = start_ts + pd.Timedelta(seconds=int(horizon.total_seconds() * fast_ratio))
                sub = df.loc[start_ts:cut_ts]
            else:
                sub = df.loc[start_ts:end_ts]
            if sub.empty:
                continue
            try:
                m = backtest_long_short(
                    sub,
                    params["tenkan"], params["kijun"], params["senkou_b"], params["shift"], params["atr_mult"],
                    loss_mult=(1.0 if loss_mult is None else float(loss_mult)),
                    symbol=sym, timeframe=timeframe
                )
            except Exception as e:
                # Pruning agressif si erreur
                raise optuna.TrialPruned() if optuna else e
            # Gating de risque (prune si run invalide)
            if (m.get("liquidations", 0) or m.get("margin_calls", 0) or float(m.get("min_equity", 1.0)) < 0.6 or float(m.get("max_drawdown", 0.0)) > 0.6):
                if optuna:
                    raise optuna.TrialPruned()
                else:
                    continue
            cagr_list.append(float(m.get("CAGR", 0.0)))
            sharpe_list.append(float(m.get("sharpe_proxy", 0.0)))
            dd_list.append(float(m.get("max_drawdown", 0.0)))
            trades_list.append(int(m.get("trades", 0)))
            # Reporting/pruning ASHA
            fold_idx += 1
            interim = float(np.mean(sharpe_list) - 0.2 * np.mean(dd_list) + 0.2 * np.mean(cagr_list))
            if optuna:
                trial.report(interim, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    return compute_score_optuna(cagr_list, sharpe_list, dd_list, trades_list)

def optuna_optimize_profile(profile_name: str, n_trials: int = 2000, seed: int | None = None, out_dir: str = "outputs",
                            use_cache: bool = True, start_year: int | None = None, end_year: int | None = None,
                            jobs: int = 1, fast_ratio: float = 1.0):
    if optuna is None:
        raise RuntimeError("Optuna non disponible. Installez-le (pip install optuna).")
    if profile_name not in PROFILES:
        raise ValueError(f"Profil inconnu: {profile_name}")
    cfg = PROFILES[profile_name]
    os.makedirs(out_dir, exist_ok=True)
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    symbols = cfg["symbols"]; timeframe = cfg["timeframe"]; years_back = cfg["years_back"]
    end_dt = datetime.utcnow(); start_dt = end_dt - timedelta(days=int(365.25 * years_back))
    since_ms = utc_ms(start_dt); until_ms = utc_ms(end_dt)
    ex = ccxt.binance({"enableRateLimit": True})
    market_data = {}
    for sym in symbols:
        log(f"Téléchargement {sym} {timeframe} sur ~{years_back} ans…")
        df = fetch_ohlcv_range(ex, sym, timeframe, since_ms, until_ms, cache_dir="data", use_cache=use_cache)
        if df.empty:
            log(f"⚠️  Pas de données pour {sym}.")
            continue
        market_data[sym] = df
    if not market_data:
        raise RuntimeError("Aucune donnée chargée.")
    log(f"🧪 Optuna ASHA: trials={n_trials}, jobs={jobs}, fast_ratio={fast_ratio}")
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler(seed=seed))
    def _obj(tr):
        return optuna_objective(tr, market_data, timeframe, start_year, end_year, fast_ratio, loss_mult=1.0)
    study.optimize(_obj, n_trials=int(n_trials), n_jobs=int(jobs))
    best = study.best_trial
    log(f"🏁 OPTUNA FINI - Best score={best.value:.4f} params={best.params}")
    ts_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    # Sauvegarde JSON des meilleurs params
    best_path = os.path.join(out_dir, f"best_params_{profile_name}_{ts_label}.json")
    try:
        import json
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best.params, f, ensure_ascii=False, indent=2)
        log(f"✅ Best params sauvegardés: {best_path}")
    except Exception as e:
        log(f"⚠️  Impossible de sauvegarder best params: {e}")

def optuna_optimize_profile_per_symbol(profile_name: str, n_trials: int = 5000, seed: int | None = None, out_dir: str = "outputs",
                                       use_cache: bool = True, start_year: int | None = None, end_year: int | None = None,
                                       jobs: int = 1, fast_ratio: float = 1.0, baseline_map: dict[str, dict] | None = None,
                                       loss_mult: float | None = None):
    """Optimise indépendamment chaque paire avec Optuna (folds annuels + ASHA) et exporte les meilleurs réglages par paire.

    - n_trials est appliqué à CHAQUE paire (pour un vrai test robuste par symbole)
    - Progression affichée avec une barre simple par symbole
    - Sauvegarde un JSON des meilleurs paramètres par paire et un CSV des backtests finaux avec ces paramètres
    """
    if optuna is None:
        raise RuntimeError("Optuna non disponible. Installez-le (pip install optuna).")
    if profile_name not in PROFILES:
        raise ValueError(f"Profil inconnu: {profile_name}")

    cfg = PROFILES[profile_name]
    os.makedirs(out_dir, exist_ok=True)
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    # Ensure a locally bound variable is captured by closures to avoid NameError
    loss_mult_outer = 1.0 if (loss_mult is None) else float(loss_mult)

    symbols = cfg["symbols"]
    timeframe = cfg["timeframe"]
    years_back = cfg["years_back"]

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=int(365.25 * years_back))
    since_ms = utc_ms(start_dt)
    until_ms = utc_ms(end_dt)

    ex = ccxt.binance({"enableRateLimit": True})

    # Charger toutes les données une fois
    market_data_all = {}
    for sym in symbols:
        local_df = _load_local_csv_if_configured(sym, timeframe)
        if local_df is not None:
            log(f"Utilisation CSV local (fused) pour {sym} {timeframe} — historique complet")
            df = ensure_utc_index(local_df)
        else:
            log(f"Téléchargement {sym} {timeframe} sur ~{years_back} ans…")
            df = fetch_ohlcv_range(ex, sym, timeframe, since_ms, until_ms, cache_dir="data", use_cache=use_cache)
            df = ensure_utc_index(df)
        if df.empty:
            log(f"⚠️  Pas de données pour {sym}. Ignorer.")
            continue
        market_data_all[sym] = df
    if not market_data_all:
        raise RuntimeError("Aucune donnée chargée.")

    best_by_symbol: dict[str, dict] = {}
    # Répertoire LIVE distinct si défini (évite conflit OneDrive)
    live_dir = os.environ.get("ICHIMOKU_LIVE_DIR", out_dir)
    try:
        os.makedirs(live_dir, exist_ok=True)
    except Exception:
        live_dir = out_dir
    last_live_save = 0.0  # Force un premier write immédiat
    # Ecrire un LIVE_REPORT minimal dès le départ
    try:
        ts_iso0 = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec='seconds') + 'Z'
        live_html0 = f"""<!doctype html><html><head><meta charset=utf-8><meta http-equiv=refresh content=5><title>LIVE REPORT</title>
        <style>body{{font-family:Arial,sans-serif;padding:16px;background:#fafafa}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:6px}}</style>
        </head><body>
        <h1>LIVE — {profile_name}</h1>
        <p>Initialisation… Mis à jour: {ts_iso0}</p>
        <h2>Progression en cours</h2>
        <p>En attente du premier résultat…</p>
        <h2>Paramètres et perfs par symbole</h2>
        <table><tr><th>Symbole</th><th>Paramètres</th><th>P&L</th><th>Max DD (indexé portefeuille)</th></tr></table>
        <p><i>Ce rapport se rafraîchit toutes les 5 secondes et est réécrit côté disque régulièrement.</i></p>
        </body></html>"""
        with open(os.path.join(live_dir, "LIVE_REPORT.html"), "w", encoding="utf-8") as f:
            f.write(live_html0)
        # Ecrire aussi dans le dossier de sortie par défaut (fallback)
        try:
            with open(os.path.join(out_dir, "LIVE_REPORT.html"), "w", encoding="utf-8") as f:
                f.write(live_html0)
        except Exception:
            pass
    except Exception:
        pass
    runs_rows = []

    for idx, sym in enumerate(symbols, start=1):
        if sym not in market_data_all:
            continue
        log(f"🧪 Optuna par paire ({idx}/{len(symbols)}): {sym} — trials={n_trials}, jobs={jobs}, fast_ratio={fast_ratio}")

        # Objectif spécifique à 1 symbole
        def _obj_symbol(trial):
            # Échantillonnage élargi et indépendant; optionnellement centré autour d'un baseline par symbole
            baseline = None
            if isinstance(baseline_map, dict):
                baseline = baseline_map.get(sym)

            # Plages globales très élargies
            tenkan_min, tenkan_max = 2, 60
            kijun_min, kijun_max = 5, 120
            senkou_min, senkou_max = 10, 240
            shift_min, shift_max = 10, 60
            atr_min, atr_max = 0.5, 6.0

            # Choix du mode: local (autour du baseline) vs global
            mode = trial.suggest_categorical("mode", ["local", "global"]) if baseline else "global"

            if mode == "local" and baseline:
                # Fenêtres locales autour des valeurs baselines
                b_t = int(baseline.get("tenkan", 9))
                b_k = int(baseline.get("kijun", 26))
                b_s = int(baseline.get("senkou_b", 52))
                b_sh = int(baseline.get("shift", 26))
                b_atr = float(baseline.get("atr_mult", 3.0))

                t_low, t_high = max(tenkan_min, b_t - 10), min(tenkan_max, b_t + 10)
                tenkan = trial.suggest_int("tenkan", t_low, t_high)
                k_low, k_high = max(kijun_min, b_k - 20), min(kijun_max, b_k + 20)
                kijun = trial.suggest_int("kijun", k_low, k_high)
                s_low, s_high = max(senkou_min, b_s - 40), min(senkou_max, b_s + 40)
                senkou_b = trial.suggest_int("senkou_b", s_low, s_high)
                sh_low, sh_high = max(shift_min, b_sh - 10), min(shift_max, b_sh + 10)
                shift = trial.suggest_int("shift", sh_low, sh_high)
                atr_mult = trial.suggest_float("atr_mult", max(atr_min, b_atr - 1.0), min(atr_max, b_atr + 1.0), step=0.1)
            else:
                # Global, indépendant avec contrainte d'ordre Tenkan ≤ Kijun ≤ SenkouB
                tenkan = trial.suggest_int("tenkan", tenkan_min, tenkan_max)
                kijun = trial.suggest_int("kijun", max(tenkan, kijun_min), kijun_max)
                senkou_b = trial.suggest_int("senkou_b", max(kijun, senkou_min), senkou_max)
                shift = trial.suggest_int("shift", shift_min, shift_max)
                atr_mult = trial.suggest_float("atr_mult", atr_min, atr_max, step=0.1)

            # Appliquer la contrainte d'ordre au cas où (sécurité)
            kijun = max(tenkan, kijun)
            senkou_b = max(kijun, senkou_b)

            params = {"tenkan": int(tenkan), "kijun": int(kijun), "senkou_b": int(senkou_b), "shift": int(shift), "atr_mult": float(atr_mult)}
            cagr_list, sharpe_list, dd_list, trades_list = [], [], [], []
            df = market_data_all[sym]
            folds = make_annual_folds(df, start_year, end_year)
            total_folds = len(folds)
            done = 0
            for (start, end) in folds:
                start_ts = pd.Timestamp(start); end_ts = pd.Timestamp(end)
                if 0.0 < fast_ratio < 1.0:
                    horizon = end_ts - start_ts
                    cut_ts = start_ts + pd.Timedelta(seconds=int(horizon.total_seconds() * fast_ratio))
                    sub = df.loc[start_ts:cut_ts]
                else:
                    sub = df.loc[start_ts:end_ts]
                if sub.empty:
                    continue
                m = backtest_long_short(sub, params["tenkan"], params["kijun"], params["senkou_b"], params["shift"], params["atr_mult"], loss_mult=loss_mult_outer, symbol=sym, timeframe=timeframe)
                cagr_list.append(float(m.get("CAGR", 0.0)))
                sharpe_list.append(float(m.get("sharpe_proxy", 0.0)))
                dd_list.append(float(m.get("max_drawdown", 0.0)))
                trades_list.append(int(m.get("trades", 0)))
                done += 1
                # Rapport pour ASHA + petite progression
                interim = float(np.mean(sharpe_list) - 0.2 * np.mean(dd_list) + 0.2 * np.mean(cagr_list))
                if optuna:
                    trial.report(interim, step=done)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            return compute_score_optuna(cagr_list, sharpe_list, dd_list, trades_list)

        pruner = SuccessiveHalvingPruner(min_resource=5, reduction_factor=3)
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=200, multivariate=True)
        )

        # Plateau-stop léger: stop si aucun new best en "plateau_window" trials
        # Permettre le contrôle via variables d'environnement:
        #  - PLATEAU_WINDOW: int (par défaut 200)
        #  - PLATEAU_DISABLED: '1' pour désactiver le plateau stop (fait un plateau_window très grand)
        try:
            _env_pw = os.getenv("PLATEAU_WINDOW")
            plateau_window = int(_env_pw) if _env_pw else 200
        except Exception:
            plateau_window = 200
        _plateau_disabled = (os.getenv("PLATEAU_DISABLED") == "1")
        if _plateau_disabled:
            plateau_window = 10**9  # effet: pas de stop plateau avant n_trials
        best_value_seen = None
        last_improve_trial = -1

        # Callback de progression + autosave périodique (toutes les 30 minutes)
        def _cb(st: optuna.Study, tr: optuna.trial.FrozenTrial):
            total = max(1, int(n_trials))
            done_trials = tr.number + 1
            bar_length = 30
            filled_length = int(bar_length * done_trials // total)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            if done_trials % max(1, total // 100) == 0 or done_trials == total:
                try:
                    best_val = st.best_value
                except Exception:
                    best_val = float('nan')
                log(f"{sym} Optuna {done_trials:,}/{total:,} [{bar}] best_score≈{best_val if isinstance(best_val, float) else best_val}")

            # Plateau stop: si aucun progrès depuis plateau_window trials, on stop
            try:
                cur_best = st.best_value
            except Exception:
                cur_best = None
            nonlocal best_value_seen, last_improve_trial
            if isinstance(cur_best, float) and (best_value_seen is None or cur_best > best_value_seen):
                best_value_seen = cur_best
                last_improve_trial = tr.number
            elif last_improve_trial >= 0 and (tr.number - last_improve_trial) >= plateau_window:
                log(f"🛑 Plateau atteint: aucun nouveau best depuis {plateau_window} trials. Arrêt de {sym}.")
                try:
                    st.stop()
                except Exception:
                    pass

            # Autosave léger: toutes les 30 minutes, écrire un LIVE report sans dupliquer des fichiers
            nonlocal last_live_save
            now = time.time()
            # Écrit immédiatement au premier résultat, puis toutes les 5 minutes
            if (now - last_live_save >= 300) or (last_live_save <= 0 and best_by_symbol):
                try:
                    # Mettre à jour best courant pour ce symbole si disponible
                    try:
                        current_best = st.best_trial
                        if current_best and current_best.params:
                            # Convertir en params complets
                            def _finalize_local(p: dict) -> dict:
                                if "kijun" in p and "senkou_b" in p:
                                    return {
                                        "tenkan": int(p["tenkan"]),
                                        "kijun": int(p["kijun"]),
                                        "senkou_b": int(p["senkou_b"]),
                                        "shift": int(p["shift"]),
                                        "atr_mult": float(p["atr_mult"]),
                                    }
                                tenkan_l = int(p["tenkan"])  # requis
                                r_k = int(p.get("r_kijun", 1))
                                r_s = int(p.get("r_senkou", 1))
                                kijun_l = max(tenkan_l, r_k * tenkan_l)
                                senkou_b_l = max(kijun_l, r_s * tenkan_l)
                                return {
                                    "tenkan": tenkan_l,
                                    "kijun": int(kijun_l),
                                    "senkou_b": int(senkou_b_l),
                                    "shift": int(p["shift"]),
                                    "atr_mult": float(p["atr_mult"]),
                                }
                            best_by_symbol[sym] = _finalize_local(current_best.params)
                    except Exception:
                        pass

                    # Calculer un shared rapide si au moins 1 symbole a des params
                    shared = None
                    if best_by_symbol:
                        shared = backtest_shared_portfolio(market_data_all, best_by_symbol, timeframe=timeframe, record_curve=True)
                    # Ecrire un JSON et un HTML live (overwrite)
                    ts_iso = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec='seconds') + 'Z'
                    live_json = {
                        "best_params": best_by_symbol,
                        "shared_metrics": shared,
                        "updated_at": ts_iso,
                    }
                    try:
                        import json as _json
                        live_json_path_tmp = os.path.join(live_dir, f"LIVE_BEST_{profile_name}.json.tmp")
                        live_json_path = os.path.join(live_dir, f"LIVE_BEST_{profile_name}.json")
                        with open(live_json_path_tmp, "w", encoding="utf-8") as f:
                            _json.dump(live_json, f, ensure_ascii=False, indent=2)
                        os.replace(live_json_path_tmp, live_json_path)
                    except Exception as _ejson:
                        log(f"⚠️  LIVE JSON non écrit: {_ejson}")

                    # Sauvegarde périodique d'un snapshot d'archive pour le MASTER REPORT (toutes les 5 min)
                    try:
                        ts_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        snapshot = dict(shared if isinstance(shared, dict) else {})
                        # Joindre aussi les paramètres courants par symbole
                        snapshot["best_params"] = best_by_symbol
                        archive_tmp = os.path.join(out_dir, f"shared_portfolio_{profile_name}_{ts_label}.json.tmp")
                        archive_path = os.path.join(out_dir, f"shared_portfolio_{profile_name}_{ts_label}.json")
                        with open(archive_tmp, "w", encoding="utf-8") as f:
                            _json.dump(snapshot, f, ensure_ascii=False, indent=2)
                        os.replace(archive_tmp, archive_path)
                        # Lien symbolique de convenance (fichier latest)
                        latest_tmp = os.path.join(out_dir, f"shared_portfolio_{profile_name}_latest.json.tmp")
                        latest_path = os.path.join(out_dir, f"shared_portfolio_{profile_name}_latest.json")
                        with open(latest_tmp, "w", encoding="utf-8") as f:
                            _json.dump(snapshot, f, ensure_ascii=False, indent=2)
                        os.replace(latest_tmp, latest_path)
                    except Exception:
                        pass

                    # HTML minimal lisible
                    def _fmt_params(pm: dict) -> str:
                        return f"Tenkan {pm.get('tenkan','?')}, Kijun {pm.get('kijun','?')}, SenkouB {pm.get('senkou_b','?')}, Shift {pm.get('shift','?')}, ATR× {float(pm.get('atr_mult', float('nan'))):.2f}"
                    rows = []
                    for s2 in sorted(best_by_symbol.keys()):
                        rows.append(f"<tr><td>{s2}</td><td>{_fmt_params(best_by_symbol[s2])}</td></tr>")
                    eq_eur = float(shared.get("equity_mult", 1.0)) * 1000.0 if isinstance(shared, dict) else float('nan')
                    dd_pct = float(shared.get("max_drawdown", 0.0)) * 100.0 if isinstance(shared, dict) else float('nan')
                    # Progression courante
                    try:
                        progress_pct = (done_trials / total) * 100.0 if total > 0 else 0.0
                    except Exception:
                        progress_pct = 0.0
                    # Per-symbol P&L and DD on portfolio equity
                    sym_pnl = {}
                    sym_dd = {}
                    try:
                        import pandas as _pd
                        if 'equity_curve' in shared and isinstance(shared['equity_curve'], list):
                            curve = _pd.DataFrame(shared['equity_curve'], columns=['ts','equity']).set_index('ts')
                        else:
                            curve = None
                    except Exception:
                        curve = None
                    # Build rows using per-symbol metrics from shared backtest if available
                    per_sym = shared.get('per_symbol', {}) if isinstance(shared, dict) else {}
                    rows = []
                    for s2 in sorted(best_by_symbol.keys()):
                        pm = best_by_symbol[s2]
                        params_html = _fmt_params(pm)
                        ps = per_sym.get(s2, {})
                        pnl_val = ps.get('pnl_eur', None)
                        dd_val = ps.get('max_dd_indexed', None)
                        pnl_html = f"{pnl_val:,.0f} €" if isinstance(pnl_val, (int, float)) else "?"
                        dd_html = f"{dd_val*100:.1f}%" if isinstance(dd_val, (int, float)) else "?"
                        rows.append(f"<tr><td>{s2}</td><td>{params_html}</td><td style='text-align:right'>{pnl_html}</td><td style='text-align:right'>{dd_html}</td></tr>")
                        live_html = f"""<!doctype html><html><head><meta charset=utf-8><meta http-equiv=refresh content=5><title>LIVE REPORT</title>
                        <style>body{{font-family:Arial,sans-serif;padding:16px;background:#fafafa}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:6px}}</style>
                        </head><body>
                        <h1>LIVE — {profile_name}</h1>
                        <p>Mis à jour: {ts_iso}</p>
                        <h2>Progression en cours</h2>
                        <p>Symbole courant: <b>{sym}</b> — Trials: <b>{done_trials:,}/{total:,}</b> — Avancement: <b>{progress_pct:.1f}%</b></p>
                        <h2>Portefeuille</h2>
                        <ul>
                          <li>Equity: <b>{'' if np.isnan(eq_eur) else '{:,.0f} €'.format(eq_eur)}</b></li>
                          <li>Max DD: <b>{'' if np.isnan(dd_pct) else '{:.1f}%'.format(dd_pct)}</b></li>
                          <li>Trades: <b>{int(shared.get('trades', 0)) if isinstance(shared, dict) else 0}</b></li>
                        </ul>
                        <h2>Paramètres et perfs par symbole</h2>
                        <table><tr><th>Symbole</th><th>Paramètres</th><th>P&L</th><th>Max DD (indexé portefeuille)</th></tr>
                        {''.join(rows)}
                        </table>
                        <p><i>Ce rapport se rafraîchit toutes les 5 secondes et est réécrit côté disque ~toutes les 5 minutes.</i></p>
                        </body></html>"""
                    try:
                        live_html_path_tmp = os.path.join(live_dir, "LIVE_REPORT.html.tmp")
                        live_html_path = os.path.join(live_dir, "LIVE_REPORT.html")
                        with open(live_html_path_tmp, "w", encoding="utf-8") as f:
                            f.write(live_html)
                        os.replace(live_html_path_tmp, live_html_path)
                        # Duplication dans out_dir pour compatibilité
                        try:
                            out_html_tmp = os.path.join(out_dir, "LIVE_REPORT.html.tmp")
                            out_html = os.path.join(out_dir, "LIVE_REPORT.html")
                            with open(out_html_tmp, "w", encoding="utf-8") as f:
                                f.write(live_html)
                            os.replace(out_html_tmp, out_html)
                        except Exception:
                            pass
                        log(f"📝 LIVE mis à jour ({ts_iso}) — {len(best_by_symbol)} symbole(s)")
                    except Exception as _ehtml:
                        log(f"⚠️  LIVE HTML non écrit: {_ehtml}")
                finally:
                    last_live_save = now

        study.optimize(_obj_symbol, n_trials=int(n_trials), n_jobs=int(jobs), callbacks=[_cb])
        best = study.best_trial
        # Convertir les paramètres Optuna (r_kijun/r_senkou) en (kijun/senkou_b)
        def _finalize(p: dict) -> dict:
            if "kijun" in p and "senkou_b" in p:
                return {
                    "tenkan": int(p["tenkan"]),
                    "kijun": int(p["kijun"]),
                    "senkou_b": int(p["senkou_b"]),
                    "shift": int(p["shift"]),
                    "atr_mult": float(p["atr_mult"]),
                }
            tenkan = int(p["tenkan"])  # requis
            r_k = int(p.get("r_kijun", 1))
            r_s = int(p.get("r_senkou", 1))
            kijun = max(tenkan, r_k * tenkan)
            senkou_b = max(kijun, r_s * tenkan)
            return {
                "tenkan": tenkan,
                "kijun": int(kijun),
                "senkou_b": int(senkou_b),
                "shift": int(p["shift"]),
                "atr_mult": float(p["atr_mult"]),
            }
        p_full = _finalize(best.params)
        log(f"🏁 {sym} — Best score={best.value:.4f} params={p_full}")
        best_by_symbol[sym] = p_full

        # Backtest final plein horizon avec les meilleurs paramètres
        df_full = market_data_all[sym]
        p = p_full
        m = backtest_long_short(df_full, p["tenkan"], p["kijun"], p["senkou_b"], p["shift"], p["atr_mult"], loss_mult=loss_mult_outer, symbol=sym, timeframe=timeframe)
        # Skip export si invalide (gating de sécurité)
        if not (m.get("liquidations", 0) or m.get("margin_calls", 0) or float(m.get("min_equity", 1.0)) < 0.6 or float(m.get("max_drawdown", 0.0)) > 0.6):
            m.update({
                "symbol": sym,
                "tenkan": p["tenkan"], "kijun": p["kijun"], "senkou_b": p["senkou_b"], "shift": p["shift"], "atr_mult": p["atr_mult"],
                "trial": -1, "generation": 0, "trader_id": 0
            })
            runs_rows.append(m)

    # Sauvegardes
    ts_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    try:
        import json
        best_map_path = os.path.join(out_dir, f"best_params_per_symbol_{profile_name}_{ts_label}.json")
        with open(best_map_path, "w", encoding="utf-8") as f:
            json.dump(best_by_symbol, f, ensure_ascii=False, indent=2)
        log(f"✅ Best params par paire sauvegardés: {best_map_path}")
    except Exception as e:
        log(f"⚠️  Impossible de sauvegarder best params par paire: {e}")

    if runs_rows:
        df_runs = pd.DataFrame(runs_rows)
        runs_path = os.path.join(out_dir, f"runs_best_per_symbol_{profile_name}_{ts_label}.csv")
        df_runs.to_csv(runs_path, index=False)
        log(f"✅ Backtests finaux exportés: {runs_path}")
    else:
        log("❌ Aucun backtest final généré.")

    # Backtest portefeuille partagé avec les meilleurs params par paire
    try:
        if best_by_symbol:
            shared = backtest_shared_portfolio(market_data_all, best_by_symbol, timeframe=timeframe, record_curve=True)
            log(f"🏦 Portefeuille partagé — equity≈{shared.get('equity_mult', float('nan')):.3f}, CAGR≈{shared.get('CAGR', 0.0):.3%}, MDD≈{shared.get('max_drawdown', 0.0):.2%}, trades={shared.get('trades', 0)}")
            import json
            shared_path = os.path.join(out_dir, f"shared_portfolio_{profile_name}_{ts_label}.json")
            with open(shared_path, "w", encoding="utf-8") as f:
                json.dump({"best_params": best_by_symbol, "shared_metrics": shared}, f, ensure_ascii=False, indent=2)
            log(f"✅ Résumé portefeuille partagé sauvegardé: {shared_path}")
    except Exception as e:
        log(f"⚠️  Erreur backtest portefeuille partagé: {e}")

def main():
    """Point d'entrée principal"""
    if len(sys.argv) < 2:
        print("Usage: python ichimoku_pipeline_web_v4_8.py <profile> [--trials N] [--seed N] [--out-dir DIR] [--no-cache] [--no-genetic] [--baseline-json PATH]")
        print("Profils disponibles:", list(PROFILES.keys()))
        print("\nOptions:")
        print("  --trials N     : Nombre de trials (défaut: 5000)")
        print("  --seed N       : Seed pour la reproductibilité")
        print("  --out-dir DIR  : Dossier de sortie (défaut: outputs)")
        print("  --no-cache     : Désactiver le cache")
        print("  --no-genetic   : Désactiver l'algorithme génétique (recherche aléatoire)")
        print("  --baseline-json: JSON { symbol: {tenkan,kijun,senkou_b,shift,atr_mult} } pour centrer la recherche")
        print("  --fixed        : Utiliser des paramètres fixes Ichimoku (9-26-52-26 par défaut)")
        print("  --tenkan N     : Tenkan pour --fixed")
        print("  --kijun N      : Kijun pour --fixed")
        print("  --senkou_b N   : Senkou B pour --fixed")
        print("  --shift N      : Shift pour --fixed")
        print("  --atr_mult X   : ATR multiplier pour --fixed (défaut 3.0)")
        print("  --loss-mult X  : Seuil quotidien k en multiples d'ATR (défaut profil)")
        sys.exit(1)
    
    profile_name = sys.argv[1]
    trials = 0
    seed = None
    out_dir = "outputs"
    use_cache = True
    use_genetic = True
    baseline_map = None
    fixed_params = None
    loss_mult = 3.0
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--trials" and i + 1 < len(sys.argv):
            trials = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--seed" and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--out-dir" and i + 1 < len(sys.argv):
            out_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--no-cache":
            use_cache = False
            i += 1
        elif sys.argv[i] == "--no-genetic":
            use_genetic = False
            i += 1
        elif sys.argv[i] == "--fixed":
            fixed_params = {"tenkan": 9, "kijun": 26, "senkou_b": 52, "shift": 26, "atr_mult": 3.0}
            i += 1
        elif sys.argv[i] == "--tenkan" and i + 1 < len(sys.argv):
            fixed_params = fixed_params or {}
            fixed_params["tenkan"] = int(sys.argv[i+1]); i += 2
        elif sys.argv[i] == "--kijun" and i + 1 < len(sys.argv):
            fixed_params = fixed_params or {}
            fixed_params["kijun"] = int(sys.argv[i+1]); i += 2
        elif sys.argv[i] == "--senkou_b" and i + 1 < len(sys.argv):
            fixed_params = fixed_params or {}
            fixed_params["senkou_b"] = int(sys.argv[i+1]); i += 2
        elif sys.argv[i] == "--shift" and i + 1 < len(sys.argv):
            fixed_params = fixed_params or {}
            fixed_params["shift"] = int(sys.argv[i+1]); i += 2
        elif sys.argv[i] == "--atr_mult" and i + 1 < len(sys.argv):
            fixed_params = fixed_params or {}
            fixed_params["atr_mult"] = float(sys.argv[i+1]); i += 2
        elif sys.argv[i] == "--loss-mult" and i + 1 < len(sys.argv):
            loss_mult = float(sys.argv[i+1]); i += 2
        elif sys.argv[i] == "--baseline-json" and i + 1 < len(sys.argv):
            try:
                import json as _json
                with open(sys.argv[i+1], "r", encoding="utf-8") as f:
                    baseline_map = _json.load(f)
            except Exception as _e:
                print(f"⚠️  Impossible de charger --baseline-json: {_e}")
            i += 2
        else:
            i += 1
    
    try:
        # Passer baseline_map via PROFILES pour optuna_optimize_profile_per_symbol
        run_profile(profile_name, trials, seed, out_dir, use_cache, use_genetic, fixed_params, baseline_map, loss_mult)
        print(f"\nDone. Outputs are in the {out_dir}\\ folder.")
        if use_genetic:
            print("Genetic algorithm enabled.")
        else:
            print("Random search enabled.")
    except Exception as e:
        try:
            print(f"Error: {str(e)}")
        except Exception:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
