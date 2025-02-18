import math
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ta.trend import EMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import threading
import queue
from scipy.signal import argrelextrema
import joblib
import warnings
import json
import os
import logging
from typing import Dict, List, Optional, Union, Tuple
from collections import deque
import hashlib
import hmac
from decimal import Decimal, ROUND_DOWN
warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingConfig:
    """Clase para manejar la configuración del bot"""
    def __init__(self, filepath: str = 'C:/Users/asusn/Desktop/Nueva carpeta/trading_bot/config/config.json'):  # Actualizado para usar la subcarpeta
        self.filepath = filepath
        self.config = self.load_config()

    def load_config(self) -> dict:
        """Carga la configuración desde un archivo JSON"""
        try:
            # Crear la carpeta config si no existe
            os.makedirs('config', exist_ok=True)
            
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            
            # Si no existe el archivo, crear uno con la configuración por defecto
            config = self.get_default_config()
            with open(self.filepath, 'w') as f:
                json.dump(config, f, indent=4)
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.get_default_config()

    def get_default_config(self) -> dict:
        """Retorna la configuración por defecto"""
        return {
            'risk_level': 0.02,  # 2% del balance por trade
            'max_position_size': 0.2,  # 20% del balance máximo por posición
            'leverage': 5,
            'min_trade_interval': 300,  # 5 minutos entre trades
            'stop_loss_multiplier': {
                'trend_stable': 2.0,
                'trend_volatile': 2.5,
                'range_stable': 1.5,
                'range_volatile': 3.0
            },
            'take_profit_multiplier': {
                'trend_stable': 3.0,
                'trend_volatile': 2.5,
                'range_stable': 2.0,
                'range_volatile': 2.0
            },
            'signal_threshold': {
                'trend_stable': 0.4,
                'trend_volatile': 0.6,
                'range_stable': 0.5,
                'range_volatile': 0.7
            }
        }

    def save_config(self) -> None:
        """Guarda la configuración actual en el archivo JSON"""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

class TradeManager:
    """Clase para manejar las operaciones de trading"""
    def __init__(self, exchange, symbol: str, config: TradingConfig):
        self.exchange = exchange
        self.symbol = symbol
        self.config = config  # Agregamos la configuración
        self.trades = deque(maxlen=1000)
        self.performance_metrics = {
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'current_drawdown': 0.0,
            'max_drawdown': 0.0
        }
        self.load_trades()

    def calculate_position_size(self, balance: float, risk_per_trade: float, 
                              stop_loss_pct: float) -> float:
        """
        Calcula el tamaño de la posición basado en el riesgo y balance
        """
        try:
            # Obtener el precio actual
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            # Calcular el riesgo en USDT
            risk_amount = balance * risk_per_trade
            
            # Calcular el tamaño en la moneda base
            position_value = risk_amount / stop_loss_pct * self.config.config['leverage']
            position_size = position_value / current_price
            
            # Obtener los límites del mercado
            market = self.exchange.market(self.symbol)
            
            # Aplicar los límites del mercado
            min_amount = market['limits']['amount']['min']
            max_amount = market['limits']['amount']['max']
            amount_step = market['precision']['amount']
            
            # Redondear al step size correcto
            position_size = self.round_to_step(position_size, amount_step)
            
            # Aplicar límites min/max
            position_size = max(min(position_size, max_amount), min_amount)
            
            logger.info(f"Calculated position size: {position_size} at price {current_price}")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def round_to_step(self, quantity: float, step_size: float) -> float:
        """
        Redondea una cantidad al step size más cercano
        """
        if step_size == 0:
            return quantity
            
        inverse = 1.0 / step_size
        return round(quantity * inverse) / inverse
        
    def round_position_size(self, size: float) -> float:
        """
        Redondea el tamaño de la posición según los límites del exchange
        """
        try:
            market = self.exchange.market(self.symbol)
            amount_step = market['precision']['amount']
            min_amount = market['limits']['amount']['min']
            max_amount = market['limits']['amount']['max']
            
            # Redondear al step size
            rounded = self.round_to_step(size, amount_step)
            
            # Aplicar límites min/max
            return max(min(rounded, max_amount), min_amount)
        except Exception as e:
            logger.error(f"Error rounding position size: {e}")
            return size

    def update_performance_metrics(self, trade: dict) -> None:
        """
        Actualiza las métricas de rendimiento con un nuevo trade
        """
        pnl = trade.get('realized_pnl', 0)
        
        if pnl > 0:
            self.performance_metrics['wins'] += 1
            self.performance_metrics['largest_win'] = max(
                self.performance_metrics['largest_win'], pnl
            )
        elif pnl < 0:
            self.performance_metrics['losses'] += 1
            self.performance_metrics['largest_loss'] = min(
                self.performance_metrics['largest_loss'], pnl
            )
            
        self.performance_metrics['total_pnl'] += pnl
        total_trades = self.performance_metrics['wins'] + self.performance_metrics['losses']
        
        if total_trades > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['wins'] / total_trades * 100
            )
            
        # Calcular drawdown
        peak = self.performance_metrics['total_pnl']
        current = self.performance_metrics['total_pnl']
        drawdown = (peak - current) / peak * 100 if peak > 0 else 0
        self.performance_metrics['current_drawdown'] = drawdown
        self.performance_metrics['max_drawdown'] = max(
            self.performance_metrics['max_drawdown'], drawdown
        )

    def save_trades(self) -> None:
        """
        Guarda el histórico de trades en un archivo
        """
        try:
            with open('trades_history.json', 'w') as f:
                json.dump(list(self.trades), f)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")

    def load_trades(self) -> None:
        """
        Carga el histórico de trades desde un archivo
        """
        try:
            if os.path.exists('trades_history.json'):
                with open('trades_history.json', 'r') as f:
                    trades = json.load(f)
                    self.trades = deque(trades, maxlen=1000)
                    for trade in trades:
                        self.update_performance_metrics(trade)
        except Exception as e:
            logger.error(f"Error loading trades: {e}")

class OptimizedBybitBot:
    def __init__(self, api_key: str, api_secret: str, symbol: str, 
                 timeframe: str = '1h'):
        """
        Inicialización para cuenta real de Bybit
        """
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'market': 'linear', # Para futuros USDT
                'marginMode': 'isolated',
                'defaultMarginMode': 'isolated'
            }
        })

        # Configuración específica para mainnet
        self.exchange.set_sandbox_mode(False)  # Asegurarse de que estamos en mainnet
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = TradingConfig()
        self.balance_manager = BalanceManager(self.config.config['initial_balance'])
        self.trade_manager = TradeManager(self.exchange, symbol, self.config)
        self.model = RandomForestClassifier(
            n_estimators=200, 
            n_jobs=-1, 
            random_state=42
        )
        self.message_queue = queue.Queue()
        self.last_trade_time = None
        self._initialize_exchange()

    def _initialize_exchange(self) -> None:
        """
        Inicializa la configuración del exchange para cuenta real
        """
        try:
            # Cargar mercados
            self.exchange.load_markets()
            logger.info("Markets loaded successfully")
            
            try:
                # Obtener información de la cuenta
                account_info = self.exchange.fetch_balance()
                logger.info(f"Account connected successfully: {account_info['total']}")
                
                # Establecer modo de trading
                try:
                    trading_params = {
                        'symbol': self.symbol,
                        'leverage': self.config.config['leverage']
                    }
                    
                    # Intentar establecer el leverage
                    self.exchange.set_leverage(self.config.config['leverage'], self.symbol)
                    logger.info(f"Leverage set to {self.config.config['leverage']}x")
                    
                except Exception as e:
                    logger.warning(f"Could not set leverage: {str(e)}")
                
                # Verificar posiciones actuales
                try:
                    positions = self.exchange.fetch_positions([self.symbol])
                    for position in positions:
                        if position['contracts'] > 0:
                            logger.info(f"Current position found: {position}")
                except Exception as e:
                    logger.warning(f"Could not fetch positions: {str(e)}")
                
            except Exception as e:
                logger.warning(f"Could not complete all initialization steps: {str(e)}")
            
            logger.info(f"Exchange initialized for {self.symbol}")
            
        except Exception as e:
            logger.error(f"Error initializing exchange: {str(e)}")
            raise
        
    def update_balance_and_risk(self) -> None:
        """
        Actualiza el balance y ajusta los parámetros de riesgo
        """
        try:
            balance_info = self.exchange.fetch_balance()
            current_balance = float(balance_info['USDT']['total'])
            
            # Actualizar balance manager
            self.balance_manager.update_balance(current_balance)
            
            # Ajustar leverage basado en el balance
            if self.config.config['dynamic_adjustment']['max_leverage_adjustment']:
                if current_balance < self.config.config['initial_balance'] * 0.8:
                    new_leverage = max(2, self.config.config['leverage'] - 1)
                elif current_balance > self.config.config['initial_balance'] * 1.2:
                    new_leverage = min(10, self.config.config['leverage'] + 1)
                else:
                    new_leverage = self.config.config['leverage']
                
                if new_leverage != self.config.config['leverage']:
                    self.config.config['leverage'] = new_leverage
                    try:
                        self.exchange.private_post_position_leverage_save({
                            'symbol': self.symbol.replace('USDT', ''),
                            'leverage': new_leverage
                        })
                        logger.info(f"Leverage adjusted to {new_leverage}x")
                    except Exception as e:
                        logger.warning(f"Could not adjust leverage: {e}")
            
            logger.info(f"Balance and risk parameters updated - Balance: {current_balance}, "
                       f"Leverage: {self.config.config['leverage']}")
            
        except Exception as e:
            logger.error(f"Error updating balance and risk: {e}")

    def calculate_position_size(self, current_price: float, risk_params: dict) -> float:
        """
        Calcula el tamaño de la posición con ajuste dinámico
        """
        try:
            # Obtener balance actual
            balance_info = self.exchange.fetch_balance()
            available_balance = float(balance_info['USDT']['free'])
            
            # Obtener tamaño seguro de posición
            risk_amount = self.balance_manager.get_safe_position_size(
                available_balance,
                self.config.config['risk_level']
            )
            
            # Calcular cantidad base en el activo
            base_quantity = (risk_amount * self.config.config['leverage']) / current_price
            
            # Obtener límites del mercado
            market = self.exchange.market(self.symbol)
            min_amount = market['limits']['amount']['min']
            max_amount = market['limits']['amount']['max']
            
            # Aplicar límites y precisión
            quantity = float(self.exchange.amount_to_precision(self.symbol, base_quantity))
            quantity = max(min(quantity, max_amount), min_amount)
            
            logger.info(f"Position size calculated - Base: {base_quantity}, "
                       f"Final: {quantity}, Risk Amount: {risk_amount}")
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def execute_trade(self, signal: float, risk_params: dict) -> Optional[dict]:
        """
        Ejecuta una operación de trading con gestión de riesgos
        """
        try:
            logger.info(f"Attempting to execute trade with signal: {signal} and risk params: {risk_params}")
            
            # Actualizar balance y riesgo
            self.update_balance_and_risk()
            
            # Verificar intervalo mínimo entre trades
            if (self.last_trade_time and 
                time.time() - self.last_trade_time < self.config.config['min_trade_interval']):
                logger.info("Trade skipped: minimum interval not elapsed")
                return None

            # Obtener información del mercado y precio actual
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
                
                # Calcular cantidad ajustada
                quantity = self.calculate_position_size(current_price, risk_params)
                
                if quantity <= 0:
                    logger.error("Invalid position size calculated")
                    return None
                
                logger.info(f"Executing trade - Price: {current_price}, Quantity: {quantity}")
                
                # Crear orden principal
                side = 'buy' if signal > 0 else 'sell'
                order_params = {
                    'category': 'linear',
                    'positionIdx': 0,  # 0 para modo unidireccional
                    'leverage': str(self.config.config['leverage'])
                }
                
                order = self.exchange.create_market_order(
                    self.symbol,
                    side,
                    quantity,
                    params=order_params
                )
                
                logger.info(f"Market order created: {order}")
                
                if order and order.get('id'):
                    entry_price = float(order.get('price', current_price))
                    
                    # Calcular niveles de SL/TP
                    if side == 'buy':
                        stop_loss = entry_price * (1 - risk_params['stop_loss_pct'])
                        take_profit = entry_price * (1 + risk_params['take_profit_pct'])
                    else:
                        stop_loss = entry_price * (1 + risk_params['stop_loss_pct'])
                        take_profit = entry_price * (1 - risk_params['take_profit_pct'])
                    
                    # Redondear precios
                    stop_loss = float(self.exchange.price_to_precision(self.symbol, stop_loss))
                    take_profit = float(self.exchange.price_to_precision(self.symbol, take_profit))
                    
                    logger.info(f"Setting SL at {stop_loss} and TP at {take_profit}")
                    
                    # Colocar SL
                    try:
                        sl_params = {
                            'category': 'linear',
                            'positionIdx': 0,
                            'stopPrice': stop_loss,
                            'basePrice': entry_price,
                            'triggerBy': 'LastPrice',
                            'timeInForce': 'GTC'
                        }
                        
                        sl_order = self.exchange.create_order(
                            self.symbol,
                            'stop_market',
                            'sell' if side == 'buy' else 'buy',
                            quantity,
                            None,
                            params=sl_params
                        )
                        logger.info(f"Stop loss order placed: {sl_order}")
                    except Exception as e:
                        logger.error(f"Error placing stop loss: {e}")
                    
                    # Colocar TP
                    try:
                        tp_params = {
                            'category': 'linear',
                            'positionIdx': 0,
                            'timeInForce': 'GTC'
                        }
                        
                        tp_order = self.exchange.create_order(
                            self.symbol,
                            'limit',
                            'sell' if side == 'buy' else 'buy',
                            quantity,
                            take_profit,
                            params=tp_params
                        )
                        logger.info(f"Take profit order placed: {tp_order}")
                    except Exception as e:
                        logger.error(f"Error placing take profit: {e}")
                    
                    # Registrar trade
                    trade = {
                        'timestamp': datetime.now().isoformat(),
                        'side': side,
                        'size': quantity,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'status': 'open',
                        'order_id': order['id']
                    }
                    
                    self.trade_manager.trades.append(trade)
                    self.trade_manager.save_trades()
                    self.last_trade_time = time.time()
                    
                    logger.info(f"Trade successfully executed and recorded: {trade}")
                    return trade
                
            except Exception as e:
                logger.error(f"Error in trade execution: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error in execute_trade: {e}")
            return None
    def __init__(self, api_key: str, api_secret: str, symbol: str, 
                 timeframe: str = '1h'):
        """
        Inicialización mejorada del bot de trading
        """
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'test': False,
                'adjustForTimeDifference': True,
                'recvWindow': 60000
            }
        })
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = TradingConfig()
        self.trade_manager = TradeManager(self.exchange, symbol, self.config)
        self.model = RandomForestClassifier(
            n_estimators=200, 
            n_jobs=-1, 
            random_state=42
        )
        self.message_queue = queue.Queue()
        self.last_trade_time = None
        self._initialize_exchange()

    def _initialize_exchange(self) -> None:
        """
        Inicializa la configuración del exchange
        """
        try:
            # Cargar mercados
            self.exchange.load_markets()
            logger.info("Markets loaded successfully")
            
            # Validar que el símbolo existe
            if self.symbol not in self.exchange.markets:
                raise ValueError(f"Symbol {self.symbol} not found in available markets")
            
            # Configurar leverage y modo de trading
            try:
                # Configurar modo de posición
                self.exchange.private_post_position_switch_mode({
                    'coin': self.symbol.replace('USDT', ''),
                    'mode': 1  # 1 para modo aislado
                })
                logger.info("Position mode set to isolated")
                
                # Configurar leverage
                self.exchange.private_post_position_leverage_save({
                    'symbol': self.symbol.replace('USDT', ''),
                    'leverage': self.config.config['leverage']
                })
                logger.info(f"Leverage set to {self.config.config['leverage']}x")
                
            except Exception as e:
                logger.warning(f"Could not complete leverage/margin configuration: {e}")
            
            logger.info(f"Exchange initialized successfully for {self.symbol}")
            
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            raise

    def _initialize_exchange(self) -> None:
        """
        Inicializa la configuración del exchange
        """
        try:
            self.exchange.load_markets()
            # Configurar el apalancamiento para el mercado lineal
            try:
                # Configurar margen aislado
                self.exchange.set_position_mode(hedged=False)  # Asegurarse de que estamos en modo no-hedge
                
                # Establecer el tipo de margen y leverage
                params = {
                    'symbol': self.symbol,
                    'leverage': self.config.config['leverage'],
                    'marginType': 'isolated'
                }
                
                self.exchange.privatePutPositionLeverage(params)
                logger.info(f"Successfully set leverage to {self.config.config['leverage']}x and margin type to isolated")
                
            except Exception as e:
                logger.warning(f"Could not set leverage and margin type: {e}")
                # No lanzamos error aquí para permitir que el bot siga funcionando
            logger.info(f"Exchange initialized for {self.symbol}")
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            raise

    def fetch_data(self, limit: int = 100) -> pd.DataFrame:
        """
        Obtiene y preprocesa los datos OHLCV
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, 
                timeframe=self.timeframe, 
                limit=limit
            )
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cálculo de medias móviles exponenciales
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega indicadores técnicos avanzados al DataFrame
        """
        try:
            # Indicadores de tendencia
            macd = MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            ichimoku = IchimokuIndicator(df['high'], df['low'])
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            
            # Indicadores de momentum
            df['rsi'] = RSIIndicator(df['close']).rsi()
            stoch = StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Indicadores de volatilidad
            bb = BollingerBands(df['close'])
            df['bb_high'] = bb.bollinger_hband()
            df['bb_mid'] = bb.bollinger_mavg()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
            
            # Indicadores de volumen
            df['vwap'] = VolumeWeightedAveragePrice(
                df['high'], df['low'], df['close'], df['volume']
            ).volume_weighted_average_price()
            
            df['obv'] = OnBalanceVolumeIndicator(
                df['close'], df['volume']
            ).on_balance_volume()
            
            return df
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            raise

    def calculate_market_regime(self, df: pd.DataFrame) -> str:
        """
        Identifica el régimen actual del mercado
        """
        try:
            # Volatilidad
            df['volatility'] = df['bb_width']
            df['volatility_percentile'] = (
                df['volatility'].rolling(20).rank(pct=True)
            )
            
            # Fuerza de la tendencia
            df['trend_strength'] = abs(df['ema_20'] - df['ema_50']) / df['close']
            df['trend_strength_percentile'] = (
                df['trend_strength'].rolling(20).rank(pct=True)
            )
            
            vol_high = df['volatility_percentile'].iloc[-1] > 0.7
            trend_high = df['trend_strength_percentile'].iloc[-1] > 0.7
            
            if trend_high:
                return 'trend_volatile' if vol_high else 'trend_stable'
            else:
                return 'range_volatile' if vol_high else 'range_stable'
        except Exception as e:
            logger.error(f"Error calculating market regime: {e}")
            return 'range_stable'  # Default seguro

    def calculate_signal_strength(self, df: pd.DataFrame, 
                                market_regime: str) -> float:
        """
        Calcula la fuerza de la señal de trading
        """
        try:
            signals = {}
            
            # Señales base
            signals['rsi'] = (
                1 if df['rsi'].iloc[-1] < 30 else 
                -1 if df['rsi'].iloc[-1] > 70 else 0
            )
            
            signals['macd'] = (
                1 if df['macd_diff'].iloc[-1] > 0 else -1
            )
            
            # Señales específicas del régimen
            if market_regime.startswith('trend'):
                # Señales para mercados en tendencia
                signals['ema'] = (
                    1 if df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1] else -1
                )
                
                signals['ichimoku'] = (
                    1 if (df['close'].iloc[-1] > df['ichimoku_a'].iloc[-1] and
                         df['ichimoku_a'].iloc[-1] > df['ichimoku_b'].iloc[-1])
                    else -1
                )
            else:
                # Señales para mercados en rango
                signals['bb'] = (
                    1 if df['close'].iloc[-1] < df['bb_low'].iloc[-1]
                    else -1 if df['close'].iloc[-1] > df['bb_high'].iloc[-1]
                    else 0
                )
                
                signals['stoch'] = (
                    1 if (df['stoch_k'].iloc[-1] < 20 and 
                         df['stoch_d'].iloc[-1] < 20)
                    else -1 if (df['stoch_k'].iloc[-1] > 80 and 
                                df['stoch_d'].iloc[-1] > 80)
                    else 0
                )

            # Ponderación de señales
            weights = {
                'trend_stable': {'rsi': 0.2, 'macd': 0.3, 'ema': 0.3, 'ichimoku': 0.2},
                'trend_volatile': {'rsi': 0.15, 'macd': 0.25, 'ema': 0.35, 'ichimoku': 0.25},
                'range_stable': {'rsi': 0.3, 'macd': 0.2, 'bb': 0.3, 'stoch': 0.2},
                'range_volatile': {'rsi': 0.25, 'macd': 0.15, 'bb': 0.35, 'stoch': 0.25}
            }

            # Calcular señal final ponderada
            regime_weights = weights[market_regime]
            final_signal = 0
            for signal_name, signal_value in signals.items():
                if signal_name in regime_weights:
                    final_signal += signal_value * regime_weights[signal_name]

            return final_signal

        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0

    def calculate_risk_params(self, df: pd.DataFrame, market_regime: str) -> dict:
        """
        Calcula los parámetros de riesgo basados en la volatilidad actual
        """
        try:
            atr = AverageTrueRange(
                df['high'], df['low'], df['close']
            ).average_true_range()
            
            current_atr = atr.iloc[-1]
            avg_atr = atr.mean()
            volatility_factor = current_atr / avg_atr if avg_atr > 0 else 1.0

            # Obtener multiplicadores base del config
            stop_loss_mult = self.config.config['stop_loss_multiplier'][market_regime]
            take_profit_mult = self.config.config['take_profit_multiplier'][market_regime]

            # Ajustar por volatilidad
            return {
                'stop_loss_pct': stop_loss_mult * current_atr / df['close'].iloc[-1],
                'take_profit_pct': take_profit_mult * current_atr / df['close'].iloc[-1],
                'volatility_factor': volatility_factor
            }

        except Exception as e:
            logger.error(f"Error calculating risk parameters: {e}")
            return {
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'volatility_factor': 1.0
            }

    def execute_trade(self, signal: float, risk_params: dict) -> Optional[dict]:
        """
        Ejecuta una operación de trading con gestión de riesgos
        """
        logger.info(f"Attempting to execute trade with signal: {signal} and risk params: {risk_params}")
        try:
            # Verificar intervalo mínimo entre trades
            if (self.last_trade_time and 
                time.time() - self.last_trade_time < self.config.config['min_trade_interval']):
                return None

            # Obtener balance y calcular tamaño de posición
            balance = float(self.exchange.fetch_balance()['total']['USDT'])
            position_size = self.trade_manager.calculate_position_size(
                balance,
                self.config.config['risk_level'],
                risk_params['stop_loss_pct']
            )

            # Verificar posición existente
            current_position = self.exchange.fetch_position(self.symbol)
            if current_position and abs(float(current_position['size'])) > 0:
                # Cerrar posición existente
                side = 'sell' if current_position['side'] == 'long' else 'buy'
                self.exchange.create_market_order(
                    self.symbol,
                    side,
                    abs(float(current_position['size'])),
                    {'reduce_only': True}
                )

            # Crear nueva posición
            side = 'buy' if signal > 0 else 'sell'
            order = self.exchange.create_market_order(
                self.symbol,
                side,
                position_size,
                {
                    'leverage': self.config.config['leverage'],
                    'time_in_force': 'GoodTillCancel'
                }
            )

            if order:
                entry_price = float(order['price'])
                
                # Calcular niveles de stop loss y take profit
                if side == 'buy':
                    stop_loss = entry_price * (1 - risk_params['stop_loss_pct'])
                    take_profit = entry_price * (1 + risk_params['take_profit_pct'])
                else:
                    stop_loss = entry_price * (1 + risk_params['stop_loss_pct'])
                    take_profit = entry_price * (1 - risk_params['take_profit_pct'])

                # Colocar órdenes de stop loss y take profit
                self.exchange.create_order(
                    self.symbol,
                    'stop',
                    'sell' if side == 'buy' else 'buy',
                    position_size,
                    stop_loss,
                    {'stopPrice': stop_loss}
                )

                self.exchange.create_order(
                    self.symbol,
                    'limit',
                    'sell' if side == 'buy' else 'buy',
                    position_size,
                    take_profit
                )

                # Registrar trade
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'side': side,
                    'size': position_size,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'status': 'open'
                }
                
                self.trade_manager.trades.append(trade)
                self.trade_manager.save_trades()
                self.last_trade_time = time.time()

                return trade

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None

    def run_bot(self, stop_event: threading.Event) -> None:
        """
        Ejecuta el bot de trading en un ciclo continuo
        """
        while not stop_event.is_set():
            try:
                # Obtener y procesar datos
                df = self.fetch_data()
                df = self.add_technical_indicators(df)
                
                # Analizar mercado y calcular señales
                market_regime = self.calculate_market_regime(df)
                signal = self.calculate_signal_strength(df, market_regime)
                
                # Verificar si la señal supera el umbral
                if abs(signal) > self.config.config['signal_threshold'][market_regime]:
                    # Calcular parámetros de riesgo
                    risk_params = self.calculate_risk_params(df, market_regime)
                    
                    # Ejecutar trade
                    trade = self.execute_trade(signal, risk_params)
                    
                    if trade:
                        logger.info(f"Trade executed: {trade}")
                        self.message_queue.put(f"New trade: {trade['side']} {trade['size']} {self.symbol}")

                time.sleep(10)  # Esperar antes del siguiente ciclo

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.message_queue.put(f"Error: {str(e)}")
                time.sleep(60)  # Esperar más tiempo en caso de error

    def save_state(self) -> None:
        """
        Guarda el estado actual del bot
        """
        try:
            state = {
                'last_trade_time': self.last_trade_time,
                'performance_metrics': self.trade_manager.performance_metrics,
                'config': self.config.config
            }
            
            with open('bot_state.json', 'w') as f:
                json.dump(state, f)
                
            # Guardar modelo si existe
            if hasattr(self, 'model'):
                joblib.dump(self.model, 'trading_model.joblib')
                
        except Exception as e:
            logger.error(f"Error saving bot state: {e}")

    def load_state(self) -> None:
        """
        Carga el estado guardado del bot
        """
        try:
            if os.path.exists('bot_state.json'):
                with open('bot_state.json', 'r') as f:
                    state = json.load(f)
                    self.last_trade_time = state.get('last_trade_time')
                    self.trade_manager.performance_metrics = state.get('performance_metrics', {})
                    self.config.config.update(state.get('config', {}))
                    
            if os.path.exists('trading_model.joblib'):
                self.model = joblib.load('trading_model.joblib')
                
        except Exception as e:
            logger.error(f"Error loading bot state: {e}")

# Interfaz Streamlit mejorada
def create_streamlit_interface():
    st.set_page_config(
        page_title="Advanced Trading Bot Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar para configuración
    st.sidebar.title("Bot Configuration")
    
    # Campos de configuración con valores por defecto seguros
    api_key = st.sidebar.text_input("API Key", type="password")
    api_secret = st.sidebar.text_input("API Secret", type="password")
    symbol = st.sidebar.text_input("Trading Symbol", value="BTCUSDT")
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
        index=3
    )

    # Botones de control
    col1, col2 = st.sidebar.columns(2)
    start_bot = col1.button("Start Bot")
    stop_bot = col2.button("Stop Bot")

    # Configuración de riesgo
    st.sidebar.subheader("Risk Management")
    risk_level = st.sidebar.slider(
        "Risk per trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=2.0,
        step=0.1
    )
    leverage = st.sidebar.slider(
        "Leverage",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    # Inicialización del bot en session_state
    if 'bot' not in st.session_state:
        if api_key and api_secret:
            try:
                st.session_state.bot = OptimizedBybitBot(
                    api_key, api_secret, symbol, timeframe
                )
                st.session_state.bot.config.config['risk_level'] = risk_level / 100
                st.session_state.bot.config.config['leverage'] = leverage
            except Exception as e:
                st.error(f"Error initializing bot: {e}")
                return

    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = threading.Event()

    if 'bot_thread' not in st.session_state:
        st.session_state.bot_thread = None

    # Manejo de inicio/parada del bot
    if start_bot:
        if not st.session_state.bot_thread or not st.session_state.bot_thread.is_alive():
            st.session_state.stop_event.clear()
            st.session_state.bot_thread = threading.Thread(
                target=st.session_state.bot.run_bot,
                args=(st.session_state.stop_event,),
                daemon=True
            )
            st.session_state.bot_thread.start()
            st.sidebar.success("Bot started successfully!")

    if stop_bot:
        st.session_state.stop_event.set()
        if st.session_state.bot_thread:
            st.session_state.bot_thread.join(timeout=1)
            st.session_state.bot_thread = None
        st.sidebar.warning("Bot stopped!")

    # Pestañas principales
    tabs = st.tabs([
        "Dashboard",
        "Trade History",
        "Performance Metrics",
        "Market Analysis",
        "Logs"
    ])

    # Tab 1: Dashboard
    with tabs[0]:
        st.header("Trading Dashboard")
        
        try:
            if hasattr(st.session_state, 'bot'):
                df = st.session_state.bot.fetch_data()
                df = st.session_state.bot.add_technical_indicators(df)

                # Gráfico principal
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=('Price', 'Volume'),
                    row_heights=[0.7, 0.3]
                )

                # Candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name="OHLC"
                    ),
                    row=1, col=1
                )

                # Bollinger Bands
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['bb_high'],
                        name="BB Upper",
                        line=dict(color='gray', dash='dash')
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['bb_low'],
                        name="BB Lower",
                        line=dict(color='gray', dash='dash'),
                        fill='tonexty'
                    ),
                    row=1, col=1
                )

                # Volume
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['volume'],
                        name="Volume"
                    ),
                    row=2, col=1
                )

                fig.update_layout(
                    height=800,
                    title_text=f"{symbol} {timeframe} Chart",
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )

                st.plotly_chart(fig, use_container_width=True)

                # Métricas actuales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Current Price",
                        f"${df['close'].iloc[-1]:.2f}",
                        f"{((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "24h Volume",
                        f"${df['volume'].sum():.2f}"
                    )
                
                with col3:
                    st.metric(
                        "RSI",
                        f"{df['rsi'].iloc[-1]:.2f}"
                    )
                
                with col4:
                    market_regime = st.session_state.bot.calculate_market_regime(df)
                    st.metric(
                        "Market Regime",
                        market_regime.replace('_', ' ').title()
                    )

        except Exception as e:
            st.error(f"Error updating dashboard: {e}")

    # Tab 2: Trade History
    with tabs[1]:
        st.header("Trade History")
        if hasattr(st.session_state, 'bot'):
            trades_df = pd.DataFrame(st.session_state.bot.trade_manager.trades)
            if not trades_df.empty:
                # Agregar filtros
                status_filter = st.multiselect(
                    "Filter by Status",
                    trades_df['status'].unique(),
                    default=trades_df['status'].unique()
                )
                
                filtered_df = trades_df[trades_df['status'].isin(status_filter)]
                st.dataframe(
                    filtered_df.style.format({
                        'entry_price': '${:.2f}',
                        'stop_loss': '${:.2f}',
                        'take_profit': '${:.2f}',
                        'size': '{:.4f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No trades executed yet.")

    # Tab 3: Performance Metrics
    with tabs[2]:
        st.header("Performance Metrics")
        if hasattr(st.session_state, 'bot'):
            metrics = st.session_state.bot.trade_manager.performance_metrics
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total PnL", f"${metrics['total_pnl']:.2f}")
                st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
                st.metric("Total Trades", f"{metrics['wins'] + metrics['losses']}")
            
            with col2:
                st.metric("Largest Win", f"${metrics['largest_win']:.2f}")
                st.metric("Average Win", f"${metrics['avg_win']:.2f}")
                st.metric("Total Wins", f"{metrics['wins']}")
            
            with col3:
                st.metric("Largest Loss", f"${metrics['largest_loss']:.2f}")
                st.metric("Average Loss", f"${metrics['avg_loss']:.2f}")
                st.metric("Total Losses", f"{metrics['losses']}")
            
            # Gráfico de PnL acumulado
            if not trades_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trades_df['timestamp'],
                    y=trades_df['realized_pnl'].cumsum(),
                    mode='lines',
                    name='Cumulative PnL'
                ))
                fig.update_layout(
                    title="Cumulative Profit/Loss Over Time",
                    xaxis_title="Date",
                    yaxis_title="Cumulative PnL ($)"
                )
                st.plotly_chart(fig, use_container_width=True)

    # Tab 4: Market Analysis
    with tabs[3]:
        st.header("Market Analysis")
        if hasattr(st.session_state, 'bot'):
            try:
                # Mostrar indicadores técnicos actuales
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Technical Indicators")
                    indicators_df = pd.DataFrame({
                        'Indicator': [
                            'RSI',
                            'MACD',
                            'MACD Signal',
                            'Stochastic K',
                            'Stochastic D',
                            'BB Width'
                        ],
                        'Value': [
                            f"{df['rsi'].iloc[-1]:.2f}",
                            f"{df['macd'].iloc[-1]:.2f}",
                            f"{df['macd_signal'].iloc[-1]:.2f}",
                            f"{df['stoch_k'].iloc[-1]:.2f}",
                            f"{df['stoch_d'].iloc[-1]:.2f}",
                            f"{df['bb_width'].iloc[-1]:.2f}"
                        ]
                    })
                    st.dataframe(indicators_df, use_container_width=True)
                
                with col2:
                    st.subheader("Market Conditions")
                    market_df = pd.DataFrame({
                        'Metric': [
                            'Trend Strength',
                            'Volatility',
                            'Volume Trend',
                            'Price Trend'
                        ],
                        'Status': [
                            'Strong' if df['trend_strength'].iloc[-1] > df['trend_strength'].mean() else 'Weak',
                            'High' if df['volatility'].iloc[-1] > df['volatility'].mean() else 'Low',
                            'Increasing' if df['volume'].iloc[-1] > df['volume'].mean() else 'Decreasing',
                            'Bullish' if df['close'].iloc[-1] > df['ema_20'].iloc[-1] else 'Bearish'
                        ]
                    })
                    st.dataframe(market_df, use_container_width=True)

                # Gráfico de correlación de indicadores
                st.subheader("Indicator Correlation Matrix")
                corr_cols = ['rsi', 'macd', 'stoch_k', 'bb_width', 'close']
                corr_matrix = df[corr_cols].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                ))
                fig.update_layout(title="Indicator Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error in market analysis: {e}")

    # Tab 5: Logs
    with tabs[4]:
        st.header("System Logs")
        if hasattr(st.session_state, 'bot'):
            # Mostrar logs del sistema
            logs = []
            while not st.session_state.bot.message_queue.empty():
                logs.append(st.session_state.bot.message_queue.get())
            
            if logs:
                for log in logs:
                    st.text(log)
            else:
                st.info("No recent logs available.")

if __name__ == "__main__":
    create_streamlit_interface()
