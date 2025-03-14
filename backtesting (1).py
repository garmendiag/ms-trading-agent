"""
Backtesting Framework for MANUS Trading Agent

This module implements a backtesting framework to test and validate
the MANUS trading agent with historical data.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class BacktestingEngine:
    """
    Class for backtesting the MANUS trading agent.
    """
    
    def __init__(self, data_manager=None, indicators_manager=None, 
                 signal_manager=None, decision_manager=None,
                 risk_manager=None, cache_dir=None, output_dir=None):
        """
        Initialize the backtesting engine.
        
        Args:
            data_manager: Data collection manager instance.
            indicators_manager: Technical indicators manager instance.
            signal_manager: Signal generation manager instance.
            decision_manager: Decision logic manager instance.
            risk_manager: Risk management instance.
            cache_dir (str, optional): Directory to cache data. Defaults to None.
            output_dir (str, optional): Directory to save output files. Defaults to None.
        """
        self.data_manager = data_manager
        self.indicators_manager = indicators_manager
        self.signal_manager = signal_manager
        self.decision_manager = decision_manager
        self.risk_manager = risk_manager
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize backtesting results
        self.results = {
            'trades': [],
            'performance': {
                'NQ-ES': {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit_loss': 0,
                    'max_drawdown': 0,
                    'win_rate': 0
                },
                'MES-MNQ': {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit_loss': 0,
                    'max_drawdown': 0,
                    'win_rate': 0
                }
            },
            'equity_curve': [],
            'drawdown_curve': [],
            'signals': []
        }
    
    def load_historical_data(self, start_date, end_date, interval='1h'):
        """
        Load historical data for backtesting.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            interval (str, optional): Data interval. Defaults to '1h'.
        
        Returns:
            dict: Dictionary containing historical data.
        """
        if self.data_manager:
            # Use data manager to load historical data
            print(f"Loading historical data from {start_date} to {end_date} with interval {interval}")
            
            # This is a placeholder for actual data loading
            # In a real implementation, you would use the data manager to load historical data
            
            # For now, we'll use a simplified approach to load ES and NQ data
            from ..data_collection.market_index import MarketIndexCollector
            collector = MarketIndexCollector(cache_dir=self.cache_dir)
            
            # Convert date strings to period
            start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            days = (end - start).days
            period = f"{days}d"
            
            # Load ES data
            es_data = collector.get_historical_data('es', period=period, interval=interval)
            
            # Load NQ data
            nq_data = collector.get_historical_data('nq', period=period, interval=interval)
            
            # Load VIX data
            vix_data = collector.get_historical_data('^VIX', period=period, interval=interval)
            
            return {
                'es': es_data,
                'nq': nq_data,
                'vix': vix_data
            }
        else:
            print("Warning: No data manager available. Using simulated data.")
            return self._generate_simulated_data(start_date, end_date, interval)
    
    def _generate_simulated_data(self, start_date, end_date, interval='1h'):
        """
        Generate simulated data for testing.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            interval (str, optional): Data interval. Defaults to '1h'.
        
        Returns:
            dict: Dictionary containing simulated data.
        """
        # Convert date strings to datetime objects
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        # Determine number of periods based on interval
        if interval == '1h':
            # Assuming 8 trading hours per day, 5 days per week
            periods = int((end - start).days * 8 * 5 / 7)
        elif interval == '1d':
            # Assuming 5 trading days per week
            periods = int((end - start).days * 5 / 7)
        else:
            # Default to hourly
            periods = int((end - start).days * 8 * 5 / 7)
        
        # Generate date range
        date_range = pd.date_range(start=start, periods=periods, freq=interval)
        
        # Generate simulated ES data
        es_data = pd.DataFrame(index=date_range)
        es_data['Open'] = np.random.normal(5000, 50, len(date_range))
        es_data['High'] = es_data['Open'] + np.random.normal(10, 5, len(date_range))
        es_data['Low'] = es_data['Open'] - np.random.normal(10, 5, len(date_range))
        es_data['Close'] = np.random.normal(es_data['Open'], 15, len(date_range))
        es_data['Volume'] = np.random.normal(1000000, 200000, len(date_range))
        
        # Generate simulated NQ data
        nq_data = pd.DataFrame(index=date_range)
        nq_data['Open'] = np.random.normal(17500, 150, len(date_range))
        nq_data['High'] = nq_data['Open'] + np.random.normal(30, 15, len(date_range))
        nq_data['Low'] = nq_data['Open'] - np.random.normal(30, 15, len(date_range))
        nq_data['Close'] = np.random.normal(nq_data['Open'], 45, len(date_range))
        nq_data['Volume'] = np.random.normal(800000, 150000, len(date_range))
        
        # Generate simulated VIX data
        vix_data = pd.DataFrame(index=date_range)
        vix_data['Open'] = np.random.normal(15, 3, len(date_range))
        vix_data['High'] = vix_data['Open'] + np.random.normal(1, 0.5, len(date_range))
        vix_data['Low'] = vix_data['Open'] - np.random.normal(1, 0.5, len(date_range))
        vix_data['Close'] = np.random.normal(vix_data['Open'], 1.5, len(date_range))
        vix_data['Volume'] = np.random.normal(500000, 100000, len(date_range))
        
        return {
            'es': es_data,
            'nq': nq_data,
            'vix': vix_data
        }
    
    def run_backtest(self, historical_data, start_date=None, end_date=None):
        """
        Run backtest using historical data.
        
        Args:
            historical_data (dict): Dictionary containing historical data.
            start_date (str, optional): Start date for backtest in 'YYYY-MM-DD' format.
            end_date (str, optional): End date for backtest in 'YYYY-MM-DD' format.
        
        Returns:
            dict: Dictionary containing backtest results.
        """
        # Extract data
        es_data = historical_data.get('es')
        nq_data = historical_data.get('nq')
        vix_data = historical_data.get('vix')
        
        if es_data is None or nq_data is None:
            print("Error: Missing required data for backtest")
            return None
        
        # Filter data by date range if specified
        if start_date:
            start = pd.to_datetime(start_date)
            es_data = es_data[es_data.index >= start]
            nq_data = nq_data[nq_data.index >= start]
            if vix_data is not None:
                vix_data = vix_data[vix_data.index >= start]
        
        if end_date:
            end = pd.to_datetime(end_date)
            es_data = es_data[es_data.index <= end]
            nq_data = nq_data[nq_data.index <= end]
            if vix_data is not None:
                vix_data = vix_data[vix_data.index <= end]
        
        # Initialize positions and performance tracking
        positions = {
            'ES': {'direction': 'none', 'quantity': 0, 'entry_price': 0, 'entry_time': None},
            'NQ': {'direction': 'none', 'quantity': 0, 'entry_price': 0, 'entry_time': None},
            'MES': {'direction': 'none', 'quantity': 0, 'entry_price': 0, 'entry_time': None},
            'MNQ': {'direction': 'none', 'quantity': 0, 'entry_price': 0, 'entry_time': None}
        }
        
        performance = {
            'NQ-ES': {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'profit_loss': 0,
                'max_drawdown': 0,
                'equity': [0],
                'drawdown': [0]
            },
            'MES-MNQ': {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'profit_loss': 0,
                'max_drawdown': 0,
                'equity': [0],
                'drawdown': [0]
            }
        }
        
        trades = []
        signals = []
        
        # Initialize risk state
        risk_state = {
            'NQ-ES': {
                'current_loss': 0,
                'max_loss_reached': False,
                'daily_loss': 0,
                'daily_loss_reached': False,
                'trailing_stop_level': None,
                'risk_level': 'normal'
            },
            'MES-MNQ': {
                'current_loss': 0,
                'max_loss_reached': False,
                'daily_loss': 0,
                'daily_loss_reached': False,
                'trailing_stop_level': None,
                'risk_level': 'normal'
            },
            'trading_enabled': True
        }
        
        # Run backtest for each time period
        for i in range(len(es_data)):
            # Get current data point
            current_time = es_data.index[i]
            es_price = es_data['Close'].iloc[i]
            nq_price = nq_data['Close'].iloc[i]
            
            # Create market prices dictionary
            market_prices = {
                'ES': es_price,
                'NQ': nq_price,
                'MES': es_price,
                'MNQ': nq_price
            }
            
            # Calculate technical indicators
            if self.indicators_manager:
                # Use a window of data for indicator calculation
                window_start = max(0, i - 100)
                es_window = es_data.iloc[window_start:i+1]
                
                indicators_data = self.indicators_manager.calculate_all_indicators(es_window, pip_value=0.25)
                all_signals = self.indicators_manager.get_all_signals(indicators_data)
                combined_signal = self.indicators_manager.generate_combined_signal(all_signals)
                
                # Format technical indicators for MANUS
                technical_indicators = {
                    'rsi': all_signals['rsi'],
                    'market_structure': all_signals['market_structure'],
                    'order_blocks': all_signals['order_blocks'],
                    'optimal_trade_entry': all_signals['optimal_trade_entry'],
                    'liquidity_model': all_signals['liquidity_model'],
                    'combined_signal': combined_signal
                }
            else:
                # Use simplified indicators
                technical_indicators = {
                    'combined_signal': {
                        'bias': 'Bullish' if es_data['Close'].iloc[i] > es_data['Close'].iloc[i-1] else 'Bearish',
                        'signal_strength': 60,
                        'confidence': 'Medium'
                    }
                }
            
            # Generate trading signal
            if self.signal_manager:
                # Create market data dictionary
                market_data = {
                    'market_data': {
                        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'es_price': es_price,
                        'nq_price': nq_price,
                        'vix_level': vix_data['Close'].iloc[i] if vix_data is not None else 15,
                        'vix_change': (vix_data['Close'].iloc[i] / vix_data['Close'].iloc[i-1] - 1) * 100 if vix_data is not None else 0,
                        'market_trend': 'bullish' if es_data['Close'].iloc[i] > es_data['Close'].iloc[i-5] else 'bearish'
                    },
                    'news_sentiment': {
                        'overall_sentiment': 0.2,
                        'sentiment_label': 'Slightly Positive'
                    }
                }
                
                # Generate signal
                signal_data = self.signal_manager.generate_trading_signal(market_data, es_window)
                execution_signal = signal_data['execution_signal']
            else:
                # Generate simplified signal based on price movement
                if es_data['Close'].iloc[i] > es_data['Close'].iloc[i-1]:
                    # Bullish signal
                    execution_signal = {
                        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'market_direction': 'rise',
                        'strategy_nq_es': {
                            'signal': 'Market Rises: 1 ES long, 1 NQ short',
                            'positions': {
                                'ES': {'direction': 'long', 'quantity': 1},
                                'NQ': {'direction': 'short', 'quantity': 1}
                            }
                        },
                        'strategy_mes_mnq': {
                            'signal': 'Market Rises: 1 MES long, 2 MNQ short',
                            'positions': {
                                'MES': {'direction': 'long', 'quantity': 1},
                                'MNQ': {'direction': 'short', 'quantity': 2}
                            }
                        }
                    }
                else:
                    # Bearish signal
                    execution_signal = {
                        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'market_direction': 'fall',
                        'strategy_nq_es': {
                            'signal': 'Market Falls: 1 ES short, 1 NQ long',
                            'positions': {
                                'ES': {'direction': 'short', 'quantity': 1},
                                'NQ': {'direction': 'long', 'quantity': 1}
                            }
                        },
                        'strategy_mes_mnq': {
                            'signal': 'Market Falls: 1 MES short, 2 MNQ long',
                            'positions': {
                                'MES': {'direction': 'short', 'quantity': 1},
                                'MNQ': {'direction': 'long', 'quantity': 2}
                            }
                        }
                    }
            
            # Record signal
            signals.append({
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'es_price': es_price,
                'nq_price': nq_price,
                'signal': execution_signal
            })
            
            # Check risk limits
            if self.risk_manager:
                # Update risk state
                positions_data = {
                    'positions': positions,
                    'nq_es_pl': self._calculate_position_pl('NQ-ES', positions, market_prices),
                    'mes_mnq_pl': self._calculate_position_pl('MES-MNQ', positions, market_prices)
                }
                
                performance_data = {
                    'NQ-ES': {
                        'profit_loss': performance['NQ-ES']['profit_loss']
                    },
                    'MES-MNQ': {
                        'profit_loss': performance['MES-MNQ']['profit_loss']
                    }
                }
                
                risk_state = self.risk_manager.update_risk_state(performance_data, positions_data)
                
                # Apply stop-loss rules
                positions_to_close = self.risk_manager.apply_stop_loss(positions_data, market_prices)
                
                # Close positions if needed
                for symbol, close_info in positions_to_close.items():
                    # Determine which strategy this symbol belongs to
                    strategy = 'NQ-ES' if symbol in ['ES', 'NQ'] else 'MES-MNQ'
                    
                    # Calculate profit/loss
                    position = positions[symbol]
                    if position['direction'] == 'long':
                        pl = (market_prices[symbol] - position['entry_price']) * position['quantity']
                    else:  # short
                        pl = (position['entry_price'] - market_prices[symbol]) * position['quantity']
                    
                    # Apply contract multiplier
                    if symbol == 'ES':
                        pl *= 50  # ES contract multiplier
                    elif symbol == 'NQ':
                        pl *= 20  # NQ contract multiplier
                    elif symbol == 'MES':
                        pl *= 5  # MES contract multiplier (1/10 of ES)
                    elif symbol == 'MNQ':
                        pl *= 2  # MNQ contract multiplier (1/10 of NQ)
                    
                    # Record trade
                    trades.append({
                        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'symbol': symbol,
                        'direction': position['direction'],
                        'quantity': position['quantity'],
                        'entry_price': position['entry_price'],
                        'entry_time': position['entry_time'],
                        'exit_price': market_prices[symbol],
                        'exit_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'profit_loss': pl,
                        'reason': close_info['reason'],
                        'strategy': strategy
                    })
                    
                    # Update performance
                    performance[strategy]['trades'] += 1
                    performance[strategy]['profit_loss'] += pl
                    
                    if pl > 0:
                        performance[strategy]['wins'] += 1
                    else:
                        performance[strategy]['losses'] += 1
                    
                    # Update equity curve
                    performance[strategy]['equity'].append(performance[strategy]['profit_loss'])
                    
                    # Update drawdown
                    max_equity = max(performance[strategy]['equity'])
                    current_drawdown = max_equity - performance[strategy]['profit_loss']
                    performance[strategy]['drawdown'].append(current_drawdown)
                    
                    if current_drawdown > performance[strategy]['max_drawdown']:
                        performance[strategy]['max_drawdown'] = current_drawdown
                    
                    # Reset position
                    positions[symbol] = {
                        'direction': 'none',
                        'quantity': 0,
                        'entry_price': 0,
                        'entry_time': None
                    }
            
            # Check if trading is enabled
            if not risk_state['trading_enabled']:
                continue
            
            # Execute trades based on signal
            # NQ-ES strategy
            nq_es_signal = execution_signal.get('strategy_nq_es', {}).get('signal', 'Hold: No clear signal')
            nq_es_positions = execution_signal.get('strategy_nq_es', {}).get('positions', {})
            
            if 'Hold' not in nq_es_signal:
                # Check if we need to close existing positions
                for symbol in ['ES', 'NQ']:
                    if positions[symbol]['direction'] != 'none':
                        new_direction = nq_es_positions.get(symbol, {}).get('direction', 'none')
                        new_quantity = nq_es_positions.get(symbol, {}).get('quantity', 0)
                        
                        if positions[symbol]['direction'] != new_direction or positions[symbol]['quantity'] != new_quantity:
                            # Close position
                            position = positions[symbol]
                            if position['direction'] == 'long':
                                pl = (market_prices[symbol] - position['entry_price']) * position['quantity']
                            else:  # short
                                pl = (position['entry_price'] - market_prices[symbol]) * position['quantity']
                            
                            # Apply contract multiplier
                            if symbol == 'ES':
                                pl *= 50  # ES contract multiplier
                            elif symbol == 'NQ':
                                pl *= 20  # NQ contract multiplier
                            
                            # Record trade
                            trades.append({
                                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'symbol': symbol,
                                'direction': position['direction'],
                                'quantity': position['quantity'],
                                'entry_price': position['entry_price'],
                                'entry_time': position['entry_time'],
                                'exit_price': market_prices[symbol],
                                'exit_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'profit_loss': pl,
                                'reason': 'Signal change',
                                'strategy': 'NQ-ES'
                            })
                            
                            # Update performance
                            performance['NQ-ES']['trades'] += 1
                            performance['NQ-ES']['profit_loss'] += pl
                            
                            if pl > 0:
                                performance['NQ-ES']['wins'] += 1
                            else:
                                performance['NQ-ES']['losses'] += 1
                            
                            # Reset position
                            positions[symbol] = {
                                'direction': 'none',
                                'quantity': 0,
                                'entry_price': 0,
                                'entry_time': None
                            }
                
                # Open new positions
                for symbol in ['ES', 'NQ']:
                    direction = nq_es_positions.get(symbol, {}).get('direction', 'none')
                    quantity = nq_es_positions.get(symbol, {}).get('quantity', 0)
                    
                    if direction != 'none' and quantity > 0 and positions[symbol]['direction'] == 'none':
                        # Open position
                        positions[symbol] = {
                            'direction': direction,
                            'quantity': quantity,
                            'entry_price': market_prices[symbol],
                            'entry_time': current_time.strftime('%Y-%m-%d %H:%M:%S')
                        }
            
            # MES-MNQ strategy
            mes_mnq_signal = execution_signal.get('strategy_mes_mnq', {}).get('signal', 'Hold: No clear signal')
            mes_mnq_positions = execution_signal.get('strategy_mes_mnq', {}).get('positions', {})
            
            if 'Hold' not in mes_mnq_signal:
                # Check if we need to close existing positions
                for symbol in ['MES', 'MNQ']:
                    if positions[symbol]['direction'] != 'none':
                        new_direction = mes_mnq_positions.get(symbol, {}).get('direction', 'none')
                        new_quantity = mes_mnq_positions.get(symbol, {}).get('quantity', 0)
                        
                        if positions[symbol]['direction'] != new_direction or positions[symbol]['quantity'] != new_quantity:
                            # Close position
                            position = positions[symbol]
                            if position['direction'] == 'long':
                                pl = (market_prices[symbol] - position['entry_price']) * position['quantity']
                            else:  # short
                                pl = (position['entry_price'] - market_prices[symbol]) * position['quantity']
                            
                            # Apply contract multiplier
                            if symbol == 'MES':
                                pl *= 5  # MES contract multiplier
                            elif symbol == 'MNQ':
                                pl *= 2  # MNQ contract multiplier
                            
                            # Record trade
                            trades.append({
                                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'symbol': symbol,
                                'direction': position['direction'],
                                'quantity': position['quantity'],
                                'entry_price': position['entry_price'],
                                'entry_time': position['entry_time'],
                                'exit_price': market_prices[symbol],
                                'exit_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'profit_loss': pl,
                                'reason': 'Signal change',
                                'strategy': 'MES-MNQ'
                            })
                            
                            # Update performance
                            performance['MES-MNQ']['trades'] += 1
                            performance['MES-MNQ']['profit_loss'] += pl
                            
                            if pl > 0:
                                performance['MES-MNQ']['wins'] += 1
                            else:
                                performance['MES-MNQ']['losses'] += 1
                            
                            # Reset position
                            positions[symbol] = {
                                'direction': 'none',
                                'quantity': 0,
                                'entry_price': 0,
                                'entry_time': None
                            }
                
                # Open new positions
                for symbol in ['MES', 'MNQ']:
                    direction = mes_mnq_positions.get(symbol, {}).get('direction', 'none')
                    quantity = mes_mnq_positions.get(symbol, {}).get('quantity', 0)
                    
                    if direction != 'none' and quantity > 0 and positions[symbol]['direction'] == 'none':
                        # Open position
                        positions[symbol] = {
                            'direction': direction,
                            'quantity': quantity,
                            'entry_price': market_prices[symbol],
                            'entry_time': current_time.strftime('%Y-%m-%d %H:%M:%S')
                        }
            
            # Update equity curve
            performance['NQ-ES']['equity'].append(performance['NQ-ES']['profit_loss'])
            performance['MES-MNQ']['equity'].append(performance['MES-MNQ']['profit_loss'])
            
            # Update drawdown
            max_equity_nq_es = max(performance['NQ-ES']['equity'])
            current_drawdown_nq_es = max_equity_nq_es - performance['NQ-ES']['profit_loss']
            performance['NQ-ES']['drawdown'].append(current_drawdown_nq_es)
            
            if current_drawdown_nq_es > performance['NQ-ES']['max_drawdown']:
                performance['NQ-ES']['max_drawdown'] = current_drawdown_nq_es
            
            max_equity_mes_mnq = max(performance['MES-MNQ']['equity'])
            current_drawdown_mes_mnq = max_equity_mes_mnq - performance['MES-MNQ']['profit_loss']
            performance['MES-MNQ']['drawdown'].append(current_drawdown_mes_mnq)
            
            if current_drawdown_mes_mnq > performance['MES-MNQ']['max_drawdown']:
                performance['MES-MNQ']['max_drawdown'] = current_drawdown_mes_mnq
        
        # Close any remaining positions at the end of the backtest
        for symbol, position in positions.items():
            if position['direction'] != 'none':
                # Determine which strategy this symbol belongs to
                strategy = 'NQ-ES' if symbol in ['ES', 'NQ'] else 'MES-MNQ'
                
                # Calculate profit/loss
                if position['direction'] == 'long':
                    pl = (market_prices[symbol] - position['entry_price']) * position['quantity']
                else:  # short
                    pl = (position['entry_price'] - market_prices[symbol]) * position['quantity']
                
                # Apply contract multiplier
                if symbol == 'ES':
                    pl *= 50  # ES contract multiplier
                elif symbol == 'NQ':
                    pl *= 20  # NQ contract multiplier
                elif symbol == 'MES':
                    pl *= 5  # MES contract multiplier
                elif symbol == 'MNQ':
                    pl *= 2  # MNQ contract multiplier
                
                # Record trade
                trades.append({
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': symbol,
                    'direction': position['direction'],
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'entry_time': position['entry_time'],
                    'exit_price': market_prices[symbol],
                    'exit_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'profit_loss': pl,
                    'reason': 'End of backtest',
                    'strategy': strategy
                })
                
                # Update performance
                performance[strategy]['trades'] += 1
                performance[strategy]['profit_loss'] += pl
                
                if pl > 0:
                    performance[strategy]['wins'] += 1
                else:
                    performance[strategy]['losses'] += 1
        
        # Calculate win rates
        for strategy in ['NQ-ES', 'MES-MNQ']:
            if performance[strategy]['trades'] > 0:
                performance[strategy]['win_rate'] = performance[strategy]['wins'] / performance[strategy]['trades'] * 100
        
        # Store results
        self.results = {
            'trades': trades,
            'performance': {
                'NQ-ES': {
                    'trades': performance['NQ-ES']['trades'],
                    'wins': performance['NQ-ES']['wins'],
                    'losses': performance['NQ-ES']['losses'],
                    'profit_loss': performance['NQ-ES']['profit_loss'],
                    'max_drawdown': performance['NQ-ES']['max_drawdown'],
                    'win_rate': performance['NQ-ES']['win_rate']
                },
                'MES-MNQ': {
                    'trades': performance['MES-MNQ']['trades'],
                    'wins': performance['MES-MNQ']['wins'],
                    'losses': performance['MES-MNQ']['losses'],
                    'profit_loss': performance['MES-MNQ']['profit_loss'],
                    'max_drawdown': performance['MES-MNQ']['max_drawdown'],
                    'win_rate': performance['MES-MNQ']['win_rate']
                }
            },
            'equity_curve': {
                'NQ-ES': performance['NQ-ES']['equity'],
                'MES-MNQ': performance['MES-MNQ']['equity']
            },
            'drawdown_curve': {
                'NQ-ES': performance['NQ-ES']['drawdown'],
                'MES-MNQ': performance['MES-MNQ']['drawdown']
            },
            'signals': signals
        }
        
        return self.results
    
    def _calculate_position_pl(self, strategy, positions, market_prices):
        """
        Calculate profit/loss for current positions.
        
        Args:
            strategy (str): Strategy name ('NQ-ES' or 'MES-MNQ').
            positions (dict): Dictionary containing positions.
            market_prices (dict): Dictionary containing current market prices.
        
        Returns:
            float: Current profit/loss.
        """
        total_pl = 0
        
        if strategy == 'NQ-ES':
            symbols = ['ES', 'NQ']
        else:  # MES-MNQ
            symbols = ['MES', 'MNQ']
        
        for symbol in symbols:
            position = positions[symbol]
            if position['direction'] == 'none' or position['quantity'] == 0:
                continue
            
            # Calculate profit/loss
            if position['direction'] == 'long':
                pl = (market_prices[symbol] - position['entry_price']) * position['quantity']
            else:  # short
                pl = (position['entry_price'] - market_prices[symbol]) * position['quantity']
            
            # Apply contract multiplier
            if symbol == 'ES':
                pl *= 50  # ES contract multiplier
            elif symbol == 'NQ':
                pl *= 20  # NQ contract multiplier
            elif symbol == 'MES':
                pl *= 5  # MES contract multiplier (1/10 of ES)
            elif symbol == 'MNQ':
                pl *= 2  # MNQ contract multiplier (1/10 of NQ)
            
            total_pl += pl
        
        return total_pl
    
    def generate_backtest_report(self, output_file=None):
        """
        Generate a backtest report.
        
        Args:
            output_file (str, optional): File to save the report to. If None, generates a timestamped filename.
        
        Returns:
            str: Path to the saved report file.
        """
        if output_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"backtest_report_{timestamp}.md"
        
        if self.output_dir:
            filepath = os.path.join(self.output_dir, output_file)
        else:
            filepath = output_file
        
        # Generate report content
        report = f"""# MANUS Trading System Backtest Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary

### NQ-ES Strategy
- Total Trades: {self.results['performance']['NQ-ES']['trades']}
- Wins: {self.results['performance']['NQ-ES']['wins']}
- Losses: {self.results['performance']['NQ-ES']['losses']}
- Win Rate: {self.results['performance']['NQ-ES']['win_rate']:.2f}%
- Total P/L: ${self.results['performance']['NQ-ES']['profit_loss']:.2f}
- Max Drawdown: ${self.results['performance']['NQ-ES']['max_drawdown']:.2f}

### MES-MNQ Strategy
- Total Trades: {self.results['performance']['MES-MNQ']['trades']}
- Wins: {self.results['performance']['MES-MNQ']['wins']}
- Losses: {self.results['performance']['MES-MNQ']['losses']}
- Win Rate: {self.results['performance']['MES-MNQ']['win_rate']:.2f}%
- Total P/L: ${self.results['performance']['MES-MNQ']['profit_loss']:.2f}
- Max Drawdown: ${self.results['performance']['MES-MNQ']['max_drawdown']:.2f}

## Overall Performance
- Total P/L: ${self.results['performance']['NQ-ES']['profit_loss'] + self.results['performance']['MES-MNQ']['profit_loss']:.2f}
"""
        
        # Add trade history
        report += "\n## Trade History\n\n"
        
        if self.results['trades']:
            report += "| Timestamp | Symbol | Direction | Quantity | Entry Price | Exit Price | P/L | Reason | Strategy |\n"
            report += "|-----------|--------|-----------|----------|-------------|------------|-----|--------|----------|\n"
            
            for trade in self.results['trades']:
                report += f"| {trade['timestamp']} | {trade['symbol']} | {trade['direction']} | {trade['quantity']} | {trade['entry_price']:.2f} | {trade['exit_price']:.2f} | ${trade['profit_loss']:.2f} | {trade['reason']} | {trade['strategy']} |\n"
        else:
            report += "No trades recorded during backtest.\n"
        
        # Write report to file
        with open(filepath, 'w') as f:
            f.write(report)
        
        # Generate performance charts
        chart_path = self.generate_performance_charts()
        if chart_path:
            # Update the report with chart reference
            with open(filepath, 'a') as f:
                f.write(f"\n\n## Performance Charts\n\n![Performance Charts]({os.path.basename(chart_path)})\n")
        
        return filepath
    
    def generate_performance_charts(self, output_file=None):
        """
        Generate performance charts.
        
        Args:
            output_file (str, optional): File to save the charts to. If None, generates a timestamped filename.
        
        Returns:
            str: Path to the saved chart file.
        """
        if not self.output_dir:
            return None
        
        if output_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"backtest_charts_{timestamp}.png"
        
        filepath = os.path.join(self.output_dir, output_file)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot equity curves
        ax1.plot(self.results['equity_curve']['NQ-ES'], label='NQ-ES Equity', color='blue')
        ax1.plot(self.results['equity_curve']['MES-MNQ'], label='MES-MNQ Equity', color='green')
        
        # Add combined equity curve
        combined_equity = [a + b for a, b in zip(self.results['equity_curve']['NQ-ES'], self.results['equity_curve']['MES-MNQ'])]
        ax1.plot(combined_equity, label='Combined Equity', color='red', linewidth=2)
        
        # Set title and labels
        ax1.set_title('Equity Curves')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Profit/Loss ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdown curves
        ax2.plot(self.results['drawdown_curve']['NQ-ES'], label='NQ-ES Drawdown', color='blue')
        ax2.plot(self.results['drawdown_curve']['MES-MNQ'], label='MES-MNQ Drawdown', color='green')
        
        # Add combined drawdown curve
        combined_drawdown = [max(a, b) for a, b in zip(self.results['drawdown_curve']['NQ-ES'], self.results['drawdown_curve']['MES-MNQ'])]
        ax2.plot(combined_drawdown, label='Combined Drawdown', color='red', linewidth=2)
        
        # Set title and labels
        ax2.set_title('Drawdown Curves')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Drawdown ($)')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(filepath)
        plt.close()
        
        return filepath
    
    def optimize_parameters(self, historical_data, parameter_ranges, metric='profit_loss'):
        """
        Optimize strategy parameters using grid search.
        
        Args:
            historical_data (dict): Dictionary containing historical data.
            parameter_ranges (dict): Dictionary containing parameter ranges to test.
            metric (str, optional): Metric to optimize. Defaults to 'profit_loss'.
        
        Returns:
            dict: Dictionary containing optimization results.
        """
        # This is a placeholder for parameter optimization
        # In a real implementation, you would perform grid search over parameter ranges
        
        print("Parameter optimization not implemented yet")
        return None

# Example usage
if __name__ == "__main__":
    # Create backtesting engine
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    
    backtest_engine = BacktestingEngine(
        cache_dir=cache_dir,
        output_dir=output_dir
    )
    
    # Load historical data
    historical_data = backtest_engine.load_historical_data(
        start_date='2023-01-01',
        end_date='2023-12-31',
        interval='1h'
    )
    
    # Run backtest
    results = backtest_engine.run_backtest(historical_data)
    
    # Print results
    print("Backtest Results:")
    print(f"  NQ-ES Trades: {results['performance']['NQ-ES']['trades']}")
    print(f"  NQ-ES Win Rate: {results['performance']['NQ-ES']['win_rate']:.2f}%")
    print(f"  NQ-ES P/L: ${results['performance']['NQ-ES']['profit_loss']:.2f}")
    print(f"  MES-MNQ Trades: {results['performance']['MES-MNQ']['trades']}")
    print(f"  MES-MNQ Win Rate: {results['performance']['MES-MNQ']['win_rate']:.2f}%")
    print(f"  MES-MNQ P/L: ${results['performance']['MES-MNQ']['profit_loss']:.2f}")
    
    # Generate backtest report
    report_file = backtest_engine.generate_backtest_report()
    print(f"\nBacktest report saved to: {report_file}")
