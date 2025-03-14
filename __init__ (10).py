"""
Technical Indicators Module Integration

This module integrates all the technical indicators into a single interface
for the MANUS trading agent. It provides a unified way to calculate and
analyze various technical indicators, including ICT concepts.
"""

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import all technical indicator modules
from .rsi_indicator import RSIIndicator
from .ict_order_blocks import OrderBlocksIndicator
from .ict_market_structure import MarketStructureIndicator
from .ict_optimal_trade_entry import OptimalTradeEntryIndicator
from .ict_liquidity_model import LiquidityModelIndicator

class TechnicalIndicatorsManager:
    """
    Class for managing and integrating all technical indicators.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the technical indicators manager.
        
        Args:
            output_dir (str, optional): Directory to save plots and output files.
                                       Defaults to None.
        """
        self.output_dir = output_dir
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize indicators
        self.rsi = RSIIndicator(period=14, overbought=70, oversold=30)
        self.order_blocks = OrderBlocksIndicator(lookback=10, threshold_pips=5, min_size_pips=10)
        self.market_structure = MarketStructureIndicator(lookback=10, swing_threshold=0.001)
        self.optimal_trade_entry = OptimalTradeEntryIndicator(lookback=15)
        self.liquidity_model = LiquidityModelIndicator(lookback=15, equal_level_pips=2, liquidity_threshold=3)
    
    def calculate_all_indicators(self, price_data, pip_value=0.0001):
        """
        Calculate all technical indicators for the given price data.
        
        Args:
            price_data (pandas.DataFrame): Price data with OHLC columns.
            pip_value (float, optional): Value of 1 pip in price units. Defaults to 0.0001.
        
        Returns:
            dict: Dictionary containing all calculated indicators.
        """
        # Calculate RSI
        rsi_data = self.rsi.calculate(price_data)
        rsi_signals = self.rsi.generate_signals(rsi_data)
        
        # Calculate Order Blocks
        ob_data = self.order_blocks.calculate(price_data, pip_value)
        ob_signals = self.order_blocks.generate_signals(ob_data)
        
        # Calculate Market Structure
        ms_data = self.market_structure.calculate(price_data)
        ms_signals = self.market_structure.generate_signals(ms_data)
        
        # Calculate Optimal Trade Entry
        ote_data = self.optimal_trade_entry.calculate(price_data, ms_data)
        ote_signals = self.optimal_trade_entry.generate_signals(ote_data)
        
        # Calculate Liquidity Model
        liq_data = self.liquidity_model.calculate(price_data, pip_value)
        liq_signals = self.liquidity_model.generate_signals(liq_data)
        
        # Combine all indicators into a single DataFrame
        # Start with the original price data
        result = price_data.copy()
        
        # Add RSI columns
        rsi_cols = [col for col in rsi_signals.columns if col not in result.columns and col.startswith('RSI')]
        for col in rsi_cols:
            result[col] = rsi_signals[col]
        
        # Add Order Blocks columns
        ob_cols = [col for col in ob_signals.columns if col not in result.columns and (col.startswith('OB') or col.startswith('Bullish_OB') or col.startswith('Bearish_OB'))]
        for col in ob_cols:
            result[col] = ob_signals[col]
        
        # Add Market Structure columns
        ms_cols = [col for col in ms_signals.columns if col not in result.columns and (col.startswith('MS') or col.startswith('Swing') or col.startswith('Higher') or col.startswith('Lower'))]
        for col in ms_cols:
            result[col] = ms_signals[col]
        
        # Add Optimal Trade Entry columns
        ote_cols = [col for col in ote_signals.columns if col not in result.columns and col.startswith('OTE')]
        for col in ote_cols:
            result[col] = ote_signals[col]
        
        # Add Liquidity Model columns
        liq_cols = [col for col in liq_signals.columns if col not in result.columns and (col.startswith('Liquidity') or col.startswith('Buy_Liquidity') or col.startswith('Sell_Liquidity'))]
        for col in liq_cols:
            result[col] = liq_signals[col]
        
        return {
            'combined_data': result,
            'rsi_data': rsi_signals,
            'ob_data': ob_signals,
            'ms_data': ms_signals,
            'ote_data': ote_signals,
            'liq_data': liq_signals
        }
    
    def get_all_signals(self, data):
        """
        Get all current trading signals from the calculated indicators.
        
        Args:
            data (dict): Dictionary containing all calculated indicators.
        
        Returns:
            dict: Dictionary containing all current signals.
        """
        # Get current signals from each indicator
        rsi_signal = self.rsi.get_current_signal(data['rsi_data'])
        ob_signal = self.order_blocks.get_current_signal(data['ob_data'])
        ms_signal = self.market_structure.get_current_signal(data['ms_data'])
        ote_signal = self.optimal_trade_entry.get_current_signal(data['ote_data'])
        liq_signal = self.liquidity_model.get_current_signal(data['liq_data'])
        
        # Combine all signals
        return {
            'rsi': rsi_signal,
            'order_blocks': ob_signal,
            'market_structure': ms_signal,
            'optimal_trade_entry': ote_signal,
            'liquidity_model': liq_signal
        }
    
    def generate_combined_signal(self, signals):
        """
        Generate a combined trading signal based on all individual signals.
        
        Args:
            signals (dict): Dictionary containing all current signals.
        
        Returns:
            dict: Dictionary containing the combined signal.
        """
        # Count bullish and bearish signals
        bullish_count = 0
        bearish_count = 0
        
        # Track signal strengths
        bullish_strength = 0
        bearish_strength = 0
        
        # RSI signal
        if signals['rsi']['signal'] in ['Buy', 'Buy_Divergence']:
            bullish_count += 1
            bullish_strength += signals['rsi']['signal_strength']
        elif signals['rsi']['signal'] in ['Sell', 'Sell_Divergence']:
            bearish_count += 1
            bearish_strength += signals['rsi']['signal_strength']
        
        # Order Blocks signal
        if signals['order_blocks']['signal'] in ['Buy', 'Buy_Approaching']:
            bullish_count += 1
            bullish_strength += signals['order_blocks']['signal_strength']
        elif signals['order_blocks']['signal'] in ['Sell', 'Sell_Approaching']:
            bearish_count += 1
            bearish_strength += signals['order_blocks']['signal_strength']
        
        # Market Structure signal
        if signals['market_structure']['signal'].startswith('Buy'):
            bullish_count += 1
            bullish_strength += signals['market_structure']['signal_strength']
        elif signals['market_structure']['signal'].startswith('Sell'):
            bearish_count += 1
            bearish_strength += signals['market_structure']['signal_strength']
        
        # Optimal Trade Entry signal
        if signals['optimal_trade_entry']['signal'] in ['Buy', 'Buy_Approaching']:
            bullish_count += 1
            bullish_strength += signals['optimal_trade_entry']['signal_strength']
        elif signals['optimal_trade_entry']['signal'] in ['Sell', 'Sell_Approaching']:
            bearish_count += 1
            bearish_strength += signals['optimal_trade_entry']['signal_strength']
        
        # Liquidity Model signal
        if signals['liquidity_model']['signal'] in ['Buy_After_Sweep', 'Buy_Approaching_Liquidity']:
            bullish_count += 1
            bullish_strength += signals['liquidity_model']['signal_strength']
        elif signals['liquidity_model']['signal'] in ['Sell_After_Sweep', 'Sell_Approaching_Liquidity']:
            bearish_count += 1
            bearish_strength += signals['liquidity_model']['signal_strength']
        
        # Determine overall bias
        if bullish_count > bearish_count:
            bias = "Bullish"
        elif bearish_count > bullish_count:
            bias = "Bearish"
        else:
            # If counts are equal, use strength to determine bias
            if bullish_strength > bearish_strength:
                bias = "Bullish"
            elif bearish_strength > bullish_strength:
                bias = "Bearish"
            else:
                bias = "Neutral"
        
        # Calculate overall signal strength
        if bias == "Bullish":
            signal_strength = bullish_strength / max(1, bullish_count)
        elif bias == "Bearish":
            signal_strength = bearish_strength / max(1, bearish_count)
        else:
            signal_strength = 0
        
        # Determine confidence level
        if signal_strength > 80:
            confidence = "Very High"
        elif signal_strength > 60:
            confidence = "High"
        elif signal_strength > 40:
            confidence = "Medium"
        elif signal_strength > 20:
            confidence = "Low"
        else:
            confidence = "Very Low"
        
        # Generate the combined signal
        if bias == "Bullish" and signal_strength > 50:
            combined_signal = "Market Rises: 1 ES long, 1 NQ short"
            mes_mnq_signal = "Market Rises: 1 MES long, 2 MNQ short"
        elif bias == "Bearish" and signal_strength > 50:
            combined_signal = "Market Falls: 1 ES short, 1 NQ long"
            mes_mnq_signal = "Market Falls: 1 MES short, 2 MNQ long"
        else:
            combined_signal = "Hold: No clear signal"
            mes_mnq_signal = "Hold: No clear signal"
        
        return {
            'bias': bias,
            'signal': combined_signal,
            'mes_mnq_signal': mes_mnq_signal,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def plot_all_indicators(self, data, output_dir=None):
        """
        Generate plots for all indicators.
        
        Args:
            data (dict): Dictionary containing all calculated indicators.
            output_dir (str, optional): Directory to save plots. Defaults to self.output_dir.
        
        Returns:
            list: List of saved plot file paths.
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        plot_files = []
        
        # Plot RSI
        rsi_fig = self.rsi.plot(data['rsi_data'], output_dir=output_dir, filename="rsi_plot.png")
        if output_dir:
            plot_files.append(os.path.join(output_dir, "rsi_plot.png"))
        
        # Plot Order Blocks
        ob_fig = self.order_blocks.plot(data['ob_data'], output_dir=output_dir, filename="order_blocks_plot.png")
        if output_dir:
            plot_files.append(os.path.join(output_dir, "order_blocks_plot.png"))
        
        # Plot Market Structure
        ms_fig = self.market_structure.plot(data['ms_data'], output_dir=output_dir, filename="market_structure_plot.png")
        if output_dir:
            plot_files.append(os.path.join(output_dir, "market_structure_plot.png"))
        
        # Plot Optimal Trade Entry
        ote_fig = self.optimal_trade_entry.plot(data['ote_data'], output_dir=output_dir, filename="optimal_trade_entry_plot.png")
        if output_dir:
            plot_files.append(os.path.join(output_dir, "optimal_trade_entry_plot.png"))
        
        # Plot Liquidity Model
        liq_fig = self.liquidity_model.plot(data['liq_data'], output_dir=output_dir, filename="liquidity_model_plot.png")
        if output_dir:
            plot_files.append(os.path.join(output_dir, "liquidity_model_plot.png"))
        
        # Plot combined chart
        combined_fig = self.plot_combined_chart(data['combined_data'], output_dir=output_dir, filename="combined_indicators_plot.png")
        if output_dir:
            plot_files.append(os.path.join(output_dir, "combined_indicators_plot.png"))
        
        return plot_files
    
    def plot_combined_chart(self, data, output_dir=None, filename=None):
        """
        Generate a combined chart with price and key signals from all indicators.
        
        Args:
            data (pandas.DataFrame): Combined DataFrame with all indicators.
            output_dir (str, optional): Directory to save the plot. Defaults to None.
            filename (str, optional): Filename for the plot. Defaults to None.
        
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price on top subplot
        ax1.plot(data.index, data['Close'], label='Close Price')
        
        # Plot Order Blocks
        bullish_obs = data[data['Bullish_OB'] & data['OB_Active']]
        bearish_obs = data[data['Bearish_OB'] & data['OB_Active']]
        
        for idx, row in bullish_obs.iterrows():
            ax1.axhspan(row['OB_Low'], row['OB_High'], 
                       xmin=max(0, (idx-5)/len(data)), xmax=min(1, (idx+5)/len(data)),
                       color='green', alpha=0.2)
        
        for idx, row in bearish_obs.iterrows():
            ax1.axhspan(row['OB_Low'], row['OB_High'], 
                       xmin=max(0, (idx-5)/len(data)), xmax=min(1, (idx+5)/len(data)),
                       color='red', alpha=0.2)
        
        # Plot OTE levels
        for i in range(len(data)):
            if data['OTE_Buy_Active'].iloc[i] and not np.isnan(data['OTE_Buy_Zone_Low'].iloc[i]):
                ax1.axhspan(data['OTE_Buy_Zone_Low'].iloc[i], data['OTE_Buy_Zone_High'].iloc[i], 
                           xmin=i/len(data), xmax=(i+1)/len(data),
                           color='green', alpha=0.1)
            
            if data['OTE_Sell_Active'].iloc[i] and not np.isnan(data['OTE_Sell_Zone_Low'].iloc[i]):
                ax1.axhspan(data['OTE_Sell_Zone_Low'].iloc[i], data['OTE_Sell_Zone_High'].iloc[i], 
                           xmin=i/len(data), xmax=(i+1)/len(data),
                           color='red', alpha=0.1)
        
        # Plot Liquidity levels
        buy_liquidity = data[data['Buy_Liquidity'] & data['Liquidity_Active']]
        sell_liquidity = data[data['Sell_Liquidity'] & data['Liquidity_Active']]
        
        for idx, row in buy_liquidity.iterrows():
            ax1.axhline(y=row['Liquidity_Level'], 
                       xmin=max(0, (idx-5)/len(data)), xmax=min(1, (idx+5)/len(data)),
                       color='blue', linestyle='-', linewidth=1, alpha=0.7)
        
        for idx, row in sell_liquidity.iterrows():
            ax1.axhline(y=row['Liquidity_Level'], 
                       xmin=max(0, (idx-5)/len(data)), xmax=min(1, (idx+5)/len(data)),
                       color='purple', linestyle='-', linewidth=1, alpha=0.7)
        
        # Plot Market Structure swing points
        swing_highs = data[data['Swing_High']]
        swing_lows = data[data['Swing_Low']]
        
        if not swing_highs.empty:
            ax1.scatter(swing_highs.index, swing_highs['High'], 
                       color='red', marker='o', s=30, alpha=0.7)
        
        if not swing_lows.empty:
            ax1.scatter(swing_lows.index, swing_lows['Low'], 
                       color='green', marker='o', s=30, alpha=0.7)
        
        # Plot combined buy signals
        buy_signals = data[(data['RSI_Trade_Signal'] == 'Buy') | 
                          (data['OB_Trade_Signal'] == 'Buy') | 
                          (data['MS_Trade_Signal'].str.startswith('Buy')) | 
                          (data['OTE_Trade_Signal'] == 'Buy') | 
                          (data['Liquidity_Trade_Signal'].str.startswith('Buy'))]
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['Close'], 
                       color='green', marker='^', s=100, label='Buy Signals')
        
        # Plot combined sell signals
        sell_signals = data[(data['RSI_Trade_Signal'] == 'Sell') | 
                           (data['OB_Trade_Signal'] == 'Sell') | 
                           (data['MS_Trade_Signal'].str.startswith('Sell')) | 
                           (data['OTE_Trade_Signal'] == 'Sell') | 
                           (data['Liquidity_Trade_Signal'].str.startswith('Sell'))]
        
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['Close'], 
                       color='red', marker='v', s=100, label='Sell Signals')
        
        # Set title and labels for price subplot
        ax1.set_title('Combined Technical Indicators')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # Plot RSI on bottom subplot
        if 'RSI_14' in data.columns:
            ax2.plot(data.index, data['RSI_14'], label='RSI-14', color='purple')
            
            # Add overbought and oversold lines
            ax2.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
            ax2.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
            ax2.axhline(y=50, color='k', linestyle='-', alpha=0.3)
            
            # Set title and labels for RSI subplot
            ax2.set_title('RSI (14)')
            ax2.set_ylabel('RSI Value')
            ax2.grid(True)
            ax2.set_ylim(0, 100)
            ax2.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot if output_dir is provided
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            if filename is None:
                filename = "combined_indicators_plot.png"
            
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
        
        return fig
    
    def format_signals_for_manus(self, price_data, symbol='ES'):
        """
        Format all technical indicators and signals for MANUS input.
        
        Args:
            price_data (pandas.DataFrame): Price data with OHLC columns.
            symbol (str, optional): Symbol being analyzed. Defaults to 'ES'.
        
        Returns:
            dict: Dictionary containing formatted data for MANUS.
        """
        # Calculate all indicators
        pip_value = 0.25 if symbol in ['ES', 'NQ'] else 0.0001  # Adjust pip value based on symbol
        indicators_data = self.calculate_all_indicators(price_data, pip_value)
        
        # Get all signals
        all_signals = self.get_all_signals(indicators_data)
        
        # Generate combined signal
        combined_signal = self.generate_combined_signal(all_signals)
        
        # Format the data for MANUS
        current_price = price_data['Close'].iloc[-1]
        current_time = price_data.index[-1]
        
        # Format RSI data
        rsi_value = indicators_data['rsi_data']['RSI_14'].iloc[-1] if 'RSI_14' in indicators_data['rsi_data'].columns else None
        rsi_signal = all_signals['rsi']['signal']
        
        # Format Market Structure data
        ms_trend = all_signals['market_structure']['trend']
        
        # Format Order Blocks data
        active_obs = all_signals['order_blocks']['active_obs']
        ob_bias = all_signals['order_blocks']['bias']
        
        # Format OTE data
        buy_ote_active = all_signals['optimal_trade_entry']['buy_ote_active']
        sell_ote_active = all_signals['optimal_trade_entry']['sell_ote_active']
        
        # Format Liquidity Model data
        liquidity_bias = all_signals['liquidity_model']['bias']
        recent_sweep_up = all_signals['liquidity_model']['recent_sweep_up']
        recent_sweep_down = all_signals['liquidity_model']['recent_sweep_down']
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'current_time': current_time,
            'technical_indicators': {
                'rsi': {
                    'value': rsi_value,
                    'signal': rsi_signal
                },
                'market_structure': {
                    'trend': ms_trend
                },
                'order_blocks': {
                    'active_count': active_obs,
                    'bias': ob_bias
                },
                'optimal_trade_entry': {
                    'buy_active': buy_ote_active,
                    'sell_active': sell_ote_active
                },
                'liquidity_model': {
                    'bias': liquidity_bias,
                    'recent_sweep_up': recent_sweep_up,
                    'recent_sweep_down': recent_sweep_down
                }
            },
            'combined_signal': combined_signal
        }

# Example usage
if __name__ == "__main__":
    # Import necessary modules for testing
    import sys
    sys.path.append('..')
    from data_collection.market_index import MarketIndexCollector
    
    # Create a market index collector
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    collector = MarketIndexCollector(cache_dir=cache_dir)
    
    # Get ES futures data
    es_data = collector.get_historical_data('es', period="10d", interval="1h")
    
    # Create technical indicators manager
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    manager = TechnicalIndicatorsManager(output_dir=output_dir)
    
    # Calculate all indicators
    indicators_data = manager.calculate_all_indicators(es_data, pip_value=0.25)
    
    # Get all signals
    all_signals = manager.get_all_signals(indicators_data)
    
    # Generate combined signal
    combined_signal = manager.generate_combined_signal(all_signals)
    
    # Print combined signal
    print("Combined Signal:")
    for key, value in combined_signal.items():
        print(f"  {key}: {value}")
    
    # Generate plots
    plot_files = manager.plot_all_indicators(indicators_data)
    print(f"\nGenerated {len(plot_files)} plot files")
    
    # Format signals for MANUS
    manus_data = manager.format_signals_for_manus(es_data, symbol='ES')
    print("\nFormatted data for MANUS:")
    print(f"  Symbol: {manus_data['symbol']}")
    print(f"  Current Price: {manus_data['current_price']}")
    print(f"  Combined Signal: {manus_data['combined_signal']['signal']}")
    print(f"  MES-MNQ Signal: {manus_data['combined_signal']['mes_mnq_signal']}")
    print(f"  Confidence: {manus_data['combined_signal']['confidence']}")
