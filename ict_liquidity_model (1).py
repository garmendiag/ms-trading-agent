"""
ICT Liquidity Model Indicator Module

This module implements the ICT (Inner Circle Trader) Liquidity Model concept
with various customization options and signal generation capabilities for
the MANUS trading agent. It identifies liquidity pools, liquidity sweeps,
and generates trading signals based on liquidity concepts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class LiquidityModelIndicator:
    """
    Class for identifying and analyzing ICT Liquidity Model.
    """
    
    def __init__(self, lookback=15, equal_level_pips=2, liquidity_threshold=3):
        """
        Initialize the Liquidity Model indicator.
        
        Args:
            lookback (int, optional): Lookback period for identifying liquidity levels. Defaults to 15.
            equal_level_pips (float, optional): Threshold in pips to identify equal levels. Defaults to 2.
            liquidity_threshold (int, optional): Minimum number of equal levels to form liquidity. Defaults to 3.
        """
        self.lookback = lookback
        self.equal_level_pips = equal_level_pips
        self.liquidity_threshold = liquidity_threshold
    
    def calculate(self, price_data, pip_value=0.0001):
        """
        Identify Liquidity zones in the given price data.
        
        Args:
            price_data (pandas.DataFrame): Price data with OHLC columns.
            pip_value (float, optional): Value of 1 pip in price units. Defaults to 0.0001.
        
        Returns:
            pandas.DataFrame: DataFrame with Liquidity Model information.
        """
        # Make a copy of the data to avoid modifying the original
        result = price_data.copy()
        
        # Convert equal level threshold from pips to price units
        equal_level_threshold = self.equal_level_pips * pip_value
        
        # Initialize liquidity columns
        result['Buy_Liquidity'] = False
        result['Sell_Liquidity'] = False
        result['Liquidity_Level'] = np.nan
        result['Liquidity_Type'] = ''
        result['Liquidity_Strength'] = 0
        result['Liquidity_Sweep_Up'] = False
        result['Liquidity_Sweep_Down'] = False
        
        # Identify equal highs and lows (liquidity zones)
        for i in range(self.lookback, len(result) - 1):
            # Find equal highs (sell liquidity)
            equal_highs = []
            for j in range(i-self.lookback, i):
                if abs(result['High'].iloc[j] - result['High'].iloc[i]) < equal_level_threshold:
                    equal_highs.append(j)
            
            # Find equal lows (buy liquidity)
            equal_lows = []
            for j in range(i-self.lookback, i):
                if abs(result['Low'].iloc[j] - result['Low'].iloc[i]) < equal_level_threshold:
                    equal_lows.append(j)
            
            # Mark sell liquidity if enough equal highs are found
            if len(equal_highs) >= self.liquidity_threshold:
                result['Sell_Liquidity'].iloc[i] = True
                result['Liquidity_Level'].iloc[i] = result['High'].iloc[i]
                result['Liquidity_Type'].iloc[i] = 'Sell'
                result['Liquidity_Strength'].iloc[i] = len(equal_highs)
            
            # Mark buy liquidity if enough equal lows are found
            if len(equal_lows) >= self.liquidity_threshold:
                result['Buy_Liquidity'].iloc[i] = True
                result['Liquidity_Level'].iloc[i] = result['Low'].iloc[i]
                result['Liquidity_Type'].iloc[i] = 'Buy'
                result['Liquidity_Strength'].iloc[i] = len(equal_lows)
        
        # Identify liquidity sweeps
        for i in range(self.lookback + 1, len(result)):
            # Liquidity sweep up (price breaks above sell liquidity)
            if result['Sell_Liquidity'].iloc[i-1]:
                liquidity_level = result['Liquidity_Level'].iloc[i-1]
                if result['High'].iloc[i] > liquidity_level:
                    result['Liquidity_Sweep_Up'].iloc[i] = True
            
            # Liquidity sweep down (price breaks below buy liquidity)
            if result['Buy_Liquidity'].iloc[i-1]:
                liquidity_level = result['Liquidity_Level'].iloc[i-1]
                if result['Low'].iloc[i] < liquidity_level:
                    result['Liquidity_Sweep_Down'].iloc[i] = True
        
        # Identify active liquidity levels (not yet swept)
        result['Liquidity_Active'] = False
        
        # For each liquidity level, check if it's still active
        for i in range(len(result)-1, self.lookback-1, -1):
            if result['Buy_Liquidity'].iloc[i] or result['Sell_Liquidity'].iloc[i]:
                liquidity_level = result['Liquidity_Level'].iloc[i]
                liquidity_type = result['Liquidity_Type'].iloc[i]
                
                # Check subsequent price action for sweep
                if liquidity_type == 'Buy':
                    # Buy liquidity is swept if price trades below its level
                    swept = (result['Low'].iloc[i+1:] < liquidity_level).any()
                    if not swept:
                        result['Liquidity_Active'].iloc[i] = True
                
                elif liquidity_type == 'Sell':
                    # Sell liquidity is swept if price trades above its level
                    swept = (result['High'].iloc[i+1:] > liquidity_level).any()
                    if not swept:
                        result['Liquidity_Active'].iloc[i] = True
        
        return result
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Liquidity Model.
        
        Args:
            data (pandas.DataFrame): DataFrame with Liquidity Model information.
        
        Returns:
            pandas.DataFrame: DataFrame with trading signals.
        """
        # Make a copy of the data
        result = data.copy()
        
        # Initialize signal column
        result['Liquidity_Trade_Signal'] = 'Hold'
        
        # Generate signals based on liquidity sweeps and approaches
        for i in range(1, len(result)):
            # Buy signal after liquidity sweep down (institutional buying)
            if result['Liquidity_Sweep_Down'].iloc[i]:
                # Check if price has started to move up after the sweep
                if i < len(result) - 1 and result['Close'].iloc[i+1] > result['Close'].iloc[i]:
                    result['Liquidity_Trade_Signal'].iloc[i] = 'Buy_After_Sweep'
            
            # Sell signal after liquidity sweep up (institutional selling)
            elif result['Liquidity_Sweep_Up'].iloc[i]:
                # Check if price has started to move down after the sweep
                if i < len(result) - 1 and result['Close'].iloc[i+1] < result['Close'].iloc[i]:
                    result['Liquidity_Trade_Signal'].iloc[i] = 'Sell_After_Sweep'
            
            # Buy signal when approaching buy liquidity from above
            elif i > 0 and result['Buy_Liquidity'].iloc[i-1] and result['Liquidity_Active'].iloc[i-1]:
                liquidity_level = result['Liquidity_Level'].iloc[i-1]
                distance_pct = (result['Close'].iloc[i] - liquidity_level) / result['Close'].iloc[i] * 100
                if 0 < distance_pct < 0.5:  # Within 0.5% above buy liquidity
                    result['Liquidity_Trade_Signal'].iloc[i] = 'Buy_Approaching_Liquidity'
            
            # Sell signal when approaching sell liquidity from below
            elif i > 0 and result['Sell_Liquidity'].iloc[i-1] and result['Liquidity_Active'].iloc[i-1]:
                liquidity_level = result['Liquidity_Level'].iloc[i-1]
                distance_pct = (liquidity_level - result['Close'].iloc[i]) / result['Close'].iloc[i] * 100
                if 0 < distance_pct < 0.5:  # Within 0.5% below sell liquidity
                    result['Liquidity_Trade_Signal'].iloc[i] = 'Sell_Approaching_Liquidity'
        
        return result
    
    def plot(self, data, output_dir=None, filename=None):
        """
        Plot price with Liquidity levels and signals.
        
        Args:
            data (pandas.DataFrame): DataFrame with price and Liquidity data.
            output_dir (str, optional): Directory to save the plot. Defaults to None.
            filename (str, optional): Filename for the plot. Defaults to None.
        
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price
        ax.plot(data.index, data['Close'], label='Close Price')
        
        # Plot buy liquidity levels
        buy_liquidity = data[data['Buy_Liquidity'] & data['Liquidity_Active']]
        for idx, row in buy_liquidity.iterrows():
            ax.axhline(y=row['Liquidity_Level'], 
                      xmin=max(0, (idx-5)/len(data)), xmax=min(1, (idx+5)/len(data)),
                      color='green', linestyle='-', linewidth=2, alpha=0.7)
        
        # Plot sell liquidity levels
        sell_liquidity = data[data['Sell_Liquidity'] & data['Liquidity_Active']]
        for idx, row in sell_liquidity.iterrows():
            ax.axhline(y=row['Liquidity_Level'], 
                      xmin=max(0, (idx-5)/len(data)), xmax=min(1, (idx+5)/len(data)),
                      color='red', linestyle='-', linewidth=2, alpha=0.7)
        
        # Plot liquidity sweeps
        sweep_up = data[data['Liquidity_Sweep_Up']]
        if not sweep_up.empty:
            ax.scatter(sweep_up.index, sweep_up['High'], 
                      color='purple', marker='x', s=100, label='Liquidity Sweep Up')
        
        sweep_down = data[data['Liquidity_Sweep_Down']]
        if not sweep_down.empty:
            ax.scatter(sweep_down.index, sweep_down['Low'], 
                      color='blue', marker='x', s=100, label='Liquidity Sweep Down')
        
        # Plot buy signals
        buy_signals = data[data['Liquidity_Trade_Signal'] == 'Buy_After_Sweep']
        if not buy_signals.empty:
            ax.scatter(buy_signals.index, buy_signals['Close'], 
                      color='green', marker='^', s=100, label='Buy After Sweep')
        
        # Plot sell signals
        sell_signals = data[data['Liquidity_Trade_Signal'] == 'Sell_After_Sweep']
        if not sell_signals.empty:
            ax.scatter(sell_signals.index, sell_signals['Close'], 
                      color='red', marker='v', s=100, label='Sell After Sweep')
        
        # Plot approaching signals
        buy_approaching = data[data['Liquidity_Trade_Signal'] == 'Buy_Approaching_Liquidity']
        if not buy_approaching.empty:
            ax.scatter(buy_approaching.index, buy_approaching['Close'], 
                      color='lime', marker='^', s=60, label='Buy Approaching Liquidity')
        
        sell_approaching = data[data['Liquidity_Trade_Signal'] == 'Sell_Approaching_Liquidity']
        if not sell_approaching.empty:
            ax.scatter(sell_approaching.index, sell_approaching['Close'], 
                      color='lightcoral', marker='v', s=60, label='Sell Approaching Liquidity')
        
        # Set title and labels
        ax.set_title('Price with ICT Liquidity Model')
        ax.set_ylabel('Price')
        ax.grid(True)
        ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot if output_dir is provided
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            if filename is None:
                filename = "ict_liquidity_model_plot.png"
            
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
        
        return fig
    
    def get_active_liquidity_levels(self, data):
        """
        Get information about active liquidity levels.
        
        Args:
            data (pandas.DataFrame): DataFrame with Liquidity Model information.
        
        Returns:
            dict: Dictionary with active liquidity level information.
        """
        # Filter for active liquidity levels
        active_liquidity = data[data['Liquidity_Active']]
        
        if active_liquidity.empty:
            return {
                'buy_liquidity_count': 0,
                'sell_liquidity_count': 0,
                'active_levels': []
            }
        
        # Count by type
        buy_liquidity_count = active_liquidity[active_liquidity['Liquidity_Type'] == 'Buy'].shape[0]
        sell_liquidity_count = active_liquidity[active_liquidity['Liquidity_Type'] == 'Sell'].shape[0]
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        
        # Prepare list of active levels with details
        active_levels = []
        for idx, row in active_liquidity.iterrows():
            # Calculate distance to current price
            if row['Liquidity_Type'] == 'Buy':
                distance_pct = (current_price - row['Liquidity_Level']) / current_price * 100
                status = 'Above'
            else:  # Sell
                distance_pct = (row['Liquidity_Level'] - current_price) / current_price * 100
                status = 'Below'
            
            active_levels.append({
                'index': idx,
                'date': data.index[idx].strftime('%Y-%m-%d %H:%M:%S') if hasattr(data.index[idx], 'strftime') else str(data.index[idx]),
                'type': row['Liquidity_Type'],
                'level': row['Liquidity_Level'],
                'strength': row['Liquidity_Strength'],
                'distance_percent': distance_pct,
                'status': status
            })
        
        # Sort by distance (closest first)
        active_levels.sort(key=lambda x: abs(x['distance_percent']))
        
        return {
            'buy_liquidity_count': buy_liquidity_count,
            'sell_liquidity_count': sell_liquidity_count,
            'active_levels': active_levels
        }
    
    def get_recent_sweeps(self, data, lookback_bars=10):
        """
        Get information about recent liquidity sweeps.
        
        Args:
            data (pandas.DataFrame): DataFrame with Liquidity Model information.
            lookback_bars (int, optional): Number of bars to look back. Defaults to 10.
        
        Returns:
            dict: Dictionary with recent sweep information.
        """
        if len(data) < lookback_bars:
            return {
                'recent_sweep_up': False,
                'recent_sweep_down': False,
                'sweeps': []
            }
        
        # Check for recent sweeps
        recent_data = data.iloc[-lookback_bars:]
        recent_sweep_up = recent_data['Liquidity_Sweep_Up'].any()
        recent_sweep_down = recent_data['Liquidity_Sweep_Down'].any()
        
        # Get details of recent sweeps
        sweeps = []
        for i in range(len(data)-1, max(0, len(data)-lookback_bars-1), -1):
            if data['Liquidity_Sweep_Up'].iloc[i]:
                sweeps.append({
                    'index': i,
                    'date': data.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(data.index[i], 'strftime') else str(data.index[i]),
                    'type': 'Sweep Up',
                    'price': data['High'].iloc[i]
                })
            elif data['Liquidity_Sweep_Down'].iloc[i]:
                sweeps.append({
                    'index': i,
                    'date': data.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(data.index[i], 'strftime') else str(data.index[i]),
                    'type': 'Sweep Down',
                    'price': data['Low'].iloc[i]
                })
        
        return {
            'recent_sweep_up': recent_sweep_up,
            'recent_sweep_down': recent_swee<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>