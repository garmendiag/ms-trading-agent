"""
ICT Order Blocks Indicator Module

This module implements the ICT (Inner Circle Trader) Order Blocks concept
with various customization options and signal generation capabilities for
the MANUS trading agent.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class OrderBlocksIndicator:
    """
    Class for identifying and analyzing ICT Order Blocks.
    """
    
    def __init__(self, lookback=10, threshold_pips=5, min_size_pips=10):
        """
        Initialize the Order Blocks indicator.
        
        Args:
            lookback (int, optional): Lookback period for identifying swing points. Defaults to 10.
            threshold_pips (float, optional): Threshold in pips to identify significant price moves. Defaults to 5.
            min_size_pips (float, optional): Minimum size of order block in pips. Defaults to 10.
        """
        self.lookback = lookback
        self.threshold_pips = threshold_pips
        self.min_size_pips = min_size_pips
    
    def calculate(self, price_data, pip_value=0.0001):
        """
        Identify Order Blocks in the given price data.
        
        Args:
            price_data (pandas.DataFrame): Price data with OHLC columns.
            pip_value (float, optional): Value of 1 pip in price units. Defaults to 0.0001.
        
        Returns:
            pandas.DataFrame: DataFrame with Order Block information.
        """
        # Make a copy of the data to avoid modifying the original
        result = price_data.copy()
        
        # Convert threshold and min size from pips to price units
        threshold = self.threshold_pips * pip_value
        min_size = self.min_size_pips * pip_value
        
        # Initialize order block columns
        result['Bullish_OB'] = False
        result['Bearish_OB'] = False
        result['OB_High'] = np.nan
        result['OB_Low'] = np.nan
        result['OB_Size'] = np.nan
        result['OB_Type'] = ''
        result['OB_Strength'] = np.nan
        
        # Calculate price swings
        result['Swing_High'] = False
        result['Swing_Low'] = False
        
        # Identify swing points
        for i in range(self.lookback, len(result) - self.lookback):
            # Check for swing high
            if result['High'].iloc[i] == result['High'].iloc[i-self.lookback:i+self.lookback+1].max():
                result['Swing_High'].iloc[i] = True
            
            # Check for swing low
            if result['Low'].iloc[i] == result['Low'].iloc[i-self.lookback:i+self.lookback+1].min():
                result['Swing_Low'].iloc[i] = True
        
        # Identify bullish order blocks (last down candle before a significant move up)
        for i in range(self.lookback, len(result) - 1):
            if result['Swing_Low'].iloc[i]:
                # Look for the last down candle before the swing low
                for j in range(i-1, max(0, i-self.lookback), -1):
                    if result['Close'].iloc[j] < result['Open'].iloc[j]:  # Down candle
                        # Check if the candle size is significant
                        candle_size = result['High'].iloc[j] - result['Low'].iloc[j]
                        if candle_size >= min_size:
                            result['Bullish_OB'].iloc[j] = True
                            result['OB_High'].iloc[j] = result['High'].iloc[j]
                            result['OB_Low'].iloc[j] = result['Low'].iloc[j]
                            result['OB_Size'].iloc[j] = candle_size / pip_value  # Size in pips
                            result['OB_Type'].iloc[j] = 'Bullish'
                            
                            # Calculate strength based on subsequent price movement
                            price_move = result['High'].iloc[i:i+self.lookback].max() - result['Low'].iloc[i]
                            result['OB_Strength'].iloc[j] = price_move / pip_value  # Strength in pips
                            break
        
        # Identify bearish order blocks (last up candle before a significant move down)
        for i in range(self.lookback, len(result) - 1):
            if result['Swing_High'].iloc[i]:
                # Look for the last up candle before the swing high
                for j in range(i-1, max(0, i-self.lookback), -1):
                    if result['Close'].iloc[j] > result['Open'].iloc[j]:  # Up candle
                        # Check if the candle size is significant
                        candle_size = result['High'].iloc[j] - result['Low'].iloc[j]
                        if candle_size >= min_size:
                            result['Bearish_OB'].iloc[j] = True
                            result['OB_High'].iloc[j] = result['High'].iloc[j]
                            result['OB_Low'].iloc[j] = result['Low'].iloc[j]
                            result['OB_Size'].iloc[j] = candle_size / pip_value  # Size in pips
                            result['OB_Type'].iloc[j] = 'Bearish'
                            
                            # Calculate strength based on subsequent price movement
                            price_move = result['High'].iloc[i] - result['Low'].iloc[i:i+self.lookback].min()
                            result['OB_Strength'].iloc[j] = price_move / pip_value  # Strength in pips
                            break
        
        # Identify active order blocks (not yet mitigated)
        result['OB_Active'] = False
        result['OB_Mitigation_Percent'] = np.nan
        
        # For each order block, check if it's still active
        for i in range(len(result)):
            if result['Bullish_OB'].iloc[i] or result['Bearish_OB'].iloc[i]:
                ob_high = result['OB_High'].iloc[i]
                ob_low = result['OB_Low'].iloc[i]
                ob_type = result['OB_Type'].iloc[i]
                
                # Check subsequent price action for mitigation
                if ob_type == 'Bullish':
                    # Bullish OB is mitigated if price trades below its low
                    mitigated = (result['Low'].iloc[i+1:] < ob_low).any()
                    if not mitigated:
                        result['OB_Active'].iloc[i] = True
                        # Calculate how much of the OB has been mitigated
                        lowest_subsequent = result['Low'].iloc[i+1:].min()
                        mitigation_percent = max(0, min(100, (ob_high - lowest_subsequent) / (ob_high - ob_low) * 100))
                        result['OB_Mitigation_Percent'].iloc[i] = mitigation_percent
                
                elif ob_type == 'Bearish':
                    # Bearish OB is mitigated if price trades above its high
                    mitigated = (result['High'].iloc[i+1:] > ob_high).any()
                    if not mitigated:
                        result['OB_Active'].iloc[i] = True
                        # Calculate how much of the OB has been mitigated
                        highest_subsequent = result['High'].iloc[i+1:].max()
                        mitigation_percent = max(0, min(100, (highest_subsequent - ob_low) / (ob_high - ob_low) * 100))
                        result['OB_Mitigation_Percent'].iloc[i] = mitigation_percent
        
        return result
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Order Blocks.
        
        Args:
            data (pandas.DataFrame): DataFrame with Order Block information.
        
        Returns:
            pandas.DataFrame: DataFrame with trading signals.
        """
        # Make a copy of the data
        result = data.copy()
        
        # Initialize signal column
        result['OB_Trade_Signal'] = 'Hold'
        
        # Generate signals based on price approaching active order blocks
        for i in range(1, len(result)):
            # Check for active order blocks
            active_obs = result[result['OB_Active']].index
            
            if len(active_obs) > 0:
                current_price = result['Close'].iloc[i]
                
                # Check each active order block
                for ob_idx in active_obs:
                    if ob_idx >= i:  # Skip future order blocks
                        continue
                    
                    ob_high = result['OB_High'].iloc[ob_idx]
                    ob_low = result['OB_Low'].iloc[ob_idx]
                    ob_type = result['OB_Type'].iloc[ob_idx]
                    
                    # Calculate distance to order block
                    if ob_type == 'Bullish':
                        distance_pct = (current_price - ob_high) / ob_high * 100
                        # Bullish signal if price is approaching bullish order block from above
                        if -1 < distance_pct < 0:
                            result['OB_Trade_Signal'].iloc[i] = 'Buy_Approaching'
                        # Stronger signal if price is entering the order block
                        elif ob_low <= current_price <= ob_high:
                            result['OB_Trade_Signal'].iloc[i] = 'Buy'
                    
                    elif ob_type == 'Bearish':
                        distance_pct = (current_price - ob_low) / ob_low * 100
                        # Bearish signal if price is approaching bearish order block from below
                        if 0 < distance_pct < 1:
                            result['OB_Trade_Signal'].iloc[i] = 'Sell_Approaching'
                        # Stronger signal if price is entering the order block
                        elif ob_low <= current_price <= ob_high:
                            result['OB_Trade_Signal'].iloc[i] = 'Sell'
        
        return result
    
    def plot(self, data, output_dir=None, filename=None):
        """
        Plot price with Order Blocks and signals.
        
        Args:
            data (pandas.DataFrame): DataFrame with price and Order Block data.
            output_dir (str, optional): Directory to save the plot. Defaults to None.
            filename (str, optional): Filename for the plot. Defaults to None.
        
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price
        ax.plot(data.index, data['Close'], label='Close Price')
        
        # Plot bullish order blocks
        bullish_obs = data[data['Bullish_OB']]
        for idx, row in bullish_obs.iterrows():
            if row['OB_Active']:
                color = 'green'
                alpha = 0.3
            else:
                color = 'lightgreen'
                alpha = 0.1
            
            ax.axhspan(row['OB_Low'], row['OB_High'], 
                      xmin=idx/len(data), xmax=(idx+1)/len(data),
                      color=color, alpha=alpha)
        
        # Plot bearish order blocks
        bearish_obs = data[data['Bearish_OB']]
        for idx, row in bearish_obs.iterrows():
            if row['OB_Active']:
                color = 'red'
                alpha = 0.3
            else:
                color = 'lightcoral'
                alpha = 0.1
            
            ax.axhspan(row['OB_Low'], row['OB_High'], 
                      xmin=idx/len(data), xmax=(idx+1)/len(data),
                      color=color, alpha=alpha)
        
        # Plot buy signals
        buy_signals = data[data['OB_Trade_Signal'] == 'Buy']
        if not buy_signals.empty:
            ax.scatter(buy_signals.index, buy_signals['Close'], 
                      color='green', marker='^', s=100, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = data[data['OB_Trade_Signal'] == 'Sell']
        if not sell_signals.empty:
            ax.scatter(sell_signals.index, sell_signals['Close'], 
                      color='red', marker='v', s=100, label='Sell Signal')
        
        # Plot approaching signals
        buy_approaching = data[data['OB_Trade_Signal'] == 'Buy_Approaching']
        if not buy_approaching.empty:
            ax.scatter(buy_approaching.index, buy_approaching['Close'], 
                      color='lime', marker='^', s=60, label='Buy Approaching')
        
        sell_approaching = data[data['OB_Trade_Signal'] == 'Sell_Approaching']
        if not sell_approaching.empty:
            ax.scatter(sell_approaching.index, sell_approaching['Close'], 
                      color='lightcoral', marker='v', s=60, label='Sell Approaching')
        
        # Set title and labels
        ax.set_title('Price with ICT Order Blocks')
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
                filename = "ict_order_blocks_plot.png"
            
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
        
        return fig
    
    def get_active_order_blocks(self, data):
        """
        Get information about active order blocks.
        
        Args:
            data (pandas.DataFrame): DataFrame with Order Block information.
        
        Returns:
            dict: Dictionary with active order block information.
        """
        # Filter for active order blocks
        active_obs = data[data['OB_Active']]
        
        if active_obs.empty:
            return {
                'bullish_count': 0,
                'bearish_count': 0,
                'active_blocks': []
            }
        
        # Count by type
        bullish_count = active_obs[active_obs['OB_Type'] == 'Bullish'].shape[0]
        bearish_count = active_obs[active_obs['OB_Type'] == 'Bearish'].shape[0]
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        
        # Prepare list of active blocks with details
        active_blocks = []
        for idx, row in active_obs.iterrows():
            # Calculate distance to current price
            if row['OB_Type'] == 'Bullish':
                distance_pct = (current_price - row['OB_High']) / row['OB_High'] * 100
                status = 'Above' if current_price > row['OB_High'] else 'Inside' if current_price >= row['OB_Low'] else 'Below'
            else:  # Bearish
                distance_pct = (current_price - row['OB_Low']) / row['OB_Low'] * 100
                status = 'Below' if current_price < row['OB_Low'] else 'Inside' if current_price <= row['OB_High'] else 'Above'
            
            active_blocks.append({
                'index': idx,
                'date': data.index[idx].strftime('%Y-%m-%d %H:%M:%S') if hasattr(data.index[idx], 'strftime') else str(data.index[idx]),
                'type': row['OB_Type'],
                'high': row['OB_High'],
                'low': row['OB_Low'],
                'size': row['OB_Size'],
                'strength': row['OB_Strength'],
                'mitigation_percent': row['OB_Mitigation_Percent'],
                'distance_percent': distance_pct,
                'status': status
            })
        
        return {
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'active_blocks': active_blocks
        }
    
    def get_current_signal(self, data):
        """
        Get the current trading signal based on Order Blocks.
        
        Args:
            data (pandas.DataFrame): DataFrame with Order Block information.
        
        Returns:
            dict: Dictionary with current Order Block signal information.
        """
        if len(data) < 2:
            return {
                'signal': 'Insufficient Data',
                'active_obs': 0,
                'sign<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>