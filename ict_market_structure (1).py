"""
ICT Market Structure Indicator Module

This module implements the ICT (Inner Circle Trader) Market Structure concept
with various customization options and signal generation capabilities for
the MANUS trading agent. It identifies Market Structure Shifts (MSS) and
higher highs/higher lows or lower highs/lower lows patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class MarketStructureIndicator:
    """
    Class for identifying and analyzing ICT Market Structure.
    """
    
    def __init__(self, lookback=10, swing_threshold=0.001):
        """
        Initialize the Market Structure indicator.
        
        Args:
            lookback (int, optional): Lookback period for identifying swing points. Defaults to 10.
            swing_threshold (float, optional): Minimum price change to identify a swing point as percentage. Defaults to 0.001 (0.1%).
        """
        self.lookback = lookback
        self.swing_threshold = swing_threshold
    
    def calculate(self, price_data):
        """
        Identify Market Structure patterns in the given price data.
        
        Args:
            price_data (pandas.DataFrame): Price data with OHLC columns.
        
        Returns:
            pandas.DataFrame: DataFrame with Market Structure information.
        """
        # Make a copy of the data to avoid modifying the original
        result = price_data.copy()
        
        # Initialize market structure columns
        result['Swing_High'] = False
        result['Swing_Low'] = False
        result['Higher_High'] = False
        result['Lower_Low'] = False
        result['Higher_Low'] = False
        result['Lower_High'] = False
        result['MSS_Bullish'] = False
        result['MSS_Bearish'] = False
        result['MS_Trend'] = 'Neutral'
        
        # Identify swing points
        for i in range(self.lookback, len(result) - self.lookback):
            # Check for swing high
            if result['High'].iloc[i] == result['High'].iloc[i-self.lookback:i+self.lookback+1].max():
                # Ensure the swing is significant enough
                if (result['High'].iloc[i] - result['Low'].iloc[i-self.lookback:i+self.lookback+1].min()) / result['High'].iloc[i] >= self.swing_threshold:
                    result['Swing_High'].iloc[i] = True
            
            # Check for swing low
            if result['Low'].iloc[i] == result['Low'].iloc[i-self.lookback:i+self.lookback+1].min():
                # Ensure the swing is significant enough
                if (result['High'].iloc[i-self.lookback:i+self.lookback+1].max() - result['Low'].iloc[i]) / result['Low'].iloc[i] >= self.swing_threshold:
                    result['Swing_Low'].iloc[i] = True
        
        # Identify higher highs, lower lows, etc.
        for i in range(2*self.lookback, len(result)):
            # Find previous swing high
            prev_swing_high = None
            for j in range(i-1, max(0, i-5*self.lookback), -1):
                if result['Swing_High'].iloc[j]:
                    prev_swing_high = j
                    break
            
            # Find previous swing low
            prev_swing_low = None
            for j in range(i-1, max(0, i-5*self.lookback), -1):
                if result['Swing_Low'].iloc[j]:
                    prev_swing_low = j
                    break
            
            # Check for higher high
            if result['Swing_High'].iloc[i] and prev_swing_high is not None:
                if result['High'].iloc[i] > result['High'].iloc[prev_swing_high]:
                    result['Higher_High'].iloc[i] = True
                else:
                    result['Lower_High'].iloc[i] = True
            
            # Check for lower low
            if result['Swing_Low'].iloc[i] and prev_swing_low is not None:
                if result['Low'].iloc[i] < result['Low'].iloc[prev_swing_low]:
                    result['Lower_Low'].iloc[i] = True
                else:
                    result['Higher_Low'].iloc[i] = True
        
        # Identify market structure shifts
        for i in range(3*self.lookback, len(result)):
            # Find recent higher highs and lower lows
            recent_higher_high = result['Higher_High'].iloc[i-3*self.lookback:i].any()
            recent_lower_low = result['Lower_Low'].iloc[i-3*self.lookback:i].any()
            recent_higher_low = result['Higher_Low'].iloc[i-3*self.lookback:i].any()
            recent_lower_high = result['Lower_High'].iloc[i-3*self.lookback:i].any()
            
            # Bullish MSS: Lower Low followed by Higher High
            if result['Lower_Low'].iloc[i-3*self.lookback:i-self.lookback].any() and result['Higher_High'].iloc[i]:
                result['MSS_Bullish'].iloc[i] = True
            
            # Bearish MSS: Higher High followed by Lower Low
            if result['Higher_High'].iloc[i-3*self.lookback:i-self.lookback].any() and result['Lower_Low'].iloc[i]:
                result['MSS_Bearish'].iloc[i] = True
            
            # Determine trend based on recent structure
            if recent_higher_high and recent_higher_low and not (recent_lower_low or recent_lower_high):
                result['MS_Trend'].iloc[i] = 'Bullish'
            elif recent_lower_low and recent_lower_high and not (recent_higher_high or recent_higher_low):
                result['MS_Trend'].iloc[i] = 'Bearish'
            elif result['MSS_Bullish'].iloc[i]:
                result['MS_Trend'].iloc[i] = 'Bullish_Shift'
            elif result['MSS_Bearish'].iloc[i]:
                result['MS_Trend'].iloc[i] = 'Bearish_Shift'
            else:
                # Determine trend based on most recent swing points
                recent_swings = []
                for j in range(i, max(0, i-5*self.lookback), -1):
                    if result['Swing_High'].iloc[j] or result['Swing_Low'].iloc[j]:
                        if result['Swing_High'].iloc[j]:
                            recent_swings.append(('High', j, result['High'].iloc[j]))
                        else:
                            recent_swings.append(('Low', j, result['Low'].iloc[j]))
                    
                    if len(recent_swings) >= 4:
                        break
                
                if len(recent_swings) >= 4:
                    # Check if we have higher highs and higher lows
                    highs = [p[2] for p in recent_swings if p[0] == 'High']
                    lows = [p[2] for p in recent_swings if p[0] == 'Low']
                    
                    if len(highs) >= 2 and len(lows) >= 2:
                        if highs[0] > highs[-1] and lows[0] > lows[-1]:
                            result['MS_Trend'].iloc[i] = 'Bullish'
                        elif highs[0] < highs[-1] and lows[0] < lows[-1]:
                            result['MS_Trend'].iloc[i] = 'Bearish'
        
        return result
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Market Structure.
        
        Args:
            data (pandas.DataFrame): DataFrame with Market Structure information.
        
        Returns:
            pandas.DataFrame: DataFrame with trading signals.
        """
        # Make a copy of the data
        result = data.copy()
        
        # Initialize signal column
        result['MS_Trade_Signal'] = 'Hold'
        
        # Generate signals based on market structure shifts and trends
        for i in range(1, len(result)):
            # Bullish MSS signal
            if result['MSS_Bullish'].iloc[i]:
                result['MS_Trade_Signal'].iloc[i] = 'Buy_MSS'
            
            # Bearish MSS signal
            elif result['MSS_Bearish'].iloc[i]:
                result['MS_Trade_Signal'].iloc[i] = 'Sell_MSS'
            
            # Higher High after Higher Low (bullish continuation)
            elif result['Higher_High'].iloc[i] and result['Higher_Low'].iloc[i-self.lookback:i].any():
                result['MS_Trade_Signal'].iloc[i] = 'Buy_Continuation'
            
            # Lower Low after Lower High (bearish continuation)
            elif result['Lower_Low'].iloc[i] and result['Lower_High'].iloc[i-self.lookback:i].any():
                result['MS_Trade_Signal'].iloc[i] = 'Sell_Continuation'
            
            # Higher Low in bullish trend (potential entry)
            elif result['Higher_Low'].iloc[i] and result['MS_Trend'].iloc[i-1] == 'Bullish':
                result['MS_Trade_Signal'].iloc[i] = 'Buy_HL'
            
            # Lower High in bearish trend (potential entry)
            elif result['Lower_High'].iloc[i] and result['MS_Trend'].iloc[i-1] == 'Bearish':
                result['MS_Trade_Signal'].iloc[i] = 'Sell_LH'
        
        return result
    
    def plot(self, data, output_dir=None, filename=None):
        """
        Plot price with Market Structure and signals.
        
        Args:
            data (pandas.DataFrame): DataFrame with price and Market Structure data.
            output_dir (str, optional): Directory to save the plot. Defaults to None.
            filename (str, optional): Filename for the plot. Defaults to None.
        
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price
        ax.plot(data.index, data['Close'], label='Close Price')
        
        # Plot swing highs and lows
        swing_highs = data[data['Swing_High']]
        swing_lows = data[data['Swing_Low']]
        
        if not swing_highs.empty:
            ax.scatter(swing_highs.index, swing_highs['High'], 
                      color='red', marker='o', s=50, label='Swing High')
        
        if not swing_lows.empty:
            ax.scatter(swing_lows.index, swing_lows['Low'], 
                      color='green', marker='o', s=50, label='Swing Low')
        
        # Plot higher highs and higher lows
        higher_highs = data[data['Higher_High']]
        higher_lows = data[data['Higher_Low']]
        
        if not higher_highs.empty:
            ax.scatter(higher_highs.index, higher_highs['High'], 
                      color='darkred', marker='o', s=70, label='Higher High')
        
        if not higher_lows.empty:
            ax.scatter(higher_lows.index, higher_lows['Low'], 
                      color='darkgreen', marker='o', s=70, label='Higher Low')
        
        # Plot lower highs and lower lows
        lower_highs = data[data['Lower_High']]
        lower_lows = data[data['Lower_Low']]
        
        if not lower_highs.empty:
            ax.scatter(lower_highs.index, lower_highs['High'], 
                      color='orange', marker='o', s=70, label='Lower High')
        
        if not lower_lows.empty:
            ax.scatter(lower_lows.index, lower_lows['Low'], 
                      color='lime', marker='o', s=70, label='Lower Low')
        
        # Plot market structure shifts
        bullish_mss = data[data['MSS_Bullish']]
        bearish_mss = data[data['MSS_Bearish']]
        
        if not bullish_mss.empty:
            ax.scatter(bullish_mss.index, bullish_mss['Close'], 
                      color='blue', marker='^', s=120, label='Bullish MSS')
        
        if not bearish_mss.empty:
            ax.scatter(bearish_mss.index, bearish_mss['Close'], 
                      color='purple', marker='v', s=120, label='Bearish MSS')
        
        # Plot buy signals
        buy_signals = data[data['MS_Trade_Signal'].str.startswith('Buy')]
        if not buy_signals.empty:
            ax.scatter(buy_signals.index, buy_signals['Close'] - (data['High'].max() - data['Low'].min()) * 0.02, 
                      color='green', marker='^', s=100, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = data[data['MS_Trade_Signal'].str.startswith('Sell')]
        if not sell_signals.empty:
            ax.scatter(sell_signals.index, sell_signals['Close'] + (data['High'].max() - data['Low'].min()) * 0.02, 
                      color='red', marker='v', s=100, label='Sell Signal')
        
        # Set title and labels
        ax.set_title('Price with ICT Market Structure')
        ax.set_ylabel('Price')
        ax.grid(True)
        ax.legend(loc='upper left')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot if output_dir is provided
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            if filename is None:
                filename = "ict_market_structure_plot.png"
            
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
        
        return fig
    
    def get_current_market_structure(self, data):
        """
        Get information about the current market structure.
        
        Args:
            data (pandas.DataFrame): DataFrame with Market Structure information.
        
        Returns:
            dict: Dictionary with current market structure information.
        """
        if len(data) < 3*self.lookback:
            return {
                'trend': 'Insufficient Data',
                'recent_shifts': [],
                'recent_swings': []
            }
        
        # Get the most recent trend
        current_trend = data['MS_Trend'].iloc[-1]
        
        # Find recent market structure shifts
        recent_shifts = []
        for i in range(len(data)-1, max(0, len(data)-20), -1):
            if data['MSS_Bullish'].iloc[i]:
                recent_shifts.append({
                    'type': 'Bullish MSS',
                    'index': i,
                    'date': data.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(data.index[i], 'strftime') else str(data.index[i]),
                    'price': data['Close'].iloc[i]
                })
            elif data['MSS_Bearish'].iloc[i]:
                recent_shifts.append({
                    'type': 'Bearish MSS',
                    'index': i,
                    'date': data.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(data.index[i], 'strftime') else str(data.index[i]),
                    'price': data['Close'].iloc[i]
                })
            
            if len(recent_shifts) >= 3:
                break
        
        # Find recent swing points
        recent_swings = []
        for i in range(len(data)-1, max(0, len(data)-20), -1):
            if data['Swing_High'].iloc[i]:
                swing_type = 'Higher High' if data['Higher_High'].iloc[i] else 'Lower High' if data['Lower_High'].iloc[i] else 'Swing High'
                recent_swings.append({
                    'type': swing_type,
                    'index': i,
                    'date': data.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(data.index[i], 'strftime') else str(data.index[i]),
                    'price': data['High'].iloc[i]
                })
            elif data['Swing_Low'].iloc[i]:
                swing_type = 'Higher Low' if data['Higher_Low'].iloc[i] else 'Lower Low' if data['Lower_Low'].iloc[i] else 'Swing Low'
                recent_swings.append({
                    'type': swing_type,
                    'index': i,
                    'date': data.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(data.index[i], 'strftime') else str(data.index[i]),
                    'price': data['Low'].iloc[i]
                })
            
            if len(recent_swings) >= 5:
                break
        
        # Count recent higher highs, higher lows, etc.
        recent_hh = sum(1 for s in recent_swings if s['type'] == 'Higher High')
        recent_hl = sum(1 for s in recent_swings if s['type'] == 'Higher Low')
        recent_lh = sum(1 for s in recent_swings if s['type'] == 'Low<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>