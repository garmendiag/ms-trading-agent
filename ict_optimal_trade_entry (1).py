"""
ICT Optimal Trade Entry (OTE) Indicator Module

This module implements the ICT (Inner Circle Trader) Optimal Trade Entry (OTE)
concept with various customization options and signal generation capabilities
for the MANUS trading agent.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class OptimalTradeEntryIndicator:
    """
    Class for identifying and analyzing ICT Optimal Trade Entry (OTE) levels.
    """
    
    def __init__(self, lookback=20, fib_levels=None):
        """
        Initialize the Optimal Trade Entry indicator.
        
        Args:
            lookback (int, optional): Lookback period for identifying swing points. Defaults to 20.
            fib_levels (list, optional): Fibonacci levels to use for OTE. Defaults to [0.5, 0.618, 0.65, 0.786].
        """
        self.lookback = lookback
        self.fib_levels = fib_levels or [0.5, 0.618, 0.65, 0.786]
    
    def calculate(self, price_data, ms_data=None):
        """
        Calculate Optimal Trade Entry (OTE) levels.
        
        Args:
            price_data (pandas.DataFrame): Price data with OHLC columns.
            ms_data (pandas.DataFrame, optional): Market Structure data. If None, will calculate internally.
        
        Returns:
            pandas.DataFrame: DataFrame with OTE information.
        """
        # Make a copy of the data to avoid modifying the original
        result = price_data.copy()
        
        # If market structure data is not provided, calculate it
        if ms_data is None:
            # Import here to avoid circular imports
            from .ict_market_structure import MarketStructureIndicator
            ms_indicator = MarketStructureIndicator(lookback=self.lookback)
            ms_data = ms_indicator.calculate(price_data)
        
        # Initialize OTE columns
        result['OTE_Buy_Level'] = np.nan
        result['OTE_Sell_Level'] = np.nan
        result['OTE_Buy_Zone_High'] = np.nan
        result['OTE_Buy_Zone_Low'] = np.nan
        result['OTE_Sell_Zone_High'] = np.nan
        result['OTE_Sell_Zone_Low'] = np.nan
        result['OTE_Buy_Active'] = False
        result['OTE_Sell_Active'] = False
        result['OTE_Buy_Hit'] = False
        result['OTE_Sell_Hit'] = False
        
        # Calculate OTE levels
        for i in range(2*self.lookback, len(result)):
            # Bullish OTE: After a bullish MSS, look for a retracement to 50-78.6% of the previous swing
            if ms_data['MSS_Bullish'].iloc[i]:
                # Find the most recent swing low before the MSS
                swing_low_idx = None
                for j in range(i-1, max(0, i-2*self.lookback), -1):
                    if ms_data['Swing_Low'].iloc[j]:
                        swing_low_idx = j
                        break
                
                # Find the most recent swing high before the swing low
                swing_high_idx = None
                if swing_low_idx is not None:
                    for j in range(swing_low_idx-1, max(0, swing_low_idx-2*self.lookback), -1):
                        if ms_data['Swing_High'].iloc[j]:
                            swing_high_idx = j
                            break
                
                # Calculate OTE buy levels (50-78.6% retracement)
                if swing_low_idx is not None and swing_high_idx is not None:
                    swing_high = result['High'].iloc[swing_high_idx]
                    swing_low = result['Low'].iloc[swing_low_idx]
                    swing_range = swing_high - swing_low
                    
                    # Calculate OTE levels for each Fibonacci level
                    ote_levels = [swing_high - level * swing_range for level in self.fib_levels]
                    
                    # Set the main OTE level (average of all levels)
                    result['OTE_Buy_Level'].iloc[i] = sum(ote_levels) / len(ote_levels)
                    
                    # Set the OTE zone (min to max of the levels)
                    result['OTE_Buy_Zone_High'].iloc[i] = max(ote_levels)
                    result['OTE_Buy_Zone_Low'].iloc[i] = min(ote_levels)
                    
                    # Mark the OTE as active
                    result['OTE_Buy_Active'].iloc[i] = True
            
            # Bearish OTE: After a bearish MSS, look for a retracement to 50-78.6% of the previous swing
            if ms_data['MSS_Bearish'].iloc[i]:
                # Find the most recent swing high before the MSS
                swing_high_idx = None
                for j in range(i-1, max(0, i-2*self.lookback), -1):
                    if ms_data['Swing_High'].iloc[j]:
                        swing_high_idx = j
                        break
                
                # Find the most recent swing low before the swing high
                swing_low_idx = None
                if swing_high_idx is not None:
                    for j in range(swing_high_idx-1, max(0, swing_high_idx-2*self.lookback), -1):
                        if ms_data['Swing_Low'].iloc[j]:
                            swing_low_idx = j
                            break
                
                # Calculate OTE sell levels (50-78.6% retracement)
                if swing_high_idx is not None and swing_low_idx is not None:
                    swing_high = result['High'].iloc[swing_high_idx]
                    swing_low = result['Low'].iloc[swing_low_idx]
                    swing_range = swing_high - swing_low
                    
                    # Calculate OTE levels for each Fibonacci level
                    ote_levels = [swing_low + level * swing_range for level in self.fib_levels]
                    
                    # Set the main OTE level (average of all levels)
                    result['OTE_Sell_Level'].iloc[i] = sum(ote_levels) / len(ote_levels)
                    
                    # Set the OTE zone (min to max of the levels)
                    result['OTE_Sell_Zone_Low'].iloc[i] = min(ote_levels)
                    result['OTE_Sell_Zone_High'].iloc[i] = max(ote_levels)
                    
                    # Mark the OTE as active
                    result['OTE_Sell_Active'].iloc[i] = True
        
        # Propagate OTE levels forward until they are hit or a new OTE is formed
        for i in range(len(result)-2, 2*self.lookback-1, -1):
            # If this row doesn't have an OTE level, copy from the next row
            if np.isnan(result['OTE_Buy_Level'].iloc[i]) and not np.isnan(result['OTE_Buy_Level'].iloc[i+1]):
                result['OTE_Buy_Level'].iloc[i] = result['OTE_Buy_Level'].iloc[i+1]
                result['OTE_Buy_Zone_High'].iloc[i] = result['OTE_Buy_Zone_High'].iloc[i+1]
                result['OTE_Buy_Zone_Low'].iloc[i] = result['OTE_Buy_Zone_Low'].iloc[i+1]
                result['OTE_Buy_Active'].iloc[i] = result['OTE_Buy_Active'].iloc[i+1]
            
            if np.isnan(result['OTE_Sell_Level'].iloc[i]) and not np.isnan(result['OTE_Sell_Level'].iloc[i+1]):
                result['OTE_Sell_Level'].iloc[i] = result['OTE_Sell_Level'].iloc[i+1]
                result['OTE_Sell_Zone_High'].iloc[i] = result['OTE_Sell_Zone_High'].iloc[i+1]
                result['OTE_Sell_Zone_Low'].iloc[i] = result['OTE_Sell_Zone_Low'].iloc[i+1]
                result['OTE_Sell_Active'].iloc[i] = result['OTE_Sell_Active'].iloc[i+1]
        
        # Check if OTE levels have been hit
        for i in range(1, len(result)):
            # Check if buy OTE has been hit
            if result['OTE_Buy_Active'].iloc[i-1] and not np.isnan(result['OTE_Buy_Zone_Low'].iloc[i-1]):
                if result['Low'].iloc[i] <= result['OTE_Buy_Zone_High'].iloc[i-1] and result['Low'].iloc[i] >= result['OTE_Buy_Zone_Low'].iloc[i-1]:
                    result['OTE_Buy_Hit'].iloc[i] = True
                    # Deactivate the OTE once it's hit
                    result['OTE_Buy_Active'].iloc[i] = False
                else:
                    # Propagate active status
                    result['OTE_Buy_Active'].iloc[i] = result['OTE_Buy_Active'].iloc[i-1]
            
            # Check if sell OTE has been hit
            if result['OTE_Sell_Active'].iloc[i-1] and not np.isnan(result['OTE_Sell_Zone_Low'].iloc[i-1]):
                if result['High'].iloc[i] >= result['OTE_Sell_Zone_Low'].iloc[i-1] and result['High'].iloc[i] <= result['OTE_Sell_Zone_High'].iloc[i-1]:
                    result['OTE_Sell_Hit'].iloc[i] = True
                    # Deactivate the OTE once it's hit
                    result['OTE_Sell_Active'].iloc[i] = False
                else:
                    # Propagate active status
                    result['OTE_Sell_Active'].iloc[i] = result['OTE_Sell_Active'].iloc[i-1]
        
        return result
    
    def generate_signals(self, data):
        """
        Generate trading signals based on OTE levels.
        
        Args:
            data (pandas.DataFrame): DataFrame with OTE information.
        
        Returns:
            pandas.DataFrame: DataFrame with trading signals.
        """
        # Make a copy of the data
        result = data.copy()
        
        # Initialize signal column
        result['OTE_Trade_Signal'] = 'Hold'
        
        # Generate signals based on OTE hits
        for i in range(1, len(result)):
            # Buy signal when price hits buy OTE
            if result['OTE_Buy_Hit'].iloc[i]:
                result['OTE_Trade_Signal'].iloc[i] = 'Buy'
            
            # Sell signal when price hits sell OTE
            elif result['OTE_Sell_Hit'].iloc[i]:
                result['OTE_Trade_Signal'].iloc[i] = 'Sell'
            
            # Approaching buy OTE
            elif result['OTE_Buy_Active'].iloc[i] and not np.isnan(result['OTE_Buy_Zone_High'].iloc[i]):
                distance = (result['Close'].iloc[i] - result['OTE_Buy_Zone_High'].iloc[i]) / result['Close'].iloc[i] * 100
                if 0 < distance < 1:  # Within 1% above the OTE zone
                    result['OTE_Trade_Signal'].iloc[i] = 'Buy_Approaching'
            
            # Approaching sell OTE
            elif result['OTE_Sell_Active'].iloc[i] and not np.isnan(result['OTE_Sell_Zone_Low'].iloc[i]):
                distance = (result['OTE_Sell_Zone_Low'].iloc[i] - result['Close'].iloc[i]) / result['Close'].iloc[i] * 100
                if 0 < distance < 1:  # Within 1% below the OTE zone
                    result['OTE_Trade_Signal'].iloc[i] = 'Sell_Approaching'
        
        return result
    
    def plot(self, data, output_dir=None, filename=None):
        """
        Plot price with OTE levels and signals.
        
        Args:
            data (pandas.DataFrame): DataFrame with price and OTE data.
            output_dir (str, optional): Directory to save the plot. Defaults to None.
            filename (str, optional): Filename for the plot. Defaults to None.
        
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price
        ax.plot(data.index, data['Close'], label='Close Price')
        
        # Plot OTE buy zones
        for i in range(len(data)):
            if data['OTE_Buy_Active'].iloc[i] and not np.isnan(data['OTE_Buy_Zone_Low'].iloc[i]):
                ax.axhspan(data['OTE_Buy_Zone_Low'].iloc[i], data['OTE_Buy_Zone_High'].iloc[i], 
                          xmin=i/len(data), xmax=(i+1)/len(data),
                          color='green', alpha=0.2)
                
                # Plot the main OTE level
                ax.axhline(y=data['OTE_Buy_Level'].iloc[i], 
                          xmin=i/len(data), xmax=(i+1)/len(data),
                          color='green', linestyle='--', alpha=0.7)
        
        # Plot OTE sell zones
        for i in range(len(data)):
            if data['OTE_Sell_Active'].iloc[i] and not np.isnan(data['OTE_Sell_Zone_Low'].iloc[i]):
                ax.axhspan(data['OTE_Sell_Zone_Low'].iloc[i], data['OTE_Sell_Zone_High'].iloc[i], 
                          xmin=i/len(data), xmax=(i+1)/len(data),
                          color='red', alpha=0.2)
                
                # Plot the main OTE level
                ax.axhline(y=data['OTE_Sell_Level'].iloc[i], 
                          xmin=i/len(data), xmax=(i+1)/len(data),
                          color='red', linestyle='--', alpha=0.7)
        
        # Plot buy signals
        buy_signals = data[data['OTE_Trade_Signal'] == 'Buy']
        if not buy_signals.empty:
            ax.scatter(buy_signals.index, buy_signals['Close'], 
                      color='green', marker='^', s=100, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = data[data['OTE_Trade_Signal'] == 'Sell']
        if not sell_signals.empty:
            ax.scatter(sell_signals.index, sell_signals['Close'], 
                      color='red', marker='v', s=100, label='Sell Signal')
        
        # Plot approaching signals
        buy_approaching = data[data['OTE_Trade_Signal'] == 'Buy_Approaching']
        if not buy_approaching.empty:
            ax.scatter(buy_approaching.index, buy_approaching['Close'], 
                      color='lime', marker='^', s=60, label='Buy Approaching')
        
        sell_approaching = data[data['OTE_Trade_Signal'] == 'Sell_Approaching']
        if not sell_approaching.empty:
            ax.scatter(sell_approaching.index, sell_approaching['Close'], 
                      color='lightcoral', marker='v', s=60, label='Sell Approaching')
        
        # Set title and labels
        ax.set_title('Price with ICT Optimal Trade Entry (OTE) Levels')
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
                filename = "ict_ote_plot.png"
            
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
        
        return fig
    
    def get_active_ote_levels(self, data):
        """
        Get information about active OTE levels.
        
        Args:
            data (pandas.DataFrame): DataFrame with OTE information.
        
        Returns:
            dict: Dictionary with active OTE level information.
        """
        if len(data) < 1:
            return {
                'buy_ote_active': False,
                'sell_ote_active': False,
                'buy_ote_level': None,
                'sell_ote_level': None
            }
        
        # Get the most recent data point
        latest = data.iloc[-1]
        
        # Check if there are active OTE levels
        buy_ote_active = latest['OTE_Buy_Active']
        sell_ote_active = latest['OTE_Sell_Active']
        
        # Get the OTE levels
        buy_ote_level = latest['OTE_Buy_Level'] if not np.isnan(latest['OTE_Buy_Level']) else None
        buy_ote_zone_high = latest['OTE_Buy_Zone_High'] if not np.isnan(latest['OTE_Buy_Zone_High']) else None
        buy_ote_zone_low = latest['OTE_Buy_Zone_Low'] if not np.isnan(latest['OTE_Buy_Zone_Low']) else None
        
        sell_ote_level = latest['OTE_Sell_Level'] if not np.isnan(latest['OTE_Sell_Level']) else None
        sell_ote_zone_high = latest['OTE_Sell_Zone_High'] if not np.isnan(latest['OTE_Sell_Zone_High']) else None
        sell_ote_zone_low = latest['OTE_Sell_Zone_Low'] if not np.isnan(latest['OTE_Sell_Zone_Low']) else None
        
        # Calculate distance to current price
        current_price = latest['Close']
        
        buy_distance_pct = None
        if buy_ote_level is not None:
   <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>