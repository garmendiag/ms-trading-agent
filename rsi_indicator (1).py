"""
RSI (Relative Strength Index) Indicator Module

This module implements the RSI technical indicator with various customization
options and signal generation capabilities for the MANUS trading agent.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class RSIIndicator:
    """
    Class for calculating and analyzing Relative Strength Index (RSI).
    """
    
    def __init__(self, period=14, overbought=70, oversold=30):
        """
        Initialize the RSI indicator.
        
        Args:
            period (int, optional): RSI calculation period. Defaults to 14.
            overbought (int, optional): Overbought threshold. Defaults to 70.
            oversold (int, optional): Oversold threshold. Defaults to 30.
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def calculate(self, price_data, price_column='Close'):
        """
        Calculate RSI values for the given price data.
        
        Args:
            price_data (pandas.DataFrame): Price data.
            price_column (str, optional): Column name for price data. Defaults to 'Close'.
        
        Returns:
            pandas.DataFrame: DataFrame with RSI values and signals.
        """
        if price_column not in price_data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        
        # Make a copy of the data to avoid modifying the original
        result = price_data.copy()
        
        # Calculate price changes
        delta = result[price_column].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Add RSI to the result DataFrame
        result[f'RSI_{self.period}'] = rsi
        
        # Add RSI signals
        result['RSI_Signal'] = 'Neutral'
        result.loc[result[f'RSI_{self.period}'] > self.overbought, 'RSI_Signal'] = 'Overbought'
        result.loc[result[f'RSI_{self.period}'] < self.oversold, 'RSI_Signal'] = 'Oversold'
        
        # Add RSI divergence signals
        result = self._calculate_divergence(result, price_column)
        
        return result
    
    def _calculate_divergence(self, data, price_column, window=10):
        """
        Calculate RSI divergence signals.
        
        Args:
            data (pandas.DataFrame): DataFrame with price and RSI data.
            price_column (str): Column name for price data.
            window (int, optional): Window for finding local extrema. Defaults to 10.
        
        Returns:
            pandas.DataFrame: DataFrame with divergence signals.
        """
        # Initialize divergence columns
        data['RSI_Bullish_Divergence'] = False
        data['RSI_Bearish_Divergence'] = False
        
        rsi_col = f'RSI_{self.period}'
        
        # We need at least 2*window+1 data points to calculate divergence
        if len(data) < 2*window+1:
            return data
        
        # Find local price lows and RSI lows for bullish divergence
        for i in range(window, len(data) - window):
            # Check if this is a local price low
            if data[price_column].iloc[i] == data[price_column].iloc[i-window:i+window+1].min():
                # Look back for another local low within a reasonable range
                for j in range(i - 5*window, i - window):
                    if j < 0:
                        continue
                    
                    if data[price_column].iloc[j] == data[price_column].iloc[max(0, j-window):j+window+1].min():
                        # Check for bullish divergence: price makes lower low but RSI makes higher low
                        if (data[price_column].iloc[i] < data[price_column].iloc[j] and 
                            data[rsi_col].iloc[i] > data[rsi_col].iloc[j]):
                            data['RSI_Bullish_Divergence'].iloc[i] = True
                        break
        
        # Find local price highs and RSI highs for bearish divergence
        for i in range(window, len(data) - window):
            # Check if this is a local price high
            if data[price_column].iloc[i] == data[price_column].iloc[i-window:i+window+1].max():
                # Look back for another local high within a reasonable range
                for j in range(i - 5*window, i - window):
                    if j < 0:
                        continue
                    
                    if data[price_column].iloc[j] == data[price_column].iloc[max(0, j-window):j+window+1].max():
                        # Check for bearish divergence: price makes higher high but RSI makes lower high
                        if (data[price_column].iloc[i] > data[price_column].iloc[j] and 
                            data[rsi_col].iloc[i] < data[rsi_col].iloc[j]):
                            data['RSI_Bearish_Divergence'].iloc[i] = True
                        break
        
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI values.
        
        Args:
            data (pandas.DataFrame): DataFrame with RSI values.
        
        Returns:
            pandas.DataFrame: DataFrame with trading signals.
        """
        rsi_col = f'RSI_{self.period}'
        
        # Make a copy of the data
        result = data.copy()
        
        # Initialize signal column
        result['RSI_Trade_Signal'] = 'Hold'
        
        # Generate signals based on RSI crossing thresholds
        for i in range(1, len(result)):
            # Bullish signal: RSI crosses above oversold threshold
            if (result[rsi_col].iloc[i-1] < self.oversold and 
                result[rsi_col].iloc[i] > self.oversold):
                result['RSI_Trade_Signal'].iloc[i] = 'Buy'
            
            # Bearish signal: RSI crosses below overbought threshold
            elif (result[rsi_col].iloc[i-1] > self.overbought and 
                  result[rsi_col].iloc[i] < self.overbought):
                result['RSI_Trade_Signal'].iloc[i] = 'Sell'
            
            # Additional signal: Bullish divergence
            elif result['RSI_Bullish_Divergence'].iloc[i]:
                result['RSI_Trade_Signal'].iloc[i] = 'Buy_Divergence'
            
            # Additional signal: Bearish divergence
            elif result['RSI_Bearish_Divergence'].iloc[i]:
                result['RSI_Trade_Signal'].iloc[i] = 'Sell_Divergence'
        
        return result
    
    def plot(self, data, output_dir=None, filename=None):
        """
        Plot price and RSI with signals.
        
        Args:
            data (pandas.DataFrame): DataFrame with price and RSI data.
            output_dir (str, optional): Directory to save the plot. Defaults to None.
            filename (str, optional): Filename for the plot. Defaults to None.
        
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        rsi_col = f'RSI_{self.period}'
        
        if rsi_col not in data.columns:
            raise ValueError(f"RSI column '{rsi_col}' not found in data")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price on top subplot
        ax1.plot(data.index, data['Close'], label='Close Price')
        
        # Plot buy signals
        buy_signals = data[data['RSI_Trade_Signal'] == 'Buy']
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['Close'], 
                       color='green', marker='^', s=100, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = data[data['RSI_Trade_Signal'] == 'Sell']
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['Close'], 
                       color='red', marker='v', s=100, label='Sell Signal')
        
        # Plot divergence signals
        buy_div_signals = data[data['RSI_Trade_Signal'] == 'Buy_Divergence']
        if not buy_div_signals.empty:
            ax1.scatter(buy_div_signals.index, buy_div_signals['Close'], 
                       color='lime', marker='^', s=120, label='Bullish Divergence')
        
        sell_div_signals = data[data['RSI_Trade_Signal'] == 'Sell_Divergence']
        if not sell_div_signals.empty:
            ax1.scatter(sell_div_signals.index, sell_div_signals['Close'], 
                       color='darkred', marker='v', s=120, label='Bearish Divergence')
        
        # Set title and labels for price subplot
        ax1.set_title('Price with RSI Signals')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # Plot RSI on bottom subplot
        ax2.plot(data.index, data[rsi_col], label=f'RSI-{self.period}', color='purple')
        
        # Add overbought and oversold lines
        ax2.axhline(y=self.overbought, color='r', linestyle='--', label=f'Overbought ({self.overbought})')
        ax2.axhline(y=self.oversold, color='g', linestyle='--', label=f'Oversold ({self.oversold})')
        ax2.axhline(y=50, color='k', linestyle='-', alpha=0.3)
        
        # Set title and labels for RSI subplot
        ax2.set_title(f'RSI ({self.period})')
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
                filename = f"rsi_{self.period}_plot.png"
            
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
        
        return fig
    
    def get_current_signal(self, data):
        """
        Get the current trading signal based on RSI.
        
        Args:
            data (pandas.DataFrame): DataFrame with RSI values.
        
        Returns:
            dict: Dictionary with current RSI signal information.
        """
        if len(data) < 2:
            return {
                'signal': 'Insufficient Data',
                'rsi_value': None,
                'signal_strength': 0
            }
        
        rsi_col = f'RSI_{self.period}'
        
        # Get the most recent values
        current_rsi = data[rsi_col].iloc[-1]
        current_signal = data['RSI_Trade_Signal'].iloc[-1]
        
        # Calculate signal strength (0-100)
        if current_signal == 'Buy' or current_signal == 'Buy_Divergence':
            # For buy signals, strength increases as RSI gets lower
            signal_strength = max(0, min(100, 100 - current_rsi))
        elif current_signal == 'Sell' or current_signal == 'Sell_Divergence':
            # For sell signals, strength increases as RSI gets higher
            signal_strength = max(0, min(100, current_rsi))
        else:
            # For hold signals, strength is based on distance from 50
            signal_strength = max(0, min(100, 2 * abs(current_rsi - 50)))
        
        # Determine trend direction
        if len(data) >= 5:
            rsi_trend = data[rsi_col].iloc[-5:].mean() - data[rsi_col].iloc[-1]
            if rsi_trend > 2:
                trend_direction = "Falling"
            elif rsi_trend < -2:
                trend_direction = "Rising"
            else:
                trend_direction = "Flat"
        else:
            trend_direction = "Unknown"
        
        return {
            'signal': current_signal,
            'rsi_value': current_rsi,
            'signal_strength': signal_strength,
            'trend_direction': trend_direction,
            'is_overbought': current_rsi > self.overbought,
            'is_oversold': current_rsi < self.oversold
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
    
    # Get S&P 500 data
    sp500_data = collector.get_historical_data('sp500', period="30d", interval="1h")
    
    # Create RSI indicator
    rsi = RSIIndicator(period=14, overbought=70, oversold=30)
    
    # Calculate RSI
    rsi_data = rsi.calculate(sp500_data)
    
    # Generate signals
    signal_data = rsi.generate_signals(rsi_data)
    
    # Print the last few rows
    print(signal_data[['Close', 'RSI_14', 'RSI_Signal', 'RSI_Trade_Signal']].tail())
    
    # Get current signal
    current_signal = rsi.get_current_signal(signal_data)
    print("\nCurrent RSI Signal:")
    for key, value in current_signal.items():
        print(f"  {key}: {value}")
    
    # Plot RSI
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    rsi.plot(signal_data, output_dir=output_dir)
