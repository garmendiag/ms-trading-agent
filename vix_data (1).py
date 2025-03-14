"""
VIX Data Collection Module

This module is responsible for collecting VIX (CBOE Volatility Index) data
using the Yahoo Finance API. It provides functions to fetch historical and
real-time VIX data, calculate rate of change, and store the data for use
by the trading agent.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime
import time
from pathlib import Path

class VIXDataCollector:
    """
    Class for collecting and processing VIX data from Yahoo Finance.
    """
    
    def __init__(self, cache_dir=None):
        """
        Initialize the VIX data collector.
        
        Args:
            cache_dir (str, optional): Directory to cache VIX data. 
                                      Defaults to None (no caching).
        """
        self.symbol = "^VIX"
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def get_historical_data(self, period="30d", interval="5m"):
        """
        Get historical VIX data for the specified period and interval.
        
        Args:
            period (str, optional): Time period to fetch data for. 
                                   Defaults to "30d" (30 days).
            interval (str, optional): Data interval. 
                                     Defaults to "5m" (5 minutes).
        
        Returns:
            pandas.DataFrame: DataFrame containing VIX historical data.
        """
        try:
            # Check if cached data exists and is recent enough
            if self.cache_dir:
                cache_file = Path(self.cache_dir) / f"vix_{period}_{interval}.csv"
                if cache_file.exists():
                    # Check if file is less than 1 hour old
                    file_time = datetime.datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if (datetime.datetime.now() - file_time).seconds < 3600:
                        print(f"Loading cached VIX data from {cache_file}")
                        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            
            # Fetch data from Yahoo Finance
            print(f"Fetching VIX data for period={period}, interval={interval}")
            vix_data = yf.download(self.symbol, period=period, interval=interval, progress=False)
            
            # Calculate rate of change
            vix_data['Change'] = vix_data['Close'].pct_change() * 100  # Percentage change
            vix_data['Change_1h'] = vix_data['Close'].pct_change(12) * 100  # 1-hour change (assuming 5m intervals)
            vix_data['Change_1d'] = vix_data['Close'].pct_change(78) * 100  # 1-day change (assuming 5m intervals)
            
            # Cache the data if cache_dir is specified
            if self.cache_dir:
                vix_data.to_csv(cache_file)
            
            return vix_data
        
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return pd.DataFrame()
    
    def get_current_vix(self):
        """
        Get the current VIX value and its rate of change.
        
        Returns:
            tuple: (current_vix, change_pct) containing the current VIX value
                  and its percentage change from the previous period.
        """
        try:
            # Get the most recent data
            vix_data = self.get_historical_data(period="1d", interval="5m")
            
            if vix_data.empty:
                return None, None
            
            # Get the most recent VIX value and change
            current_vix = vix_data['Close'].iloc[-1]
            change_pct = vix_data['Change'].iloc[-1]
            
            return current_vix, change_pct
        
        except Exception as e:
            print(f"Error getting current VIX: {e}")
            return None, None
    
    def get_vix_summary(self):
        """
        Get a summary of VIX data including current value, changes over different
        time periods, and a volatility assessment.
        
        Returns:
            dict: Dictionary containing VIX summary information.
        """
        try:
            vix_data = self.get_historical_data(period="5d", interval="5m")
            
            if vix_data.empty:
                return {}
            
            current_vix = vix_data['Close'].iloc[-1]
            change_5m = vix_data['Change'].iloc[-1]
            change_1h = vix_data['Change_1h'].iloc[-1]
            change_1d = vix_data['Change_1d'].iloc[-1]
            
            # Calculate average and standard deviation for volatility assessment
            avg_vix = vix_data['Close'].mean()
            std_vix = vix_data['Close'].std()
            
            # Determine volatility level
            if current_vix > avg_vix + 2*std_vix:
                volatility = "Extremely High"
            elif current_vix > avg_vix + std_vix:
                volatility = "High"
            elif current_vix < avg_vix - std_vix:
                volatility = "Low"
            elif current_vix < avg_vix - 2*std_vix:
                volatility = "Extremely Low"
            else:
                volatility = "Normal"
            
            # Determine trend based on recent changes
            if change_1h > 5:
                trend = "Strongly Rising"
            elif change_1h > 2:
                trend = "Rising"
            elif change_1h < -5:
                trend = "Strongly Falling"
            elif change_1h < -2:
                trend = "Falling"
            else:
                trend = "Stable"
            
            return {
                "current_vix": current_vix,
                "change_5m": change_5m,
                "change_1h": change_1h,
                "change_1d": change_1d,
                "avg_vix_5d": avg_vix,
                "volatility_level": volatility,
                "trend": trend
            }
        
        except Exception as e:
            print(f"Error getting VIX summary: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Create a VIX data collector with caching
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    vix_collector = VIXDataCollector(cache_dir=cache_dir)
    
    # Get historical data
    historical_data = vix_collector.get_historical_data()
    print(f"Historical data shape: {historical_data.shape}")
    print(historical_data.tail())
    
    # Get current VIX
    current_vix, change_pct = vix_collector.get_current_vix()
    print(f"Current VIX: {current_vix:.2f}, Change: {change_pct:.2f}%")
    
    # Get VIX summary
    summary = vix_collector.get_vix_summary()
    print("\nVIX Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
