"""
Market Index Data Collection Module

This module is responsible for collecting S&P 500 and Nasdaq-100 index data
using the Yahoo Finance API. It provides functions to fetch historical and
real-time index data, calculate various metrics, and store the data for use
by the trading agent.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime
from pathlib import Path

class MarketIndexCollector:
    """
    Class for collecting and processing S&P 500 and Nasdaq-100 index data.
    """
    
    # Symbols for indices and their ETF equivalents
    SYMBOLS = {
        'sp500': '^GSPC',       # S&P 500 Index
        'nasdaq100': '^NDX',    # Nasdaq-100 Index
        'spy': 'SPY',           # SPDR S&P 500 ETF
        'qqq': 'QQQ',           # Invesco QQQ Trust (Nasdaq-100 ETF)
        'es': 'ES=F',           # E-mini S&P 500 Future
        'nq': 'NQ=F',           # E-mini Nasdaq-100 Future
        'mes': 'MES=F',         # Micro E-mini S&P 500 Future
        'mnq': 'MNQ=F'          # Micro E-mini Nasdaq-100 Future
    }
    
    def __init__(self, cache_dir=None):
        """
        Initialize the market index data collector.
        
        Args:
            cache_dir (str, optional): Directory to cache index data. 
                                      Defaults to None (no caching).
        """
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def get_historical_data(self, symbol_key, period="30d", interval="5m"):
        """
        Get historical data for the specified market index or ETF.
        
        Args:
            symbol_key (str): Key for the symbol in SYMBOLS dictionary.
            period (str, optional): Time period to fetch data for. 
                                   Defaults to "30d" (30 days).
            interval (str, optional): Data interval. 
                                     Defaults to "5m" (5 minutes).
        
        Returns:
            pandas.DataFrame: DataFrame containing historical data.
        """
        if symbol_key not in self.SYMBOLS:
            raise ValueError(f"Unknown symbol key: {symbol_key}. Available keys: {list(self.SYMBOLS.keys())}")
        
        symbol = self.SYMBOLS[symbol_key]
        
        try:
            # Check if cached data exists and is recent enough
            if self.cache_dir:
                cache_file = Path(self.cache_dir) / f"{symbol_key}_{period}_{interval}.csv"
                if cache_file.exists():
                    # Check if file is less than 1 hour old
                    file_time = datetime.datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if (datetime.datetime.now() - file_time).seconds < 3600:
                        print(f"Loading cached {symbol_key} data from {cache_file}")
                        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            
            # Fetch data from Yahoo Finance
            print(f"Fetching {symbol_key} ({symbol}) data for period={period}, interval={interval}")
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            
            # Calculate additional metrics
            data['Return'] = data['Close'].pct_change() * 100  # Percentage return
            data['Return_1h'] = data['Close'].pct_change(12) * 100  # 1-hour return (assuming 5m intervals)
            data['Return_1d'] = data['Close'].pct_change(78) * 100  # 1-day return (assuming 5m intervals)
            
            # Calculate volume metrics
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()  # 20-period volume moving average
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']  # Volume ratio to moving average
            
            # Calculate price range metrics
            data['Range'] = data['High'] - data['Low']  # Daily range
            data['Range_Pct'] = data['Range'] / data['Close'] * 100  # Range as percentage of close
            
            # Cache the data if cache_dir is specified
            if self.cache_dir:
                data.to_csv(cache_file)
            
            return data
        
        except Exception as e:
            print(f"Error fetching {symbol_key} data: {e}")
            return pd.DataFrame()
    
    def get_multiple_indices(self, symbol_keys=None, period="1d", interval="5m"):
        """
        Get data for multiple indices and merge their closing prices into a single DataFrame.
        
        Args:
            symbol_keys (list, optional): List of symbol keys. If None, fetches all indices.
            period (str, optional): Time period to fetch data for.
            interval (str, optional): Data interval.
        
        Returns:
            pandas.DataFrame: DataFrame containing closing prices for all requested indices.
        """
        if symbol_keys is None:
            symbol_keys = list(self.SYMBOLS.keys())
        
        result = pd.DataFrame()
        
        for key in symbol_keys:
            try:
                df = self.get_historical_data(key, period, interval)
                if not df.empty:
                    # Add the closing price to the result DataFrame
                    result[key] = df['Close']
            except Exception as e:
                print(f"Error processing {key}: {e}")
        
        return result
    
    def get_current_prices(self, symbol_keys=None):
        """
        Get the current prices and metrics for specified indices.
        
        Args:
            symbol_keys (list, optional): List of symbol keys. If None, fetches all indices.
        
        Returns:
            dict: Dictionary containing current prices and metrics for each index.
        """
        if symbol_keys is None:
            symbol_keys = list(self.SYMBOLS.keys())
        
        result = {}
        
        for key in symbol_keys:
            try:
                df = self.get_historical_data(key, period="1d", interval="5m")
                if not df.empty:
                    result[key] = {
                        'price': df['Close'].iloc[-1],
                        'change_pct': df['Return'].iloc[-1],
                        'change_1h': df['Return_1h'].iloc[-1],
                        'change_1d': df['Return_1d'].iloc[-1],
                        'volume': df['Volume'].iloc[-1],
                        'volume_ratio': df['Volume_Ratio'].iloc[-1],
                        'range_pct': df['Range_Pct'].iloc[-1],
                        'timestamp': df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                    }
            except Exception as e:
                print(f"Error getting current price for {key}: {e}")
        
        return result
    
    def get_es_nq_spread(self):
        """
        Calculate the spread between ES and NQ futures, which is useful for
        the ES-NQ trading strategy.
        
        Returns:
            dict: Dictionary containing spread information.
        """
        try:
            # Get data for ES and NQ futures
            es_data = self.get_historical_data('es', period="5d", interval="5m")
            nq_data = self.get_historical_data('nq', period="5d", interval="5m")
            
            if es_data.empty or nq_data.empty:
                return {}
            
            # Align the data
            combined = pd.DataFrame({
                'ES': es_data['Close'],
                'NQ': nq_data['Close']
            })
            combined = combined.dropna()
            
            if combined.empty:
                return {}
            
            # Calculate the spread (NQ/ES ratio)
            combined['Spread'] = combined['NQ'] / combined['ES']
            
            # Calculate spread metrics
            current_spread = combined['Spread'].iloc[-1]
            spread_mean = combined['Spread'].mean()
            spread_std = combined['Spread'].std()
            z_score = (current_spread - spread_mean) / spread_std
            
            # Determine if spread is extreme
            if z_score > 2:
                spread_status = "Extremely Wide"
            elif z_score > 1:
                spread_status = "Wide"
            elif z_score < -2:
                spread_status = "Extremely Narrow"
            elif z_score < -1:
                spread_status = "Narrow"
            else:
                spread_status = "Normal"
            
            return {
                'current_spread': current_spread,
                'mean_spread': spread_mean,
                'std_spread': spread_std,
                'z_score': z_score,
                'status': spread_status,
                'es_price': combined['ES'].iloc[-1],
                'nq_price': combined['NQ'].iloc[-1],
                'timestamp': combined.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            }
        
        except Exception as e:
            print(f"Error calculating ES-NQ spread: {e}")
            return {}
    
    def get_market_summary(self):
        """
        Get a comprehensive summary of market indices including current prices,
        changes, and relative performance.
        
        Returns:
            dict: Dictionary containing market summary information.
        """
        try:
            # Get current prices for all indices
            prices = self.get_current_prices()
            
            # Get ES-NQ spread
            spread = self.get_es_nq_spread()
            
            # Determine market trend based on S&P 500 and Nasdaq-100 performance
            if prices.get('sp500') and prices.get('nasdaq100'):
                sp500_change = prices['sp500']['change_1h']
                nasdaq_change = prices['nasdaq100']['change_1h']
                
                # Overall market trend
                if sp500_change > 1 and nasdaq_change > 1:
                    market_trend = "Strongly Bullish"
                elif sp500_change > 0.3 and nasdaq_change > 0.3:
                    market_trend = "Bullish"
                elif sp500_change < -1 and nasdaq_change < -1:
                    market_trend = "Strongly Bearish"
                elif sp500_change < -0.3 and nasdaq_change < -0.3:
                    market_trend = "Bearish"
                else:
                    market_trend = "Neutral"
                
                # Relative performance (Nasdaq vs S&P)
                if nasdaq_change > sp500_change + 0.5:
                    relative_performance = "Nasdaq Outperforming"
                elif sp500_change > nasdaq_change + 0.5:
                    relative_performance = "S&P Outperforming"
                else:
                    relative_performance = "Balanced Performance"
            else:
                market_trend = "Unknown"
                relative_performance = "Unknown"
            
            return {
                'prices': prices,
                'es_nq_spread': spread,
                'market_trend': market_trend,
                'relative_performance': relative_performance,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        except Exception as e:
            print(f"Error getting market summary: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Create a market index collector with caching
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    collector = MarketIndexCollector(cache_dir=cache_dir)
    
    # Get S&P 500 data
    sp500_data = collector.get_historical_data('sp500')
    print(f"S&P 500 data shape: {sp500_data.shape}")
    print(sp500_data.tail())
    
    # Get current prices
    prices = collector.get_current_prices(['sp500', 'nasdaq100', 'es', 'nq'])
    print("\nCurrent Prices:")
    for key, info in prices.items():
        print(f"  {key}: {info['price']:.2f} (change: {info['change_pct']:.2f}%)")
    
    # Get ES-NQ spread
    spread = collector.get_es_nq_spread()
    print("\nES-NQ Spread:")
    for key, value in spread.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Get market summary
    summary = collector.get_market_summary()
    print("\nMarket Summary:")
    print(f"  Market Trend: {summary.get('market_trend')}")
    print(f"  Relative Performance: {summary.get('relative_performance')}")
