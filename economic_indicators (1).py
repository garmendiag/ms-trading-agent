"""
Economic Indicators Data Collection Module

This module is responsible for collecting U.S. economic indicators data
using the FRED API. It provides functions to fetch data for key economic
indicators such as PMI, NFP, and Fed Funds Rate, and store the data for
use by the trading agent.
"""

import pandas as pd
import numpy as np
import os
import datetime
from fredapi import Fred
from pathlib import Path

class EconomicIndicatorsCollector:
    """
    Class for collecting and processing economic indicators data from FRED.
    """
    
    # FRED Series IDs for key economic indicators
    SERIES_IDS = {
        'pmi': 'NAPM',                # ISM Manufacturing PMI
        'nfp': 'PAYEMS',              # Nonfarm Payroll Employment
        'fed_funds_rate': 'FEDFUNDS', # Federal Funds Effective Rate
        'cpi': 'CPIAUCSL',            # Consumer Price Index for All Urban Consumers
        'gdp': 'GDP',                 # Gross Domestic Product
        'unemployment': 'UNRATE',     # Unemployment Rate
        'retail_sales': 'RSAFS',      # Retail Sales
        'industrial_production': 'INDPRO', # Industrial Production Index
        'housing_starts': 'HOUST',    # Housing Starts
        'consumer_sentiment': 'UMCSENT' # University of Michigan Consumer Sentiment
    }
    
    def __init__(self, api_key=None, cache_dir=None):
        """
        Initialize the economic indicators data collector.
        
        Args:
            api_key (str, optional): FRED API key. If None, will look for
                                    FRED_API_KEY environment variable.
            cache_dir (str, optional): Directory to cache economic data.
                                      Defaults to None (no caching).
        """
        # Use provided API key or get from environment variable
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        
        if not self.api_key:
            print("Warning: No FRED API key provided. Using demo mode with limited functionality.")
            # In demo mode, we'll use pre-cached data or mock data
        else:
            self.fred = Fred(api_key=self.api_key)
        
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def get_indicator_data(self, indicator, start_date=None, end_date=None, frequency=None):
        """
        Get data for a specific economic indicator.
        
        Args:
            indicator (str): Indicator name (must be in SERIES_IDS).
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            frequency (str, optional): Data frequency ('d', 'w', 'm', 'q', 'a').
        
        Returns:
            pandas.DataFrame: DataFrame containing indicator data.
        """
        if indicator not in self.SERIES_IDS:
            raise ValueError(f"Unknown indicator: {indicator}. Available indicators: {list(self.SERIES_IDS.keys())}")
        
        series_id = self.SERIES_IDS[indicator]
        
        # Check if cached data exists and is recent enough
        if self.cache_dir:
            cache_file = Path(self.cache_dir) / f"{indicator}.csv"
            if cache_file.exists():
                # For economic indicators, data that's less than 1 day old is considered fresh
                file_time = datetime.datetime.fromtimestamp(cache_file.stat().st_mtime)
                if (datetime.datetime.now() - file_time).days < 1:
                    print(f"Loading cached {indicator} data from {cache_file}")
                    return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # If no API key or in demo mode, use mock data
        if not self.api_key:
            print(f"No API key available. Using mock data for {indicator}.")
            return self._get_mock_data(indicator)
        
        try:
            # Fetch data from FRED
            print(f"Fetching {indicator} data from FRED")
            data = self.fred.get_series(
                series_id, 
                observation_start=start_date,
                observation_end=end_date,
                frequency=frequency
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[indicator])
            
            # Calculate rate of change
            df[f"{indicator}_change"] = df[indicator].pct_change() * 100
            
            # Cache the data if cache_dir is specified
            if self.cache_dir:
                df.to_csv(Path(self.cache_dir) / f"{indicator}.csv")
            
            return df
        
        except Exception as e:
            print(f"Error fetching {indicator} data: {e}")
            # Fall back to mock data if API call fails
            return self._get_mock_data(indicator)
    
    def get_multiple_indicators(self, indicators=None, start_date=None, end_date=None):
        """
        Get data for multiple economic indicators and merge them into a single DataFrame.
        
        Args:
            indicators (list, optional): List of indicator names. If None, fetches all available indicators.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
        
        Returns:
            pandas.DataFrame: DataFrame containing data for all requested indicators.
        """
        if indicators is None:
            indicators = list(self.SERIES_IDS.keys())
        
        # Initialize with the first indicator
        result = self.get_indicator_data(indicators[0], start_date, end_date)
        
        # Merge with remaining indicators
        for indicator in indicators[1:]:
            df = self.get_indicator_data(indicator, start_date, end_date)
            result = pd.merge(result, df, left_index=True, right_index=True, how='outer')
        
        return result
    
    def get_latest_values(self, indicators=None):
        """
        Get the latest values for specified economic indicators.
        
        Args:
            indicators (list, optional): List of indicator names. If None, fetches all available indicators.
        
        Returns:
            dict: Dictionary containing the latest values for each indicator.
        """
        if indicators is None:
            indicators = list(self.SERIES_IDS.keys())
        
        result = {}
        
        for indicator in indicators:
            try:
                df = self.get_indicator_data(indicator)
                if not df.empty:
                    latest_value = df[indicator].iloc[-1]
                    latest_change = df[f"{indicator}_change"].iloc[-1]
                    latest_date = df.index[-1]
                    
                    result[indicator] = {
                        'value': latest_value,
                        'change': latest_change,
                        'date': latest_date.strftime('%Y-%m-%d'),
                        'days_old': (datetime.datetime.now().date() - latest_date.date()).days
                    }
            except Exception as e:
                print(f"Error getting latest value for {indicator}: {e}")
        
        return result
    
    def _get_mock_data(self, indicator):
        """
        Generate mock data for demonstration purposes when API key is not available.
        
        Args:
            indicator (str): Indicator name.
        
        Returns:
            pandas.DataFrame: DataFrame containing mock data.
        """
        # Create date range for the past year
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Generate mock values based on indicator type
        if indicator == 'pmi':
            # PMI typically ranges from 40 to 60
            values = np.random.normal(50, 3, size=len(dates))
        elif indicator == 'nfp':
            # NFP in thousands, typically ranges from -500k to +500k
            values = np.random.normal(100, 150, size=len(dates)) * 1000
        elif indicator == 'fed_funds_rate':
            # Fed funds rate typically ranges from 0 to 5%
            values = np.linspace(2.0, 3.5, len(dates)) + np.random.normal(0, 0.1, size=len(dates))
        else:
            # Generic values for other indicators
            values = np.random.normal(100, 10, size=len(dates))
        
        # Create DataFrame
        df = pd.DataFrame(values, index=dates, columns=[indicator])
        
        # Calculate rate of change
        df[f"{indicator}_change"] = df[indicator].pct_change() * 100
        
        return df

# Example usage
if __name__ == "__main__":
    # Create an economic indicators collector with caching
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    collector = EconomicIndicatorsCollector(cache_dir=cache_dir)
    
    # Get PMI data
    pmi_data = collector.get_indicator_data('pmi')
    print(f"PMI data shape: {pmi_data.shape}")
    print(pmi_data.tail())
    
    # Get multiple indicators
    indicators = ['pmi', 'fed_funds_rate', 'unemployment']
    multi_data = collector.get_multiple_indicators(indicators)
    print(f"\nMultiple indicators data shape: {multi_data.shape}")
    print(multi_data.tail())
    
    # Get latest values
    latest = collector.get_latest_values()
    print("\nLatest indicator values:")
    for indicator, info in latest.items():
        print(f"  {indicator}: {info['value']:.2f} (change: {info['change']:.2f}%, as of {info['date']}, {info['days_old']} days old)")
