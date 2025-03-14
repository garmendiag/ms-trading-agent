"""
Data Collection Module Integration

This module integrates all the data collection components into a single interface
for the MANUS trading agent. It provides a unified way to collect and process
data from various sources.
"""

import os
import datetime
import pandas as pd
from pathlib import Path

# Import all data collection modules
from .vix_data import VIXDataCollector
from .economic_indicators import EconomicIndicatorsCollector
from .market_index import MarketIndexCollector
from .news_sentiment import NewsSentimentAnalyzer
from .tradingview_indicators import TradingViewIndicators

class DataCollectionManager:
    """
    Class for managing and integrating all data collection components.
    """
    
    def __init__(self, cache_dir=None, api_keys=None):
        """
        Initialize the data collection manager.
        
        Args:
            cache_dir (str, optional): Directory to cache data.
                                      Defaults to None (no caching).
            api_keys (dict, optional): Dictionary of API keys for various services.
                                      Defaults to None.
        """
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize API keys
        self.api_keys = api_keys or {}
        
        # Initialize data collectors
        self.vix_collector = VIXDataCollector(cache_dir=cache_dir)
        
        self.economic_collector = EconomicIndicatorsCollector(
            api_key=self.api_keys.get('fred'),
            cache_dir=cache_dir
        )
        
        self.market_collector = MarketIndexCollector(cache_dir=cache_dir)
        
        self.news_analyzer = NewsSentimentAnalyzer(
            api_key=self.api_keys.get('news_api'),
            cache_dir=cache_dir
        )
        
        self.tv_indicators = TradingViewIndicators(
            api_key=self.api_keys.get('tradingview'),
            cache_dir=cache_dir
        )
    
    def collect_all_data(self):
        """
        Collect data from all sources and return a comprehensive market snapshot.
        
        Returns:
            dict: Dictionary containing data from all sources.
        """
        # Get VIX data
        vix_summary = self.vix_collector.get_vix_summary()
        
        # Get economic indicators
        economic_data = self.economic_collector.get_latest_values()
        
        # Get market index data
        market_summary = self.market_collector.get_market_summary()
        
        # Get news sentiment
        news_sentiment = self.news_analyzer.get_market_sentiment()
        
        # Get ICT signals for ES and NQ
        es_signals = self.tv_indicators.get_ict_signals('ES1!', '1h')
        nq_signals = self.tv_indicators.get_ict_signals('NQ1!', '1h')
        
        # Combine all data
        return {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'vix_data': vix_summary,
            'economic_data': economic_data,
            'market_data': market_summary,
            'news_sentiment': news_sentiment,
            'ict_signals': {
                'es': es_signals,
                'nq': nq_signals
            }
        }
    
    def format_data_for_manus(self, data=None):
        """
        Format collected data into a structured prompt for MANUS.
        
        Args:
            data (dict, optional): Data to format. If None, collects new data.
        
        Returns:
            str: Formatted prompt for MANUS.
        """
        if data is None:
            data = self.collect_all_data()
        
        # Extract key data points
        vix_data = data.get('vix_data', {})
        economic_data = data.get('economic_data', {})
        market_data = data.get('market_data', {})
        news_sentiment = data.get('news_sentiment', {})
        ict_signals = data.get('ict_signals', {})
        
        # Format VIX data
        vix_level = vix_data.get('current_vix', 'N/A')
        vix_change = vix_data.get('change_1h', 'N/A')
        vix_trend = vix_data.get('trend', 'N/A')
        
        # Format economic indicators
        pmi = economic_data.get('pmi', {}).get('value', 'N/A') if 'pmi' in economic_data else 'N/A'
        nfp = economic_data.get('nfp', {}).get('value', 'N/A') if 'nfp' in economic_data else 'N/A'
        fed_rate = economic_data.get('fed_funds_rate', {}).get('value', 'N/A') if 'fed_funds_rate' in economic_data else 'N/A'
        
        # Format market data
        prices = market_data.get('prices', {})
        sp500_price = prices.get('sp500', {}).get('price', 'N/A') if 'sp500' in prices else 'N/A'
        nasdaq_price = prices.get('nasdaq100', {}).get('price', 'N/A') if 'nasdaq100' in prices else 'N/A'
        es_price = prices.get('es', {}).get('price', 'N/A') if 'es' in prices else 'N/A'
        nq_price = prices.get('nq', {}).get('price', 'N/A') if 'nq' in prices else 'N/A'
        
        market_trend = market_data.get('market_trend', 'N/A')
        relative_performance = market_data.get('relative_performance', 'N/A')
        
        # Format news sentiment
        sentiment_score = news_sentiment.get('overall_sentiment', 'N/A')
        sentiment_label = news_sentiment.get('sentiment_label', 'N/A')
        
        # Format ICT signals
        es_bias = ict_signals.get('es', {}).get('bias', 'N/A')
        nq_bias = ict_signals.get('nq', {}).get('bias', 'N/A')
        es_signals_list = ict_signals.get('es', {}).get('signals', [])
        nq_signals_list = ict_signals.get('nq', {}).get('signals', [])
        
        # Build the prompt
        prompt = f"""
Current Data Snapshot:
- VIX Level: {vix_level}, Rate of Change: {vix_change}%, Trend: {vix_trend}
- Economic Indicators: PMI: {pmi}, NFP: {nfp}, Fed Funds Rate: {fed_rate}%
- S&P 500 Price: {sp500_price}, Nasdaq-100 Price: {nasdaq_price}
- ES Futures: {es_price}, NQ Futures: {nq_price}
- Market Trend: {market_trend}, Relative Performance: {relative_performance}
- News Sentiment: {sentiment_score} ({sentiment_label})
- ICT Analysis for ES: {es_bias} bias, Signals: {', '.join(es_signals_list) if es_signals_list else 'None'}
- ICT Analysis for NQ: {nq_bias} bias, Signals: {', '.join(nq_signals_list) if nq_signals_list else 'None'}

Based on this data, what is the predicted intraday market direction? 
If the market is likely to fall, recommend: "Market Falls: 1 ES short, 1 NQ long" 
If the market is likely to rise, recommend: "Market Rises: 1 ES long, 1 NQ short".

Also, consider MES–MNQ strategy alternatives with respective positions if that model is activated.
"""
        
        return prompt
    
    def save_data_snapshot(self, data=None, filename=None):
        """
        Save a snapshot of collected data to a file.
        
        Args:
            data (dict, optional): Data to save. If None, collects new data.
            filename (str, optional): Filename to save to. If None, generates a timestamped filename.
        
        Returns:
            str: Path to the saved file.
        """
        if data is None:
            data = self.collect_all_data()
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"market_snapshot_{timestamp}.json"
        
        if self.cache_dir:
            filepath = os.path.join(self.cache_dir, filename)
        else:
            filepath = filename
        
        # Convert data to JSON-compatible format
        import json
        
        # Custom JSON encoder to handle non-serializable objects
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (datetime.datetime, datetime.date)):
                    return obj.isoformat()
                elif pd.isna(obj):
                    return None
                return str(obj)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=CustomEncoder)
        
        return filepath

# Example usage
if __name__ == "__main__":
    # Create a data collection manager with caching
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    # API keys (replace with actual keys in production)
    api_keys = {
        'fred': os.environ.get('FRED_API_KEY'),
        'news_api': os.environ.get('NEWS_API_KEY'),
        'tradingview': os.environ.get('TRADINGVIEW_API_KEY')
    }
    
    manager = DataCollectionManager(cache_dir=cache_dir, api_keys=api_keys)
    
    # Collect all data
    data = manager.collect_all_data()
    print("Data collected from all sources")
    
    # Format data for MANUS
    prompt = manager.format_data_for_manus(data)
    print("\nMANUS Prompt:")
    print(prompt)
    
    # Save data snapshot
    filepath = manager.save_data_snapshot(data)
    print(f"\nData snapshot saved to: {filepath}")
