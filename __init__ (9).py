"""
Signal Generation Module Integration

This module integrates the MANUS signal generation with data collection
and technical indicators to provide a complete signal generation pipeline.
"""

import os
import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Import required modules
from .manus_signal import MANUSSignalGenerator

class SignalGenerationManager:
    """
    Class for managing the signal generation process.
    """
    
    def __init__(self, data_manager=None, indicators_manager=None, cache_dir=None, output_dir=None):
        """
        Initialize the signal generation manager.
        
        Args:
            data_manager: Data collection manager instance.
            indicators_manager: Technical indicators manager instance.
            cache_dir (str, optional): Directory to cache data. Defaults to None.
            output_dir (str, optional): Directory to save output files. Defaults to None.
        """
        self.data_manager = data_manager
        self.indicators_manager = indicators_manager
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MANUS signal generator
        self.manus_generator = MANUSSignalGenerator(cache_dir=cache_dir)
    
    def generate_trading_signal(self, market_data=None, price_data=None, pip_value=0.25):
        """
        Generate a trading signal using the complete pipeline.
        
        Args:
            market_data (dict, optional): Market data dictionary. If None, will collect data.
            price_data (pandas.DataFrame, optional): Price data. If None, will use ES data.
            pip_value (float, optional): Value of 1 pip in price units. Defaults to 0.25.
        
        Returns:
            dict: Dictionary containing the generated signal and execution data.
        """
        # Collect market data if not provided
        if market_data is None and self.data_manager:
            market_data = self.data_manager.collect_all_data()
        
        # Get price data if not provided
        if price_data is None and self.data_manager:
            from data_collection.market_index import MarketIndexCollector
            collector = MarketIndexCollector(cache_dir=self.cache_dir)
            price_data = collector.get_historical_data('es', period="10d", interval="1h")
        
        # Calculate technical indicators
        if self.indicators_manager and price_data is not None:
            indicators_data = self.indicators_manager.calculate_all_indicators(price_data, pip_value)
            all_signals = self.indicators_manager.get_all_signals(indicators_data)
            combined_signal = self.indicators_manager.generate_combined_signal(all_signals)
            
            # Format technical indicators for MANUS
            technical_indicators = {
                'rsi': all_signals['rsi'],
                'market_structure': all_signals['market_structure'],
                'order_blocks': all_signals['order_blocks'],
                'optimal_trade_entry': all_signals['optimal_trade_entry'],
                'liquidity_model': all_signals['liquidity_model'],
                'combined_signal': combined_signal
            }
        else:
            # If no indicators manager or price data, use empty dict
            technical_indicators = {}
        
        # Create prompt for MANUS
        prompt = self.manus_generator.create_prompt(
            market_data=market_data.get('market_data', {}) if market_data else {},
            technical_indicators=technical_indicators,
            news_sentiment=market_data.get('news_sentiment', {}) if market_data else {}
        )
        
        # Generate timestamp for cache key
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        cache_key = f"signal_{timestamp}"
        
        # Generate signal
        signal_data = self.manus_generator.generate_signal(prompt, cache_key=cache_key)
        
        # Format signal for execution
        execution_signal = self.manus_generator.format_signal_for_execution(signal_data['parsed_signal'])
        
        # Save signal history
        filepath = self.manus_generator.save_signal_history(signal_data)
        
        # Return the complete signal data
        return {
            'prompt': prompt,
            'manus_response': signal_data['response'],
            'parsed_signal': signal_data['parsed_signal'],
            'execution_signal': execution_signal,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'history_file': filepath
        }
    
    def generate_signal_report(self, signal_data, output_file=None):
        """
        Generate a human-readable report from the signal data.
        
        Args:
            signal_data (dict): Dictionary containing signal data.
            output_file (str, optional): File to save the report to. If None, generates a timestamped filename.
        
        Returns:
            str: Path to the saved report file.
        """
        if output_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"signal_report_{timestamp}.md"
        
        if self.output_dir:
            filepath = os.path.join(self.output_dir, output_file)
        else:
            filepath = output_file
        
        # Extract data
        timestamp = signal_data.get('timestamp', 'Unknown')
        parsed_signal = signal_data.get('parsed_signal', {})
        execution_signal = signal_data.get('execution_signal', {})
        
        # Generate report content
        report = f"""# Trading Signal Report
Generated: {timestamp}

## Market Direction
**Prediction:** {parsed_signal.get('market_direction', 'Unknown').capitalize()}

## Trading Signals
**NQ-ES Strategy:** {execution_signal.get('strategy_nq_es', {}).get('signal', 'Unknown')}
- ES: {execution_signal.get('strategy_nq_es', {}).get('positions', {}).get('ES', {}).get('direction', 'none')} {execution_signal.get('strategy_nq_es', {}).get('positions', {}).get('ES', {}).get('quantity', 0)}
- NQ: {execution_signal.get('strategy_nq_es', {}).get('positions', {}).get('NQ', {}).get('direction', 'none')} {execution_signal.get('strategy_nq_es', {}).get('positions', {}).get('NQ', {}).get('quantity', 0)}

**MES-MNQ Strategy:** {execution_signal.get('strategy_mes_mnq', {}).get('signal', 'Unknown')}
- MES: {execution_signal.get('strategy_mes_mnq', {}).get('positions', {}).get('MES', {}).get('direction', 'none')} {execution_signal.get('strategy_mes_mnq', {}).get('positions', {}).get('MES', {}).get('quantity', 0)}
- MNQ: {execution_signal.get('strategy_mes_mnq', {}).get('positions', {}).get('MNQ', {}).get('direction', 'none')} {execution_signal.get('strategy_mes_mnq', {}).get('positions', {}).get('MNQ', {}).get('quantity', 0)}

## Analysis
{parsed_signal.get('reasoning', 'No analysis provided.')}

## MANUS Response
```
{signal_data.get('manus_response', 'No response available.')}
```

## Prompt Used
```
{signal_data.get('prompt', 'No prompt available.')}
```
"""
        
        # Write report to file
        with open(filepath, 'w') as f:
            f.write(report)
        
        return filepath

# Example usage
if __name__ == "__main__":
    # Import necessary modules for testing
    import sys
    sys.path.append('..')
    from data_collection import DataCollectionManager
    from technical_indicators import TechnicalIndicatorsManager
    
    # Create data collection manager
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    data_manager = DataCollectionManager(cache_dir=cache_dir)
    
    # Create technical indicators manager
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    indicators_manager = TechnicalIndicatorsManager(output_dir=output_dir)
    
    # Create signal generation manager
    signal_manager = SignalGenerationManager(
        data_manager=data_manager,
        indicators_manager=indicators_manager,
        cache_dir=cache_dir,
        output_dir=output_dir
    )
    
    # Generate trading signal
    signal_data = signal_manager.generate_trading_signal()
    
    # Print the execution signal
    print("Trading Signal:")
    print(f"  Market Direction: {signal_data['parsed_signal']['market_direction']}")
    print(f"  NQ-ES Strategy: {signal_data['execution_signal']['strategy_nq_es']['signal']}")
    print(f"  MES-MNQ Strategy: {signal_data['execution_signal']['strategy_mes_mnq']['signal']}")
    
    # Generate signal report
    report_file = signal_manager.generate_signal_report(signal_data)
    print(f"\nSignal report saved to: {report_file}")
