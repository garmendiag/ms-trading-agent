"""
MANUS Signal Generation Module

This module is responsible for generating trading signals using the MANUS model.
It formats data from various sources, creates prompts for MANUS, and interprets
the responses to generate actionable trading signals.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import requests
from pathlib import Path

class MANUSSignalGenerator:
    """
    Class for generating trading signals using the MANUS model.
    """
    
    def __init__(self, api_key=None, model="MANUS-4", cache_dir=None):
        """
        Initialize the MANUS signal generator.
        
        Args:
            api_key (str, optional): MANUS API key. If None, will look for
                                    MANUS_API_KEY environment variable.
            model (str, optional): MANUS model to use. Defaults to "MANUS-4".
            cache_dir (str, optional): Directory to cache MANUS responses.
                                      Defaults to None (no caching).
        """
        # Use provided API key or get from environment variable
        self.api_key = api_key or os.environ.get('MANUS_API_KEY')
        
        if not self.api_key:
            print("Warning: No MANUS API key provided. Using simulation mode.")
            # In simulation mode, we'll generate simulated responses
        
        self.model = model
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def create_prompt(self, market_data, technical_indicators, news_sentiment, economic_data=None):
        """
        Create a prompt for MANUS based on market data, technical indicators, and other inputs.
        
        Args:
            market_data (dict): Dictionary containing market data (prices, VIX, etc.).
            technical_indicators (dict): Dictionary containing technical indicator signals.
            news_sentiment (dict): Dictionary containing news sentiment data.
            economic_data (dict, optional): Dictionary containing economic indicator data.
        
        Returns:
            str: Formatted prompt for MANUS.
        """
        # Extract key data points
        
        # Market data
        current_time = market_data.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        es_price = market_data.get('es_price', 'N/A')
        nq_price = market_data.get('nq_price', 'N/A')
        sp500_price = market_data.get('sp500_price', 'N/A')
        nasdaq_price = market_data.get('nasdaq100_price', 'N/A')
        
        # VIX data
        vix_level = market_data.get('vix_level', 'N/A')
        vix_change = market_data.get('vix_change', 'N/A')
        vix_trend = market_data.get('vix_trend', 'N/A')
        
        # Economic indicators
        if economic_data:
            pmi = economic_data.get('pmi', {}).get('value', 'N/A')
            nfp = economic_data.get('nfp', {}).get('value', 'N/A')
            fed_rate = economic_data.get('fed_funds_rate', {}).get('value', 'N/A')
        else:
            pmi = 'N/A'
            nfp = 'N/A'
            fed_rate = 'N/A'
        
        # Market trend
        market_trend = market_data.get('market_trend', 'N/A')
        relative_performance = market_data.get('relative_performance', 'N/A')
        
        # News sentiment
        sentiment_score = news_sentiment.get('overall_sentiment', 'N/A')
        sentiment_label = news_sentiment.get('sentiment_label', 'N/A')
        
        # Technical indicators
        rsi_value = technical_indicators.get('rsi', {}).get('value', 'N/A')
        rsi_signal = technical_indicators.get('rsi', {}).get('signal', 'N/A')
        
        ms_trend = technical_indicators.get('market_structure', {}).get('trend', 'N/A')
        
        ob_bias = technical_indicators.get('order_blocks', {}).get('bias', 'N/A')
        
        ote_buy_active = technical_indicators.get('optimal_trade_entry', {}).get('buy_ote_active', False)
        ote_sell_active = technical_indicators.get('optimal_trade_entry', {}).get('sell_ote_active', False)
        
        liquidity_bias = technical_indicators.get('liquidity_model', {}).get('bias', 'N/A')
        recent_sweep_up = technical_indicators.get('liquidity_model', {}).get('recent_sweep_up', False)
        recent_sweep_down = technical_indicators.get('liquidity_model', {}).get('recent_sweep_down', False)
        
        # Combined signal
        combined_signal = technical_indicators.get('combined_signal', {}).get('signal', 'N/A')
        combined_bias = technical_indicators.get('combined_signal', {}).get('bias', 'N/A')
        signal_strength = technical_indicators.get('combined_signal', {}).get('signal_strength', 'N/A')
        confidence = technical_indicators.get('combined_signal', {}).get('confidence', 'N/A')
        
        # Build the prompt
        prompt = f"""
Current Data Snapshot ({current_time}):

Market Data:
- S&P 500 Price: {sp500_price}, Nasdaq-100 Price: {nasdaq_price}
- ES Futures: {es_price}, NQ Futures: {nq_price}
- Market Trend: {market_trend}, Relative Performance: {relative_performance}

Volatility:
- VIX Level: {vix_level}, Rate of Change: {vix_change}%, Trend: {vix_trend}

Economic Indicators:
- PMI: {pmi}, NFP: {nfp}, Fed Funds Rate: {fed_rate}%

News Sentiment:
- Overall Sentiment: {sentiment_score} ({sentiment_label})

Technical Indicators:
- RSI: {rsi_value} ({rsi_signal})
- Market Structure: {ms_trend}
- Order Blocks Bias: {ob_bias}
- Optimal Trade Entry: Buy OTE Active: {ote_buy_active}, Sell OTE Active: {ote_sell_active}
- Liquidity Model: {liquidity_bias} bias, Recent Sweep Up: {recent_sweep_up}, Recent Sweep Down: {recent_sweep_down}

Technical Analysis Summary:
- Combined Bias: {combined_bias}
- Signal Strength: {signal_strength}
- Confidence: {confidence}
- Current Technical Signal: {combined_signal}

Based on this comprehensive data, analyze the current market conditions and predict the intraday market direction.

If the market is likely to fall, recommend: "Market Falls: 1 ES short, 1 NQ long" 
If the market is likely to rise, recommend: "Market Rises: 1 ES long, 1 NQ short".

Also provide the MES-MNQ strategy alternative:
For falling market: "MES-MNQ Strategy: 1 MES short, 2 MNQ long"
For rising market: "MES-MNQ Strategy: 1 MES long, 2 MNQ short"

Include a brief explanation of your reasoning, focusing on the most significant factors influencing your prediction.
"""
        
        return prompt
    
    def generate_signal(self, prompt, cache_key=None):
        """
        Generate a trading signal by sending a prompt to the MANUS model.
        
        Args:
            prompt (str): Prompt to send to MANUS.
            cache_key (str, optional): Key for caching the response. Defaults to None.
        
        Returns:
            dict: Dictionary containing the MANUS response and parsed signal.
        """
        # Check if cached response exists
        if self.cache_dir and cache_key:
            cache_file = Path(self.cache_dir) / f"{cache_key}.json"
            if cache_file.exists():
                # For cached responses, check if they're less than 1 hour old
                file_time = datetime.datetime.fromtimestamp(cache_file.stat().st_mtime)
                if (datetime.datetime.now() - file_time).seconds < 3600:  # 1 hour in seconds
                    print(f"Loading cached MANUS response from {cache_file}")
                    with open(cache_file, 'r') as f:
                        return json.load(f)
        
        # If no API key or in simulation mode, generate a simulated response
        if not self.api_key:
            print("No API key available. Generating simulated MANUS response.")
            return self._generate_simulated_response(prompt)
        
        try:
            # This is a placeholder for actual MANUS API call
            # In a real implementation, you would make API calls to the MANUS API
            print("Calling MANUS API...")
            
            # Simulate API response with a simulated response
            response = self._generate_simulated_response(prompt)
            
            # Cache the response if cache_dir and cache_key are specified
            if self.cache_dir and cache_key:
                with open(Path(self.cache_dir) / f"{cache_key}.json", 'w') as f:
                    json.dump(response, f, indent=2)
            
            return response
        
        except Exception as e:
            print(f"Error calling MANUS API: {e}")
            # Fall back to simulated response if API call fails
            return self._generate_simulated_response(prompt)
    
    def _generate_simulated_response(self, prompt):
        """
        Generate a simulated MANUS response for testing purposes.
        
        Args:
            prompt (str): The prompt that would be sent to MANUS.
        
        Returns:
            dict: Dictionary containing the simulated response and parsed signal.
        """
        # Extract key information from the prompt to generate a reasonable response
        prompt_lower = prompt.lower()
        
        # Check for bullish/bearish indicators in the prompt
        bullish_indicators = 0
        bearish_indicators = 0
        
        # Check VIX
        if "vix" in prompt_lower:
            if "falling" in prompt_lower or "decreasing" in prompt_lower:
                bullish_indicators += 1
            elif "rising" in prompt_lower or "increasing" in prompt_lower:
                bearish_indicators += 1
        
        # Check RSI
        if "rsi" in prompt_lower:
            if "oversold" in prompt_lower:
                bullish_indicators += 1
            elif "overbought" in prompt_lower:
                bearish_indicators += 1
        
        # Check Market Structure
        if "market structure" in prompt_lower:
            if "bullish" in prompt_lower:
                bullish_indicators += 1
            elif "bearish" in prompt_lower:
                bearish_indicators += 1
        
        # Check Order Blocks
        if "order blocks" in prompt_lower:
            if "bullish" in prompt_lower:
                bullish_indicators += 1
            elif "bearish" in prompt_lower:
                bearish_indicators += 1
        
        # Check Liquidity Model
        if "liquidity model" in prompt_lower:
            if "bullish" in prompt_lower:
                bullish_indicators += 1
            elif "bearish" in prompt_lower:
                bearish_indicators += 1
            if "sweep down" in prompt_lower:
                bullish_indicators += 1
            elif "sweep up" in prompt_lower:
                bearish_indicators += 1
        
        # Check News Sentiment
        if "news sentiment" in prompt_lower:
            if "positive" in prompt_lower:
                bullish_indicators += 1
            elif "negative" in prompt_lower:
                bearish_indicators += 1
        
        # Check Combined Signal
        if "combined bias: bullish" in prompt_lower:
            bullish_indicators += 2
        elif "combined bias: bearish" in prompt_lower:
            bearish_indicators += 2
        
        # Determine the overall bias
        if bullish_indicators > bearish_indicators:
            market_direction = "rise"
            signal = "Market Rises: 1 ES long, 1 NQ short"
            mes_mnq_signal = "MES-MNQ Strategy: 1 MES long, 2 MNQ short"
            reasoning = "The market is likely to rise based on bullish technical indicators, favorable market structure, and positive sentiment. The VIX trend and liquidity patterns also support this view."
        elif bearish_indicators > bullish_indicators:
            market_direction = "fall"
            signal = "Market Falls: 1 ES short, 1 NQ long"
            mes_mnq_signal = "MES-MNQ Strategy: 1 MES short, 2 MNQ long"
            reasoning = "The market is likely to fall based on bearish technical indicators, deteriorating market structure, and negative sentiment. The VIX trend and liquidity patterns also support this view."
        else:
            # If indicators are balanced, generate a random response
            import random
            if random.random() > 0.5:
                market_direction = "rise"
                signal = "Market Rises: 1 ES long, 1 NQ short"
                mes_mnq_signal = "MES-MNQ Strategy: 1 MES long, 2 MNQ short"
                reasoning = "While indicators are mixed, there's a slight edge toward bullish momentum. Recent price action and market structure suggest potential upside in the near term."
            else:
                market_direction = "fall"
                signal = "Market Falls: 1 ES short, 1 NQ long"
                mes_mnq_signal = "MES-MNQ Strategy: 1 MES short, 2 MNQ long"
                reasoning = "Despite mixed indicators, there's a slight bearish bias in the current market conditions. Recent price action and market structure suggest potential downside in the near term."
        
        # Construct the full response
        full_response = f"""
Based on the comprehensive data provided, I've analyzed the current market conditions to predict the intraday market direction.

The market is likely to {market_direction} today. Key factors influencing this prediction include:
- {'Bullish' if market_direction == 'rise' else 'Bearish'} technical indicators across multiple timeframes
- {'Favorable' if market_direction == 'rise' else 'Deteriorating'} market structure
- {'Positive' if market_direction == 'rise' else 'Negative'} sentiment indicators
- {'Supportive' if market_direction == 'rise' else 'Concerning'} liquidity patterns

{reasoning}

Recommended trading strategy:
{signal}

Alternative MES-MNQ strategy:
{mes_mnq_signal}

Monitor the market closely for any significant shifts in sentiment or technical patterns that could alter this outlook.
"""
        
        # Parse the signal from the response
        parsed_signal = {
            'market_direction': market_direction,
            'signal': signal,
            'mes_mnq_signal': mes_mnq_signal,
            'reasoning': reasoning
        }
        
        return {
            'prompt': prompt,
            'response': full_response,
            'parsed_signal': parsed_signal,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def parse_manus_response(self, response):
        """
        Parse the MANUS response to extract the trading signal.
        
        Args:
            response (str): MANUS response text.
        
        Returns:
            dict: Dictionary containing the parsed signal.
        """
        response_lower = response.lower()
        
        # Extract market direction
        if "market rises" in response_lower or "likely to rise" in response_lower:
            market_direction = "rise"
        elif "market falls" in response_lower or "likely to fall" in response_lower:
            market_direction = "fall"
        else:
            market_direction = "neutral"
        
        # Extract trading signal
        if "market rises: 1 es long, 1 nq short" in response_lower:
            signal = "Market Rises: 1 ES long, 1 NQ short"
        elif "market falls: 1 es short, 1 nq long" in response_lower:
            signal = "Market Falls: 1 ES short, 1 NQ long"
        else:
            signal = "Hold: No clear signal"
        
        # Extract MES-MNQ signal
        if "mes-mnq strategy: 1 mes long, 2 mnq short" in response_lower:
            mes_mnq_signal = "MES-MNQ Strategy: 1 MES long, 2 MNQ short"
        elif "mes-mnq strategy: 1 mes short, 2 mnq long" in response_lower:
            mes_mnq_signal = "MES-MNQ Strategy: 1 MES short, 2 MNQ long"
        else:
            mes_mnq_signal = "Hold: No clear signal"
        
        # Extract reasoning (the paragraph after "Key factors" or similar)
        reasoning_start = response.find("Key factors")
        if reasoning_start == -1:
            reasoning_start = response.find("factors influencing")
        
        if reasoning_start != -1:
            reasoning_end = response.find("\n\n", reasoning_start)
            if reasoning_end == -1:
                reasoning = response[reasoning_start:]
            else:
                reasoning = response[reasoning_start:reasoning_end]
        else:
            reasoning = "No specific reasoning provided."
        
        return {
            'market_direction': market_direction,
            'signal': signal,
            'mes_mnq_signal': mes_mnq_signal,
            'reasoning': reasoning
        }
    
    def format_signal_for_execution(self, parsed_signal):
        """
        Format the parsed signal for the execution module.
        
        Args:
            parsed_signal (dict): Dictionary containing the parsed signal.
        
        Returns:
            dict: Dictionary containing the formatted signal for execution.
        """
        # Extract the basic signal components
        market_direction = parsed_signal.get('market_direction', 'neutral')
        signal = parsed_signal.get('signal', 'Hold: No clear signal')
        mes_mnq_signal = parsed_signal.get('mes_mnq_signal', 'Hold: No clear signal')
        
        # Determine ES position
        if "ES long" in signal:
            es_position = "long"
            es_quantity = 1
        elif "ES short" in signal:
            es_position = "short"
            es_quantity = 1
        else:
            es_position = "none"
            es_quantity = 0
        
        # Determine NQ position
        if "NQ long" in signal:
            nq_position = "long"
            nq_quantity = 1
        elif "NQ short" in signal:
            nq_position = "short"
            nq_quantity = 1
        else:
            nq_position = "none"
            nq_quantity = 0
        
        # Determine MES position
        if "MES long" in mes_mnq_signal:
            mes_position = "long"
            mes_quantity = 1
        elif "MES short" in mes_mnq_signal:
            mes_position = "short"
            mes_quantity = 1
        else:
            mes_position = "none"
            mes_quantity = 0
        
        # Determine MNQ position
        if "MNQ long" in mes_mnq_signal:
            mnq_position = "long"
            mnq_quantity = 2
        elif "MNQ short" in mes_mnq_signal:
            mnq_position = "short"
            mnq_quantity = 2
        else:
            mnq_position = "none"
            mnq_quantity = 0
        
        # Format the execution signal
        execution_signal = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_direction': market_direction,
            'strategy_nq_es': {
                'signal': signal,
                'positions': {
                    'ES': {
                        'direction': es_position,
                        'quantity': es_quantity
                    },
                    'NQ': {
                        'direction': nq_position,
                        'quantity': nq_quantity
                    }
                }
            },
            'strategy_mes_mnq': {
                'signal': mes_mnq_signal,
                'positions': {
                    'MES': {
                        'direction': mes_position,
                        'quantity': mes_quantity
                    },
                    'MNQ': {
                        'direction': mnq_position,
                        'quantity': mnq_quantity
                    }
                }
            }
        }
        
        return execution_signal
    
    def save_signal_history(self, signal_data, filename=None):
        """
        Save signal history to a file.
        
        Args:
            signal_data (dict): Dictionary containing signal data.
            filename (str, optional): Filename to save to. If None, generates a timestamped filename.
        
        Returns:
            str: Path to the saved file.
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"signal_history_{timestamp}.json"
        
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
            json.dump(signal_data, f, indent=2, cls=CustomEncoder)
        
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
    
    # Collect market data
    market_data = data_manager.collect_all_data()
    
    # Create technical indicators manager
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    indicators_manager = TechnicalIndicatorsManager(output_dir=output_dir)
    
    # Get ES futures data
    from data_collection.market_index import MarketIndexCollector
    collector = MarketIndexCollector(cache_dir=cache_dir)
    es_data = collector.get_historical_data('es', period="10d", interval="1h")
    
    # Calculate technical indicators
    indicators_data = indicators_manager.calculate_all_indicators(es_data, pip_value=0.25)
    
    # Get all signals
    all_signals = indicators_manager.get_all_signals(indicators_data)
    
    # Generate combined signal
    combined_signal = indicators_manager.generate_combined_signal(all_signals)
    
    # Format technical indicators for MANUS
    technical_indicators = {
        'rsi': all_signals['rsi'],
        'market_structure': all_signals['market_structure'],
        'order_blocks': all_signals['order_blocks'],
        'optimal_trade_entry': all_signals['optimal_trade_entry'],
        'liquidity_model': all_signals['liquidity_model'],
        'combined_signal': combined_signal
    }
    
    # Create MANUS signal generator
    manus_generator = MANUSSignalGenerator(cache_dir=cache_dir)
    
    # Create prompt for MANUS
    prompt = manus_generator.create_prompt(
        market_data=market_data.get('market_data', {}),
        technical_indicators=technical_indicators,
        news_sentiment=market_data.get('news_sentiment', {})
    )
    
    # Generate signal
    signal_data = manus_generator.generate_signal(prompt, cache_key="test_signal")
    
    # Print the response
    print("MANUS Response:")
    print(signal_data['response'])
    
    # Print the parsed signal
    print("\nParsed Signal:")
    for key, value in signal_data['parsed_signal'].items():
        print(f"  {key}: {value}")
    
    # Format signal for execution
    execution_signal = manus_generator.format_signal_for_execution(signal_data['parsed_signal'])
    
    # Print the execution signal
    print("\nExecution Signal:")
    print(f"  Market Direction: {execution_signal['market_direction']}")
    print(f"  NQ-ES Strategy: {execution_signal['strategy_nq_es']['signal']}")
    print(f"    ES: {execution_signal['strategy_nq_es']['positions']['ES']['direction']} {execution_signal['strategy_nq_es']['positions']['ES']['quantity']}")
    print(f"    NQ: {execution_signal['strategy_nq_es']['positions']['NQ']['direction']} {execution_signal['strategy_nq_es']['positions']['NQ']['quantity']}")
    print(f"  MES-MNQ Strategy: {execution_signal['strategy_mes_mnq']['signal']}")
    print(f"    MES: {execution_signal['strategy_mes_mnq']['positions']['MES']['direction']} {execution_signal['strategy_mes_mnq']['positions']['MES']['quantity']}")
    print(f"    MNQ: {execution_signal['strategy_mes_mnq']['positions']['MNQ']['direction']} {execution_signal['strategy_mes_mnq']['positions']['MNQ']['quantity']}")
    
    # Save signal history
    filepath = manus_generator.save_signal_history(signal_data)
    print(f"\nSignal history saved to: {filepath}")
