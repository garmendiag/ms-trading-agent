"""
Decision Logic Module

This module implements the decision logic for the MANUS trading agent.
It integrates signal generation with trade execution to create a complete
trading system with risk management.
"""

import os
import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Import required modules
from ..signal_generation import SignalGenerationManager
from .trade_execution import TradeExecutor

class DecisionLogicManager:
    """
    Class for managing the decision logic and trade execution process.
    """
    
    def __init__(self, data_manager=None, indicators_manager=None, 
                 signal_manager=None, trade_executor=None,
                 cache_dir=None, output_dir=None):
        """
        Initialize the decision logic manager.
        
        Args:
            data_manager: Data collection manager instance.
            indicators_manager: Technical indicators manager instance.
            signal_manager: Signal generation manager instance.
            trade_executor: Trade executor instance.
            cache_dir (str, optional): Directory to cache data. Defaults to None.
            output_dir (str, optional): Directory to save output files. Defaults to None.
        """
        self.data_manager = data_manager
        self.indicators_manager = indicators_manager
        self.signal_manager = signal_manager
        self.trade_executor = trade_executor
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize signal manager if not provided
        if self.signal_manager is None and data_manager is not None and indicators_manager is not None:
            from ..signal_generation import SignalGenerationManager
            self.signal_manager = SignalGenerationManager(
                data_manager=data_manager,
                indicators_manager=indicators_manager,
                cache_dir=cache_dir,
                output_dir=output_dir
            )
        
        # Initialize trade executor if not provided
        if self.trade_executor is None:
            from .trade_execution import TradeExecutor
            self.trade_executor = TradeExecutor(
                demo_mode=True,
                cache_dir=cache_dir
            )
    
    def run_trading_cycle(self, market_data=None, price_data=None):
        """
        Run a complete trading cycle: collect data, generate signals, execute trades.
        
        Args:
            market_data (dict, optional): Market data dictionary. If None, will collect data.
            price_data (pandas.DataFrame, optional): Price data. If None, will use ES data.
        
        Returns:
            dict: Dictionary containing the results of the trading cycle.
        """
        # Collect market data if not provided
        if market_data is None and self.data_manager:
            market_data = self.data_manager.collect_all_data()
        
        # Get price data if not provided
        if price_data is None and self.data_manager:
            from ..data_collection.market_index import MarketIndexCollector
            collector = MarketIndexCollector(cache_dir=self.cache_dir)
            price_data = collector.get_historical_data('es', period="10d", interval="1h")
        
        # Generate trading signal
        if self.signal_manager:
            signal_data = self.signal_manager.generate_trading_signal(market_data, price_data)
            execution_signal = signal_data['execution_signal']
        else:
            print("Warning: No signal manager available. Using default signal.")
            execution_signal = self._generate_default_signal()
        
        # Execute trades
        if self.trade_executor:
            execution_result = self.trade_executor.execute_signal(execution_signal)
        else:
            print("Warning: No trade executor available. Skipping trade execution.")
            execution_result = None
        
        # Generate reports
        reports = {}
        
        if self.signal_manager and signal_data:
            signal_report = self.signal_manager.generate_signal_report(signal_data)
            reports['signal_report'] = signal_report
        
        if self.trade_executor:
            performance_report = self.trade_executor.generate_performance_report()
            reports['performance_report'] = performance_report
        
        # Save trading cycle results
        cycle_result = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_data': market_data,
            'signal_data': signal_data if self.signal_manager else None,
            'execution_result': execution_result,
            'reports': reports
        }
        
        self._save_cycle_result(cycle_result)
        
        return cycle_result
    
    def _generate_default_signal(self):
        """
        Generate a default signal for testing.
        
        Returns:
            dict: Dictionary containing a default execution signal.
        """
        return {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_direction': 'neutral',
            'strategy_nq_es': {
                'signal': 'Hold: No clear signal',
                'positions': {
                    'ES': {
                        'direction': 'none',
                        'quantity': 0
                    },
                    'NQ': {
                        'direction': 'none',
                        'quantity': 0
                    }
                }
            },
            'strategy_mes_mnq': {
                'signal': 'Hold: No clear signal',
                'positions': {
                    'MES': {
                        'direction': 'none',
                        'quantity': 0
                    },
                    'MNQ': {
                        'direction': 'none',
                        'quantity': 0
                    }
                }
            }
        }
    
    def _save_cycle_result(self, cycle_result):
        """
        Save trading cycle result to file.
        
        Args:
            cycle_result (dict): Dictionary containing cycle result.
        
        Returns:
            str: Path to the saved file.
        """
        if not self.cache_dir:
            return None
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"trading_cycle_{timestamp}.json"
        filepath = os.path.join(self.cache_dir, filename)
        
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
        
        # Create a simplified version of the cycle result for saving
        simplified_result = {
            'timestamp': cycle_result['timestamp'],
            'signal': cycle_result.get('signal_data', {}).get('parsed_signal', {}),
            'execution': cycle_result.get('execution_result', {}),
            'reports': cycle_result.get('reports', {})
        }
        
        with open(filepath, 'w') as f:
            json.dump(simplified_result, f, indent=2, cls=CustomEncoder)
        
        return filepath
    
    def get_current_positions(self):
        """
        Get current positions.
        
        Returns:
            dict: Dictionary containing current positions.
        """
        if self.trade_executor:
            return self.trade_executor.get_position_summary()
        else:
            return {
                'positions': {
                    'ES': {'direction': 'none', 'quantity': 0, 'entry_price': 0},
                    'NQ': {'direction': 'none', 'quantity': 0, 'entry_price': 0},
                    'MES': {'direction': 'none', 'quantity': 0, 'entry_price': 0},
                    'MNQ': {'direction': 'none', 'quantity': 0, 'entry_price': 0}
                },
                'nq_es_pl': 0,
                'mes_mnq_pl': 0
            }
    
    def get_performance_summary(self):
        """
        Get performance summary.
        
        Returns:
            dict: Dictionary containing performance summary.
        """
        if self.trade_executor:
            return self.trade_executor.get_performance_summary()
        else:
            return {
                'NQ-ES': {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'profit_loss': 0,
                    'max_drawdown': 0
                },
                'MES-MNQ': {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'profit_loss': 0,
                    'max_drawdown': 0
                },
                'total_profit_loss': 0
            }
    
    def generate_system_status_report(self, output_file=None):
        """
        Generate a system status report.
        
        Args:
            output_file (str, optional): File to save the report to. If None, generates a timestamped filename.
        
        Returns:
            str: Path to the saved report file.
        """
        if output_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"system_status_{timestamp}.md"
        
        if self.output_dir:
            filepath = os.path.join(self.output_dir, output_file)
        else:
            filepath = output_file
        
        # Get current positions
        positions = self.get_current_positions()
        
        # Get performance summary
        performance = self.get_performance_summary()
        
        # Generate report content
        report = f"""# MANUS Trading System Status Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Components Status

| Component | Status |
|-----------|--------|
| Data Collection | {'Active' if self.data_manager else 'Not Available'} |
| Technical Indicators | {'Active' if self.indicators_manager else 'Not Available'} |
| Signal Generation | {'Active' if self.signal_manager else 'Not Available'} |
| Trade Execution | {'Active' if self.trade_executor else 'Not Available'} |

## Current Positions

### NQ-ES Strategy
- ES: {positions['positions']['ES']['direction']} {positions['positions']['ES']['quantity']} @ {positions['positions']['ES']['entry_price']}
- NQ: {positions['positions']['NQ']['direction']} {positions['positions']['NQ']['quantity']} @ {positions['positions']['NQ']['entry_price']}
- Current P/L: ${positions['nq_es_pl']:.2f}

### MES-MNQ Strategy
- MES: {positions['positions']['MES']['direction']} {positions['positions']['MES']['quantity']} @ {positions['positions']['MES']['entry_price']}
- MNQ: {positions['positions']['MNQ']['direction']} {positions['positions']['MNQ']['quantity']} @ {positions['positions']['MNQ']['entry_price']}
- Current P/L: ${positions['mes_mnq_pl']:.2f}

## Performance Summary

### NQ-ES Strategy
- Total Trades: {performance['NQ-ES']['trades']}
- Wins: {performance['NQ-ES']['wins']}
- Losses: {performance['NQ-ES']['losses']}
- Win Rate: {performance['NQ-ES']['win_rate']:.2f}%
- Total P/L: ${performance['NQ-ES']['profit_loss']:.2f}
- Max Drawdown: ${performance['NQ-ES']['max_drawdown']:.2f}

### MES-MNQ Strategy
- Total Trades: {performance['MES-MNQ']['trades']}
- Wins: {performance['MES-MNQ']['wins']}
- Losses: {performance['MES-MNQ']['losses']}
- Win Rate: {performance['MES-MNQ']['win_rate']:.2f}%
- Total P/L: ${performance['MES-MNQ']['profit_loss']:.2f}
- Max Drawdown: ${performance['MES-MNQ']['max_drawdown']:.2f}

## Overall Performance
- Total P/L: ${performance['total_profit_loss']:.2f}

## Risk Management Status
- NQ-ES Max Loss Limit: $5,000
- MES-MNQ Max Loss Limit: $3,000
- NQ-ES Current Loss: ${max(0, -performance['NQ-ES']['profit_loss']):.2f}
- MES-MNQ Current Loss: ${max(0, -performance['MES-MNQ']['profit_loss']):.2f}
- Risk Status: {'Warning' if max(0, -performance['NQ-ES']['profit_loss']) > 3000 or max(0, -performance['MES-MNQ']['profit_loss']) > 2000 else 'Normal'}
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
    from signal_generation import SignalGenerationManager
    
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
    
    # Create trade executor
    trade_executor = TradeExecutor(
        demo_mode=True,
        cache_dir=cache_dir
    )
    
    # Create decision logic manager
    decision_manager = DecisionLogicManager(
        data_manager=data_manager,
        indicators_manager=indicators_manager,
        signal_manager=signal_manager,
        trade_executor=trade_executor,
        cache_dir=cache_dir,
        output_dir=output_dir
    )
    
    # Run trading cycle
    cycle_result = decision_manager.run_trading_cycle()
    
    # Print results
    print("Trading Cycle Results:")
    print(f"  Timestamp: {cycle_result['timestamp']}")
    
    if cycle_result.get('signal_data'):
        print(f"  Signal: {cycle_result['signal_data']['parsed_signal']['signal']}")
    
    if cycle_result.get('execution_result'):
        print(f"  NQ-ES Signal: {cycle_result['execution_result']['nq_es_signal']}")
        print(f"  MES-MNQ Signal: {cycle_result['execution_result']['mes_mnq_signal']}")
    
    # Print reports
    if cycle_result.get('reports'):
        print("\nGenerated Reports:")
        for report_type, report_path in cycle_result['reports'].items():
            print(f"  {report_type}: {report_path}")
    
    # Generate system status report
    status_report = decision_manager.generate_system_status_report()
    print(f"\nSystem status report saved to: {status_report}")
