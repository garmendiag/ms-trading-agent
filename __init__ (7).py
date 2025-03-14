"""
Risk Management Module Integration

This module integrates the risk management functionality with the decision logic
and trade execution to create a complete trading system with advanced risk controls.
"""

import os
import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Import required modules
from .risk_manager import RiskManager

class RiskManagementIntegrator:
    """
    Class for integrating risk management with the trading system.
    """
    
    def __init__(self, decision_manager=None, trade_executor=None, 
                 max_loss_nq_es=5000, max_loss_mes_mnq=3000, 
                 daily_loss_limit=2000, trailing_stop_pct=0.5,
                 cache_dir=None, output_dir=None):
        """
        Initialize the risk management integrator.
        
        Args:
            decision_manager: Decision logic manager instance.
            trade_executor: Trade executor instance.
            max_loss_nq_es (float, optional): Maximum loss allowed for NQ-ES strategy in USD. Defaults to 5000.
            max_loss_mes_mnq (float, optional): Maximum loss allowed for MES-MNQ strategy in USD. Defaults to 3000.
            daily_loss_limit (float, optional): Maximum daily loss allowed in USD. Defaults to 2000.
            trailing_stop_pct (float, optional): Trailing stop percentage. Defaults to 0.5.
            cache_dir (str, optional): Directory to cache data. Defaults to None.
            output_dir (str, optional): Directory to save output files. Defaults to None.
        """
        self.decision_manager = decision_manager
        self.trade_executor = trade_executor
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            max_loss_nq_es=max_loss_nq_es,
            max_loss_mes_mnq=max_loss_mes_mnq,
            daily_loss_limit=daily_loss_limit,
            trailing_stop_pct=trailing_stop_pct,
            cache_dir=cache_dir,
            output_dir=output_dir
        )
        
        # Initialize alert system
        self.alerts = []
    
    def run_risk_managed_trading_cycle(self, market_data=None, price_data=None):
        """
        Run a complete trading cycle with risk management.
        
        Args:
            market_data (dict, optional): Market data dictionary. If None, will collect data.
            price_data (pandas.DataFrame, optional): Price data. If None, will use ES data.
        
        Returns:
            dict: Dictionary containing the results of the trading cycle.
        """
        # Clear alerts for this cycle
        self.alerts = []
        
        # Get current performance and positions
        if self.trade_executor:
            performance_data = self.trade_executor.get_performance_summary()
            positions_data = self.trade_executor.get_position_summary()
        else:
            # Default empty data if no trade executor
            performance_data = {
                'NQ-ES': {'profit_loss': 0},
                'MES-MNQ': {'profit_loss': 0}
            }
            positions_data = {
                'positions': {},
                'nq_es_pl': 0,
                'mes_mnq_pl': 0
            }
        
        # Update risk state
        risk_state = self.risk_manager.update_risk_state(performance_data, positions_data)
        
        # Check for stop-loss triggers
        if self.trade_executor:
            # Get current market prices (in a real system, would fetch from market data provider)
            market_prices = self._get_market_prices()
            
            # Apply stop-loss rules
            positions_to_close = self.risk_manager.apply_stop_loss(positions_data, market_prices)
            
            # Close positions if needed
            if positions_to_close:
                for symbol, close_info in positions_to_close.items():
                    self._add_alert(f"Stop-loss triggered for {symbol}: {close_info['reason']}")
                    
                    # Close the position
                    if hasattr(self.trade_executor, '_close_position'):
                        close_result = self.trade_executor._close_position(symbol, market_prices)
                        self._add_alert(f"Closed {symbol} position: P/L ${close_result.get('profit_loss', 0):.2f}")
        
        # Run the trading cycle if trading is enabled
        if risk_state['trading_enabled']:
            # Run the decision logic manager's trading cycle
            if self.decision_manager:
                cycle_result = self.decision_manager.run_trading_cycle(market_data, price_data)
            else:
                cycle_result = {
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'message': 'No decision manager available'
                }
        else:
            # Trading is disabled due to risk limits
            self._add_alert("Trading disabled due to risk limits")
            cycle_result = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'message': 'Trading disabled due to risk limits',
                'risk_state': risk_state
            }
        
        # Generate risk report
        risk_report = self.risk_manager.generate_risk_report()
        
        # Add risk information to cycle result
        cycle_result['risk_state'] = risk_state
        cycle_result['risk_report'] = risk_report
        cycle_result['alerts'] = self.alerts
        
        return cycle_result
    
    def _get_market_prices(self):
        """
        Get current market prices.
        
        Returns:
            dict: Dictionary containing current market prices.
        """
        # In a real implementation, these would be fetched from a market data provider
        return {
            'ES': 5000.00,
            'NQ': 17500.00,
            'MES': 5000.00,
            'MNQ': 17500.00
        }
    
    def _add_alert(self, message):
        """
        Add an alert message.
        
        Args:
            message (str): Alert message.
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.alerts.append({
            'timestamp': timestamp,
            'message': message
        })
        print(f"ALERT [{timestamp}]: {message}")
    
    def check_risk_before_trade(self, strategy, signal):
        """
        Check if a trade is allowed based on risk state.
        
        Args:
            strategy (str): Strategy name ('NQ-ES' or 'MES-MNQ').
            signal (dict): Trading signal.
        
        Returns:
            tuple: (allowed, reason) where allowed is a boolean and reason is a string.
        """
        return self.risk_manager.check_trade_allowed(strategy, signal)
    
    def generate_risk_alerts_report(self, output_file=None):
        """
        Generate a report of risk alerts.
        
        Args:
            output_file (str, optional): File to save the report to. If None, generates a timestamped filename.
        
        Returns:
            str: Path to the saved report file.
        """
        if output_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"risk_alerts_{timestamp}.md"
        
        if self.output_dir:
            filepath = os.path.join(self.output_dir, output_file)
        else:
            filepath = output_file
        
        # Generate report content
        report = f"""# Risk Alerts Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Current Alerts

"""
        
        if self.alerts:
            report += "| Timestamp | Message |\n"
            report += "|-----------|----------|\n"
            
            for alert in self.alerts:
                report += f"| {alert['timestamp']} | {alert['message']} |\n"
        else:
            report += "No alerts in the current trading cycle.\n"
        
        # Write report to file
        with open(filepath, 'w') as f:
            f.write(report)
        
        return filepath
    
    def get_risk_summary(self):
        """
        Get a summary of the current risk state.
        
        Returns:
            dict: Dictionary containing risk summary.
        """
        # Get the current risk state
        risk_state = self.risk_manager.risk_state
        
        # Create a simplified summary
        summary = {
            'trading_enabled': risk_state['trading_enabled'],
            'nq_es_risk': {
                'current_loss': risk_state['NQ-ES']['current_loss'],
                'daily_loss': risk_state['NQ-ES']['daily_loss'],
                'risk_level': risk_state['NQ-ES']['risk_level'],
                'max_loss_reached': risk_state['NQ-ES']['max_loss_reached'],
                'max_loss_limit': self.risk_manager.max_loss_nq_es
            },
            'mes_mnq_risk': {
                'current_loss': risk_state['MES-MNQ']['current_loss'],
                'daily_loss': risk_state['MES-MNQ']['daily_loss'],
                'risk_level': risk_state['MES-MNQ']['risk_level'],
                'max_loss_reached': risk_state['MES-MNQ']['max_loss_reached'],
                'max_loss_limit': self.risk_manager.max_loss_mes_mnq
            },
            'daily_loss_limit': self.risk_manager.daily_loss_limit,
            'last_update': risk_state['last_update'],
            'alert_count': len(self.alerts)
        }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Import necessary modules for testing
    import sys
    sys.path.append('..')
    from decision_logic import DecisionLogicManager
    from decision_logic.trade_execution import TradeExecutor
    
    # Create cache and output directories
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    
    # Create trade executor
    trade_executor = TradeExecutor(
        demo_mode=True,
        cache_dir=cache_dir
    )
    
    # Create decision logic manager
    decision_manager = DecisionLogicManager(
        trade_executor=trade_executor,
        cache_dir=cache_dir,
        output_dir=output_dir
    )
    
    # Create risk management integrator
    risk_integrator = RiskManagementIntegrator(
        decision_manager=decision_manager,
        trade_executor=trade_executor,
        max_loss_nq_es=5000,
        max_loss_mes_mnq=3000,
        daily_loss_limit=2000,
        trailing_stop_pct=0.5,
        cache_dir=cache_dir,
        output_dir=output_dir
    )
    
    # Run risk-managed trading cycle
    cycle_result = risk_integrator.run_risk_managed_trading_cycle()
    
    # Print results
    print("Risk-Managed Trading Cycle Results:")
    print(f"  Timestamp: {cycle_result.get('timestamp', 'N/A')}")
    print(f"  Trading Enabled: {cycle_result['risk_state']['trading_enabled']}")
    
    if cycle_result.get('alerts'):
        print("\nAlerts:")
        for alert in cycle_result['alerts']:
            print(f"  {alert['timestamp']}: {alert['message']}")
    
    # Get risk summary
    risk_summary = risk_integrator.get_risk_summary()
    print("\nRisk Summary:")
    print(f"  Trading Enabled: {risk_summary['trading_enabled']}")
    print(f"  NQ-ES Risk Level: {risk_summary['nq_es_risk']['risk_level']}")
    print(f"  MES-MNQ Risk Level: {risk_summary['mes_mnq_risk']['risk_level']}")
    
    # Generate risk alerts report
    alerts_report = risk_integrator.generate_risk_alerts_report()
    print(f"\nRisk alerts report saved to: {alerts_report}")
    
    # Print risk report path
    print(f"Risk report saved to: {cycle_result['risk_report']}")
