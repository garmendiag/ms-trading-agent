"""
Risk Management Module

This module implements risk management functionality for the MANUS trading agent.
It provides advanced risk control features including maximum loss limits,
stop-loss rules, and performance monitoring.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class RiskManager:
    """
    Class for managing trading risk.
    """
    
    def __init__(self, max_loss_nq_es=5000, max_loss_mes_mnq=3000, 
                 daily_loss_limit=2000, trailing_stop_pct=0.5,
                 cache_dir=None, output_dir=None):
        """
        Initialize the risk manager.
        
        Args:
            max_loss_nq_es (float, optional): Maximum loss allowed for NQ-ES strategy in USD. Defaults to 5000.
            max_loss_mes_mnq (float, optional): Maximum loss allowed for MES-MNQ strategy in USD. Defaults to 3000.
            daily_loss_limit (float, optional): Maximum daily loss allowed in USD. Defaults to 2000.
            trailing_stop_pct (float, optional): Trailing stop percentage. Defaults to 0.5.
            cache_dir (str, optional): Directory to cache risk data. Defaults to None.
            output_dir (str, optional): Directory to save output files. Defaults to None.
        """
        self.max_loss_nq_es = max_loss_nq_es
        self.max_loss_mes_mnq = max_loss_mes_mnq
        self.daily_loss_limit = daily_loss_limit
        self.trailing_stop_pct = trailing_stop_pct
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize risk state
        self.risk_state = {
            'NQ-ES': {
                'current_loss': 0,
                'max_loss_reached': False,
                'daily_loss': 0,
                'daily_loss_reached': False,
                'trailing_stop_level': None,
                'risk_level': 'normal'  # normal, warning, critical
            },
            'MES-MNQ': {
                'current_loss': 0,
                'max_loss_reached': False,
                'daily_loss': 0,
                'daily_loss_reached': False,
                'trailing_stop_level': None,
                'risk_level': 'normal'  # normal, warning, critical
            },
            'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'trading_enabled': True
        }
        
        # Initialize risk history
        self.risk_history = {
            'NQ-ES': [],
            'MES-MNQ': [],
            'system': []
        }
        
        # Load existing risk state if available
        self._load_state()
    
    def _load_state(self):
        """
        Load existing risk state from cache.
        """
        if not self.cache_dir:
            return
        
        # Load risk state
        risk_state_file = os.path.join(self.cache_dir, 'risk_state.json')
        if os.path.exists(risk_state_file):
            try:
                with open(risk_state_file, 'r') as f:
                    self.risk_state = json.load(f)
                print(f"Loaded risk state from {risk_state_file}")
            except Exception as e:
                print(f"Error loading risk state: {e}")
        
        # Load risk history
        risk_history_file = os.path.join(self.cache_dir, 'risk_history.json')
        if os.path.exists(risk_history_file):
            try:
                with open(risk_history_file, 'r') as f:
                    self.risk_history = json.load(f)
                print(f"Loaded risk history from {risk_history_file}")
            except Exception as e:
                print(f"Error loading risk history: {e}")
    
    def _save_state(self):
        """
        Save current risk state to cache.
        """
        if not self.cache_dir:
            return
        
        # Update last update timestamp
        self.risk_state['last_update'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save risk state
        risk_state_file = os.path.join(self.cache_dir, 'risk_state.json')
        try:
            with open(risk_state_file, 'w') as f:
                json.dump(self.risk_state, f, indent=2)
            print(f"Saved risk state to {risk_state_file}")
        except Exception as e:
            print(f"Error saving risk state: {e}")
        
        # Save risk history
        risk_history_file = os.path.join(self.cache_dir, 'risk_history.json')
        try:
            with open(risk_history_file, 'w') as f:
                json.dump(self.risk_history, f, indent=2)
            print(f"Saved risk history to {risk_history_file}")
        except Exception as e:
            print(f"Error saving risk history: {e}")
    
    def update_risk_state(self, performance_data, positions_data):
        """
        Update risk state based on performance and positions data.
        
        Args:
            performance_data (dict): Dictionary containing performance data.
            positions_data (dict): Dictionary containing positions data.
        
        Returns:
            dict: Updated risk state.
        """
        # Extract performance data
        nq_es_pl = performance_data.get('NQ-ES', {}).get('profit_loss', 0)
        mes_mnq_pl = performance_data.get('MES-MNQ', {}).get('profit_loss', 0)
        
        # Extract positions data
        nq_es_position_pl = positions_data.get('nq_es_pl', 0)
        mes_mnq_position_pl = positions_data.get('mes_mnq_pl', 0)
        
        # Calculate current loss (negative P/L)
        nq_es_current_loss = max(0, -nq_es_pl)
        mes_mnq_current_loss = max(0, -mes_mnq_pl)
        
        # Update daily loss (simplified - in a real system, would track by actual trading day)
        # Check if it's a new day
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        last_update_date = datetime.datetime.strptime(self.risk_state['last_update'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        
        if current_date != last_update_date:
            # Reset daily loss for a new day
            self.risk_state['NQ-ES']['daily_loss'] = 0
            self.risk_state['MES-MNQ']['daily_loss'] = 0
            self.risk_state['NQ-ES']['daily_loss_reached'] = False
            self.risk_state['MES-MNQ']['daily_loss_reached'] = False
        
        # Update daily loss with any new losses
        if nq_es_position_pl < 0:
            self.risk_state['NQ-ES']['daily_loss'] += abs(nq_es_position_pl)
        
        if mes_mnq_position_pl < 0:
            self.risk_state['MES-MNQ']['daily_loss'] += abs(mes_mnq_position_pl)
        
        # Update current loss
        self.risk_state['NQ-ES']['current_loss'] = nq_es_current_loss
        self.risk_state['MES-MNQ']['current_loss'] = mes_mnq_current_loss
        
        # Check if max loss is reached
        self.risk_state['NQ-ES']['max_loss_reached'] = nq_es_current_loss >= self.max_loss_nq_es
        self.risk_state['MES-MNQ']['max_loss_reached'] = mes_mnq_current_loss >= self.max_loss_mes_mnq
        
        # Check if daily loss limit is reached
        self.risk_state['NQ-ES']['daily_loss_reached'] = self.risk_state['NQ-ES']['daily_loss'] >= self.daily_loss_limit
        self.risk_state['MES-MNQ']['daily_loss_reached'] = self.risk_state['MES-MNQ']['daily_loss'] >= self.daily_loss_limit
        
        # Update trailing stop levels
        # For positions with profit, set trailing stop at entry price + profit * trailing_stop_pct
        for symbol, position in positions_data.get('positions', {}).items():
            if position['direction'] == 'none' or position['quantity'] == 0:
                continue
            
            # Determine which strategy this symbol belongs to
            strategy = 'NQ-ES' if symbol in ['ES', 'NQ'] else 'MES-MNQ'
            
            # Calculate position P/L
            position_pl = 0
            if symbol in ['ES', 'NQ']:
                position_pl = nq_es_position_pl
            else:
                position_pl = mes_mnq_position_pl
            
            # Update trailing stop if position is profitable
            if position_pl > 0:
                trailing_stop = position_pl * (1 - self.trailing_stop_pct)
                if self.risk_state[strategy]['trailing_stop_level'] is None or trailing_stop > self.risk_state[strategy]['trailing_stop_level']:
                    self.risk_state[strategy]['trailing_stop_level'] = trailing_stop
        
        # Update risk level
        for strategy in ['NQ-ES', 'MES-MNQ']:
            max_loss = self.max_loss_nq_es if strategy == 'NQ-ES' else self.max_loss_mes_mnq
            current_loss = self.risk_state[strategy]['current_loss']
            
            if current_loss >= max_loss * 0.8:
                self.risk_state[strategy]['risk_level'] = 'critical'
            elif current_loss >= max_loss * 0.5:
                self.risk_state[strategy]['risk_level'] = 'warning'
            else:
                self.risk_state[strategy]['risk_level'] = 'normal'
        
        # Update trading enabled status
        self.risk_state['trading_enabled'] = not (
            self.risk_state['NQ-ES']['max_loss_reached'] or
            self.risk_state['MES-MNQ']['max_loss_reached'] or
            self.risk_state['NQ-ES']['daily_loss_reached'] or
            self.risk_state['MES-MNQ']['daily_loss_reached']
        )
        
        # Add to risk history
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.risk_history['NQ-ES'].append({
            'timestamp': timestamp,
            'current_loss': nq_es_current_loss,
            'daily_loss': self.risk_state['NQ-ES']['daily_loss'],
            'risk_level': self.risk_state['NQ-ES']['risk_level'],
            'max_loss_reached': self.risk_state['NQ-ES']['max_loss_reached'],
            'daily_loss_reached': self.risk_state['NQ-ES']['daily_loss_reached']
        })
        
        self.risk_history['MES-MNQ'].append({
            'timestamp': timestamp,
            'current_loss': mes_mnq_current_loss,
            'daily_loss': self.risk_state['MES-MNQ']['daily_loss'],
            'risk_level': self.risk_state['MES-MNQ']['risk_level'],
            'max_loss_reached': self.risk_state['MES-MNQ']['max_loss_reached'],
            'daily_loss_reached': self.risk_state['MES-MNQ']['daily_loss_reached']
        })
        
        self.risk_history['system'].append({
            'timestamp': timestamp,
            'trading_enabled': self.risk_state['trading_enabled'],
            'total_current_loss': nq_es_current_loss + mes_mnq_current_loss,
            'total_daily_loss': self.risk_state['NQ-ES']['daily_loss'] + self.risk_state['MES-MNQ']['daily_loss']
        })
        
        # Save updated state
        self._save_state()
        
        return self.risk_state
    
    def check_trade_allowed(self, strategy, signal):
        """
        Check if a trade is allowed based on risk state.
        
        Args:
            strategy (str): Strategy name ('NQ-ES' or 'MES-MNQ').
            signal (dict): Trading signal.
        
        Returns:
            tuple: (allowed, reason) where allowed is a boolean and reason is a string.
        """
        # Check if trading is enabled
        if not self.risk_state['trading_enabled']:
            return False, "Trading disabled due to risk limits"
        
        # Check strategy-specific risk limits
        if strategy == 'NQ-ES':
            if self.risk_state['NQ-ES']['max_loss_reached']:
                return False, f"NQ-ES max loss limit (${self.max_loss_nq_es}) reached"
            
            if self.risk_state['NQ-ES']['daily_loss_reached']:
                return False, f"NQ-ES daily loss limit (${self.daily_loss_limit}) reached"
            
            if self.risk_state['NQ-ES']['risk_level'] == 'critical':
                return False, "NQ-ES risk level critical"
        
        elif strategy == 'MES-MNQ':
            if self.risk_state['MES-MNQ']['max_loss_reached']:
                return False, f"MES-MNQ max loss limit (${self.max_loss_mes_mnq}) reached"
            
            if self.risk_state['MES-MNQ']['daily_loss_reached']:
                return False, f"MES-MNQ daily loss limit (${self.daily_loss_limit}) reached"
            
            if self.risk_state['MES-MNQ']['risk_level'] == 'critical':
                return False, "MES-MNQ risk level critical"
        
        # If no risk limits are triggered, allow the trade
        return True, "Trade allowed"
    
    def apply_stop_loss(self, positions_data, market_prices):
        """
        Apply stop-loss rules to current positions.
        
        Args:
            positions_data (dict): Dictionary containing positions data.
            market_prices (dict): Dictionary containing current market prices.
        
        Returns:
            dict: Dictionary containing positions to close due to stop-loss.
        """
        positions_to_close = {}
        
        # Check each position
        for symbol, position in positions_data.get('positions', {}).items():
            if position['direction'] == 'none' or position['quantity'] == 0:
                continue
            
            # Determine which strategy this symbol belongs to
            strategy = 'NQ-ES' if symbol in ['ES', 'NQ'] else 'MES-MNQ'
            
            # Get current price
            price = market_prices.get(symbol, 0)
            
            # Calculate position P/L
            if position['direction'] == 'long':
                pl = (price - position['entry_price']) * position['quantity']
            else:  # short
                pl = (position['entry_price'] - price) * position['quantity']
            
            # Apply contract multiplier
            if symbol == 'ES':
                pl *= 50  # ES contract multiplier
            elif symbol == 'NQ':
                pl *= 20  # NQ contract multiplier
            elif symbol == 'MES':
                pl *= 5  # MES contract multiplier (1/10 of ES)
            elif symbol == 'MNQ':
                pl *= 2  # MNQ contract multiplier (1/10 of NQ)
            
            # Check if trailing stop is hit
            trailing_stop_level = self.risk_state[strategy]['trailing_stop_level']
            if trailing_stop_level is not None and pl < trailing_stop_level:
                positions_to_close[symbol] = {
                    'reason': 'Trailing stop hit',
                    'current_pl': pl,
                    'stop_level': trailing_stop_level
                }
                continue
            
            # Check if max loss per position is hit (simplified example)
            max_loss_per_position = 1000  # $1000 max loss per position
            if pl < -max_loss_per_position:
                positions_to_close[symbol] = {
                    'reason': 'Max loss per position hit',
                    'current_pl': pl,
                    'max_loss': max_loss_per_position
                }
                continue
            
            # Check if risk level is critical
            if self.risk_state[strategy]['risk_level'] == 'critical' and pl < 0:
                positions_to_close[symbol] = {
                    'reason': 'Risk level critical with losing position',
                    'current_pl': pl,
                    'risk_level': 'critical'
                }
                continue
        
        return positions_to_close
    
    def generate_risk_report(self, output_file=None):
        """
        Generate a risk management report.
        
        Args:
            output_file (str, optional): File to save the report to. If None, generates a timestamped filename.
        
        Returns:
            str: Path to the saved report file.
        """
        if output_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"risk_report_{timestamp}.md"
        
        if self.output_dir:
            filepath = os.path.join(self.output_dir, output_file)
        else:
            filepath = output_file
        
        # Generate report content
        report = f"""# Risk Management Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Risk State Summary

| Strategy | Current Loss | Daily Loss | Risk Level | Max Loss Reached | Daily Loss Reached |
|----------|--------------|------------|------------|------------------|-------------------|
| NQ-ES | ${self.risk_state['NQ-ES']['current_loss']:.2f} | ${self.risk_state['NQ-ES']['daily_loss']:.2f} | {self.risk_state['NQ-ES']['risk_level'].upper()} | {'Yes' if self.risk_state['NQ-ES']['max_loss_reached'] else 'No'} | {'Yes' if self.risk_state['NQ-ES']['daily_loss_reached'] else 'No'} |
| MES-MNQ | ${self.risk_state['MES-MNQ']['current_loss']:.2f} | ${self.risk_state['MES-MNQ']['daily_loss']:.2f} | {self.risk_state['MES-MNQ']['risk_level'].upper()} | {'Yes' if self.risk_state['MES-MNQ']['max_loss_reached'] else 'No'} | {'Yes' if self.risk_state['MES-MNQ']['daily_loss_reached'] else 'No'} |

## Risk Limits

- NQ-ES Max Loss Limit: ${self.max_loss_nq_es:.2f}
- MES-MNQ Max Loss Limit: ${self.max_loss_mes_mnq:.2f}
- Daily Loss Limit: ${self.daily_loss_limit:.2f}
- Trailing Stop Percentage: {self.trailing_stop_pct * 100:.1f}%

## Trading Status

- Trading Enabled: {'Yes' if self.risk_state['trading_enabled'] else 'No'}
- Last Update: {self.risk_state['last_update']}

## Trailing Stop Levels

- NQ-ES Trailing Stop: ${self.risk_state['NQ-ES']['trailing_stop_level'] if self.risk_state['NQ-ES']['trailing_stop_level'] is not None else 'Not set'}
- MES-MNQ Trailing Stop: ${self.risk_state['MES-MNQ']['trailing_stop_level'] if self.risk_state['MES-MNQ']['trailing_stop_level'] is not None else 'Not set'}

## Risk History

"""
        
        # Add risk history if available
        if self.risk_history['system']:
            # Get the last 10 entries
            recent_history = self.risk_history['system'][-10:]
            
            report += "### Recent System Risk History\n\n"
            report += "| Timestamp | Trading Enabled | Total Current Loss | Total Daily Loss |\n"
            report += "|-----------|-----------------|-------------------|----------------|\n"
            
            for entry in recent_history:
                report += f"| {entry['timestamp']} | {'Yes' if entry['trading_enabled'] else 'No'} | ${entry['total_current_loss']:.2f} | ${entry['total_daily_loss']:.2f} |\n"
        
        # Write report to file
        with open(filepath, 'w') as f:
            f.write(report)
        
        # Generate risk charts
        if self.output_dir and len(self.risk_history['system']) > 1:
            chart_path = self.generate_risk_charts()
            if chart_path:
                report += f"\n\n## Risk Charts\n\n![Risk Charts]({os.path.basename(chart_path)})\n"
                
                # Update the report file with the chart reference
                with open(filepath, 'w') as f:
                    f.write(report)
        
        return filepath
    
    def generate_risk_charts(self, output_file=None):
        """
        Generate charts visualizing risk metrics.
        
        Args:
            output_file (str, optional): File to save the charts to. If None, generates a timestamped filename.
        
        Returns:
            str: Path to the saved chart file.
        """
        if not self.output_dir:
            return None
        
        if output_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"risk_charts_{timestamp}.png"
        
        filepath = os.path.join(self.output_dir, output_file)
        
        # Extract data for plotting
        if len(self.risk_history['system']) < 2:
            print("Not enough risk history data for charts")
            return None
        
        # Convert timestamps to datetime objects
        timestamps = [datetime.datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S') 
                     for entry in self.risk_history['system']]
        
        # Extract loss data
        nq_es_current_loss = [entry['current_loss'] for entry in self.risk_history['NQ-ES']]
        mes_mnq_current_loss = [entry['current_loss'] for entry in self.risk_history['MES-MNQ']]
        total_current_loss = [entry['total_current_loss'] for entry in self.risk_history['system']]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot current loss over time
        ax1.plot(timestamps, nq_es_current_loss, label='NQ-ES Current Loss', color='blue')
        ax1.plot(timestamps, mes_mnq_current_loss, label='MES-MNQ Current Loss', color='green')
        ax1.plot(timestamps, total_current_loss, label='Total Current Loss', color='red', linewidth=2)
        
        # Add risk limit lines
        ax1.axhline(y=self.max_loss_nq_es, color='blue', linestyle='--', label=f'NQ-ES Max Loss Limit (${self.max_loss_nq_es})')
        ax1.axhline(y=self.max_loss_mes_mnq, color='green', linestyle='--', label=f'MES-MNQ Max Loss Limit (${self.max_loss_mes_mnq})')
        
        # Set title and labels
        ax1.set_title('Current Loss Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Loss ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Extract daily loss data
        nq_es_daily_loss = [entry['daily_loss'] for entry in self.risk_history['NQ-ES']]
        mes_mnq_daily_loss = [entry['daily_loss'] for entry in self.risk_history['MES-MNQ']]
        total_daily_loss = [entry['total_daily_loss'] for entry in self.risk_history['system']]
        
        # Plot daily loss over time
        ax2.plot(timestamps, nq_es_daily_loss, label='NQ-ES Daily Loss', color='blue')
        ax2.plot(timestamps, mes_mnq_daily_loss, label='MES-MNQ Daily Loss', color='green')
        ax2.plot(timestamps, total_daily_loss, label='Total Daily Loss', color='red', linewidth=2)
        
        # Add daily loss limit line
        ax2.axhline(y=self.daily_loss_limit, color='red', linestyle='--', label=f'Daily Loss Limit (${self.daily_loss_limit})')
        
        # Set title and labels
        ax2.set_title('Daily Loss Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Loss ($)')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(filepath)
        plt.close()
        
        return filepath

# Example usage
if __name__ == "__main__":
    # Create risk manager
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    
    risk_manager = RiskManager(
        max_loss_nq_es=5000,
        max_loss_mes_mnq=3000,
        daily_loss_limit=2000,
        trailing_stop_pct=0.5,
        cache_dir=cache_dir,
        output_dir=output_dir
    )
    
    # Create sample performance data
    performance_data = {
        'NQ-ES': {
            'trades': 10,
            'wins': 6,
            'losses': 4,
            'win_rate': 60.0,
            'profit_loss': -1500,
            'max_drawdown': 2000
        },
        'MES-MNQ': {
            'trades': 8,
            'wins': 5,
            'losses': 3,
            'win_rate': 62.5,
            'profit_loss': 800,
            'max_drawdown': 1000
        }
    }
    
    # Create sample positions data
    positions_data = {
        'positions': {
            'ES': {'direction': 'long', 'quantity': 1, 'entry_price': 4950},
            'NQ': {'direction': 'short', 'quantity': 1, 'entry_price': 17600},
            'MES': {'direction': 'none', 'quantity': 0, 'entry_price': 0},
            'MNQ': {'direction': 'long', 'quantity': 2, 'entry_price': 17400}
        },
        'nq_es_pl': -1200,
        'mes_mnq_pl': 600
    }
    
    # Update risk state
    risk_state = risk_manager.update_risk_state(performance_data, positions_data)
    
    # Print risk state
    print("Risk State:")
    print(f"  NQ-ES Current Loss: ${risk_state['NQ-ES']['current_loss']}")
    print(f"  NQ-ES Risk Level: {risk_state['NQ-ES']['risk_level']}")
    print(f"  MES-MNQ Current Loss: ${risk_state['MES-MNQ']['current_loss']}")
    print(f"  MES-MNQ Risk Level: {risk_state['MES-MNQ']['risk_level']}")
    print(f"  Trading Enabled: {risk_state['trading_enabled']}")
    
    # Check if trades are allowed
    nq_es_allowed, nq_es_reason = risk_manager.check_trade_allowed('NQ-ES', {})
    mes_mnq_allowed, mes_mnq_reason = risk_manager.check_trade_allowed('MES-MNQ', {})
    
    print("\nTrade Permissions:")
    print(f"  NQ-ES: {nq_es_allowed} - {nq_es_reason}")
    print(f"  MES-MNQ: {mes_mnq_allowed} - {mes_mnq_reason}")
    
    # Apply stop-loss rules
    market_prices = {
        'ES': 4900,
        'NQ': 17700,
        'MES': 4900,
        'MNQ': 17500
    }
    
    positions_to_close = risk_manager.apply_stop_loss(positions_data, market_prices)
    
    print("\nPositions to Close:")
    if positions_to_close:
        for symbol, close_info in positions_to_close.items():
            print(f"  {symbol}: {close_info['reason']} (P/L: ${close_info['current_pl']:.2f})")
    else:
        print("  None")
    
    # Generate risk report
    report_file = risk_manager.generate_risk_report()
    print(f"\nRisk report saved to: {report_file}")
