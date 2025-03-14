#!/usr/bin/env python3
"""
MANUS Trading Agent - Main Script

This script is the entry point for the MANUS Trading Agent system.
It initializes all components and starts the trading system.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from data_collection import DataCollectionManager
from technical_indicators import TechnicalIndicatorsManager
from signal_generation import SignalGenerationManager
from decision_logic import DecisionLogicManager
from risk_management import RiskManagementIntegrator
from testing import TestingManager

def setup_logging(log_dir, level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_dir (str): Directory to save log files.
        level (int, optional): Logging level. Defaults to logging.INFO.
    
    Returns:
        logging.Logger: Configured logger.
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'manus_trading_agent_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('manus_trading_agent')

def load_config(config_file):
    """
    Load configuration from JSON file.
    
    Args:
        config_file (str): Path to configuration file.
    
    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration.")
        return {}

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='MANUS Trading Agent')
    
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    
    parser.add_argument('--mode', type=str, default='development',
                        choices=['development', 'production', 'backtest', 'walk-forward', 'monte-carlo'],
                        help='Operation mode')
    
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for backtesting (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for backtesting (YYYY-MM-DD)')
    
    parser.add_argument('--window-size', type=int, default=30,
                        help='Window size for walk-forward testing (days)')
    
    parser.add_argument('--step-size', type=int, default=7,
                        help='Step size for walk-forward testing (days)')
    
    parser.add_argument('--backtest-file', type=str, default=None,
                        help='Backtest results file for Monte Carlo simulation')
    
    parser.add_argument('--num-simulations', type=int, default=1000,
                        help='Number of simulations for Monte Carlo')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    return parser.parse_args()

def main():
    """
    Main function to run the MANUS Trading Agent.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    cache_dir = os.path.join(data_dir, 'cache')
    output_dir = os.path.join(base_dir, 'output')
    log_dir = os.path.join(base_dir, 'logs')
    
    # Create directories if they don't exist
    for directory in [data_dir, cache_dir, output_dir, log_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_dir, level=log_level)
    
    # Log startup information
    logger.info(f"Starting MANUS Trading Agent in {args.mode} mode")
    logger.info(f"Using configuration from {args.config}")
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        
        # Initialize data collection manager
        data_manager = DataCollectionManager(
            config.get('data_collection', {}),
            cache_dir=cache_dir
        )
        
        # Initialize technical indicators manager
        indicators_manager = TechnicalIndicatorsManager(
            config.get('technical_indicators', {}),
            data_manager=data_manager
        )
        
        # Initialize signal generation manager
        signal_manager = SignalGenerationManager(
            config.get('signal_generation', {}),
            data_manager=data_manager,
            indicators_manager=indicators_manager,
            cache_dir=cache_dir
        )
        
        # Initialize decision logic manager
        decision_manager = DecisionLogicManager(
            config.get('trade_execution', {}),
            signal_manager=signal_manager
        )
        
        # Initialize risk management integrator
        risk_manager = RiskManagementIntegrator(
            config.get('risk_management', {}),
            decision_manager=decision_manager
        )
        
        # Initialize testing manager
        testing_manager = TestingManager(
            data_manager=data_manager,
            indicators_manager=indicators_manager,
            signal_manager=signal_manager,
            decision_manager=decision_manager,
            risk_manager=risk_manager,
            cache_dir=cache_dir,
            output_dir=output_dir
        )
        
        # Run in specified mode
        if args.mode == 'development' or args.mode == 'production':
            # Run in live trading mode
            logger.info(f"Running in {args.mode} mode")
            
            # Set demo mode based on mode
            demo_mode = args.mode == 'development'
            decision_manager.set_demo_mode(demo_mode)
            
            # Start trading loop
            logger.info("Starting trading loop...")
            
            try:
                while True:
                    # Collect data
                    logger.info("Collecting data...")
                    data = data_manager.collect_all_data()
                    
                    # Calculate indicators
                    logger.info("Calculating indicators...")
                    indicators = indicators_manager.calculate_all_indicators(data)
                    
                    # Generate signals
                    logger.info("Generating signals...")
                    signals = signal_manager.generate_signals(data, indicators)
                    
                    # Execute trades
                    logger.info("Executing trades...")
                    trades = decision_manager.execute_trades(signals)
                    
                    # Manage risk
                    logger.info("Managing risk...")
                    risk_status = risk_manager.check_risk_limits()
                    
                    # Log status
                    logger.info(f"Current risk status: {risk_status}")
                    
                    # Sleep until next cycle
                    import time
                    time.sleep(60)  # Sleep for 1 minute
            
            except KeyboardInterrupt:
                logger.info("Trading loop interrupted by user")
            
            # Clean up and exit
            logger.info("Cleaning up and exiting...")
            
        elif args.mode == 'backtest':
            # Run in backtesting mode
            logger.info("Running in backtest mode")
            
            # Check if start and end dates are provided
            if not args.start_date or not args.end_date:
                logger.error("Start and end dates are required for backtesting")
                return
            
            # Run backtest
            logger.info(f"Running backtest from {args.start_date} to {args.end_date}")
            backtest_results = testing_manager.run_historical_backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                interval='1h'
            )
            
            # Log backtest results
            logger.info("Backtest completed")
            logger.info(f"NQ-ES P/L: ${backtest_results['results']['performance']['NQ-ES']['profit_loss']:.2f}")
            logger.info(f"MES-MNQ P/L: ${backtest_results['results']['performance']['MES-MNQ']['profit_loss']:.2f}")
            logger.info(f"Total P/L: ${backtest_results['results']['performance']['NQ-ES']['profit_loss'] + backtest_results['results']['performance']['MES-MNQ']['profit_loss']:.2f}")
            
            # Save backtest report
            logger.info(f"Backtest report saved to: {backtest_results['report_file']}")
            
        elif args.mode == 'walk-forward':
            # Run in walk-forward testing mode
            logger.info("Running in walk-forward testing mode")
            
            # Check if start and end dates are provided
            if not args.start_date or not args.end_date:
                logger.error("Start and end dates are required for walk-forward testing")
                return
            
            # Run walk-forward test
            logger.info(f"Running walk-forward test from {args.start_date} to {args.end_date}")
            walk_forward_results = testing_manager.run_walk_forward_test(
                start_date=args.start_date,
                end_date=args.end_date,
                window_size=args.window_size,
                step_size=args.step_size,
                interval='1h'
            )
            
            # Log walk-forward results
            logger.info("Walk-forward test completed")
            logger.info(f"NQ-ES P/L: ${walk_forward_results['aggregate_results']['NQ-ES']['profit_loss']:.2f}")
            logger.info(f"MES-MNQ P/L: ${walk_forward_results['aggregate_results']['MES-MNQ']['profit_loss']:.2f}")
            logger.info(f"Total P/L: ${walk_forward_results['aggregate_results']['NQ-ES']['profit_loss'] + walk_forward_results['aggregate_results']['MES-MNQ']['profit_loss']:.2f}")
            
            # Save walk-forward report
            logger.info(f"Walk-forward report saved to: {walk_forward_results['report_file']}")
            
        elif args.mode == 'monte-carlo':
            # Run in Monte Carlo simulation mode
            logger.info("Running in Monte Carlo simulation mode")
            
            # Check if backtest file is provided
            if not args.backtest_file:
                logger.error("Backtest results file is required for Monte Carlo simulation")
                return
            
            # Load backtest results
            try:
                with open(args.backtest_file, 'r') as f:
                    backtest_results = json.load(f)
            except Exception as e:
                logger.error(f"Error loading backtest results: {e}")
                return
            
            # Run Monte Carlo simulation
            logger.info(f"Running Monte Carlo simulation with {args.num_simulations} simulations")
            monte_carlo_results = testing_manager.run_monte_carlo_simulation(
                base_results=backtest_results,
                num_simulations=args.num_simulations
            )
            
            # Log Monte Carlo results
            logger.info("Monte Carlo simulation completed")
            logger.info(f"NQ-ES Mean Final Equity: ${monte_carlo_results['monte_carlo_stats']['NQ-ES']['mean_final_equity']:.2f}")
            logger.info(f"MES-MNQ Mean Final Equity: ${monte_carlo_results['monte_carlo_stats']['MES-MNQ']['mean_final_equity']:.2f}")
            logger.info(f"Combined Mean Final Equity: ${monte_carlo_results['monte_carlo_stats']['combined']['mean_final_equity']:.2f}")
            
            # Save Monte Carlo report
            logger.info(f"Monte Carlo report saved to: {monte_carlo_results['report_file']}")
        
        # Generate validation report
        logger.info("Generating validation report...")
        validation_report = testing_manager.generate_validation_report()
        logger.info(f"Validation report saved to: {validation_report}")
        
    except Exception as e:
        logger.error(f"Error running MANUS Trading Agent: {e}", exc_info=True)
    
    logger.info("MANUS Trading Agent completed")

if __name__ == "__main__":
    main()
