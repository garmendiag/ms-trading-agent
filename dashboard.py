#!/usr/bin/env python3
"""
MANUS Trading Agent - Web Dashboard

This script creates a Streamlit web dashboard for the MANUS Trading Agent.
It allows users to interact with the trading system through a user-friendly interface.
"""

import os
import sys
import json
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="MANUS Trading Agent Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
try:
    from data_collection import DataCollectionManager
    from technical_indicators import TechnicalIndicatorsManager
    from signal_generation import SignalGenerationManager
    from decision_logic import DecisionLogicManager
    from risk_management import RiskManagementIntegrator
    from testing import TestingManager
    
    MODULES_LOADED = True
except ImportError:
    MODULES_LOADED = False
    st.warning("Some modules could not be imported. Running in demo mode.")

# Define color scheme
COLORS = {
    'primary': '#1E88E5',
    'secondary': '#FFC107',
    'success': '#4CAF50',
    'danger': '#F44336',
    'warning': '#FF9800',
    'info': '#2196F3',
    'light': '#F5F5F5',
    'dark': '#212121',
    'background': '#FAFAFA',
    'text': '#212121',
    'bullish': '#4CAF50',
    'bearish': '#F44336',
    'neutral': '#9E9E9E'
}

# Define demo data
def generate_demo_data():
    """Generate demo data for the dashboard."""
    # Generate date range
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    # Generate price data
    np.random.seed(42)
    prices = np.random.normal(loc=0, scale=1, size=len(date_range))
    prices = np.cumsum(prices) + 4500  # Start around 4500 (ES price)
    
    # Generate OHLC data
    data = pd.DataFrame({
        'timestamp': date_range,
        'open': prices,
        'high': prices + np.random.uniform(0, 10, size=len(date_range)),
        'low': prices - np.random.uniform(0, 10, size=len(date_range)),
        'close': prices + np.random.normal(loc=0, scale=3, size=len(date_range)),
        'volume': np.random.randint(1000, 10000, size=len(date_range))
    })
    
    # Set timestamp as index
    data.set_index('timestamp', inplace=True)
    
    # Generate VIX data
    vix_data = pd.DataFrame({
        'timestamp': date_range,
        'value': np.random.normal(loc=15, scale=3, size=len(date_range))
    })
    vix_data.set_index('timestamp', inplace=True)
    
    # Generate economic indicators
    economic_data = {
        'PMI': 52.8,
        'NFP': 200000,
        'Fed Funds Rate': 4.5,
        'Unemployment Rate': 3.8,
        'CPI': 2.5,
        'GDP Growth': 2.1,
        'Retail Sales': 0.7
    }
    
    # Generate news sentiment
    # Calculate how many news items we'll have (one every 8 hours)
    news_timestamps = date_range[::8]
    
    # Create headlines list with exactly the same length as timestamps
    headlines = [
        "Fed signals potential rate cut in upcoming meeting",
        "Strong jobs report exceeds expectations",
        "Inflation data shows cooling trend",
        "Market volatility increases amid geopolitical tensions",
        "Tech sector leads market rally",
        "Oil prices surge on supply concerns",
        "Consumer confidence reaches 5-year high",
        "Retail sales data disappoints analysts",
        "Housing market shows signs of cooling",
        "Manufacturing sector expansion continues",
        "GDP growth exceeds expectations",
        "Corporate earnings beat estimates",
        "Market awaits Fed decision on interest rates",
        "Treasury yields rise on inflation concerns",
        "Global markets react to central bank policies",
        "Economic outlook remains positive despite challenges",
        "Market sentiment improves on trade developments",
        "Sector rotation observed as investors adjust portfolios",
        "Market breadth improves as rally broadens",
        "Technical indicators suggest potential market reversal",
        "Analysts revise market forecasts upward",
        "Institutional investors increase equity allocations",
        "Market internals show improving trend",
        "Economic data supports continued growth narrative"
    ]
    
    # Make sure headlines list is exactly the same length as timestamps
    if len(headlines) > len(news_timestamps):
        headlines = headlines[:len(news_timestamps)]
    elif len(headlines) < len(news_timestamps):
        # Repeat headlines if we have more timestamps than headlines
        headlines = (headlines * (len(news_timestamps) // len(headlines) + 1))[:len(news_timestamps)]
    
    # Generate sentiment values with exactly the same length as timestamps
    sentiments = np.random.uniform(-1, 1, size=len(news_timestamps))
    
    # Create the DataFrame with arrays of equal length
    news_data = pd.DataFrame({
        'timestamp': news_timestamps,
        'headline': headlines,
        'sentiment': sentiments
    })
    news_data.set_index('timestamp', inplace=True)
    
    # Generate trading signals
    signal_data = pd.DataFrame({
        'timestamp': date_range[::12],  # One signal every 12 hours
        'signal': np.random.choice(['Market Falls: 1 ES short, 1 NQ long', 
                                   'Market Rises: 1 ES long, 1 NQ short'], 
                                   size=len(date_range[::12])),
        'confidence': np.random.uniform(0.6, 0.95, size=len(date_range[::12]))
    })
    signal_data.set_index('timestamp', inplace=True)
    
    # Generate trade data
    trade_data = pd.DataFrame({
        'timestamp': date_range[::12],  # One trade every 12 hours
        'strategy': np.random.choice(['NQ-ES', 'MES-MNQ'], size=len(date_range[::12])),
        'direction': np.random.choice(['long', 'short'], size=len(date_range[::12])),
        'entry_price': np.random.uniform(4400, 4600, size=len(date_range[::12])),
        'exit_price': np.random.uniform(4400, 4600, size=len(date_range[::12])),
        'profit_loss': np.random.uniform(-500, 1000, size=len(date_range[::12])),
        'duration': np.random.randint(1, 24, size=len(date_range[::12]))
    })
    trade_data.set_index('timestamp', inplace=True)
    
    # Calculate trade metrics
    trade_metrics = {
        'total_trades': len(trade_data),
        'winning_trades': sum(trade_data['profit_loss'] > 0),
        'losing_trades': sum(trade_data['profit_loss'] <= 0),
        'win_rate': sum(trade_data['profit_loss'] > 0) / len(trade_data) * 100,
        'total_profit_loss': trade_data['profit_loss'].sum(),
        'average_profit_loss': trade_data['profit_loss'].mean(),
        'max_profit': trade_data['profit_loss'].max(),
        'max_loss': trade_data['profit_loss'].min(),
        'profit_factor': trade_data[trade_data['profit_loss'] > 0]['profit_loss'].sum() / abs(trade_data[trade_data['profit_loss'] <= 0]['profit_loss'].sum()) if abs(trade_data[trade_data['profit_loss'] <= 0]['profit_loss'].sum()) > 0 else float('inf'),
        'average_duration': trade_data['duration'].mean()
    }
    
    # Generate risk metrics
    risk_metrics = {
        'current_risk_level': 'Normal',
        'max_drawdown': 1250.0,
        'current_drawdown': 320.0,
        'daily_loss_limit': 5000.0,
        'current_daily_loss': 320.0,
        'max_loss_nq_es': 5000.0,
        'current_loss_nq_es': 120.0,
        'max_loss_mes_mnq': 3000.0,
        'current_loss_mes_mnq': 200.0
    }
    
    # Generate equity curve
    equity_data = pd.DataFrame({
        'timestamp': date_range,
        'equity': 100000 + np.cumsum(np.random.normal(loc=50, scale=200, size=len(date_range)))
    })
    equity_data.set_index('timestamp', inplace=True)
    
    # Generate drawdown curve
    equity_values = equity_data['equity'].values
    running_max = np.maximum.accumulate(equity_values)
    drawdown = (running_max - equity_values) / running_max * 100
    drawdown_data = pd.DataFrame({
        'timestamp': date_range,
        'drawdown': drawdown
    })
    drawdown_data.set_index('timestamp', inplace=True)
    
    # Return all demo data
    return {
        'market_data': data,
        'vix_data': vix_data,
        'economic_data': economic_data,
        'news_data': news_data,
        'signal_data': signal_data,
        'trade_data': trade_data,
        'trade_metrics': trade_metrics,
        'risk_metrics': risk_metrics,
        'equity_data': equity_data,
        'drawdown_data': drawdown_data
    }

# Initialize session state
if 'demo_data' not in st.session_state:
    st.session_state.demo_data = generate_demo_data()

if 'data_manager' not in st.session_state:
    if MODULES_LOADED:
        try:
            st.session_state.data_manager = DataCollectionManager(cache_dir='data/cache')
        except:
            st.session_state.data_manager = None
    else:
        st.session_state.data_manager = None

if 'indicators_manager' not in st.session_state:
    if MODULES_LOADED and st.session_state.data_manager:
        try:
            st.session_state.indicators_manager = TechnicalIndicatorsManager(data_manager=st.session_state.data_manager)
        except:
            st.session_state.indicators_manager = None
    else:
        st.session_state.indicators_manager = None

if 'signal_manager' not in st.session_state:
    if MODULES_LOADED and st.session_state.data_manager and st.session_state.indicators_manager:
        try:
            st.session_state.signal_manager = SignalGenerationManager(
                data_manager=st.session_state.data_manager,
                indicators_manager=st.session_state.indicators_manager,
                cache_dir='data/cache'
            )
        except:
            st.session_state.signal_manager = None
    else:
        st.session_state.signal_manager = None

if 'decision_manager' not in st.session_state:
    if MODULES_LOADED and st.session_state.signal_manager:
        try:
            st.session_state.decision_manager = DecisionLogicManager(
                signal_manager=st.session_state.signal_manager
            )
        except:
            st.session_state.decision_manager = None
    else:
        st.session_state.decision_manager = None

if 'risk_manager' not in st.session_state:
    if MODULES_LOADED and st.session_state.decision_manager:
        try:
            st.session_state.risk_manager = RiskManagementIntegrator(
                decision_manager=st.session_state.decision_manager
            )
        except:
            st.session_state.risk_manager = None
    else:
        st.session_state.risk_manager = None

if 'testing_manager' not in st.session_state:
    if MODULES_LOADED and st.session_state.data_manager and st.session_state.indicators_manager and st.session_state.signal_manager and st.session_state.decision_manager and st.session_state.risk_manager:
        try:
            st.session_state.testing_manager = TestingManager(
                data_manager=st.session_state.data_manager,
                indicators_manager=st.session_state.indicators_manager,
                signal_manager=st.session_state.signal_manager,
                decision_manager=st.session_state.decision_manager,
                risk_manager=st.session_state.risk_manager,
                cache_dir='data/cache',
                output_dir='output'
            )
        except:
            st.session_state.testing_manager = None
    else:
        st.session_state.testing_manager = None

# Define functions for dashboard components
def create_market_chart(data):
    """Create a market chart with price and volume."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02, 
                        row_heights=[0.8, 0.2])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume',
            marker_color=COLORS['secondary']
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Market Data',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_vix_chart(data):
    """Create a VIX chart."""
    fig = go.Figure()
    
    # Add VIX line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['value'],
            mode='lines',
            name='VIX',
            line=dict(color=COLORS['danger'], width=2)
        )
    )
    
    # Add moving average
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['value'].rolling(window=20).mean(),
            mode='lines',
            name='20-period MA',
            line=dict(color=COLORS['primary'], width=1, dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        title='VIX (Volatility Index)',
        xaxis_title='Date',
        yaxis_title='VIX',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_news_sentiment_chart(data):
    """Create a news sentiment chart."""
    fig = go.Figure()
    
    # Add sentiment scatter plot
    colors = [COLORS['danger'] if s < 0 else COLORS['success'] for s in data['sentiment']]
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['sentiment'],
            mode='markers',
            name='Sentiment',
            marker=dict(
                color=colors,
                size=10,
                line=dict(width=1, color='black')
            ),
            text=data['headline'],
            hovertemplate='%{text}<br>Sentiment: %{y:.2f}<extra></extra>'
        )
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Update layout
    fig.update_layout(
        title='News Sentiment',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='closest',
        yaxis=dict(range=[-1.1, 1.1])
    )
    
    return fig

def create_signal_chart(data):
    """Create a trading signal chart."""
    fig = go.Figure()
    
    # Separate bullish and bearish signals
    bullish_signals = data[data['signal'].str.contains('Rises')]
    bearish_signals = data[data['signal'].str.contains('Falls')]
    
    # Add bullish signals
    fig.add_trace(
        go.Scatter(
            x=bullish_signals.index,
            y=bullish_signals['confidence'],
            mode='markers',
            name='Bullish Signals'),
            marker=dict(
                color=COLORS['bullish'],
                size=12,
                symbol='triangle-up',
                line=dict(width=1, color='black')
            ),
            text=bullish_signals['signal'],
          hovertemplate='%{text}<br>{Confidence: %} <response clipped> <NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>
      )
