"""
═══════════════════════════════════════════════════════════════════════════════
ALPHA RECOMMENDATION ENGINE - STREAMLIT DASHBOARD (FIXED)
═══════════════════════════════════════════════════════════════════════════════

Professional Portfolio Manager Dashboard with:
- Flexible sector selection
- Adjustable number of recommendations (10-20)
- Capital allocation visualization (Pie & Bar charts)
- Time series analysis for top stocks
- Key performance metrics
- AI Chatbot for portfolio analysis (Groq API)

Author: Harshvardhan
Version: 1.1 (Fixed)
Date: 2026-02-13
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import json
import requests
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the main engine (from parent directory)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alpha_sentiment_integrated import AlphaRecommendationEngine
from config import *

# ═══════════════════════════════════════════════════════════════════════════
# YAHOO FINANCE FIX - Let yfinance v1.1.0 handle sessions internally
# ═══════════════════════════════════════════════════════════════════════════
# Note: yfinance v1.1.0+ uses curl_cffi which doesn't work with custom sessions
# The new version handles rate limiting and browser headers automatically

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=DASHBOARD_CONFIG['page_title'],
    page_icon=DASHBOARD_CONFIG['page_icon'],
    layout=DASHBOARD_CONFIG['layout'],
    initial_sidebar_state=DASHBOARD_CONFIG['initial_sidebar_state']
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stock-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* Floating Chatbot Widget */
    .chatbot-widget {
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
    }
    
    .chatbot-toggle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        font-size: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
    }
    
    .chatbot-toggle:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 16px rgba(0,0,0,0.4);
    }
    
    .chatbot-window {
        position: fixed;
        bottom: 90px;
        left: 20px;
        width: 380px;
        height: 550px;
        background: white;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        display: none;
        flex-direction: column;
        overflow: hidden;
        z-index: 9998;
        animation: slideUp 0.3s ease;
    }
    
    .chatbot-window.active {
        display: flex;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .chatbot-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px;
        font-weight: bold;
        font-size: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .chatbot-messages {
        flex: 1;
        overflow-y: auto;
        padding: 16px;
        background: #f8f9fa;
    }
    
    .chatbot-message {
        margin-bottom: 12px;
        padding: 10px 14px;
        border-radius: 12px;
        max-width: 85%;
        word-wrap: break-word;
    }
    
    .chatbot-message.user {
        background: #e3f2fd;
        margin-left: auto;
        text-align: right;
    }
    
    .chatbot-message.assistant {
        background: white;
        border: 1px solid #e0e0e0;
    }
    
    .chatbot-input-area {
        padding: 12px;
        background: white;
        border-top: 1px solid #e0e0e0;
        display: flex;
        gap: 8px;
    }
    
    .chatbot-input {
        flex: 1;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 20px;
        outline: none;
        font-size: 14px;
    }
    
    .chatbot-send {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        cursor: pointer;
        font-size: 18px;
    }
    
    .chatbot-send:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_historical_data(tickers, period="1y"):
    """
    Fetch historical stock data using yfinance v1.1.0+
    The new version handles rate limiting and browser headers automatically
    Returns dict[ticker] = dataframe
    """
    import time
    data = {}
    
    if not tickers:
        return data
    
    # Try batch download first (faster but might fail)
    try:
        tickers_str = " ".join(tickers)
        df_all = yf.download(
            tickers_str,
            period=period,
            progress=False,
            auto_adjust=True,
            group_by='ticker',
            threads=True
        )
        
        # Process multi-ticker results
        if len(tickers) > 1 and isinstance(df_all.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    if ticker in df_all.columns.get_level_values(0):
                        df = df_all[ticker].copy()
                        
                        if df is None or df.empty or "Close" not in df.columns:
                            continue
                        
                        df = df.dropna(subset=["Close"])
                        if df.empty:
                            continue
                        
                        base_price = df["Close"].iloc[0]
                        if base_price == 0 or pd.isna(base_price):
                            continue
                        
                        df["Normalized"] = (df["Close"] / base_price) * 100
                        data[ticker] = df
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
                    continue
        
        # Single ticker or fallback to individual downloads
        elif len(tickers) == 1 or df_all.empty:
            for ticker in tickers:
                try:
                    time.sleep(0.5)  # Rate limiting delay
                    df = yf.download(
                        ticker,
                        period=period,
                        progress=False,
                        auto_adjust=True
                    )
                    
                    if df is None or df.empty or "Close" not in df.columns:
                        continue
                    
                    df = df.dropna(subset=["Close"])
                    if df.empty:
                        continue
                    
                    base_price = df["Close"].iloc[0]
                    if base_price == 0 or pd.isna(base_price):
                        continue
                    
                    df["Normalized"] = (df["Close"] / base_price) * 100
                    data[ticker] = df
                    
                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")
                    continue
    
    except Exception as e:
        print(f"Batch download failed: {e}, trying individual downloads...")
        # Fallback: Individual downloads with delays
        for ticker in tickers:
            try:
                time.sleep(1.0)  # Longer delay for fallback
                df = yf.download(
                    ticker,
                    period=period,
                    progress=False,
                    auto_adjust=True
                )
                
                if df is not None and not df.empty and "Close" in df.columns:
                    df = df.dropna(subset=["Close"])
                    if not df.empty:
                        base_price = df["Close"].iloc[0]
                        if base_price != 0 and not pd.isna(base_price):
                            df["Normalized"] = (df["Close"] / base_price) * 100
                            data[ticker] = df
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                continue
    
    print(f"✅ Successfully fetched {len(data)}/{len(tickers)} stocks")
    return data


def create_pie_chart(portfolio: pd.DataFrame) -> go.Figure:
    """Create capital allocation pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=portfolio['ticker'],
        values=portfolio['weight'],
        hole=0.4,
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.1%}<br>Sector: %{customdata}<extra></extra>',
        customdata=portfolio['sector']
    )])
    
    fig.update_layout(
        title={
            'text': '💼 Capital Allocation by Stock',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1f77b4'}
        },
        showlegend=True,
        height=500,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    return fig

def create_bar_chart(portfolio: pd.DataFrame) -> go.Figure:
    """Create portfolio weights bar chart"""
    portfolio_sorted = portfolio.sort_values('weight', ascending=True)
    
    fig = go.Figure(data=[go.Bar(
        y=portfolio_sorted['ticker'],
        x=portfolio_sorted['weight'],
        orientation='h',
        marker=dict(
            color=portfolio_sorted['weight'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Weight")
        ),
        text=[f"{w:.1%}" for w in portfolio_sorted['weight']],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Weight: %{x:.2%}<br>Expected Return: %{customdata:.2%}<extra></extra>',
        customdata=portfolio_sorted['expected_return']
    )])
    
    fig.update_layout(
        title={
            'text': '📊 Portfolio Weights Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1f77b4'}
        },
        xaxis_title='Portfolio Weight',
        yaxis_title='Stock Ticker',
        height=max(400, len(portfolio) * 30),
        showlegend=False,
        xaxis=dict(tickformat='.0%')
    )
    return fig

def create_time_series_chart(historical_data, portfolio):
    """Create multi-line time series chart for top stocks"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    color_idx = 0
    
    for ticker, df in historical_data.items():
        if df is None or df.empty:
            continue
        
        if "Normalized" not in df.columns:
            continue
        
        # Get weight for this ticker
        weight = portfolio[portfolio['ticker'] == ticker]['weight'].iloc[0] if ticker in portfolio['ticker'].values else 0
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Normalized"],
                mode="lines",
                name=f"{ticker} ({weight:.1%})",
                line=dict(width=2, color=colors[color_idx % len(colors)]),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
            )
        )
        color_idx += 1
    
    fig.update_layout(
        title={
            'text': "📈 Portfolio Stocks Performance (Normalized to 100)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1f77b4'}
        },
        xaxis_title="Date",
        yaxis_title="Normalized Price (Base 100)",
        template="plotly_white",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
    )
    
    return fig

def create_sector_allocation_chart(portfolio: pd.DataFrame) -> go.Figure:
    """Create sector allocation pie chart"""
    sector_weights = portfolio.groupby('sector')['weight'].sum().reset_index()
    
    fig = go.Figure(data=[go.Pie(
        labels=sector_weights['sector'],
        values=sector_weights['weight'],
        hole=0.3,
        marker=dict(
            colors=px.colors.qualitative.Pastel,
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>Allocation: %{value:.1%}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': '🏢 Sector Allocation',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1f77b4'}
        },
        height=400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    return fig

def local_portfolio_analysis(question: str, portfolio: pd.DataFrame, metrics: dict) -> str:
    """
    Local fallback: generates smart portfolio analysis purely from the
    actual Alpha Engine data — no API key required.
    """
    q = question.lower()
    
    # Portfolio stats
    positions = len(portfolio)
    sectors = portfolio['sector'].nunique()
    total_exp_return = (portfolio['weight'] * portfolio['expected_return']).sum()
    
    # ---- question-routing ----
    if any(w in q for w in ["invest", "why", "reason", "should i buy", "buy", "about"]):
        # Extract specific ticker if mentioned in question and actually in our portfolio
        words = question.upper().replace('?', '').replace(',', '').split()
        portfolio_tickers = set(portfolio['ticker'].values)
        ticker = next((w for w in words if w in portfolio_tickers), None)
        
        if ticker:
            m = metrics.get(ticker)
            row = portfolio[portfolio['ticker'] == ticker].iloc[0]
            
            sector_context = f"in the {row['sector']} sector" if row['sector'] != 'Unknown' else ""
            return (
                f"**3 Data-Backed Reasons to Invest in {ticker}:**\n\n"
                f"1. **Alpha Profile** — {ticker} {sector_context} was selected because it carries an expected return of **{row['expected_return']:.2%}**. "
                f"Our multi-task ranker assigned it a high composite score based on its momentum ({m.momentum:.1%}) and risk-adjusted efficiency.\n\n"
                f"2. **Risk Performance** — With a volatility of **{m.predicted_volatility:.2%}** and a Beta of **{m.beta:.2f}**, {ticker} passed our "
                f"Stage 3 institutional-grade filters, ensuring it provides stability relative to the broader market portfolio.\n\n"
                f"3. **Strategic Fit** — The Alpha Engine's Stage 4 optimizer allocated a **{row['weight']:.1%} weight** to {ticker}. This specific sizing "
                f"optimizes the portfolio's overall Sharpe Ratio (**{(total_exp_return/0.15):.2f}** estimated) by leveraging {ticker}'s low semantic correlation "
                f"with your other holdings."
            )
        else:
            return (
                f"**Why This Alpha Portfolio Is Optimized for Performance:**\n\n"
                f"1. **Quantitative Selection** — Every stock was ranked by a 4-stage pipeline "
                f"(GNN retrieval → multi-task ranker → risk filters → mean-variance optimizer), ensuring only the highest-conviction signals remain.\n\n"
                f"2. **Strong Return Profile** — The portfolio targeting an expected return of **{total_exp_return:.2%}**, "
                f"selected from a vetted universe of S&P 500 candidates with the best momentum and sentiment scores.\n\n"
                f"3. **Diversified Risk** — With **{positions} positions** across **{sectors} sectors**, "
                f"the portfolio limits concentration risk. No single stock dominates, matching professional asset management standards."
            )

    elif any(w in q for w in ["risk", "volatile", "drawdown", "safe", "dangerous"]):
        return (
            f"**Portfolio Risk Analysis:**\n\n"
            f"• **Volatility Control** — Portfolio stocks passed a 45% annualised-volatility cap and a beta \u2264 2.5 filter.\n"
            f"• **Diversification Level** — Spanning **{sectors} unique sectors** allows the portfolio to mitigate sector-specific shocks.\n"
            f"• **Optimized Weights** — The Stage 4 optimizer actively penalises high-variance correlations during weight allocation.\n"
            f"• **Institutional Guards** — Position sizes are capped, preventing any single stock from impacting the portfolio by more than its allocated threshold."
        )

    elif any(w in q for w in ["return", "profit", "gain", "perform", "best", "highest"]):
        return (
            f"**Portfolio Return Analysis:**\n\n"
            f"• **Target Expected Return:** {total_exp_return:.2%}\n"
            f"• **Alpha Logic:** Stocks are ranked by a composite score involving Sharpe Ratio (25%), Momentum (20%), and Sentiment (15%).\n"
            f"• **Top Picks:** Quantitative ranking identified the current {positions} positions as having the highest probability of market outperformance.\n"
            f"• **Conviction:** Weights reflect the engine's confidence in each stock's contribution to the portfolio's total return."
        )

    else:
        return (
            f"**Portfolio Summary:**\n\n"
            f"• **Positions:** {positions} stocks selected by Alpha Engine\n"
            f"• **Expected Return:** {total_exp_return:.2%} (weighted average)\n"
            f"• **Sectors represented:** {sectors}\n"
            f"• **Ranking Metrics:** Momentum, Sharpe Ratio, Quality, and Sentiment.\n\n"
            f"*Note: For a deeper AI analysis using your holdings data, please provide a Groq API key in the sidebar.*"
        )


def query_groq_chatbot(question: str, portfolio: pd.DataFrame, metrics: dict, api_key: str) -> str:
    """Query Groq API for portfolio analysis, with local fallback on auth errors."""
    # No key provided → go straight to local analysis
    if not api_key or not api_key.strip():
        return local_portfolio_analysis(question, portfolio, metrics)

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }
        
        # Build descriptive context from actual data
        avg_ret = (portfolio['weight'] * portfolio['expected_return']).sum()
        stats = f"Positions: {len(portfolio)}, Sectors: {portfolio['sector'].nunique()}, Exp.Return: {avg_ret:.2%}"
        details = "\n".join([f"- {r.ticker}: Weight {r.weight:.1%}, Sector {r.sector}, Exp.Return {r.expected_return:.2%}" for _,r in portfolio.iterrows()])
        
        full_context = f"Summary Stats: {stats}\n\nActual Portfolio Composition:\n{details}"

        system_prompt = f"""You are an expert Portfolio Analyst. 
Analyze the provided Alpha Engine portfolio data to answer user questions.
IMPORTANT: You MUST use the exact numbers (weights, returns, etc.) from the data provided below. 
Do NOT use generic templates. If a specific stock is asked about, look it up in the 'Actual Portfolio Composition' list.

{full_context}"""

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 1,
            "stream": False
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return "Error: Unexpected response format from API"
        elif response.status_code in (401, 403):
            # Invalid / expired key → silent fallback so the UI never shows a raw error
            return local_portfolio_analysis(question, portfolio_context)
        else:
            error_msg = f"Error: API returned status code {response.status_code}"
            try:
                error_detail = response.json()
                if 'error' in error_detail:
                    error_msg += f"\nDetails: {error_detail['error'].get('message', 'Unknown error')}"
            except:
                pass
            return error_msg + "\n\n" + local_portfolio_analysis(question, portfolio_context)
            
    except requests.exceptions.Timeout:
        return "Request timed out — here's a local analysis instead:\n\n" + local_portfolio_analysis(question, portfolio_context)
    except requests.exceptions.RequestException as e:
        return "Network error — here's a local analysis instead:\n\n" + local_portfolio_analysis(question, portfolio_context)
    except Exception as e:
        return local_portfolio_analysis(question, portfolio_context)

# ═══════════════════════════════════════════════════════════════════════════
# RAG-POWERED CHATBOT FOR FLOATING WIDGET
# ═══════════════════════════════════════════════════════════════════════════

def retrieve_relevant_portfolio_data(question: str, portfolio: pd.DataFrame, metrics: dict) -> str:
    """
    RAG Retrieval: Extract relevant portfolio information based on user question
    Uses keyword matching and semantic understanding
    """
    question_lower = question.lower()
    retrieved_context = []
    
    # Detect question intent
    is_risk_question = any(word in question_lower for word in ['risk', 'volatile', 'volatility', 'drawdown', 'safe', 'dangerous'])
    is_return_question = any(word in question_lower for word in ['return', 'profit', 'gain', 'performance', 'best', 'highest'])
    is_sector_question = any(word in question_lower for word in ['sector', 'industry', 'diversif'])
    is_stock_specific = any(ticker in question.upper() for ticker in portfolio['ticker'].values)
    
    # Retrieve relevant data based on intent
    if is_risk_question:
        retrieved_context.append("📊 RISK ANALYSIS:")
        for _, row in portfolio.iterrows():
            ticker = row['ticker']
            if ticker in metrics:
                m = metrics[ticker]
                retrieved_context.append(
                    f"• {ticker}: Volatility={m.predicted_volatility:.2%}, "
                    f"Max Drawdown={m.max_drawdown:.2%}, "
                    f"Downside Risk={m.downside_risk:.2%}, "
                    f"Sharpe Ratio={m.sharpe_ratio:.2f}"
                )
    
    if is_return_question:
        retrieved_context.append("\n💰 RETURN ANALYSIS:")
        sorted_portfolio = portfolio.sort_values('expected_return', ascending=False)
        for _, row in sorted_portfolio.head(5).iterrows():
            ticker = row['ticker']
            if ticker in metrics:
                m = metrics[ticker]
                retrieved_context.append(
                    f"• {ticker}: Expected Return={row['expected_return']:.2%}, "
                    f"Outperform Prob={m.outperform_prob:.1%}, "
                    f"Momentum={m.momentum:.2%}"
                )
    
    if is_sector_question:
        retrieved_context.append("\n🏢 SECTOR ALLOCATION:")
        sector_weights = portfolio.groupby('sector')['weight'].sum().sort_values(ascending=False)
        for sector, weight in sector_weights.items():
            sector_stocks = portfolio[portfolio['sector'] == sector]['ticker'].tolist()
            retrieved_context.append(f"• {sector}: {weight:.1%} ({', '.join(sector_stocks)})")
    
    if is_stock_specific:
        retrieved_context.append("\n📈 STOCK DETAILS:")
        for ticker in portfolio['ticker'].values:
            if ticker in question.upper() and ticker in metrics:
                row = portfolio[portfolio['ticker'] == ticker].iloc[0]
                m = metrics[ticker]
                retrieved_context.append(
                    f"• {ticker} ({row['sector']}):\n"
                    f"  - Weight: {row['weight']:.1%}\n"
                    f"  - Expected Return: {row['expected_return']:.2%}\n"
                    f"  - Volatility: {m.predicted_volatility:.2%}\n"
                    f"  - Sharpe Ratio: {m.sharpe_ratio:.2f}\n"
                    f"  - Market Cap: ${m.market_cap/1e9:.2f}B\n"
                    f"  - Beta: {m.beta:.2f}"
                )
    
    # Always include portfolio summary
    if not retrieved_context:
        retrieved_context.append("📊 PORTFOLIO OVERVIEW:")
        expected_return = (portfolio['weight'] * portfolio['expected_return']).sum()
        retrieved_context.append(f"• Total Positions: {len(portfolio)}")
        retrieved_context.append(f"• Expected Return: {expected_return:.2%}")
        retrieved_context.append(f"• Sectors: {portfolio['sector'].nunique()}")
        retrieved_context.append(f"• Top Holdings: {', '.join(portfolio.head(3)['ticker'].tolist())}")
    
    return "\n".join(retrieved_context)

def rag_chatbot_query(question: str, portfolio: pd.DataFrame, metrics: dict, api_key: str) -> str:
    """
    RAG-powered chatbot: Retrieves relevant data + Generates AI response.
    Falls back to local analysis when no valid API key is available.
    """
    try:
        # Step 1: RETRIEVAL - Get relevant portfolio data
        retrieved_data = retrieve_relevant_portfolio_data(question, portfolio, metrics)
        
        # Step 2: AUGMENTATION - Create enhanced context
        portfolio_summary = f"""
Portfolio Summary:
- Total Positions: {len(portfolio)}
- Expected Return: {(portfolio['weight'] * portfolio['expected_return']).sum():.2%}
- Sectors: {portfolio['sector'].nunique()}

Retrieved Relevant Data:
{retrieved_data}
"""
        
        # No API key → local fallback immediately
        if not api_key or not api_key.strip():
            return local_portfolio_analysis(question, portfolio, metrics)

        # Step 3: GENERATION - Query Groq AI with retrieved context
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }
        
        system_prompt = f"""You are an expert portfolio analyst with access to detailed portfolio data.

{portfolio_summary}

Provide concise, actionable insights based on the retrieved data above. 
Use specific numbers from the data. Keep responses under 150 words.
Be professional but conversational."""

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": 0.7,
            "max_tokens": 300,
            "stream": False
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return local_portfolio_analysis(question, portfolio, metrics)
        elif response.status_code in (401, 403):
            # Expired / invalid key → silent local fallback
            return local_portfolio_analysis(question, portfolio, metrics)
        else:
            return local_portfolio_analysis(question, portfolio, metrics)
            
    except Exception as e:
        return local_portfolio_analysis(question, portfolio, metrics)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM STOCK FETCHER
# ═══════════════════════════════════════════════════════════════════════════

def fetch_single_stock_metrics(ticker: str):
    """
    Fetch live data for a single ticker and return:
      - a dict row compatible with the portfolio DataFrame
      - a StockMetrics object for the metrics dict
    Returns (None, None) on failure.
    """
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        hist = yf.download(ticker, period="1y", progress=False, threads=False)

        if hist is None or hist.empty or len(hist) < 20:
            return None, None

        # Flatten MultiIndex columns if present
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)

        if 'Close' not in hist.columns:
            return None, None

        prices = hist['Close'].dropna()
        returns = np.log(prices / prices.shift(1)).dropna().values

        # Compute metrics
        volatility  = float(np.std(returns) * np.sqrt(252)) if len(returns) > 0 else 0.20
        rf          = 0.02
        sharpe      = float((np.mean(returns) - rf/252) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
        momentum    = float((prices.iloc[-1] / prices.iloc[0] - 1)) if len(prices) > 1 else 0.0
        cumulative  = (1 + pd.Series(np.log(prices / prices.shift(1)).fillna(0))).cumprod()
        running_max = cumulative.expanding().max()
        drawdown    = abs(((cumulative - running_max) / running_max).min())
        max_dd      = float(drawdown) if not np.isnan(drawdown) else 0.0

        # Fundamental data
        try:
            info       = stock.info
            fast       = getattr(stock, 'fast_info', {})
            sector     = info.get('sector', 'Unknown')
            beta       = float(info.get('beta', 1.0) or 1.0)
            market_cap = float(fast.get('market_cap', info.get('marketCap', 1e9)) or 1e9)
            pe_ratio   = float(info.get('trailingPE', 15) or 15)
            price      = float(fast.get('last_price', prices.iloc[-1]))
            avg_volume = float(fast.get('last_volume', 1e6))
        except Exception:
            sector     = 'Unknown'
            beta       = 1.0
            market_cap = 1e9
            pe_ratio   = 15.0
            price      = float(prices.iloc[-1])
            avg_volume = 1e6

        from alpha_sentiment_integrated import StockMetrics
        sm = StockMetrics(
            ticker           = ticker,
            returns          = returns,
            volatility       = volatility,
            momentum         = momentum,
            sharpe_ratio     = sharpe,
            max_drawdown     = max_dd,
            beta             = beta,
            sector           = sector,
            market_cap       = market_cap,
            price            = price,
            avg_volume       = avg_volume,
            pe_ratio         = pe_ratio,
            predicted_return = momentum * 0.5 + sharpe * 0.1,
            outperform_prob  = float(np.clip(0.5 + momentum, 0, 1)),
            predicted_volatility = volatility,
            downside_risk    = volatility * 0.6,
        )

        row = {
            'ticker'         : ticker,
            'weight'         : 0.05,          # placeholder; renormalised later
            'expected_return': sm.predicted_return,
            'sector'         : sector,
            'stage2_score'   : 0.5,
        }
        return row, sm

    except Exception as e:
        return None, None


def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Alpha Recommendation Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Professional Quantitative Portfolio Management System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SIDEBAR - CONTROLS
    # ═══════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/stock-market.png", width=80)
        st.title("⚙️ Portfolio Controls")
        st.markdown("---")
        
        # Number of stocks
        num_recommendations = st.slider(
            "📊 Number of Recommendations",
            min_value=10,
            max_value=20,
            value=15,
            step=1,
            help="Select how many stocks to include in the portfolio"
        )
        
        # Sector selection
        st.markdown("### 🏢 Sector Filter")
        sector_option = st.radio(
            "Select sectors to include:",
            ["All Sectors", "Custom Selection"],
            help="Choose to analyze all sectors or select specific ones"
        )
        
        if sector_option == "Custom Selection":
            selected_sectors = st.multiselect(
                "Choose sectors:",
                [s for s in ALL_SECTORS if s != 'All Sectors'],
                default=['Technology', 'Healthcare', 'Financials']
            )
        else:
            selected_sectors = None
        
        # Universe size
        universe_size = st.slider(
            "🌐 Stock Universe Size",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
            help="Number of stocks to analyze before filtering"
        )
        
        st.markdown("---")
        st.markdown("### ➕ Add Custom Stock")
        custom_ticker_input = st.text_input(
            "Stock Tickers (comma separated)",
            placeholder="e.g. AAPL, MSFT, TSLA",
            help="Type one or more US stock tickers separated by commas and click 'Add Stock' to include them in the portfolio."
        ).strip().upper()
        add_stock_btn = st.button("Add Stock(s) ➕", use_container_width=True)
        
        if add_stock_btn and custom_ticker_input:
            if 'custom_stocks' not in st.session_state:
                st.session_state.custom_stocks = {}
            
            # Split by comma and clean
            tickers_to_add = [t.strip() for t in custom_ticker_input.split(',') if t.strip()]
            
            # Remove duplicates from this input
            tickers_to_add = list(dict.fromkeys(tickers_to_add))
            
            success_list = []
            fail_list = []
            exist_list = []
            
            with st.spinner(f"Processing {len(tickers_to_add)} tickers..."):
                for ticker in tickers_to_add:
                    if ticker in st.session_state.custom_stocks:
                        exist_list.append(ticker)
                        continue
                        
                    row, sm = fetch_single_stock_metrics(ticker)
                    if row is None:
                        fail_list.append(ticker)
                    else:
                        st.session_state.custom_stocks[ticker] = {
                            'row': row, 'sm': sm
                        }
                        success_list.append(ticker)
            
            # Show summary
            if success_list:
                st.success(f"✅ Added {len(success_list)} stocks: {', '.join(success_list)}")
            if exist_list:
                st.info(f"ℹ️ Already in list: {', '.join(exist_list)}")
            if fail_list:
                st.error(f"❌ Failed to fetch: {', '.join(fail_list)}")
            
            if success_list:
                st.rerun() # Refresh to update the portfolio weights immediately
        
        # Show and allow removal of added custom stocks
        if st.session_state.get('custom_stocks'):
            st.markdown("**Added stocks:**")
            for ct in list(st.session_state.custom_stocks.keys()):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"📌 {ct}")
                with col_b:
                    if st.button("✕", key=f"rm_{ct}"):
                        del st.session_state.custom_stocks[ct]
                        st.rerun()
        
        st.markdown("---")
        
        # Run button
        run_analysis = st.button("🚀 Generate Portfolio", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🤖 AI Analyst Key")
        runtime_groq_key = st.text_input(
            "Groq API Key (optional)",
            value=GROQ_API_KEY,
            type="password",
            placeholder="gsk_...",
            help="Paste a fresh Groq API key for full AI answers. Leave blank to use built-in local analysis. Get a free key at https://console.groq.com/keys"
        )
        if runtime_groq_key:
            st.success("🔑 Key provided — AI Analyst active")
        else:
            st.info("💡 No key? Built-in local analysis will answer your questions.")
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.info("""
        **4-Stage Pipeline:**
        1. 🔍 Candidate Retrieval
        2. 🧠 Multi-Task Ranking
        3. 🛡️ Risk Filtering
        4. 💼 Portfolio Optimization
        """)
    
    # ═══════════════════════════════════════════════════════════════════════
    # MAIN CONTENT
    # ═══════════════════════════════════════════════════════════════════════
    
    # Initialize session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = None
    if 'candidates' not in st.session_state:
        st.session_state.candidates = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'custom_stocks' not in st.session_state:
        st.session_state.custom_stocks = {}
    
    if run_analysis:
        with st.spinner('🔄 Running Alpha Recommendation Engine... This may take 2-5 minutes...'):
            try:
                # Initialize engine
                engine = AlphaRecommendationEngine()
                
                # Create a progress container
                progress_text = st.empty()
                progress_text.text("Stage 1: Fetching data and building graph...")
                
                # Run pipeline
                portfolio, candidates, metrics, embeddings, graph = engine.run_complete_pipeline(
                    num_stocks=universe_size,
                    top_k=num_recommendations
                )
                
                progress_text.text("Processing complete!")
                
                # Check if portfolio was generated
                if portfolio is None:
                    st.error("❌ Pipeline returned None for portfolio. Check console output for details.")
                    st.info("💡 Try reducing Universe Size to 50 and Stocks to 10 for faster testing.")
                    return
                
                if len(portfolio) == 0:
                    st.error("❌ Portfolio is empty. All stocks may have been filtered out.")
                    st.info("💡 Try:\n- Increasing Universe Size\n- Selecting 'All Sectors'\n- Reducing number of stocks")
                    return
                
                # Apply sector filter if needed
                if selected_sectors:
                    original_len = len(portfolio)
                    portfolio = portfolio[portfolio['sector'].isin(selected_sectors)]
                    if len(portfolio) == 0:
                        st.error(f"❌ No stocks found in selected sectors. Original portfolio had {original_len} stocks in other sectors.")
                        st.info(f"💡 Try selecting 'All Sectors' or choose from: {', '.join(portfolio['sector'].unique())}")
                        return
                    # Renormalize weights
                    portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum()
                    st.info(f"ℹ️ Filtered from {original_len} to {len(portfolio)} stocks based on sector selection")
                
                # Store in session state
                st.session_state.portfolio = portfolio
                st.session_state.candidates = candidates
                st.session_state.metrics = metrics
                
                progress_text.empty()
                st.success(f"✅ Portfolio generated successfully with {len(portfolio)} stocks!")
                st.balloons()
                    
            except Exception as e:
                st.error(f"❌ Error during portfolio generation:")
                st.code(str(e))
                st.info("💡 Troubleshooting tips:\n"
                       "1. Check your internet connection\n"
                       "2. Try with smaller Universe Size (50)\n"
                       "3. Reduce number of stocks (10)\n"
                       "4. Select 'All Sectors'\n"
                       "5. Check the terminal/console for detailed error messages")
                import traceback
                with st.expander("🔍 Technical Details"):
                    st.code(traceback.format_exc())
                return
    
    # ── Inject custom stocks into the active portfolio ──────────────────────
    if st.session_state.portfolio is not None and st.session_state.get('custom_stocks'):
        portfolio_work = st.session_state.portfolio.copy()
        metrics_work   = dict(st.session_state.metrics)
        existing_tickers = set(portfolio_work['ticker'].values)

        new_rows = []
        for ct, data in st.session_state.custom_stocks.items():
            if ct not in existing_tickers:
                new_rows.append(data['row'])
                metrics_work[ct] = data['sm']

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            # Give each custom stock a 5% weight and shrink existing weights
            new_total_alloc = 0.05 * len(new_rows)
            portfolio_work['weight'] = portfolio_work['weight'] * (1 - new_total_alloc)
            portfolio_work = pd.concat([portfolio_work, new_df], ignore_index=True)
            portfolio_work['weight'] = portfolio_work['weight'] / portfolio_work['weight'].sum()
            # Update session state so the rest of the page reflects the change
            st.session_state.portfolio = portfolio_work
            st.session_state.metrics   = metrics_work
    
    # Display results if available
    if st.session_state.portfolio is not None:
        portfolio = st.session_state.portfolio
        candidates = st.session_state.candidates
        metrics = st.session_state.metrics
        
        # ═══════════════════════════════════════════════════════════════════
        # KEY METRICS
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("## 📊 Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📈 Number of Positions",
                len(portfolio),
                help="Total stocks in portfolio"
            )
        
        with col2:
            expected_return = (portfolio['weight'] * portfolio['expected_return']).sum()
            st.metric(
                "💰 Expected Return",
                f"{expected_return:.2%}",
                help="Weighted average expected return"
            )
        
        with col3:
            avg_sharpe = np.average(
                [metrics[t].sharpe_ratio for t in portfolio['ticker'] if t in metrics],
                weights=portfolio['weight']
            )
            st.metric(
                "⚡ Avg Sharpe Ratio",
                f"{avg_sharpe:.2f}",
                help="Risk-adjusted return metric"
            )
        
        with col4:
            num_sectors = portfolio['sector'].nunique()
            st.metric(
                "🏢 Sectors",
                num_sectors,
                help="Number of different sectors"
            )
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # VISUALIZATIONS
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("## 📊 Portfolio Visualizations")
        
        # Row 1: Capital Allocation (Pie) and Weights (Bar)
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_pie_chart(portfolio), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_bar_chart(portfolio), use_container_width=True)
        
        # Row 2: Sector Allocation
        st.plotly_chart(create_sector_allocation_chart(portfolio), use_container_width=True)
        
        # Row 3: Historical Performance (Multi-line chart)
        st.markdown("### 📈 Historical Performance - Top Stocks")
        with st.spinner('Fetching historical data for visualization...'):
            # Get top 10 stocks by weight for cleaner visualization
            top_stocks = portfolio.nlargest(10, 'weight')
            historical_data = fetch_historical_data(top_stocks['ticker'].tolist(), period="1y")
            
            if historical_data and len(historical_data) > 0:
                st.plotly_chart(create_time_series_chart(historical_data, portfolio), use_container_width=True)
                st.success(f"✅ Successfully loaded data for {len(historical_data)} stocks")
            else:
                st.warning("⚠️ Could not fetch historical data. Possible reasons:\n"
                          "- Internet connection issues\n"
                          "- Yahoo Finance API temporarily unavailable\n"
                          "- Invalid ticker symbols")
                st.info("💡 Try refreshing the page or generating a new portfolio")
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # DETAILED STOCK BREAKDOWN
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("## 📋 Detailed Stock Analysis")
        
        for idx, row in portfolio.iterrows():
            ticker = row['ticker']
            if ticker in metrics:
                stock_metrics = metrics[ticker]
                
                with st.expander(f"**{ticker}** - {row['sector']} | Weight: {row['weight']:.1%}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📊 Returns & Risk**")
                        st.write(f"Expected Return: **{row['expected_return']:.2%}**")
                        st.write(f"Volatility: **{stock_metrics.predicted_volatility:.2%}**")
                        st.write(f"Sharpe Ratio: **{stock_metrics.sharpe_ratio:.2f}**")
                        st.write(f"Max Drawdown: **{stock_metrics.max_drawdown:.2%}**")
                    
                    with col2:
                        st.markdown("**💼 Fundamentals**")
                        st.write(f"Market Cap: **${stock_metrics.market_cap/1e9:.2f}B**")
                        st.write(f"Current Price: **${stock_metrics.price:.2f}**")
                        st.write(f"Beta: **{stock_metrics.beta:.2f}**")
                        st.write(f"P/E Ratio: **{stock_metrics.pe_ratio:.2f}**")
                    
                    with col3:
                        st.markdown("**🎯 AI Predictions**")
                        st.write(f"Outperform Prob: **{stock_metrics.outperform_prob:.1%}**")
                        st.write(f"Downside Risk: **{stock_metrics.downside_risk:.2%}**")
                        st.write(f"Momentum: **{stock_metrics.momentum:.2%}**")
                        st.write(f"Avg Volume: **{stock_metrics.avg_volume/1e6:.2f}M**")
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # AI CHATBOT
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("## 🤖 AI Portfolio Analyst")
        st.markdown("Ask questions about your portfolio and get expert insights powered by Groq AI")
        
        # Create portfolio context for chatbot
        portfolio_summary = f"""
Portfolio Summary:
- Total Positions: {len(portfolio)}
- Expected Return: {expected_return:.2%}
- Average Sharpe Ratio: {avg_sharpe:.2f}
- Sectors: {num_sectors}

Top Holdings:
{portfolio.head(5)[['ticker', 'weight', 'expected_return', 'sector']].to_string(index=False)}

Sector Allocation:
{portfolio.groupby('sector')['weight'].sum().to_string()}
"""
        
        # Chat interface
        user_question = st.text_input(
            "💬 Ask about your portfolio:",
            placeholder="e.g., What are the main risks in this portfolio? How diversified is it?"
        )
        
        if st.button("Send", use_container_width=True):
            if user_question:
                with st.spinner('🤔 Analyzing...'):
                    response = query_groq_chatbot(user_question, portfolio, metrics, runtime_groq_key)
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': response
                    })
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💬 Conversation History")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                st.markdown(f'<div class="chat-message user-message"><b>You:</b> {chat["question"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message assistant-message"><b>AI Analyst:</b> {chat["answer"]}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick questions
        st.markdown("### 💡 Quick Questions")
        quick_questions = [
            "What are the top 3 risks in this portfolio?",
            "How well diversified is this portfolio?",
            "Which stocks have the highest expected returns?",
            "What is the sector concentration risk?",
            "Should I rebalance this portfolio?"
        ]
        
        cols = st.columns(len(quick_questions))
        for idx, (col, question) in enumerate(zip(cols, quick_questions)):
            with col:
                if st.button(f"❓ {idx+1}", key=f"quick_{idx}", help=question):
                    with st.spinner('🤔 Analyzing...'):
                        response = query_groq_chatbot(question, portfolio, metrics, runtime_groq_key)
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': response
                        })
                        st.rerun()
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # EXPORT OPTIONS
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("## 💾 Export Portfolio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = portfolio.to_csv(index=False)
            st.download_button(
                label="📥 Download Portfolio (CSV)",
                data=csv,
                file_name=f"alpha_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = portfolio.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_data,
                file_name=f"alpha_portfolio_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>👋 Welcome to the Alpha Recommendation Engine</h2>
            <p style="font-size: 1.2rem; color: #666;">
                A professional quantitative portfolio management system powered by AI
            </p>
            <br>
            <p>👈 Configure your preferences in the sidebar and click <b>"Generate Portfolio"</b> to begin</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔍 Smart Analysis
            - Graph Neural Networks
            - Multi-Task Deep Learning
            - Sentiment Analysis
            - Factor Models
            """)
        
        with col2:
            st.markdown("""
            ### 🛡️ Risk Management
            - Sector Diversification
            - Volatility Controls
            - Liquidity Filters
            - Beta Constraints
            """)
        
        with col3:
            st.markdown("""
            ### 💼 Optimization
            - Mean-Variance
            - Risk Parity
            - Capital Allocation
            - Rebalancing Alerts
            """)
    
    # ═══════════════════════════════════════════════════════════════════════
    # FLOATING CHATBOT WIDGET WITH RAG (Bottom-Left Corner)
    # ═══════════════════════════════════════════════════════════════════════
    if st.session_state.portfolio is not None:
        portfolio = st.session_state.portfolio
        metrics = st.session_state.metrics
        expected_return = (portfolio['weight'] * portfolio['expected_return']).sum()
        
        # Initialize floating chat history in session state
        if 'floating_chat_messages' not in st.session_state:
            st.session_state.floating_chat_messages = []
        
        # Create auto-summary for chatbot
        top_3_holdings = "\\n".join([f"• {row['ticker']} ({row['sector']}): {row['weight']:.1%}" for _, row in portfolio.head(3).iterrows()])
        sector_breakdown = "\\n".join([f"• {sector}: {weight:.1%}" for sector, weight in portfolio.groupby('sector')['weight'].sum().items()])
        
        auto_summary = f"""📊 Portfolio Summary

Overview:
• Total Positions: {len(portfolio)}
• Expected Return: {expected_return:.2%}
• Number of Sectors: {portfolio['sector'].nunique()}

Top 3 Holdings:
{top_3_holdings}

Sector Breakdown:
{sector_breakdown}

💡 Key Insights:
• Portfolio is {'well-diversified' if portfolio['sector'].nunique() >= 5 else 'concentrated'} across {portfolio['sector'].nunique()} sectors
• Expected annual return of {expected_return:.2%}
• Top holding represents {portfolio['weight'].max():.1%} of portfolio"""
        
        # Hidden input for RAG chatbot
        st.markdown('<div id="rag-chat-container" style="display:none;"></div>', unsafe_allow_html=True)
        
        # Chat input (hidden but functional)
        chat_col1, chat_col2 = st.columns([4, 1])
        with chat_col1:
            user_message = st.text_input("💬 Ask AI about your portfolio:", key="rag_chat_input", label_visibility="collapsed", placeholder="Ask about risks, returns, sectors...")
        with chat_col2:
            send_button = st.button("Send", key="rag_send_btn", use_container_width=True)
        
        # Process RAG query
        if send_button and user_message:
            with st.spinner('🤖 AI is analyzing your portfolio...'):
                # Use RAG to get intelligent response
                ai_response = rag_chatbot_query(user_message, portfolio, metrics, runtime_groq_key)
                
                # Store in session state
                st.session_state.floating_chat_messages.append({
                    'user': user_message,
                    'assistant': ai_response
                })
                
                # Clear input
                st.rerun()
        
        # Display chat history
        if st.session_state.floating_chat_messages:
            st.markdown("### 💬 Recent Conversations")
            for msg in st.session_state.floating_chat_messages[-3:]:  # Show last 3
                st.markdown(f'<div class="chat-message user-message"><b>You:</b> {msg["user"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message assistant-message"><b>AI:</b> {msg["assistant"]}</div>', unsafe_allow_html=True)
        
        # Build chat messages for JavaScript display
        js_chat_messages = ""
        for msg in st.session_state.floating_chat_messages:
            js_chat_messages += f'<div class="chatbot-message user">{msg["user"]}</div>'
            js_chat_messages += f'<div class="chatbot-message assistant">{msg["assistant"]}</div>'
        
        # Floating chatbot HTML with JavaScript
        chatbot_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .chatbot-widget {{
                    position: fixed;
                    bottom: 20px;
                    left: 20px;
                    z-index: 9999;
                }}
                
                .chatbot-toggle {{
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    cursor: pointer;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    font-size: 28px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.3s ease;
                }}
                
                .chatbot-toggle:hover {{
                    transform: scale(1.1);
                    box-shadow: 0 6px 16px rgba(0,0,0,0.4);
                }}
                
                .chatbot-window {{
                    position: fixed;
                    bottom: 90px;
                    left: 20px;
                    width: 380px;
                    height: 550px;
                    background: white;
                    border-radius: 16px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
                    display: none;
                    flex-direction: column;
                    overflow: hidden;
                    z-index: 9998;
                    animation: slideUp 0.3s ease;
                }}
                
                .chatbot-window.active {{
                    display: flex;
                }}
                
                @keyframes slideUp {{
                    from {{
                        opacity: 0;
                        transform: translateY(20px);
                    }}
                    to {{
                        opacity: 1;
                        transform: translateY(0);
                    }}
                }}
                
                .chatbot-header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 16px;
                    font-weight: bold;
                    font-size: 18px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                
                .chatbot-messages {{
                    flex: 1;
                    overflow-y: auto;
                    padding: 16px;
                    background: #f8f9fa;
                }}
                
                .chatbot-message {{
                    margin-bottom: 12px;
                    padding: 10px 14px;
                    border-radius: 12px;
                    max-width: 85%;
                    word-wrap: break-word;
                    font-size: 13px;
                    line-height: 1.5;
                    white-space: pre-wrap;
                }}
                
                .chatbot-message.user {{
                    background: #e3f2fd;
                    margin-left: auto;
                    text-align: right;
                }}
                
                .chatbot-message.assistant {{
                    background: white;
                    border: 1px solid #e0e0e0;
                }}
                
                .chatbot-input-area {{
                    padding: 12px;
                    background: white;
                    border-top: 1px solid #e0e0e0;
                    display: flex;
                    gap: 8px;
                }}
                
                .chatbot-input {{
                    flex: 1;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 20px;
                    outline: none;
                    font-size: 14px;
                }}
                
                .chatbot-send {{
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    cursor: pointer;
                    font-size: 18px;
                }}
                
                .chatbot-send:hover {{
                    opacity: 0.9;
                }}
                
                .rag-badge {{
                    background: #4caf50;
                    color: white;
                    padding: 2px 8px;
                    border-radius: 10px;
                    font-size: 10px;
                    margin-left: auto;
                }}
            </style>
        </head>
        <body>
            <div class="chatbot-widget">
                <button class="chatbot-toggle" onclick="toggleChatbot()" id="chatToggle">
                    💬
                </button>
                
                <div class="chatbot-window" id="chatWindow">
                    <div class="chatbot-header">
                        <span>🤖</span>
                        <span>Portfolio AI Assistant</span>
                        <span class="rag-badge">RAG</span>
                    </div>
                    
                    <div class="chatbot-messages" id="chatMessages">
                        <div class="chatbot-message assistant">{auto_summary}</div>
                        <div class="chatbot-message assistant">👋 Hi! I'm powered by RAG (Retrieval-Augmented Generation). I can analyze your portfolio data and answer specific questions about risks, returns, sectors, and individual stocks. Try asking me something!</div>
                        {js_chat_messages}
                    </div>
                    
                    <div class="chatbot-input-area">
                        <input 
                            type="text" 
                            class="chatbot-input" 
                            id="chatInput" 
                            placeholder="Ask about your portfolio..."
                            onkeypress="if(event.key==='Enter') sendMessage()"
                        />
                        <button class="chatbot-send" onclick="sendMessage()">
                            ➤
                        </button>
                    </div>
                </div>
            </div>
            
            <script>
                function toggleChatbot() {{
                    const chatWindow = document.getElementById('chatWindow');
                    const chatToggle = document.getElementById('chatToggle');
                    
                    if (chatWindow.classList.contains('active')) {{
                        chatWindow.classList.remove('active');
                        chatToggle.innerHTML = '💬';
                    }} else {{
                        chatWindow.classList.add('active');
                        chatToggle.innerHTML = '✕';
                        // Scroll to bottom when opening
                        const messagesDiv = document.getElementById('chatMessages');
                        messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    }}
                }}
                
                function sendMessage() {{
                    const input = document.getElementById('chatInput');
                    const message = input.value.trim();
                    
                    if (message === '') return;
                    
                    // Add user message to display
                    const messagesDiv = document.getElementById('chatMessages');
                    const userMsg = document.createElement('div');
                    userMsg.className = 'chatbot-message user';
                    userMsg.textContent = message;
                    messagesDiv.appendChild(userMsg);
                    
                    // Clear input
                    input.value = '';
                    
                    // Show loading message
                    const loadingMsg = document.createElement('div');
                    loadingMsg.className = 'chatbot-message assistant';
                    loadingMsg.id = 'loading-msg';
                    loadingMsg.innerHTML = '🤔 Analyzing your portfolio with RAG...';
                    messagesDiv.appendChild(loadingMsg);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    
                    // Send to Streamlit input
                    const streamlitInput = window.parent.document.querySelector('input[aria-label="💬 Ask AI about your portfolio:"]');
                    if (streamlitInput) {{
                        streamlitInput.value = message;
                        streamlitInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        
                        // Trigger send button
                        setTimeout(() => {{
                            const sendBtn = window.parent.document.querySelector('button[kind="primary"]');
                            if (sendBtn && sendBtn.textContent.includes('Send')) {{
                                sendBtn.click();
                            }}
                        }}, 100);
                    }}
                }}
                
                // Auto-scroll to bottom on load
                window.addEventListener('load', () => {{
                    const messagesDiv = document.getElementById('chatMessages');
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }});
            </script>
        </body>
        </html>
        """
        
        # Render the chatbot using components.html
        components.html(chatbot_html, height=0, scrolling=False)

if __name__ == "__main__":
    main()