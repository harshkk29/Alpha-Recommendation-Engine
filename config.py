import os
try:
    import streamlit as st
except ImportError:
    st = None

# ═══════════════════════════════════════════════════════════════════════════
# API KEYS (Prioritizes Environment Variables / Streamlit Secrets)
# ═══════════════════════════════════════════════════════════════════════════

def get_key(key_name, default=""):
    """Helper to fetch keys from secrets, environment, or default."""
    # 1. Try Streamlit Secrets
    if st is not None:
        try:
            return st.secrets.get(key_name, os.getenv(key_name, default))
        except:
            pass
    # 2. Try Environment Variables
    return os.getenv(key_name, default)

ALPHA_VANTAGE_KEY = get_key("ALPHA_VANTAGE_KEY", "JUHFG5KJIP631PTY")
NEWS_API_KEY = get_key("NEWS_API_KEY", "ef9fe4b47270440386e64edff1e11fa8")
FMP_API_KEY = get_key("FMP_API_KEY", "ef9fe4b47270440386e64edff1e11fa8")
GROQ_API_KEY = get_key("GROQ_API_KEY", "") # Recommended to use sidebar or environment variable

# ═══════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# Graph Construction
CORRELATION_THRESHOLD = 0.7
ROLLING_WINDOW = 60  # days
ALPHA_CORR = 0.4
BETA_SECTOR = 0.3
GAMMA_FACTOR = 0.3

# Node2Vec Embeddings
EMBEDDING_DIM = 32
WALK_LENGTH = 10
NUM_WALKS = 50

# Recommendations
TOP_K_RECOMMENDATIONS = 20  # Maximum recommendations
MIN_RECOMMENDATIONS = 10    # Minimum recommendations

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: MULTI-TASK RANKING WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════
RANKING_WEIGHTS = {
    'momentum': 0.20,
    'sharpe': 0.25,
    'sentiment': 0.15,
    'quality': 0.20,
    'centrality': 0.20
}

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: RISK CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════
RISK_CONSTRAINTS = {
    'max_sector_weight': 0.30,  # 30% max per sector
    'max_single_weight': 0.15,  # 15% max per stock
    'min_liquidity_adv': 1e6,   # Minimum average daily volume
    'max_volatility': 0.45,     # 45% annualized volatility cap
    'max_beta': 2.5,            # Maximum beta allowed
    'min_market_cap': 5e8       # Minimum $500M market cap
}

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: PORTFOLIO OPTIMIZATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
PORTFOLIO_PARAMS = {
    'risk_aversion': 2.5,       # γ parameter in mean-variance
    'target_volatility': 0.15,  # 15% target portfolio volatility
    'min_weight': 0.02,         # 2% minimum position
    'max_weight': 0.15,         # 15% maximum position
    'rebalance_threshold': 0.05 # 5% drift threshold
}

# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DASHBOARD_CONFIG = {
    'page_title': 'Alpha Recommendation Engine',
    'page_icon': '📈',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Available sectors for filtering
ALL_SECTORS = [
    'Technology',
    'Healthcare',
    'Financials',
    'Consumer Discretionary',
    'Communication Services',
    'Industrials',
    'Consumer Staples',
    'Energy',
    'Utilities',
    'Real Estate',
    'Materials',
    'All Sectors'
]

# Color scheme for charts
COLOR_SCHEME = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8',
    'background': '#f8f9fa'
}
