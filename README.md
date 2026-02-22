# 📈 Alpha Recommendation Engine (Quant-Focused Portfolio Dashboard)

An institutional-grade stock recommendation and portfolio optimization engine that leverages Graph Neural Networks (GNNs), Multi-Task Ranking, and Advanced NLP Sentiment Analysis.

## 🚀 Features

- **4-Stage Alpha Pipeline**:
  - **Stage 1: Candidate Retrieval**: Uses Wikipedia/S&P 500 universe with graph-based filtering.
  - **Stage 2: Multi-Task Ranking**: Ranks stocks based on Momentum, Sharpe Ratio, Quality, and Graph Centrality.
  - **Stage 3: Risk Filtration**: Institutional-grade constraints (Beta, Volatility, Liquidity, Market Cap).
  - **Stage 4: Portfolio Optimization**: Mean-Variance optimization with custom risk-aversion parameters.
- **Deep Sentiment Analysis**: Integrated **DistilBERT + t-SNE** manifold learning for high-accuracy news sentiment.
- **Interactive Dashboard**: Real-time portfolio analysis, sector breakdown, and performance visualizations.
- **AI Analyst**: Built-in RAG-powered chatbot for deep portfolio insights (Groq/Llama-3 support).
- **Custom Stock Injection**: Add and re-normalize any custom tickers into the optimized portfolio.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/harshkk29/Alpha-Recommendation-Engine.git
   cd Alpha-Recommendation-Engine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Keys**:
   Add your keys to `config.py` or provide the Groq key directly in the dashboard sidebar.

## 🖥️ Usage

Run the Streamlit dashboard:

```bash
streamlit run dashboard_app.py
```

## 📊 Technical Architecture

The engine uses a sophisticated multi-stage approach to find 'Alpha':
1. **Node2Vec Graph Embeddings**: Captures semantic relationships between companies.
2. **DistilBERT NLP**: Analyzes real-time news articles for market sentiment.
3. **Quadratic Programming**: Solves the portfolio weight allocation problem using `cvxpy`.

## 📄 License
MIT

---
*Created by Harshvardhan Khot (@harshkk29)*
