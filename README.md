#  AI-Powered Stock Price Predictor & Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED.svg)](https://www.docker.com/)
[![TensorFlow](https://img.shields.io/badge/ML-TensorFlow-FF6F00.svg)](https://www.tensorflow.org/)

---

##  The Problem
Retail investors often struggle to balance **quantitative data** (historical prices) with **qualitative news** (market sentiment). Technical indicators alone don't account for "panic selling" or "hype," while news reading is too slow for human processing. 

**The Solution:** This application bridges the gap by combining an **LSTM Neural Network** (for pattern recognition) with **VaderSentiment Analysis** (for emotional context), providing a unified "Buy/Sell/Hold" signal based on both math and mood.

---

##  Unique Selling Propositions (USP)
* **Hybrid Intelligence:** Unlike standard predictors, this integrates real-time financial news sentiment scores directly into the analysis.
* **Zero-Config Deployment:** Fully Dockerized to eliminate "it works on my machine" issues.
* **Dynamic Retraining:** The LSTM model is trained on-the-fly based on the user's selected historical window.
* **Cross-Platform ARM Compatibility:** Optimized specifically to run on modern Apple Silicon (M-series) and Linux environments.

---

##  Tech Stack
| Category | Technology |
| :--- | :--- |
| **Frontend** | Streamlit (Python-based Web Framework) |
| **Data Fetching** | YFinance (Yahoo Finance API), GNews API |
| **Deep Learning** | TensorFlow / Keras (LSTM Architecture) |
| **NLP** | VaderSentiment (Lexicon and Rule-based Sentiment Analysis) |
| **Technical Analysis** | Pandas & NumPy (SMA, EMA, RSI calculations) |
| **Visualizations** | Plotly (Interactive Candlestick & Trend Charts) |
| **Containerization** | Docker (Alpine-based Python Images) |

---

##  What the UI Provides
1.  **Sidebar Configuration:** Select your Ticker (AAPL, TSLA, BTC-USD) and training duration.
2.  **Live Market Metrics:** View Current Price, Day Change, and Volume at a glance.
3.  **Sentiment Pulse:** A gauge showing if current news headlines are Bullish, Bearish, or Neutral.
4.  **Technical Dashboard:** Interactive Plotly charts with moving averages and RSI overlays.
5.  **AI Forecast:** A visual comparison of "Actual" vs. "Predicted" prices for the upcoming period.

---

##  Containerization & Local Setup

### 1. Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed.
* Git installed.

### 2. Clone & Enter
```bash
git clone [https://github.com/bradman-14/Stock_price_prediction.git](https://github.com/bradman-14/Stock_price_prediction.git)
cd Stock_price_prediction
