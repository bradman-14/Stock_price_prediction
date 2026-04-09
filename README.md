# PRO-TRADER AI COMMAND DASHBOARD
### *An Intelligent Financial Agent for Predictive Market Analysis*

## 📝 Project Overview
The **Pro-Trader AI Command Dashboard** is a high-performance financial tool designed to bridge the gap between raw market data and actionable intelligence. In a world where stock prices are influenced as much by news headlines as by historical patterns, this project implements a **Multi-Modal AI approach** to forecast price targets.

By combining **Deep Learning (LSTM)** for technical patterns and **Natural Language Processing (NLP)** for market sentiment, the dashboard offers a holistic view of a stock's potential "Next Move."

---

## How the AI Works

### 1. The "Memory" (LSTM Neural Networks)
Unlike traditional moving averages, this project uses **Long Short-Term Memory (LSTM)** networks. 
* **The Logic:** Stock data is a "Time-Series." LSTMs are unique because they have a "forget gate," allowing the AI to remember long-term trends (like a 60-day momentum) while ignoring short-term "noise."
* **The Process:** The model analyzes the last 60 days of closing prices to predict the 61st day.

### 2. The "Emotion" (VADER Sentiment Analysis)
Markets aren't just numbers; they are driven by human emotion.
* **The Logic:** If a stock has great technicals but the news is full of "lawsuits" or "earnings misses," the price will likely drop.
* **The Process:** The system scrapes real-time headlines via the **GNews API** and calculates a "Compound Polarity Score." This score acts as a secondary weight for the AI's final verdict.

### 3. The "Translator" (Ticker Resolution Layer)
To make the app user-friendly, I implemented a search layer that allows users to type "Google" or "Reliance" instead of remembering exchange tickers. It uses the **Yahoo Finance Search API** to resolve names to symbols in real-time.

---

## Challenges Overcome (The "Mega-Cap" Problem)
During development, a significant issue was discovered: **Mega-Cap stocks (like Apple, Tesla, and Reliance)** often return "Multi-Index" dataframes that break standard scrapers, resulting in `NaN` (Not a Number) errors.

**I solved this by:**
* **Data Flattening:** Implementing a custom layer to strip complex index levels from Yahoo Finance data.
* **Resilient Pipelines:** Using Forward-Fill (`ffill`) logic to handle data gaps during market transitions.
* **Memory Optimization:** Forcing TensorFlow to clear its backend after every analysis to ensure the app remains fast on cloud servers.

---

## 📊 Visual Features
* **Interactive Dashboard:** A clean, tabular summary of Prices, AI Targets, and Risk Levels.
* **Dynamic Charting:** Matplotlib-generated graphs that overlay the **AI-Predicted Target Line** on top of historical price action.
* **Risk Categorization:** Real-time calculation of volatility to label stocks as 🟢 Low, 🟡 Moderate, or 🔴 High Risk.

---

## Technical Stack
* **Frontend:** Streamlit (UI/UX)
* **Data:** yFinance (Market Data) & GNews (Sentiment Data)
* **AI/ML:** TensorFlow/Keras (LSTM), Scikit-Learn (Scaling)
* **Language:** Python 3.11

---

## Disclaimer
*This project is an AI-driven research tool and does not constitute financial advice. Predictive models are based on probabilities and historical data; market conditions can change rapidly.*

