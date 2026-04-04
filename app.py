import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Page Setup ---
st.set_page_config(page_title="AI Stock Oracle", layout="wide")
st.title("🚀 PRO-TRADER AI COMMAND DASHBOARD")
st.markdown("### Deep Learning & Sentiment-Driven Market Analysis")

# --- Inputs in Sidebar ---
st.sidebar.header("Configuration")
API_KEY = st.sidebar.text_input("GNews API Key", value="acd998d203c26fb5e57e32f5da6a8e91", type="password")
selected_tickers = st.sidebar.multiselect(
    "Select Tickers", 
    ["HDFCBANK.NS", "RELIANCE.NS", "AAPL", "TSLA", "META", "INFY.NS", "TATASTEEL.NS"],
    default=["HDFCBANK.NS", "RELIANCE.NS", "AAPL"]
)
lookback = 60

class UltimateTradingBot:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def get_sentiment_details(self, ticker):
        query = ticker.split('.')[0]
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&token={API_KEY}&max=5"
        try:
            res = requests.get(url, timeout=5).json()
            articles = res.get("articles", [])
            if not articles: return 0.0, "😐 Neutral"
            score = np.mean([self.sia.polarity_scores(a['title'])['compound'] for a in articles])
            if score > 0.2: verdict = "🚀 Bullish"
            elif score > 0.05: verdict = "📈 Positive"
            elif score < -0.2: verdict = "📉 Panic/Fear"
            elif score < -0.05: verdict = "⚠️ Negative"
            else: verdict = "😐 Neutral"
            return round(score, 2), verdict
        except:
            return 0.0, "Limit Reached"

    def run_analysis(self, ticker):
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        score, word = self.get_sentiment_details(ticker)
        data = df[['Close']].values
        scaled_data = self.scaler.fit_transform(data)
        sent_feat = np.full((len(scaled_data), 1), score)
        combined = np.hstack((scaled_data, sent_feat))

        X, y = [], []
        for i in range(lookback, len(combined)):
            X.append(combined[i-lookback:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        # Build & Train (Same as your local code)
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0) # Epochs slightly reduced for web speed

        last_win = combined[-lookback:].reshape(1, lookback, 2)
        pred = model.predict(last_win, verbose=0)
        final_pred = self.scaler.inverse_transform(pred)[0][0]
        last_price = df['Close'].iloc[-1].item()
        move = ((final_pred - last_price) / last_price) * 100
        
        if move > 1.2 and score > 0.1: advice = "STRONG BUY"
        elif move > 0.5: advice = "BUY / HOLD"
        elif move < -1.2: advice = "SELL / EXIT"
        else: advice = "NEUTRAL"

        return {
            "Ticker": ticker, "Price": last_price, "Target": final_pred,
            "Move": move, "Sent_Score": score, "Sent_Mood": word,
            "Advice": advice, "History": df['Close'].tail(100)
        }

# --- Execution ---
if st.sidebar.button("Run Global Analysis"):
    bot = UltimateTradingBot()
    all_results = []
    
    with st.spinner("Engaging Neural Networks..."):
        for s in selected_tickers:
            res = bot.run_analysis(s)
            if res: all_results.append(res)

    # 1. Show Graphs
    st.subheader("Technical Analysis Graphs")
    cols = st.columns(len(all_results))
    for idx, r in enumerate(all_results):
        with st.container():
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(r['History'].values, color='#2c3e50', linewidth=2)
            ax.axhline(y=r['Target'], color='#e74c3c', linestyle='--')
            ax.set_title(f"{r['Ticker']} Trend")
            st.pyplot(fig)

    # 2. Show Table
    st.subheader("Final Decision Dashboard")
    summary_data = []
    for r in all_results:
        risk = "🔴 High" if abs(r['Move']) > 3 else "🟡 Mod" if abs(r['Move']) > 1.5 else "🟢 Low"
        summary_data.append([r['Ticker'], f"{r['Price']:.2f}", f"{r['Target']:.2f}", f"{r['Move']:+.2f}%", r['Sent_Mood'], risk, r['Advice']])

    df_final = pd.DataFrame(summary_data, columns=["Ticker", "Price", "Target", "Move", "Sentiment", "Risk", "Verdict"])
    st.table(df_final)


