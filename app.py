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
import tensorflow as tf

# --- Page Setup ---
st.set_page_config(page_title="Pro-Trader AI", layout="wide")
st.title(" XTI-FINANCE: STRATEGIC ALPHA SUITE")

# --- API Key Management ---
if "GNEWS_API_KEY" in st.secrets:
    API_KEY = st.secrets["GNEWS_API_KEY"]
else:
    API_KEY = st.sidebar.text_input("GNews API Key", type="password")

# --- Ticker Search Logic ---
def get_ticker_from_name(query):
    query = query.strip()
    if query.isupper() and " " not in query: return query
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5).json()
        return response['quotes'][0]['symbol']
    except: return query

# --- Sidebar: Strategic Controls ---
st.sidebar.header(" Strategy Controller")
ticker_input = st.sidebar.text_input("Enter Tickers/Names", value="Nvidia, Reliance, Apple")

# Standard yfinance periods
time_options = {
    "1 Month": "1mo", "6 Months": "6mo", "1 Year": "1y", 
    "2 Years": "2y", "5 Years": "5y", "10 Years": "10y", "MAX History": "max"
}
selected_period_label = st.sidebar.selectbox("AI Training Intelligence Depth", list(time_options.keys()), index=2)
selected_period = time_options[selected_period_label]

raw_inputs = [t.strip() for t in ticker_input.split(",") if t.strip()]
lookback = 60 

class UltimateTradingBot:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    @st.cache_data(ttl=3600)
    def get_sentiment_details(_self, ticker, api_key):
        query = ticker.split('.')[0]
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&token={api_key}&max=5"
        try:
            res = requests.get(url, timeout=5).json()
            articles = res.get("articles", [])
            if not articles: return 0.0, "😐 Neutral"
            score = np.mean([_self.sia.polarity_scores(a['title'])['compound'] for a in articles])
            return round(float(score), 2), ("🚀 Bullish" if score > 0.1 else "📉 Bearish" if score < -0.1 else "😐 Neutral")
        except: return 0.0, "API Error"

    def run_analysis(self, ticker, train_period):
        # 1. Fetch training data using the standard period
        df_train = yf.download(ticker, period=train_period, interval="1d", progress=False, auto_adjust=True)
        if df_train.empty or len(df_train) < lookback: return None
        if isinstance(df_train.columns, pd.MultiIndex): df_train.columns = df_train.columns.get_level_values(0)
        
        df_train = df_train[['Close']].ffill().dropna()
        score, word = self.get_sentiment_details(ticker, API_KEY)
        
        # 2. LSTM Pipeline
        data = df_train.values
        scaled_data = self.scaler.fit_transform(data)
        sent_feat = np.full((len(scaled_data), 1), score)
        combined = np.hstack((scaled_data, sent_feat))

        X, y = [], []
        for i in range(lookback, len(combined)):
            X.append(combined[i-lookback:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        # 3. Model Training
        tf.keras.backend.clear_session()
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=8, batch_size=32, verbose=0) 

        # 4. Predict based on training context
        last_win = combined[-lookback:].reshape(1, lookback, 2)
        pred = model.predict(last_win, verbose=0)
        final_pred = float(self.scaler.inverse_transform(pred)[0][0])
        last_price = float(df_train['Close'].iloc[-1])
        move = ((final_pred - last_price) / last_price) * 100
        
        return {
            "Ticker": ticker, "Price": last_price, "Target": final_pred,
            "Move": move, "Sent_Mood": word, "Train_Label": train_period
        }

# --- Main Execution ---
if st.sidebar.button("Execute Strategic Analysis"):
    if not API_KEY: st.error("Please enter GNews API Key.")
    else:
        bot = UltimateTradingBot()
        all_results = []
        resolved_tickers = [get_ticker_from_name(item) for item in raw_inputs]
        
        for s in resolved_tickers:
            with st.spinner(f"AI Learning {s} ({selected_period_label} patterns)..."):
                res = bot.run_analysis(s, selected_period)
                if res: all_results.append(res)

        if all_results:
            # 1. Summary Dashboard
            st.subheader(f"📊 Market Intelligence Board (Memory: {selected_period_label})")
            summary_data = [[r['Ticker'], f"{r['Price']:.2f}", f"{r['Target']:.2f}", f"{r['Move']:+.2f}%", r['Sent_Mood']] for r in all_results]
            st.table(pd.DataFrame(summary_data, columns=["Ticker", "Last Price", "AI Target", "Exp. Move", "Sentiment"]))

            # 2. Multi-Timeframe Visualization Terminal
            for r in all_results:
                st.divider()
                st.write(f"### {r['Ticker']} Visualization Suite")
                
                # These are the standard intervals provided by yfinance
                t1, t2, t3, t4, t5, t6 = st.tabs(["1 Day", "1 Week", "1 Month", "1 Year", "5 Years", "MAX"])
                
                with t1: # 1-Day Intraday
                    d = yf.download(r['Ticker'], period="1d", interval="5m", progress=False, auto_adjust=True)
                    st.line_chart(d['Close'])
                with t2: # 1-Week
                    d = yf.download(r['Ticker'], period="5d", interval="30m", progress=False, auto_adjust=True)
                    st.line_chart(d['Close'])
                with t3: # 1-Month
                    d = yf.download(r['Ticker'], period="1mo", interval="1h", progress=False, auto_adjust=True)
                    st.line_chart(d['Close'])
                with t4: # 1-Year (Standard benchmark with Target Line)
                    d = yf.download(r['Ticker'], period="1y", interval="1d", progress=False, auto_adjust=True)
                    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(d['Close'].values, color='#2c3e50', label="1 Year History")
                    ax.axhline(y=r['Target'], color='red', linestyle='--', label=f"AI Target ({selected_period_label})")
                    ax.legend()
                    st.pyplot(fig)
                with t5: # 5-Years
                    d = yf.download(r['Ticker'], period="5y", interval="1d", progress=False, auto_adjust=True)
                    st.line_chart(d['Close'])
                with t6: # MAX
                    d = yf.download(r['Ticker'], period="max", interval="1d", progress=False, auto_adjust=True)
                    st.line_chart(d['Close'])
        else:
            st.error("No data found. Check your tickers or API key.")
