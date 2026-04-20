import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# --- Page Setup ---
st.set_page_config(page_title="Pro-Trader AI", layout="wide")
st.title("STOCK PRICE PREDICTOR")

# --- API Key Management ---
if "GNEWS_API_KEY" in st.secrets:
    API_KEY = st.secrets["GNEWS_API_KEY"]
else:
    API_KEY = st.sidebar.text_input("GNews API Key", type="password")

# --- Ticker Search ---
def get_ticker_from_name(query):
    query = query.strip()
    # Updated logic: convert to upper and search if not a direct match
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5).json()
        if response['quotes']:
            return response['quotes'][0]['symbol']
    except:
        pass
    return query.upper()

# --- Sidebar ---
st.sidebar.header("Strategy Controller")
ticker_input = st.sidebar.text_input("Tickers", value="Nvidia, Reliance, Apple")

time_options = {
    "1 Month": "1mo", "6 Months": "6mo", "1 Year": "1y", 
    "2 Years": "2y", "5 Years": "5y", "10 Years": "10y", "MAX History": "max"
}
selected_period_label = st.sidebar.selectbox("AI Training Depth", list(time_options.keys()), index=5)
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
            if not articles: return 0.0, "Neutral"
            score = np.mean([_self.sia.polarity_scores(a['title'])['compound'] for a in articles])
            return round(float(score), 2), ("Bullish" if score > 0.1 else "Bearish" if score < -0.1 else "Neutral")
        except: return 0.0, "API Error"

    def run_analysis(self, ticker, train_period):
        df_train = yf.download(ticker, period=train_period, interval="1d", progress=False, auto_adjust=True)
        if df_train.empty or len(df_train) < lookback: return None
        if isinstance(df_train.columns, pd.MultiIndex): df_train.columns = df_train.columns.get_level_values(0)
        
        df_train = df_train[['Close']].ffill().dropna()
        score, word = self.get_sentiment_details(ticker, API_KEY)
        
        data = df_train.values
        scaled_data = self.scaler.fit_transform(data)
        sent_feat = np.full((len(scaled_data), 1), score)
        combined = np.hstack((scaled_data, sent_feat))

        X, y = [], []
        for i in range(lookback, len(combined)):
            X.append(combined[i-lookback:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        tf.keras.backend.clear_session()
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0) 

        last_win = combined[-lookback:].reshape(1, lookback, 2)
        pred = model.predict(last_win, verbose=0)
        final_pred = float(self.scaler.inverse_transform(pred)[0][0])
        last_price = float(df_train['Close'].iloc[-1])
        move = ((final_pred - last_price) / last_price) * 100
        
        return {
            "Ticker": ticker, "Price": last_price, "Target": final_pred,
            "Move": move, "Sent_Mood": word
        }

# --- Execution ---
if st.sidebar.button("Run Global Analysis"):
    if not API_KEY: st.error("Please enter API Key.")
    else:
        bot = UltimateTradingBot()
        all_results = []
        resolved_tickers = [get_ticker_from_name(item) for item in raw_inputs]
        
        for s in resolved_tickers:
            with st.spinner(f"Neural Engine processing {s}..."):
                res = bot.run_analysis(s, selected_period)
                if res: all_results.append(res)

        if all_results:
            st.subheader(f"Strategic Verdicts (Trained on {selected_period_label})")
            summary_data = [[r['Ticker'], f"{r['Price']:.2f}", f"{r['Target']:.2f}", f"{r['Move']:+.2f}%", r['Sent_Mood']] for r in all_results]
            st.table(pd.DataFrame(summary_data, columns=["Ticker", "Price", "AI Target", "Move %", "Sentiment"]))

            for r in all_results:
                st.divider()
                st.write(f"### {r['Ticker']} Visualization Suite")
                
                tabs = st.tabs(["1D", "1W", "1M", "1Y", "5Y", "10Y", "MAX"])
                
                # Plotly function for interactive charts with proper dates
                def plot_interactive_stock(ticker, period, interval, target):
                    d = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
                    if d.empty: return st.warning(f"No data for {period}")
                    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                    
                    fig = go.Figure()
                    
                    # Real Price Line
                    fig.add_trace(go.Scatter(
                        x=d.index, 
                        y=d['Close'], 
                        mode='lines', 
                        name='Price', 
                        line=dict(color='#00ff88', width=2)
                    ))
                    
                    # AI Target Line
                    if period != "1d":
                        fig.add_hline(
                            y=target, 
                            line_dash="dash", 
                            line_color="#ff3333", 
                            annotation_text=f"AI Target: {target:.2f}", 
                            annotation_position="top left"
                        )
                    
                    fig.update_layout(
                        title=f"{ticker} - {period} Analysis",
                        template="plotly_dark",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        hovermode="x unified",
                        height=500,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tabs[0]: plot_interactive_stock(r['Ticker'], "1d", "5m", r['Target'])
                with tabs[1]: plot_interactive_stock(r['Ticker'], "5d", "30m", r['Target'])
                with tabs[2]: plot_interactive_stock(r['Ticker'], "1mo", "1h", r['Target'])
                with tabs[3]: plot_interactive_stock(r['Ticker'], "1y", "1d", r['Target'])
                with tabs[4]: plot_interactive_stock(r['Ticker'], "5y", "1d", r['Target'])
                with tabs[5]: plot_interactive_stock(r['Ticker'], "10y", "1d", r['Target'])
                with tabs[6]: plot_interactive_stock(r['Ticker'], "max", "1d", r['Target'])
        else:
            st.error("Data error.")
