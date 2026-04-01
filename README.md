## Stock Price Prediction using LSTM

### Overview

This project predicts short-term stock prices using an LSTM (Long Short-Term Memory) neural network and generates simple trading signals (BUY / SELL / HOLD).

### Features
	•	 Fetches real-time stock data using yfinance
	•	 Uses LSTM model for prediction
	•	 Saves & reloads trained models
	•	 Visualizes actual vs predicted prices
	•	 Generates trading signals (BUY / SELL / HOLD)
	•	 Displays results in a summary table

  ### Project Workflow
	1.	Fetch historical stock data (last 2 years)
	2.	Normalize data using MinMaxScaler
	3.	Train LSTM model (or load saved model)
	4.	Predict next price using last 60 days data
	5.	Generate trading signal:
	•	🟢 BUY → Expected increase
	•	🔴 SELL → Expected decrease
	•	🟡 HOLD → Minimal change

  ### Output
	•	Graphs showing actual vs predicted prices
	•	Summary table with:
	•	Last Close Price
	•	Predicted Price
	•	Percentage Change
	•	Signal
