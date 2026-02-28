# ----------------------------------------------------
# This is train_model.py (UPDATED)
# ----------------------------------------------------

import yfinance as yf
import requests
import numpy as np
import pandas as pd
from textblob import TextBlob
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import sys

# --- CONFIG ---
import sys

TICKERS = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS","BAJAJFINSV.NS","BAJFINANCE.NS","BEL.NS","BHARTIARTL.NS",
    "CIPLA.NS","COALINDIA.NS","DRREDDY.NS","EICHERMOT.NS","ETERNAL.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS","HINDALCO.NS",
    "HINDUNILVR.NS","ICICIBANK.NS","INDIGO.NS","INFY.NS","ITC.NS","JIOFIN.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","M&M.NS","MARUTI.NS","MAXHEALTH.NS",
    "NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS","SHRIRAMFIN.NS","SUNPHARMA.NS","TATACONSUM.NS",
    "TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS","TMPV.NS","TRENT.NS","ULTRACEMCO.NS","WIPRO.NS"
]

NEWS_API_KEY = "2d365d32d1ac438498abaaed02e8c679"
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = datetime.now().strftime('%Y-%m-%d')
LOOKBACK = 60
EPOCHS = 60

# Create stock ID mapping
stock_to_id = {ticker: idx for idx, ticker in enumerate(TICKERS)}
print("Stock ID Mapping:", stock_to_id)

# --- FUNCTIONS ---
def fetch_stock_data(ticker, start, end):
    print(f"Fetching stock data for {ticker}...")
    data = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        group_by="column",
        auto_adjust=False
    )

    if data is None or data.empty:
        print("❌ No stock data received!")
        return pd.DataFrame()

    data.reset_index(inplace=True)
    # 🔒 FORCE single Close column (handles MultiIndex safely)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Adj Close" in data.columns:
        data["Close"] = data["Adj Close"]

    # Use Adj Close for better stability (splits/dividends)
    if "Adj Close" in data.columns:
        data["Close"] = data["Adj Close"]

    return data


def fetch_news_sentiment_daily(query, start, end):
    """
    Returns a dataframe with columns:
    date, sentiment
    """
    print("Fetching daily news sentiment...")

    if NEWS_API_KEY == "2d365d32d1ac438498abaaed02e8c679":
        print("NewsAPI key not set. Using neutral sentiment.")
        return pd.DataFrame(columns=["date", "sentiment"])

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "from": start,
        "to": end,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 100,   # max per request
        "apiKey": NEWS_API_KEY
    }

    try:
        res = requests.get(url, params=params, timeout=15)
        data = res.json()

        if data.get("status") != "ok":
            print("⚠️ NewsAPI error:", data)
            return pd.DataFrame(columns=["date", "sentiment"])

        articles = data.get("articles", [])
        if not articles:
            print("⚠️ No news articles found.")
            return pd.DataFrame(columns=["date", "sentiment"])

        rows = []
        for a in articles:
            title = a.get("title")
            published = a.get("publishedAt")  # ISO string

            if not title or not published:
                continue

            # Convert to date only (YYYY-MM-DD)
            dt = pd.to_datetime(published).date()

            polarity = TextBlob(title).sentiment.polarity
            rows.append((dt, polarity))

        if not rows:
            return pd.DataFrame(columns=["date", "sentiment"])

        df = pd.DataFrame(rows, columns=["date", "sentiment"])

        # Daily mean sentiment
        df = df.groupby("date", as_index=False)["sentiment"].mean()

        return df

    except Exception as e:
        print("⚠️ Sentiment fetch failed:", e)
        return pd.DataFrame(columns=["date", "sentiment"])


def prepare_features(stock_data, sentiment_df):
    print("Preparing features (MAs + daily sentiment)...")

    # Stock date column -> date only
    stock_data["date"] = pd.to_datetime(stock_data["Date"]).dt.date

    # Merge daily sentiment with stock dates
    if sentiment_df is None or sentiment_df.empty:
        stock_data["sentiment"] = 0.0
    else:
        stock_data = stock_data.merge(sentiment_df, on="date", how="left")
        stock_data["sentiment"] = stock_data["sentiment"].fillna(0.0)

    # Moving averages
    stock_data["MA5"] = stock_data["Close"].rolling(5).mean()
    stock_data["MA10"] = stock_data["Close"].rolling(10).mean()

    stock_data.dropna(inplace=True)
    return stock_data


def create_sequences(data, lookback=60, horizon=14):
    X, y = [], []
    for i in range(lookback, len(data) - horizon):
        X.append(data[i - lookback:i, :])
        y.append(data[i:i + horizon, 0])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(14)   # ✅ single-day output
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# --- MAIN SCRIPT TO RUN ---
if __name__ == "__main__":
    print(f"\n=== Starting AI Model Training ===")

    os.makedirs("ml_models", exist_ok=True)

    # 1. Fetch & combine data from all companies
    all_data = []

    for ticker in TICKERS:
        print(f"\n--- Processing {ticker} ---")

        stock_data = fetch_stock_data(ticker, TRAIN_START_DATE, TRAIN_END_DATE)
        if stock_data.empty:
            print(f"Skipping {ticker}, no data.")
            continue

        sentiment_df = fetch_news_sentiment_daily(ticker, TRAIN_START_DATE, TRAIN_END_DATE)
        stock_data = prepare_features(stock_data, sentiment_df)

        stock_id = stock_to_id[ticker]
        features_df = stock_data.loc[:, ["Close", "sentiment", "MA5", "MA10"]].copy()
        features_df["stock_id"] = stock_id
        all_data.append(features_df)

    # 🔒 SAFETY CHECK
    if not all_data:
        raise Exception("No data collected from any ticker!")

    # 🔒 CONCAT ONCE
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)

    # 🔒 FORCE FINAL FEATURE SET
    combined_data = combined_data[["Close", "sentiment", "MA5", "MA10", "stock_id"]]

    print("FINAL combined_data shape:", combined_data.shape)
    print("Columns:", combined_data.columns.tolist())

    # 2. Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_data)

    # 3. Create sequences
    X, y = create_sequences(scaled_data, LOOKBACK, 14)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 4. Build & train model
    model = build_lstm_model(input_shape=(LOOKBACK, 5))

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    )

    print("\nTraining model... This may take a few minutes.")
    model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # 5. Save model and scaler
    model.save("ml_models/stock_model.h5")
    joblib.dump(scaler, "ml_models/stock_scaler.joblib")
    joblib.dump(stock_to_id, "ml_models/stock_id_mapping.joblib")

    print("✅ Model saved.")
    print("✅ Model saved to ml_models/stock_model.h5")
    print("✅ Scaler saved to ml_models/stock_scaler.joblib")