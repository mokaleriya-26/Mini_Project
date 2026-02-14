# In predictor/views.py

from django.http import JsonResponse
from django.conf import settings # To find your model files
from tensorflow.keras.models import load_model
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
import os

COMPANY_NAMES = {
    "AXISBANK": "Axis Bank",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel",
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "WIPRO": "Wipro",
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports",
    "APOLLOHOSP": "Apollo Hospitals",
    "TATAMOTORS": "Tata Motors",
    "TATASTEEL": "Tata Steel",
    "BAJFINANCE": "Bajaj Finance",
    "ASIANPAINT": "Asian Paints",
    "AAPL": "Apple Inc"
}

# ===================== CONFIG =====================
NEWS_API_KEY = "2d365d32d1ac438498abaaed02e8c679" # <-- PUT YOUR KEY HERE
LOOKBACK = 60 # Same lookback as training

# ===================== LOAD THE "BRAIN" (ONCE, WHEN SERVER STARTS) =====================
print("Loading AI model and scaler...")
try:
    MODEL_PATH = os.path.join(settings.ML_MODELS_DIR, 'stock_model.h5')
    SCALER_PATH = os.path.join(settings.ML_MODELS_DIR, 'stock_scaler.joblib')

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("✅ Model and scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")
    model = None
    scaler = None

# ===================== HELPER FUNCTIONS =====================
def build_company_keywords(ticker, company_name):
    """
    Builds flexible keywords for matching news titles.
    Works for ALL companies.
    """
    symbol = ticker.split(".")[0].lower()

    words = company_name.lower().split()

    keywords = set()

    # Full company name
    keywords.add(company_name.lower())

    # Individual words (Axis, Bank)
    for w in words:
        if len(w) > 3:  # avoid junk like "of", "and"
            keywords.add(w)

    # Ticker symbol variations
    keywords.add(symbol)
    keywords.add(symbol.replace("bank", ""))

    return list(keywords)

def fetch_company_news(ticker, from_date, to_date):
    headers = {"User-Agent": "Mozilla/5.0"}

    # ---- company name resolution ----
    symbol = ticker.split(".")[0]
    search_query = COMPANY_NAMES.get(symbol, symbol)

    keywords = build_company_keywords(ticker, search_query)

    url = "https://newsapi.org/v2/everything"

    # ================= PASS 1 : STRICT =================
    params = {
        "q": search_query,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "publishedAt",
        "searchIn": "title",
        "pageSize": 20,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params, headers=headers, timeout=15)
    data = response.json()

    if data.get("status") != "ok":
        return 0.0, []

    articles = data.get("articles", [])
    sentiments, headlines = [], []

    for a in articles:
        title = a.get("title")
        if not title:
            continue

        title_lower = title.lower()

        if not any(k in title_lower for k in keywords):
            continue

        sentiments.append(TextBlob(title).sentiment.polarity)
        headlines.append({
            "source": a.get("source", {}).get("name", "Unknown"),
            "title": title,
            "url": a.get("url", "#")
        })

    # ================= PASS 2 : FALLBACK =================
    if not headlines:
        print("NEWS FALLBACK TRIGGERED for", ticker)

        params["searchIn"] = "title,description"
        response = requests.get(url, params=params, headers=headers, timeout=15)
        data = response.json()
        articles = data.get("articles", [])

        for a in articles:
            title = a.get("title")
            description = a.get("description", "")
            if not title:
                continue

            text = f"{title} {description}".lower()

            if not any(k in text for k in keywords):
                continue

            sentiments.append(TextBlob(title).sentiment.polarity)
            headlines.append({
                "source": a.get("source", {}).get("name", "Unknown"),
                "title": title,
                "url": a.get("url", "#")
            })

    sentiment_score = float(np.mean(sentiments)) if sentiments else 0.0
    return sentiment_score, headlines[:5]

def prepare_features(stock_data, sentiment_score):
    """Prepares data for prediction."""
    stock_data["sentiment"] = sentiment_score
    stock_data["MA5"] = stock_data["price"].rolling(5).mean()
    stock_data["MA10"] = stock_data["price"].rolling(10).mean()
    return stock_data

# ===================== THE API VIEW =====================

def safe_float(x):
    try:
        if hasattr(x, "values"):
            return float(x.values[0])
        return float(x)
    except:
        return float(0.0)

def safe_int(x):
    try:
        if hasattr(x, "values"):
            return int(x.values[0])
        return int(x)
    except:
        return int(0)

def get_stock_prediction(request, ticker):
    """
    This is the function that runs when a user visits your API URL.
    """
    if model is None or scaler is None:
        return JsonResponse({"error": "Model not loaded"}, status=500)

    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        # ----- 1. GET ALL DATA -----
        print(f"\n--- Request received for {ticker} ---")
        
        stock_data = yf.download(tickers=ticker, period="6mo", interval="1d")
        stock_data.reset_index(inplace=True)

        # 🔒 FIX: flatten MultiIndex columns (CRITICAL)
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)

        # 🔒 CREATE A SINGLE SOURCE OF TRUTH FOR PRICE
        if "Adj Close" in stock_data.columns:
            stock_data["price"] = stock_data["Adj Close"]
        elif "Close" in stock_data.columns:
            stock_data["price"] = stock_data["Close"]
        else:
            raise Exception(f"No price column found: {stock_data.columns.tolist()}")

        # 🔒 DROP ambiguity
        stock_data = stock_data.drop(columns=[c for c in ["Close", "Adj Close"] if c in stock_data.columns])

        print("DEBUG — last 5 prices:")
        print(stock_data[["Date", "price"]].tail(5))

        if stock_data.empty and ticker.endswith(".NS"):
            alt_ticker = ticker.replace(".NS", ".BO")
            stock_data = yf.download(tickers=alt_ticker, period="6mo", interval="1d")
            stock_data.reset_index(inplace=True)

            # 🔒 ALSO APPLY IT TO FALLBACK DATA
            if "Adj Close" in stock_data.columns:
                stock_data["Close"] = stock_data["Adj Close"]

        # 🔒 Always keep correct order
        stock_data = stock_data.sort_values("Date")

        if stock_data.empty:
            return JsonResponse({"error": f"No stock data found for {ticker}"}, status=404)

        news_end = datetime.now()
        news_start = news_end - timedelta(days=14) 
        sentiment, news_headlines = fetch_company_news(
            ticker, 
            news_start.strftime('%Y-%m-%d'), 
            news_end.strftime('%Y-%m-%d')
        )
        future_sentiment = 0.0
        print(f"Current sentiment for {ticker}: {sentiment:.4f}")

        # ----- 2. PREPARE DATA FOR PREDICTION -----
        stock_data_features = prepare_features(stock_data.copy(), sentiment)
        stock_data_features = stock_data_features.bfill().ffill()
        
        features_to_predict = stock_data_features[["price", "sentiment", "MA5", "MA10"]].values
        last_sequence_unscaled = features_to_predict[-LOOKBACK:]
        
        current_seq_scaled = scaler.transform(last_sequence_unscaled)
        current_seq_scaled = current_seq_scaled.reshape((1, LOOKBACK, 4))

        future_sentiment = 0.0  # IMPORTANT FIX
        # ----- 3. RUN 14-DAY PREDICTION (NEW MODEL) -----
        print("Running 14-day prediction (multi-output model)...")

        # ----- 3. RUN 14-DAY PREDICTION (CORRECT WAY) -----
        print("Running 14-day prediction (multi-output model)...")

        # Predict ONCE
        pred_scaled = model.predict(current_seq_scaled, verbose=0)[0]  # shape (14,)

        # Inverse scale correctly
        dummy = np.zeros((14, 4))
        dummy[:, 0] = pred_scaled  # Close column only

        predicted_prices = scaler.inverse_transform(dummy)[:, 0]
        predicted_prices = [float(p) for p in predicted_prices]

        # 🔒 Optional stabilizer (VERY IMPORTANT)
        last_close_series = stock_data["price"].iloc[-1]

        if hasattr(last_close_series, "values"):
            last_close = float(last_close_series.values[0])
        else:
            last_close = float(last_close_series)
        predicted_prices[0] = 0.7 * last_close + 0.3 * predicted_prices[0]

        # ----- 5. PREPARE ALL DASHBOARD DATA (WITH ALL FIXES) -----
    
       # 1. Key Stats (FIXED)
        print("DEBUG: Processing key_stats...")
        last_row = stock_data.tail(1)

        key_stats = {
            "open": safe_float(last_row["Open"]),
            "high": safe_float(last_row["High"]),
            "low": safe_float(last_row["Low"]),
            "close": safe_float(last_row["price"]),
            "volume": safe_int(last_row["Volume"]),
        }
        # 2. Historical Graph (FIXED - SAFE)
        print("DEBUG: Processing historical_graph...")
        historical_data = (
            stock_data
            .sort_values("Date")
            .dropna(subset=["price"])
            .iloc[-30:]
            .copy()
        )

        # Ensure Date is clean
        historical_data["Date"] = pd.to_datetime(historical_data["Date"])

        price_series = pd.to_numeric(historical_data["price"], errors="coerce")

        mask = price_series.notna()
        dates = historical_data.loc[mask, "Date"]
        prices = price_series.loc[mask]

        historical_graph = {
            "dates": dates.dt.strftime("%Y-%m-%d").tolist(),
            "prices": prices.round(2).tolist()
        }

        # print("HISTORICAL GRAPH DATA:", historical_graph) (safety check)

        # 3. Prediction Graph
        print("DEBUG: Processing prediction_graph...")
        future_dates_list = []
        last_date = pd.to_datetime(historical_data["Date"].iloc[-1]).date()

        next_day = last_date
        while len(future_dates_list) < 14:
            next_day = next_day + timedelta(days=1)
            # Monday=0 ... Sunday=6
            if next_day.weekday() < 5:
                future_dates_list.append(next_day.strftime('%Y-%m-%d'))
            
        prediction_graph = {
            "dates": future_dates_list,
            "prices": [float(p) for p in predicted_prices]
        }

        # ----- 6. RETURN THE FINAL, COMPLETE JSON -----
        print("--- Request complete. Sending JSON response. ---")
        return JsonResponse({
            "ticker": ticker,
            "key_stats": key_stats,
            "latest_news": news_headlines,
            "historical_graph_data": historical_graph,
            "prediction_graph_data": prediction_graph
        })

    except Exception as e:
        print(f"❌ An error occurred during prediction: {e}")
        return JsonResponse({"error": str(e)}, status=500)
    
    