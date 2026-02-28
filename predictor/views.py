# In predictor/views.py

from django.http import JsonResponse
from django.conf import settings # To find your model files
from tensorflow.keras.models import load_model
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])

COMPANY_NAMES = {
    "ADANIENT": "Adani Enterprises Limited",
    "ADANIPORTS": "Adani Ports & SEZ Limited",
    "APOLLOHOSP": "Apollo Hospitals Enterprises Limited",
    "ASIANPAINT": "Asian Paints Limited",
    "AXISBANK": "Axis Bank Limited",
    "BAJAJ-AUTO": "Bajaj Auto Limited",
    "BAJAJFINSV": "Bajaj Finserv Limited",
    "BAJFINANCE": "Bajaj Finance Limited",
    "BEL": "Bharat Electronics Limited",
    "BHARTIARTL": "Bharti Airtel Limited",
    "CIPLA": "Cipla Limited",
    "COALINDIA": "Coal India Limited",
    "DRREDDY": "Dr. Reddy's Laboratories Limited",
    "EICHERMOT": "Eicher Motors Limited",
    "ETERNAL": "Eternal Life Insurance Limited",
    "GRASIM": "Grasim Industries Limited",
    "HCLTECH": "HCL Technologies Limited",
    "HDFCBANK": "HDFC Bank Limited",
    "HDFCLIFE": "HDFC Life Insurance Company Limited",
    "HINDALCO": "Hindalco Industries Limited",
    "HINDUNILVR": "Hindustan Unilever Limited",
    "ICICIBANK": "ICICI Bank Limited",
    "INDIGO": "InterGlobe Aviation Limited",
    "INFY": "Infosys Limited",
    "ITC": "ITC Limited",
    "JIOFIN": "Jio Financial Services Limited",
    "JSWSTEEL": "JSW Steel Limited",
    "KOTAKBANK": "Kotak Mahindra Bank Limited",
    "LT": "Larsen & Toubro Limited",
    "M&M": "Mahindra & Mahindra Limited",
    "MARUTI": "Maruti Suzuki India Limited",
    "MAXHEALTH": "Max Healthcare Institute Limited",
    "NESTLEIND": "Nestle India Limited",
    "NTPC": "NTPC Limited",
    "ONGC": "Oil & Natural Gas Corporation Limited",
    "POWERGRID": "Power Grid Corporation of India Limited",
    "RELIANCE": "Reliance Industries Limited",
    "SBILIFE": "SBI Life Insurance Company Limited",
    "SBIN": "State Bank of India",
    "SHRIRAMFIN": "Shriram Finance Limited",
    "SUNPHARMA": "Sun Pharmaceuticals Industries Limited",
    "TATACONSUM": "Tata Consumer Products Limited",
    "TATASTEEL": "Tata Steel Limited",
    "TCS": "Tata Consultancy Services Limited",
    "TECHM": "Tech Mahindra Limited",
    "TITAN": "Titan Company Limited",
    "TMPV": "Tata Motors Passenger Vehicles Limited",
    "TRENT": "TRENT Limited",
    "ULTRACEMCO": "UltraTech Cement Limited",
    "WIPRO": "Wipro Limited",
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
    model.summary()
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")
    model = None
    scaler = None

print("Loading FinBERT model...")
try:
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    print("✅ FinBERT loaded successfully.")
except Exception as e:
    print("❌ FinBERT loading error:", e)
    finbert_model = None

STOCK_ID_PATH = os.path.join(settings.ML_MODELS_DIR, 'stock_id_mapping.joblib')
stock_to_id = joblib.load(STOCK_ID_PATH)
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
        if len(w) > 4:  # avoid junk like "of", "and"
            keywords.add(w)

    # Ticker symbol variations
    keywords.add(symbol)
    keywords.add(symbol.replace("bank", ""))

    return list(keywords)

financial_keywords = [
    "stock", "shares", "earnings", "results",
    "profit", "loss", "revenue", "market",
    "q1", "q2", "q3", "q4",
    "alert", "target", "buy", "sell", "hold",
    "surge", "falls", "jumps", "finance",
    "budget", "update", "guide",
    "dividend", "rate", "growth", "decline"
]

def is_financial_article(title):
    title_lower = title.lower()
    return any(word in title_lower for word in financial_keywords)


def is_article_about_company(title, company_name):
    doc = nlp(title)

    detected_orgs = [ent.text.lower() for ent in doc.ents if ent.label_ == "ORG"]

    company_clean = company_name.replace(" Limited", "").lower()
    company_words = company_clean.split()

    # 1️⃣ Check detected ORGs
    for org in detected_orgs:
        if company_clean in org:
            return True
        
        # Partial match (at least 2 words match)
        match_count = sum(1 for word in company_words if word in org)
        if match_count >= 2:
            return True

    # 2️⃣ Fallback check in title
    title_lower = title.lower()

    if company_clean in title_lower:
        return True

    match_count = sum(1 for word in company_words if word in title_lower)
    if match_count >= 2:
        return True

    return False

def get_finbert_sentiment(text):
    if finbert_model is None:
        return 0.0, "Neutral", 0.0

    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = finbert_model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)

    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    label = label_map[predicted_class.item()]
    confidence_score = confidence.item()

    # Convert to numeric score
    if label == "Positive":
        score = confidence_score
    elif label == "Negative":
        score = -confidence_score
    else:
        score = 0.0

    return score, label, confidence_score

def fetch_company_news(ticker, from_date, to_date):
    headlines = []
    sentiments = []
    seen_titles = set()

    try:
        # =============================
        # 1️⃣ YAHOO FINANCE NEWS
        # =============================
        ticker_obj = yf.Ticker(ticker)
        yahoo_news = ticker_obj.news

        for item in yahoo_news[:10]:
            title = item.get("title")
            link = item.get("link")
            publisher = item.get("publisher")
            provider_time = item.get("providerPublishTime")

            if not title:
                continue

            formatted_time = None
            if provider_time:
                dt = datetime.fromtimestamp(provider_time)
                if dt < datetime.now() - timedelta(days=30):
                    continue
                formatted_time = dt.strftime("%d %b %Y • %I:%M %p")

            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            score, label, confidence = get_finbert_sentiment(title)
            sentiments.append(score)

            headlines.append({
                "source": publisher or "Yahoo Finance",
                "title": title,
                "url": link,
                "published_at": formatted_time,
                "raw_time": dt,
                "sentiment_score": round(score, 3),
                "sentiment_label": label,
                "confidence": round(confidence * 100, 2)
            })

    except Exception as e:
        print("Yahoo news error:", e)

    try:
        # =============================
        # 2️⃣ NEWS API (FALLBACK ADDITION)
        # =============================
        symbol = ticker.split(".")[0]
        company_full = COMPANY_NAMES.get(symbol, symbol)
        company_clean = company_full.replace(" Limited", "")

        url = "https://newsapi.org/v2/everything"

        params = {
            "q": f'"{company_clean}"',
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "apiKey": NEWS_API_KEY
        }

        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        if data.get("status") == "ok":
            articles = data.get("articles", [])

            for a in articles:
                title = a.get("title")

                if not title or title in seen_titles:
                    continue

                seen_titles.add(title)
                score, label, confidence = get_finbert_sentiment(title)
                sentiments.append(score)

                published_at = a.get("publishedAt")
                formatted_time = None

                raw_dt = None
                if published_at:
                    try:
                        raw_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                        formatted_time = raw_dt.strftime("%d %b %Y • %I:%M %p")
                    except:
                        formatted_time = published_at
                else:
                    raw_dt = datetime.now()

                headlines.append({
                    "source": a.get("source", {}).get("name", "NewsAPI"),
                    "title": title,
                    "url": a.get("url", "#"),
                    "published_at": formatted_time,
                    "raw_time": raw_dt,
                    "sentiment_score": round(score, 3),
                    "sentiment_label": label,
                    "confidence": round(confidence * 100, 2)
                })

    except Exception as e:
        print("NewsAPI error:", e)

    # =============================
    # 3️⃣ FINAL PROCESSING
    # =============================
    sentiment_score = float(np.mean(sentiments)) if sentiments else 0.0

    # Sort by published time (latest first)
    headlines = sorted(
        headlines,
        key=lambda x: x.get("raw_time", datetime.min),
        reverse=True
    )
    for h in headlines:
        h.pop("raw_time", None)
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
        ticker_obj = yf.Ticker(ticker)   # ⭐ NEW
        info = ticker_obj.info           # ⭐ NEW

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
        news_start = news_end - timedelta(days=30) 
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
        
        symbol = ticker.split(".")[0]
        stock_id = stock_to_id.get(ticker, 0)
        stock_data_features["stock_id"] = stock_id

        features_to_predict = stock_data_features[["price", "sentiment", "MA5", "MA10", "stock_id"]].values
        last_sequence_unscaled = features_to_predict[-LOOKBACK:]
        
        current_seq_scaled = scaler.transform(last_sequence_unscaled)
        current_seq_scaled = current_seq_scaled.reshape((1, LOOKBACK, 5))

        future_sentiment = 0.0  # IMPORTANT FIX
        # ----- 3. RUN 14-DAY PREDICTION (CORRECT WAY) -----
        print("Running 14-day prediction (multi-output model)...")

        # Predict ONCE
        pred_scaled = model.predict(current_seq_scaled, verbose=0)[0]  # shape (14,)

        # Inverse scale correctly
        dummy = np.zeros((14, 5))
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
        # Calculate 52-week range manually (BEST METHOD)
        one_year_data = yf.download(ticker, period="1y", interval="1d")
        if isinstance(one_year_data.columns, pd.MultiIndex):
            one_year_data.columns = one_year_data.columns.get_level_values(0)
        year_low = safe_float(one_year_data["Low"].min())
        year_high = safe_float(one_year_data["High"].max())

        key_stats = {
            # Price data (from history)
            "open": safe_float(last_row["Open"]),
            "high": safe_float(last_row["High"]),
            "low": safe_float(last_row["Low"]),
            "close": safe_float(last_row["price"]),
            "volume": safe_int(last_row["Volume"]),
            "last_close": safe_float(last_row["price"]),
            # Company stats (from info)
            "market_cap": safe_int(info.get("marketCap")),
            "pe_ratio": safe_float(info.get("trailingPE")),
            "beta": safe_float(info.get("beta")),
            "eps_basic": safe_float(info.get("epsTrailingTwelveMonths")),
            "forward_pe": safe_float(info.get("forwardPE")),
            "dividend_yield": safe_float(info.get("dividendYield")),
            # 52 week range
            "days_range": f"{year_low:.2f} - {year_high:.2f}",
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
    
    