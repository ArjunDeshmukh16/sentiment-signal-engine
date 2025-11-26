from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator
from textblob import TextBlob
import altair as alt  # NEW

# =========================================================
# CONFIG: API KEY (for deployment use st.secrets)
# =========================================================
NEWSAPI_KEY = (
    st.secrets.get("NEWSAPI_KEY")
    or os.environ.get("NEWSAPI_KEY", "")
)
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Sentiment Signal Engine", layout="wide")
st.title("📈 Sentiment Signal Engine — Sector & Stock Scanner")
st.caption("Pick sectors or tickers, then click **Run Engine** in the sidebar.")

# =========================================================
# UNIVERSE: SECTORS → TICKERS
# =========================================================
SECTOR_UNIVERSE: Dict[str, List[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CSCO", "AMD", "CRM"],
    "Communication": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX"],
    "Consumer Staples": ["PEP", "KO", "WMT", "COST", "PG", "MO", "MDLZ"],
    "Financials": ["JPM", "BAC", "GS", "MS", "BLK", "AXP", "C"],
    "Healthcare": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO"],
    "Industrials": ["CAT", "HON", "BA", "UNP", "LMT", "GE", "DE"],
}

SECTOR_BY_TICKER: Dict[str, str] = {
    t: sector for sector, tickers in SECTOR_UNIVERSE.items() for t in tickers
}

LOOKBACK_DAYS_PRICE_DEFAULT = 900
NEWS_LOOKBACK_DAYS_DEFAULT = 3

DEFAULT_WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.30,
    "volume": 0.20,
    "sentiment": 0.20,
}

DEFAULT_THRESHOLDS = {
    "buy": 70.0,
    "hold": 45.0,
}

# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class NewsArticle:
    title: str
    description: str | None
    published_at: datetime
    url: str

# =========================================================
# DATA FUNCTIONS
# =========================================================
@st.cache_data(show_spinner=False)
def fetch_ohlcv(ticker: str, lookback_days: int) -> pd.DataFrame:
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days)

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise ValueError(f"No OHLCV data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]

    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False)
def fetch_news_for_ticker(ticker: str, lookback_days: int, api_key: str):
    if not api_key:
        return []

    params = {
        "q": f"{ticker} stock",
        "from": (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 50,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json().get("articles", [])
    except Exception:
        return []

    out = []
    for art in raw:
        try:
            published_at = datetime.fromisoformat(art["publishedAt"].replace("Z", "+00:00"))
            out.append(
                NewsArticle(
                    title=art.get("title", ""),
                    description=art.get("description"),
                    published_at=published_at,
                    url=art.get("url", ""),
                )
            )
        except:
            continue

    return out


def _decay_weight(timestamp: datetime, now=None, half_life_days=2.0):
    if now is None:
        now = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    delta_days = (now - timestamp).total_seconds() / 86400
    return float(np.exp(-np.log(2) * delta_days / half_life_days))


def score_headlines(headlines: List[NewsArticle], use_time_decay=True) -> float:
    if not headlines:
        return 0.0
    now = datetime.now(timezone.utc)
    scores, weights = [], []
    for art in headlines:
        txt = f"{art.title} {art.description or ''}".strip()
        if not txt:
            continue
        pol = float(TextBlob(txt).sentiment.polarity)
        scores.append(pol)
        weights.append(_decay_weight(art.published_at, now) if use_time_decay else 1.0)

    if not scores:
        return 0.0
    return float(np.clip(np.average(scores, weights=weights), -1, 1))


def add_technical_features(df):
    data = df.copy()
    data["ma20"] = data["close"].rolling(20).mean()
    data["ma50"] = data["close"].rolling(50).mean()
    data["vol20"] = data["volume"].rolling(20).mean()
    data["rsi14"] = RSIIndicator(close=data["close"], window=14).rsi()
    return data.dropna()

# =========================================================
# SCORING FUNCTIONS
# =========================================================
def trend_score(row):
    if row["ma50"] == 0:
        return 0.0
    return float(np.clip(((row["ma20"] / row["ma50"]) - 1) / 0.05, -1, 1))


def momentum_score(row):
    rsi = row["rsi14"]
    normalized = 1 - abs((rsi - 60) / 40)
    return float(np.clip((normalized * 2) - 1, -1, 1))


def volume_score(row):
    if row["vol20"] == 0:
        return 0.0
    return float(np.clip((row["volume"] / row["vol20"]) - 1, -1, 1))


def sentiment_component(val): 
    return float(np.clip(val, -1, 1))


def compute_final_score(comp, weights):
    raw = sum(comp[k] * weights.get(k, 0) for k in comp)
    raw = float(np.clip(raw, -1, 1))
    return float((raw + 1) * 50)


def classify_signal(score, thresholds):
    if score >= thresholds["buy"]:
        return "BUY"
    if score >= thresholds["hold"]:
        return "HOLD"
    return "SELL"

# =========================================================
# NEW: Compute historical signals over time
# =========================================================
def compute_signals_timeseries(df, weights, thresholds):
    data = add_technical_features(df)
    scores, signals = [], []

    for _, row in data.iterrows():
        comp = {
            "trend": trend_score(row),
            "momentum": momentum_score(row),
            "volume": volume_score(row),
            "sentiment": 0.0,
        }
        score = compute_final_score(comp, weights)
        signal = classify_signal(score, thresholds)
        scores.append(score)
        signals.append(signal)

    data["Score_0_100"] = scores
    data["Signal"] = signals
    return data

# =========================================================
# RANK UNIVERSE
# =========================================================
def score_universe(tickers, include_sentiment, lookback_days, news_days, weights, thresholds):
    rows = []
    for t in tickers:
        try:
            df = fetch_ohlcv(t, lookback_days)
            df = add_technical_features(df)
            last = df.iloc[-1]

            if include_sentiment and NEWSAPI_KEY:
                news = fetch_news_for_ticker(t, news_days, NEWSAPI_KEY)
                sent = score_headlines(news)
            else:
                sent = 0.0

            comp = {
                "trend": trend_score(last),
                "momentum": momentum_score(last),
                "volume": volume_score(last),
                "sentiment": sentiment_component(sent),
            }

            score = compute_final_score(comp, weights)
            signal = classify_signal(score, thresholds)

            rows.append({
                "Ticker": t,
                "Sector": SECTOR_BY_TICKER.get(t, "Custom"),
                "LastPrice": last["close"],
                "Trend": comp["trend"],
                "Momentum": comp["momentum"],
                "Volume": comp["volume"],
                "Sentiment": comp["sentiment"],
                "Score_0_100": score,
                "Signal": signal,
            })
        except Exception as exc:
            st.warning(f"Error scoring {t}: {exc}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("Score_0_100", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df[[
        "Rank", "Ticker", "Sector", "Score_0_100", "Signal",
        "Trend", "Momentum", "Volume", "Sentiment", "LastPrice"
    ]]

# =========================================================
# SIDEBAR UI
# =========================================================
with st.sidebar:
    st.header("Configuration")

    mode = st.radio("Universe mode", ["By Sector", "Custom tickers"])

    if mode == "By Sector":
        selected_sectors = st.multiselect(
            "Select sectors",
            list(SECTOR_UNIVERSE.keys()),
            default=["Technology", "Communication"]
        )
        tickers = sorted({t for s in selected_sectors for t in SECTOR_UNIVERSE[s]})
    else:
        raw = st.text_input("Enter tickers (comma-separated)", "AAPL, MSFT, NVDA")
        tickers = sorted({t.strip().upper() for t in raw.split(",") if t.strip()})

    lookback_price_days = st.slider("Price Lookback (days)", 200, 1200, LOOKBACK_DAYS_PRICE_DEFAULT, 50)
    news_lookback_days = st.slider("News Lookback (days)", 1, 14, NEWS_LOOKBACK_DAYS_DEFAULT)

    include_sentiment = st.checkbox("Include sentiment", value=True)

    buy_thr = st.slider("BUY threshold", 50.0, 90.0, DEFAULT_THRESHOLDS["buy"], 1.0)
    hold_thr = st.slider("HOLD threshold", 30.0, buy_thr - 1.0, DEFAULT_THRESHOLDS["hold"], 1.0)

    thresholds = {"buy": buy_thr, "hold": hold_thr}

    w_trend = st.slider("Trend weight", 0.0, 1.0, DEFAULT_WEIGHTS["trend"], 0.05)
    w_mom = st.slider("Momentum weight", 0.0, 1.0, DEFAULT_WEIGHTS["momentum"], 0.05)
    w_vol = st.slider("Volume weight", 0.0, 1.0, DEFAULT_WEIGHTS["volume"], 0.05)
    w_sent = st.slider("Sentiment weight", 0.0, 1.0, DEFAULT_WEIGHTS["sentiment"], 0.05)

    total = w_trend + w_mom + w_vol + w_sent
    weights = (
        DEFAULT_WEIGHTS if total == 0 else {
            "trend": w_trend / total,
            "momentum": w_mom / total,
            "volume": w_vol / total,
            "sentiment": w_sent / total,
        }
    )
    run_button = st.button("Run Engine")

# =========================================================
# MAIN LOGIC
# =========================================================
if run_button:
    if not tickers:
        st.warning("No tickers selected.")
        st.stop()

    if include_sentiment and not NEWSAPI_KEY:
        st.warning("No NEWSAPI_KEY found — sentiment disabled.")
        include_sentiment = False

    st.subheader("Universe Ranking")
    scores_df = score_universe(
        tickers, include_sentiment, lookback_price_days,
        news_lookback_days, weights, thresholds
    )
    st.dataframe(scores_df, use_container_width=True)

    if not scores_df.empty:
        st.subheader("Ticker Details")

        selected_ticker = st.selectbox("Select a ticker", scores_df["Ticker"].tolist())

        col_price, col_news = st.columns([2, 1])

        # -------- CHART WITH BUY/SELL SIGNALS -------- #
        with col_price:
            st.markdown(f"### {selected_ticker} — Price, Trend & Signals")

            df_price = fetch_ohlcv(selected_ticker, lookback_price_days)

            ts = compute_signals_timeseries(df_price, weights, thresholds)
            ts_reset = ts.reset_index().rename(columns={"index": "Date"})
            if "Date" not in ts_reset.columns:
                ts_reset = ts_reset.rename(columns={ts_reset.columns[0]: "Date"})

            price_line = (
                alt.Chart(ts_reset)
                .mark_line()
                .encode(
                    x="Date:T", y="close:Q",
                    tooltip=["Date:T", "close:Q", "Score_0_100:Q", "Signal:N"]
                )
            )

            ma20 = alt.Chart(ts_reset).mark_line(strokeDash=[4,2]).encode(x="Date:T", y="ma20:Q", color=alt.value("#FFA500"))
            ma50 = alt.Chart(ts_reset).mark_line(strokeDash=[2,2]).encode(x="Date:T", y="ma50:Q", color=alt.value("#808080"))

            signals = (
                alt.Chart(ts_reset)
                .mark_point(size=80, filled=True)
                .encode(
                    x="Date:T",
                    y="close:Q",
                    color=alt.Color(
                        "Signal:N",
                        scale=alt.Scale(
                            domain=["BUY", "HOLD", "SELL"],
                            range=["#16a34a", "#facc15", "#dc2626"]
                        )
                    ),
                    shape=alt.Shape(
                        "Signal:N",
                        scale=alt.Scale(
                            domain=["BUY", "HOLD", "SELL"],
                            range=["triangle-up", "circle", "triangle-down"]
                        )
                    ),
                    tooltip=["Date:T", "close:Q", "Score_0_100:Q", "Signal:N"]
                )
            )

            chart = alt.layer(price_line, ma20, ma50, signals).interactive().properties(height=400)
            st.altair_chart(chart, use_container_width=True)

        # -------- NEWS -------- #
        with col_news:
            st.markdown("### Recent Headlines")
            if include_sentiment and NEWSAPI_KEY:
                articles = fetch_news_for_ticker(selected_ticker, news_lookback_days, NEWSAPI_KEY)
                if not articles:
                    st.write("No recent news found.")
                else:
                    for art in articles[:8]:
                        st.markdown(f"- [{art.title}]({art.url})")
            else:
                st.write("Sentiment disabled or missing API key.")

else:
    st.info("Set configuration on the left and click **Run Engine**.")
