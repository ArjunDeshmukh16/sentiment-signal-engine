```python
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
import altair as alt

# =========================================================
# CONFIG: API KEY (for deployment use st.secrets)
# =========================================================
try:
    NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", default="")
except (KeyError, AttributeError, Exception):
    NEWSAPI_KEY = ""

NEWSAPI_KEY = NEWSAPI_KEY or os.environ.get("NEWSAPI_KEY", "")
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
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
    """Fetch OHLCV data via yfinance."""
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

    # Flatten multi-index if present and lowercase columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]

    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False)
def fetch_news_for_ticker(
    ticker: str,
    lookback_days: int,
    api_key: str,
) -> List[NewsArticle]:
    """Fetch recent news for a ticker via NewsAPI."""
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
        payload = resp.json()
        raw_articles = payload.get("articles", [])
    except Exception as exc:
        st.warning(f"[NEWS] Failed for {ticker}: {exc}")
        return []

    out: List[NewsArticle] = []
    for art in raw_articles:
        try:
            published_raw = art.get("publishedAt", "")
            if not published_raw:
                continue
            published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            out.append(
                NewsArticle(
                    title=art.get("title") or "",
                    description=art.get("description"),
                    published_at=published_at,
                    url=art.get("url") or "",
                )
            )
        except Exception:
            continue

    return out


def _decay_weight(
    timestamp: datetime,
    now: datetime | None = None,
    half_life_days: float = 2.0,
) -> float:
    """Exponential time-decay weight for sentiment."""
    if now is None:
        now = datetime.now(timezone.utc)

    if timestamp.tzinfo is None:
        ts_utc = timestamp.replace(tzinfo=timezone.utc)
    else:
        ts_utc = timestamp.astimezone(timezone.utc)

    delta_days = (now - ts_utc).total_seconds() / 86400
    return float(np.exp(-np.log(2) * delta_days / half_life_days))


def score_headlines(
    headlines: List[NewsArticle],
    use_time_decay: bool = True,
) -> float:
    """Aggregate sentiment for a list of headlines into [-1, 1]."""
    if not headlines:
        return 0.0

    now = datetime.now(timezone.utc)
    scores: List[float] = []
    weights: List[float] = []

    for art in headlines:
        text = f"{art.title} {art.description or ''}".strip()
        if not text:
            continue
        pol = float(TextBlob(text).sentiment.polarity)
        scores.append(pol)
        weights.append(_decay_weight(art.published_at, now) if use_time_decay else 1.0)

    if not scores:
        return 0.0

    avg = float(np.average(scores, weights=weights))
    return float(np.clip(avg, -1.0, 1.0))


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA20, MA50, RSI14, Vol20 to OHLCV data."""
    data = df.copy()
    data.columns = [c.lower() for c in data.columns]

    required = {"close", "volume"}
    if not required.issubset(set(data.columns)):
        raise ValueError("Missing required columns in OHLCV data")

    data["ma20"] = data["close"].rolling(20).mean()
    data["ma50"] = data["close"].rolling(50).mean()
    data["vol20"] = data["volume"].rolling(20).mean()

    rsi = RSIIndicator(close=data["close"], window=14)
    data["rsi14"] = rsi.rsi()

    return data.dropna()


# =========================================================
# SCORING FUNCTIONS
# =========================================================
def trend_score(row: pd.Series) -> float:
    ma20, ma50 = row["ma20"], row["ma50"]
    if ma50 == 0:
        return 0.0
    return float(np.clip(((ma20 / ma50) - 1) / 0.05, -1, 1))


def momentum_score(row: pd.Series) -> float:
    rsi = row["rsi14"]
    normalized = 1 - abs((rsi - 60) / 40)  # preference around 60
    score = (normalized * 2) - 1
    return float(np.clip(score, -1, 1))


def volume_score(row: pd.Series) -> float:
    vol, vol20 = row["volume"], row["vol20"]
    if vol20 == 0:
        return 0.0
    return float(np.clip((vol / vol20) - 1, -1, 1))


def sentiment_component(value: float) -> float:
    return float(np.clip(value, -1, 1))


def compute_final_score(
    components: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Combine components into final [0, 100] score."""
    raw = sum(float(components[k]) * weights.get(k, 0.0) for k in components)
    raw = float(np.clip(raw, -1, 1))
    return float(np.clip((raw + 1) * 50, 0, 100))


def classify_signal(score: float, thresholds: Dict[str, float]) -> str:
    """Map score → BUY/HOLD/SELL."""
    if score >= thresholds["buy"]:
        return "BUY"
    if score >= thresholds["hold"]:
        return "HOLD"
    return "SELL"


# =========================================================
# NEW: HISTORICAL SIGNALS TIMESERIES
# =========================================================
def compute_signals_timeseries(
    df: pd.DataFrame,
    weights: Dict[str, float],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    """
    Take raw OHLCV, add technicals, and compute Score_0_100 + Signal
    for each date (historical BUY/HOLD/SELL).
    Sentiment is set to 0 here because we don't have a daily time series
    of sentiment; we're focusing on technicals for the historical signals.
    """
    data = add_technical_features(df)

    scores: List[float] = []
    signals: List[str] = []

    for _, row in data.iterrows():
        components = {
            "trend": trend_score(row),
            "momentum": momentum_score(row),
            "volume": volume_score(row),
            "sentiment": 0.0,  # neutral daily sentiment for back-history
        }
        score = compute_final_score(components, weights)
        signal = classify_signal(score, thresholds)
        scores.append(score)
        signals.append(signal)

    data["Score_0_100"] = scores
    data["Signal"] = signals

    return data


# =========================================================
# UNIVERSE RANKING
# =========================================================
def score_universe(
    tickers: List[str],
    include_sentiment: bool,
    lookback_price_days: int,
    news_lookback_days: int,
    weights: Dict[str, float],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    rows = []

    for t in tickers:
        try:
            df = fetch_ohlcv(t, lookback_price_days)
            df = add_technical_features(df)
            last = df.iloc[-1]

            if include_sentiment and NEWSAPI_KEY:
                news = fetch_news_for_ticker(t, news_lookback_days, NEWSAPI_KEY)
                sent_val = score_headlines(news)
            else:
                sent_val = 0.0

            comp = {
                "trend": trend_score(last),
                "momentum": momentum_score(last),
                "volume": volume_score(last),
                "sentiment": sentiment_component(sent_val),
            }

            score = compute_final_score(comp, weights)
            signal = classify_signal(score, thresholds)

            rows.append(
                {
                    "Ticker": t,
                    "Sector": SECTOR_BY_TICKER.get(t, "Custom"),
                    "LastPrice": last["close"],
                    "Trend": comp["trend"],
                    "Momentum": comp["momentum"],
                    "Volume": comp["volume"],
                    "Sentiment": comp["sentiment"],
                    "Score_0_100": score,
                    "Signal": signal,
                }
            )
        except Exception as exc:
            st.warning(f"[RANK] Error for {t}: {exc}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("Score_0_100", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    cols = [
        "Rank", "Ticker", "Sector", "Score_0_100", "Signal",
        "Trend", "Momentum", "Volume", "Sentiment", "LastPrice",
    ]
    return df[cols]


# =========================================================
# SIDEBAR UI
# =========================================================
with st.sidebar:
    st.header("Configuration")

    mode = st.radio("Universe mode", ["By Sector", "Select Script"])

    if mode == "By Sector":
        selected_sectors = st.multiselect(
            "Select sectors",
            options=list(SECTOR_UNIVERSE.keys()),
            default=["Technology", "Communication"],
        )
        tickers = sorted({t for s in selected_sectors for t in SECTOR_UNIVERSE[s]})
    else:
        # Get all available tickers from the universe
        all_tickers = sorted({t for tickers_list in SECTOR_UNIVERSE.values() for t in tickers_list})
        selected_tickers = st.multiselect(
            "Select script",
            options=all_tickers,
            default=["AAPL", "MSFT", "NVDA"],
        )
        tickers = selected_tickers

    st.markdown(f"**Universe size:** {len(tickers)} tickers")

    lookback_price_days = st.slider(
        "Price lookback window (days)",
        min_value=200,
        max_value=1200,
        value=LOOKBACK_DAYS_PRICE_DEFAULT,
        step=50,
    )
    news_lookback_days = st.slider(
        "News lookback (days)",
        min_value=1,
        max_value=14,
        value=NEWS_LOOKBACK_DAYS_DEFAULT,
    )

    include_sentiment = st.checkbox("Include sentiment", value=True)

    buy_thr = st.slider("BUY threshold", 50.0, 90.0, DEFAULT_THRESHOLDS["buy"], 1.0)
    hold_thr = st.slider("HOLD threshold", 30.0, buy_thr - 1.0, DEFAULT_THRESHOLDS["hold"], 1.0)

    thresholds = {"buy": buy_thr, "hold": hold_thr}

    st.subheader("Weights")
    w_trend = st.slider("Trend weight", 0.0, 1.0, DEFAULT_WEIGHTS["trend"], 0.05)
    w_mom = st.slider("Momentum weight", 0.0, 1.0, DEFAULT_WEIGHTS["momentum"], 0.05)
    w_vol = st.slider("Volume weight", 0.0, 1.0, DEFAULT_WEIGHTS["volume"], 0.05)
    w_sent = st.slider("Sentiment weight", 0.0, 1.0, DEFAULT_WEIGHTS["sentiment"], 0.05)

    total_w = w_trend + w_mom + w_vol + w_sent
    if total_w == 0:
        weights = DEFAULT_WEIGHTS
    else:
        weights = {
            "trend": w_trend / total_w,
            "momentum": w_mom / total_w,
            "volume": w_vol / total_w,
            "sentiment": w_sent / total_w,
        }

    run_button = st.button("Run Engine")


# =========================================================
# MAIN PAGE LOGIC
# =========================================================
if run_button:
    if not tickers:
        st.warning("No tickers selected.")
        st.stop()

    if include_sentiment and not NEWSAPI_KEY:
        st.warning("No NEWSAPI_KEY set. Sentiment will be neutral.")
        include_sentiment = False

    st.subheader("Universe Ranking")
    scores_df = score_universe(
        tickers=tickers,
        include_sentiment=include_sentiment,
        lookback_price_days=lookback_price_days,
        news_lookback_days=news_lookback_days,
        weights=weights,
        thresholds=thresholds,
    )
    st.dataframe(scores_df, use_container_width=True)

    if not scores_df.empty:
        st.subheader("Ticker Details")

        selected_ticker = st.selectbox("Select a ticker", scores_df["Ticker"].tolist())
        if selected_ticker:
            col_price, col_news = st.columns([2, 1])

            # ================== PRICE + SIGNALS CHART ================== #
            with col_price:
                st.markdown(f"### {selected_ticker} — Price, Trend & Signals")

                df_price = fetch_ohlcv(selected_ticker, lookback_price_days)

                # Compute historical signals using your scoring engine (technicals only)
                ts = compute_signals_timeseries(df_price, weights, thresholds)

                # Reset index so Altair can use "Date" column
                ts_reset = ts.reset_index()
                # Rename the first column (which is the index) to "Date"
                if ts_reset.columns[0] != "Date":
                    ts_reset = ts_reset.rename(columns={ts_reset.columns[0]: "Date"})

                # Ensure Date is datetime
                ts_reset["Date"] = pd.to_datetime(ts_reset["Date"])

                # Show only signal transitions (when signal changes) for a clean look
                ts_reset["signal_changed"] = ts_reset["Signal"].ne(ts_reset["Signal"].shift())
                signal_markers_df = ts_reset[ts_reset["signal_changed"]].copy()
                
                # Limit to max 1-2 instances of each signal type for minimal clutter
                signal_markers_limited = []
                for signal_type in ["BUY", "SELL", "HOLD"]:
                    signal_rows = signal_markers_df[signal_markers_df["Signal"] == signal_type]
                    if len(signal_rows) > 0:
                        # Keep first occurrence and optionally the last one if there are 3+ occurrences
                        if len(signal_rows) == 1:
                            signal_markers_limited.append(signal_rows.iloc[[0]])
                        elif len(signal_rows) == 2:
                            signal_markers_limited.append(signal_rows.iloc[[0]])
                            signal_markers_limited.append(signal_rows.iloc[[1]])
                        else:  # 3 or more occurrences
                            signal_markers_limited.append(signal_rows.iloc[[0]])
                            signal_markers_limited.append(signal_rows.iloc[[-1]])
                
                if signal_markers_limited:
                    signal_markers_df = pd.concat(signal_markers_limited, ignore_index=False).sort_index()
                else:
                    # Fallback: show first point if no transitions
                    signal_markers_df = ts_reset.iloc[[0]].copy()



                # MAIN PRICE LINE
                price_line = (
                    alt.Chart(ts_reset)
                    .mark_line(color="#4ade80", strokeWidth=3)
                    .encode(
                        x=alt.X("Date:T", title="Date"),
                        y=alt.Y("close:Q", title="Price ($)"),
                    )
                )

                # ENHANCED SIGNAL MARKERS - Much larger and more visible
                signal_points = (
                    alt.Chart(signal_markers_df)
                    .mark_point(filled=True, size=400, opacity=1.0, stroke="black", strokeWidth=2)
                    .encode(
                        x="Date:T",
                        y="close:Q",
                        color=alt.Color(
                            "Signal:N",
                            scale=alt.Scale(
                                domain=["BUY", "HOLD", "SELL"],
                                range=["#00cc44", "#ffdd00", "#ff3333"],
                            ),
                            legend=alt.Legend(title="Signal Type", labelFontSize=12, titleFontSize=14),
                        ),
                        shape=alt.Shape(
                            "Signal:N",
                            scale=alt.Scale(
                                domain=["BUY", "HOLD", "SELL"],
                                range=["triangle-up", "circle", "triangle-down"],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("Date:T", format="%Y-%m-%d"),
                            alt.Tooltip("close:Q", format="$,.2f", title="Entry Price"),
                            alt.Tooltip("Score_0_100:Q", format=".1f", title="Signal Score"),
                            alt.Tooltip("Signal:N", title="Signal Type"),
                            alt.Tooltip("rsi14:Q", format=".1f", title="RSI 14"),
                            alt.Tooltip("vol20:Q", format=",.0f", title="Vol 20MA"),
                        ],
                    )
                )

                # TEXT LABELS showing entry prices on chart
                signal_labels = (
                    alt.Chart(signal_markers_df)
                    .mark_text(align="center", baseline="bottom", fontSize=11, fontWeight="bold", dy=-15)
                    .encode(
                        x="Date:T",
                        y="close:Q",
                        text=alt.Text("close:Q", format="$,.0f"),
                        color=alt.value("black"),
                    )
                )

                chart = (
                    alt.layer(price_line, signal_points, signal_labels)
                    .interactive()
                    .properties(height=480, title=f"{selected_ticker} Price Action with Signal Entry Points")
                    .resolve_scale(y='shared')
                )
                st.altair_chart(chart, use_container_width=True)

            # ================== NEWS ================== #
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
                    st.write("Sentiment disabled or no API key configured.")
else:
    st.info("Configure your universe in the sidebar and click **Run Engine**.")
```
