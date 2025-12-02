
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

# =========================================================
# NEWSAPI CONFIG (put near top of file, below NEWSAPI_ENDPOINT)
# =========================================================
NEWSAPI_DOMAINS = ",".join([
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "cnbc.com",
    "marketwatch.com",
    "seekingalpha.com",
    "fool.com",
    "barrons.com",
    "investing.com",
    "finance.yahoo.com",
    "fortune.com",
    "businessinsider.com",
    "nasdaq.com",
    "apnews.com",
])

NEWSAPI_SEARCH_IN = "title,description"
NEWSAPI_PAGE_SIZE = 40  # keep it modest to avoid rate limits
NEWSAPI_TIMEOUT = 8

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Generating Buy/Sell Trading Signals", layout="wide")

if "engine_ran" not in st.session_state:
    st.session_state["engine_ran"] = False
if "show_home" not in st.session_state:
    st.session_state["show_home"] = True


def _set_run_engine():
    st.session_state["engine_ran"] = True
    st.session_state["show_home"] = False


def _set_home():
    st.session_state["engine_ran"] = False
    st.session_state["show_home"] = True

# =========================
# BEAUTIFUL FRONT PAGE HERO
# =========================
st.markdown("""
<style>
.hero {
    padding: 40px;
    border-radius: 15px;
    background: linear-gradient(135deg, #2e2e3a 0%, #17171c 100%);
    border: 1px solid #3d3d4a;
    margin-bottom: 25px;
}
.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
}
.hero-sub {
    font-size: 18px;
    color: #cfcfcf;
    padding-top: 8px;
}
.info-box {
    padding: 18px;
    margin-top: 18px;
    background: rgba(79, 120, 203, 0.15);
    border-left: 4px solid #7493ff;
    border-radius: 8px;
    font-size: 16px;
    color: #e1e1e1;
}
</style>
""", unsafe_allow_html=True)

def render_home():
    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>Generating Buy/Sell Trading Signals from Structured Market Data</div>
        <div class='hero-sub'>
            A unified engine that fuses technical indicators, volume structure, and real-time sentiment into a single actionable score.
        </div>
        <div class='info-box'>
            <b>How to Use:</b><br>
            1. Select sectors or individual tickers in the sidebar.<br>
            2. Adjust lookback windows and factor weights.<br>
            3. Enable sentiment (requires API key).<br>
            4. Click <b>Run Engine</b> to generate rankings and signals.<br>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.title("Generating Buy/Sell Trading Signals from Structured Market Data")
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("Quick Start (Sidebar)")
        st.markdown(
            """
            - Select sectors or individual tickers.  
            - Set price/news lookback windows.  
            - Tune weights for Trend, Momentum, Volume, Sentiment.  
            - Toggle sentiment if you have a NewsAPI key.  
            - Click **Run Engine** to rank and score your universe.  
            """
        )
        st.markdown("Use the **Parameters** sidebar on the left to adjust these inputs.")

    with col_right:
        st.markdown(
            """
            ### How to Run the Signal Engine
            The interface mirrors a quant research workflow so you focus on insights, not configuration.

            **Step 1 — Select Your Universe**  
            - By Sector: scan all stocks in Technology, Communication, Financials, etc.  
            - By Script: manually pick tickers such as AAPL, NVDA, TSLA.  

            **Step 2 — Set Lookback Windows**  
            - Price lookback drives moving averages, RSI, volatility, and volume baselines.  
            - News lookback controls how many recent headlines shape sentiment.  

            **Step 3 — Configure Weights**  
            - Blend Trend, Momentum, Volume, Sentiment. Weights auto-normalize to sum to 1.  

            **Step 4 — Enable Sentiment (Optional)**  
            - Provide a NewsAPI key for real-time headlines; otherwise sentiment is neutral.  

            **Step 5 — Run the Engine**  
            - Click **Run Engine** to get rankings, ticker drill-downs, signals, RSI, volume, sentiment timeline, score breakdown, and sector heatmaps. Updates are instant when you tweak weights.  
            """
        )

        with st.expander("Usage Rules & Model Notes"):
            st.markdown(
                """
                - Research and educational analysis only; scores/signals are not investment recommendations.  
                - Sentiment can be noisy; headlines may be sarcastic or irrelevant even after smoothing.  
                - Technicals are backward-looking and cannot anticipate shocks.  
                - A BUY signal means "conditions appear favorable," not a guarantee of upside.  
                - Data quality matters; missing OHLCV or delayed news can affect stability.  
                - Adjust weights by sector; different industries react differently to sentiment vs structure.  
                - Thresholds (BUY > x, SELL < y) reflect risk appetite—tune in the sidebar.  
                """
            )

# =========================================================
# UNIVERSE: SECTORS → TICKERS
# =========================================================
SECTOR_UNIVERSE: Dict[str, List[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CSCO", "AMD", "CRM", "ORCL", "INTC", "QCOM", "TXN", "NOW", "PANW", "SNOW"],
    "Communication": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR", "EA", "ATVI", "RBLX", "SPOT"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "MELI", "ROST", "TJX", "GM", "F"],
    "Consumer Staples": ["PEP", "KO", "WMT", "COST", "PG", "MO", "MDLZ", "CL", "KMB", "KR", "ADM", "GIS", "HSY"],
    "Financials": ["JPM", "BAC", "GS", "MS", "BLK", "AXP", "C", "SCHW", "BK", "TFC", "USB", "PGR", "ICE"],
    "Healthcare": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ZTS", "BMY", "MDT", "ISRG", "REGN", "HUM"],
    "Industrials": ["CAT", "HON", "BA", "UNP", "LMT", "GE", "DE", "RTX", "GD", "ITW", "MMM", "ETN"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "HAL", "PSX"],
    "Materials": ["LIN", "APD", "SHW", "NEM", "FCX", "ECL", "NUE"],
    "Real Estate": ["PLD", "AMT", "CCI", "EQIX", "O", "SPG", "WELL"],
    "Utilities": ["NEE", "DUK", "SO", "AEP", "EXC", "XEL", "PEG"],
}

SECTOR_BY_TICKER: Dict[str, str] = {
    t: sector for sector, tickers in SECTOR_UNIVERSE.items() for t in tickers
}

TICKER_NAME: Dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "NVDA": "NVIDIA Corp.",
    "AVGO": "Broadcom Inc.",
    "ADBE": "Adobe Inc.",
    "CSCO": "Cisco Systems",
    "AMD": "Advanced Micro Devices",
    "CRM": "Salesforce Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms",
    "NFLX": "Netflix Inc.",
    "DIS": "Walt Disney Co.",
    "CMCSA": "Comcast Corp.",
    "TMUS": "T-Mobile US",
    "VZ": "Verizon Communications",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "HD": "Home Depot",
    "MCD": "McDonald's Corp.",
    "NKE": "Nike Inc.",
    "LOW": "Lowe's Cos.",
    "SBUX": "Starbucks Corp.",
    "PEP": "PepsiCo Inc.",
    "KO": "Coca-Cola Co.",
    "WMT": "Walmart Inc.",
    "COST": "Costco Wholesale",
    "PG": "Procter & Gamble",
    "MO": "Altria Group",
    "MDLZ": "Mondelez Intl.",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "BLK": "BlackRock Inc.",
    "AXP": "American Express",
    "C": "Citigroup Inc.",
    "UNH": "UnitedHealth Group",
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer Inc.",
    "ABBV": "AbbVie Inc.",
    "MRK": "Merck & Co.",
    "LLY": "Eli Lilly",
    "TMO": "Thermo Fisher Scientific",
    "CAT": "Caterpillar Inc.",
    "HON": "Honeywell Intl.",
    "BA": "Boeing Co.",
    "UNP": "Union Pacific",
    "LMT": "Lockheed Martin",
    "GE": "General Electric",
    "DE": "Deere & Co.",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron Corp.",
    "COP": "ConocoPhillips",
    "SLB": "Schlumberger",
    "EOG": "EOG Resources",
    "HAL": "Halliburton",
    "PSX": "Phillips 66",
    "LIN": "Linde plc",
    "APD": "Air Products & Chemicals",
    "SHW": "Sherwin-Williams",
    "NEM": "Newmont Corp.",
    "FCX": "Freeport-McMoRan",
    "ECL": "Ecolab Inc.",
    "NUE": "Nucor Corp.",
    "PLD": "Prologis",
    "AMT": "American Tower",
    "CCI": "Crown Castle",
    "EQIX": "Equinix",
    "O": "Realty Income",
    "SPG": "Simon Property Group",
    "WELL": "Welltower",
    "NEE": "NextEra Energy",
    "DUK": "Duke Energy",
    "SO": "Southern Co.",
    "AEP": "American Electric Power",
    "EXC": "Exelon Corp.",
    "XEL": "Xcel Energy",
    "PEG": "Public Service Enterprise Group",
    # Extras within existing sectors
    "ORCL": "Oracle Corp.",
    "INTC": "Intel Corp.",
    "QCOM": "Qualcomm",
    "TXN": "Texas Instruments",
    "NOW": "ServiceNow",
    "PANW": "Palo Alto Networks",
    "SNOW": "Snowflake",
    "T": "AT&T",
    "CHTR": "Charter Communications",
    "EA": "Electronic Arts",
    "ATVI": "Activision Blizzard",
    "RBLX": "Roblox",
    "SPOT": "Spotify",
    "BKNG": "Booking Holdings",
    "MELI": "MercadoLibre",
    "ROST": "Ross Stores",
    "TJX": "TJX Companies",
    "GM": "General Motors",
    "F": "Ford Motor",
    "CL": "Colgate-Palmolive",
    "KMB": "Kimberly-Clark",
    "KR": "Kroger Co.",
    "ADM": "Archer-Daniels-Midland",
    "GIS": "General Mills",
    "HSY": "Hershey Co.",
    "SCHW": "Charles Schwab",
    "BK": "Bank of New York Mellon",
    "TFC": "Truist Financial",
    "USB": "U.S. Bancorp",
    "PGR": "Progressive Corp.",
    "ICE": "Intercontinental Exchange",
    "RTX": "RTX Corp.",
    "GD": "General Dynamics",
    "ITW": "Illinois Tool Works",
    "MMM": "3M Company",
    "ETN": "Eaton Corp.",
    "ZTS": "Zoetis",
    "BMY": "Bristol-Myers Squibb",
    "MDT": "Medtronic",
    "ISRG": "Intuitive Surgical",
    "REGN": "Regeneron",
    "HUM": "Humana",
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
    """
    Fetch recent news for a ticker via NewsAPI.

    - Uses ticker + company name in query.
    - Restricts to major finance/business domains.
    - Silently falls back to [] on rate limit (429) or other errors
      instead of spamming the main page.
    """
    if not api_key:
        return []

    company = TICKER_NAME.get(ticker.upper(), "").strip()

    # Query: focus on finance context
    if company:
        q = f'("{ticker}" OR "{company}") AND (stock OR shares OR earnings OR company OR Inc OR Corp OR PLC)'
        q_in_title = f'"{ticker}" OR "{company}"'
    else:
        q = f'{ticker} AND (stock OR shares OR earnings)'
        q_in_title = ticker

    params = {
        "q": q,
        "qInTitle": q_in_title,
        "from": (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d"),
        "language": "en",
        "searchIn": NEWSAPI_SEARCH_IN,
        "sortBy": "publishedAt",
        "pageSize": NEWSAPI_PAGE_SIZE,
        "domains": NEWSAPI_DOMAINS,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=NEWSAPI_TIMEOUT)

        # Explicitly handle rate limiting
        if resp.status_code == 429:
            # Only show a single soft message per session, not one per ticker
            if not st.session_state.get("news_rate_limited", False):
                st.info(
                    "NewsAPI rate limit reached. "
                    "Sentiment will be set to neutral for remaining tickers."
                )
                st.session_state["news_rate_limited"] = True
            return []

        resp.raise_for_status()
        payload = resp.json()
        raw_articles = payload.get("articles", [])
    except Exception:
        # Swallow errors for UI cleanliness; just return no news = neutral sentiment
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
                    "Name": TICKER_NAME.get(t, t),
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
        "Rank", "Ticker", "Name", "Sector", "Score_0_100", "Signal",
        "Trend", "Momentum", "Volume", "Sentiment", "LastPrice",
    ]
    return df[cols]


# =========================================================
# SIDEBAR UI
# =========================================================
with st.sidebar:
    st.header("Parameters")

    mode = st.radio("Universe mode", ["By Sector", "Select Script"], index=1)

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

    thresholds = DEFAULT_THRESHOLDS

    st.subheader("Weights")
    w_trend = st.slider("Trend weight", 0.0, 1.0, DEFAULT_WEIGHTS["trend"], 0.05)
    w_mom = st.slider("Momentum weight", 0.0, 1.0, DEFAULT_WEIGHTS["momentum"], 0.05)
    w_vol = st.slider("Volume weight", 0.0, 1.0, DEFAULT_WEIGHTS["volume"], 0.05)
    w_sent = st.slider("Sentiment weight", 0.0, 1.0, DEFAULT_WEIGHTS["sentiment"], 0.05)

    total_w = w_trend + w_mom + w_vol + w_sent
    if total_w <= 0:
        weights = DEFAULT_WEIGHTS
    else:
        weights = {
            "trend": w_trend / total_w,
            "momentum": w_mom / total_w,
            "volume": w_vol / total_w,
            "sentiment": w_sent / total_w,
        }

    st.button("Run Engine", type="primary", on_click=_set_run_engine, key="run_engine_btn")
    st.button("Home", on_click=_set_home, key="home_btn")

# If on home view and engine not run, render landing content then stop
if st.session_state.get("show_home") and not st.session_state.get("engine_ran"):
    render_home()
    st.stop()


# =========================================================
# MAIN PAGE LOGIC
# =========================================================
engine_ran = st.session_state.get("engine_ran", False)
if engine_ran:
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
        # Compute shared data once
        df_price = fetch_ohlcv(selected_ticker, lookback_price_days)
        ts = compute_signals_timeseries(df_price, weights, thresholds)
        ts_reset = ts.reset_index().rename(columns={ts.reset_index().columns[0]: "Date"})
        ts_reset["Date"] = pd.to_datetime(ts_reset["Date"])

        news_articles: List[NewsArticle] = []
        if include_sentiment and NEWSAPI_KEY:
            news_articles = fetch_news_for_ticker(selected_ticker, news_lookback_days, NEWSAPI_KEY)

        tech_tab, sentiment_tab = st.tabs(["Technical Charts", "Sentiment & News"])

        with tech_tab:
        # === Price + Signals ===
            st.subheader(f"{selected_ticker} — Price, Trend & Signals")
            price_line = (
                alt.Chart(ts_reset)
                .mark_line(color="#4ade80", strokeWidth=3)
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("close:Q", title="Close"),
                )
            )

            ts_reset["signal_changed"] = ts_reset["Signal"].ne(ts_reset["Signal"].shift())
            markers = ts_reset[ts_reset["signal_changed"]]

            signal_points = (
                alt.Chart(markers)
                .mark_point(size=220, filled=True, stroke="black", strokeWidth=1.5)
                .encode(
                    x="Date:T",
                    y="close:Q",
                    color=alt.Color(
                        "Signal:N",
                        scale=alt.Scale(
                            domain=["BUY","HOLD","SELL"],
                            range=["#00cc44","#ffdd00","#ff3333"]
                        )
                    ),
                    shape=alt.Shape(
                        "Signal:N",
                        scale=alt.Scale(
                            domain=["BUY","HOLD","SELL"],
                            range=["triangle-up","circle","triangle-down"]
                        )
                    ),
                )
            )

            st.altair_chart(
                (price_line + signal_points).properties(height=360),
                use_container_width=True,
            )

            # === RSI ===
            st.subheader(f"{selected_ticker} — RSI Momentum")
            bands = pd.DataFrame({
                "level": [30, 70],
                "Label": ["Oversold (30)", "Overbought (70)"],
                "Color": ["#ef4444", "#fb923c"],
            })
            rsi_chart = (
                alt.Chart(ts_reset)
                .mark_line(color="#6366f1")
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("rsi14:Q", title="RSI(14)"),
                )
            )
            band_rules = (
                alt.Chart(bands)
                .mark_rule(strokeWidth=2)
                .encode(
                    y="level:Q",
                    color=alt.Color(
                        "Label:N",
                        scale=alt.Scale(
                            domain=list(bands["Label"]),
                            range=list(bands["Color"])
                        ),
                        legend=alt.Legend(title="RSI Zones"),
                    ),
                )
            )
            st.altair_chart((rsi_chart + band_rules).properties(height=240), use_container_width=True)

            # === Volume ===
            st.subheader(f"{selected_ticker} — Volume Structure")
            vol = alt.Chart(ts_reset).mark_bar(color="#60a5fa").encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("volume:Q", title="Volume"),
                tooltip=["Date:T","volume:Q"],
            )
            vol20 = alt.Chart(ts_reset).mark_line(color="orange", strokeWidth=3).encode(
                x="Date:T",
                y=alt.Y("vol20:Q", title="20D MA"),
            )
            st.altair_chart((vol + vol20).properties(height=240), use_container_width=True)

        with sentiment_tab:
            st.subheader(f"{selected_ticker} — News Sentiment Timeline")

            if include_sentiment and NEWSAPI_KEY:
                if not news_articles:
                    st.info("No sentiment data")
                else:
                    rows = []
                    for a in news_articles:
                        pol = TextBlob(f"{a.title} {a.description or ''}").sentiment.polarity
                        rows.append({
                            "date": a.published_at.date(),
                            "sentiment": pol,
                            "decayed": pol * _decay_weight(a.published_at),
                        })
                    sdf = (
                        pd.DataFrame(rows)
                        .groupby("date")
                        .agg(
                            sentiment=("sentiment", "mean"),
                            decayed=("decayed", "mean"),
                            headlines=("sentiment", "size"),
                        )
                        .reset_index()
                    )

                    sdf_melt = sdf.melt("date", ["sentiment", "decayed"], var_name="Series", value_name="Value")
                    sent_lines = (
                        alt.Chart(sdf_melt)
                        .mark_line(strokeWidth=3)
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("Value:Q", title="Sentiment"),
                            color=alt.Color(
                                "Series:N",
                                scale=alt.Scale(domain=["sentiment","decayed"], range=["#22c55e","orange"]),
                                legend=alt.Legend(title="Series"),
                            ),
                            tooltip=["date:T","Series:N","Value:Q"],
                        )
                    )
                    neutral = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9ca3af", strokeDash=[4,4]).encode(y="y:Q")
                    headline_bars = (
                        alt.Chart(sdf)
                        .mark_bar(color="#60a5fa", opacity=0.35)
                        .encode(
                            x="date:T",
                            y=alt.Y("headlines:Q", title="# Headlines"),
                            tooltip=["date:T","headlines:Q"],
                        )
                    )
                    st.altair_chart(
                        alt.layer(headline_bars, neutral, sent_lines)
                        .resolve_scale(y="independent")
                        .properties(height=260),
                        use_container_width=True,
                    )

                    st.markdown("**Latest headlines**")
                    sorted_news = sorted(news_articles, key=lambda a: a.published_at, reverse=True)[:10]
                    for art in sorted_news:
                        pol = TextBlob(f"{art.title} {art.description or ''}").sentiment.polarity
                        published = art.published_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                        st.markdown(
                            f"- [{art.title}]({art.url}) — {published} — sentiment {pol:+.2f}"
                        )
            else:
                st.info("Sentiment disabled or no API key")

        # === Score Breakdown ===
        st.subheader(f"{selected_ticker} — Component Score Breakdown")

        tech = add_technical_features(df_price).iloc[-1]
        sent_val = 0.0
        if include_sentiment and NEWSAPI_KEY and news_articles:
            sent_val = score_headlines(news_articles)

        comp = {
            "Trend": trend_score(tech),
            "Momentum": momentum_score(tech),
            "Volume": volume_score(tech),
            "Sentiment": sentiment_component(sent_val),
        }

        comp_df = pd.DataFrame({
            "Component": list(comp.keys()),
            "Score": list(comp.values()),
            "Weight": [weights[c.lower()] for c in comp.keys()],
        })
        comp_df["Weighted"] = comp_df["Score"] * comp_df["Weight"]

        chart = alt.Chart(comp_df).mark_bar().encode(
            x="Component:N", y="Score:Q", color="Component:N"
        )

        st.altair_chart(chart, use_container_width=True)
        st.dataframe(comp_df)

        # === Sector Heatmap ===
        st.subheader("Sector Heatmap — Composite Score")
        heat_df = scores_df[["Ticker","Sector","Score_0_100","Signal"]]
        heat = (
            alt.Chart(heat_df)
            .mark_rect(stroke="white", strokeWidth=0.5)
            .encode(
                x="Sector:N",
                y="Ticker:N",
                color=alt.Color("Score_0_100:Q", scale=alt.Scale(scheme="redyellowgreen"), legend=alt.Legend(title="Score")),
                tooltip=["Sector","Ticker","Score_0_100","Signal"]
            )
            .properties(height=300)
        )
        heat_text = (
            alt.Chart(heat_df)
            .mark_text(fontSize=11, fontWeight="bold")
            .encode(
                x="Sector:N",
                y="Ticker:N",
                text=alt.Text("Score_0_100:Q", format=".0f"),
                color=alt.value("black"),
            )
        )
        st.altair_chart(heat + heat_text, use_container_width=True)
else:
    st.info("Configure your universe in the sidebar and click **Run Engine**.")
