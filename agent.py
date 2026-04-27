"""
🔥 SMART USD & GOLD AGENT v8 (PRO TRADER EDITION)
================================================
- SCALP signal (short-term 5-30min)
- TREND signal (long-term days/weeks)
- Support & Resistance zones (auto-detected)
- RSI, Moving Averages, Pivot Points
- ATR-based Stop Loss / Take Profit
- Risk/Reward auto-calculation
- Multi-source price fetching with fallback
- Confidence max 80%
"""

import os
import sys
import time
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from typing import List, Optional, Tuple

import feedparser
import requests
from dotenv import load_dotenv


# ================= CONFIG =================

MAX_NEWS_PER_SOURCE = 3
NEWS_CACHE_SIZE = 200
SIGNAL_CHANGE_THRESHOLD = 2
HOURLY_INTERVAL = 1
REQUEST_TIMEOUT = 10
MAX_RETRIES = 2

# Confidence
CONFIDENCE_MAX = 80
CONFIDENCE_MULTIPLIER = 13

# Time windows (seconds)
SCALP_WINDOW = 1800
SHORT_WINDOW = 7200
TREND_WINDOW = 14400

# Technical analysis
SR_LOOKBACK_DAYS = 30      # Days for S/R detection
SR_TOLERANCE = 0.003       # 0.3% tolerance for grouping levels
RSI_PERIOD = 14
ATR_PERIOD = 14
MA_SHORT = 20
MA_LONG = 50

# Risk management
RISK_REWARD_TARGET = 2.0   # Aim for 2:1 R/R minimum
SL_ATR_MULTIPLIER = 1.5    # Stop loss = 1.5 * ATR
TP_ATR_MULTIPLIER = 3.0    # Take profit = 3 * ATR


# ================= LOGGING =================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("SmartAgent")


# ================= ENV =================

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    log.error("Missing BOT_TOKEN or CHAT_ID in .env")
    sys.exit(1)


# ================= SOURCES =================

RSS_SOURCES = {
    "Reuters": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    "Investing": "https://www.investing.com/rss/news_25.rss",
    "FXStreet": "https://www.fxstreet.com/rss/news",
    "Investing-Forex": "https://www.investing.com/rss/news_1.rss",
    "Investing-Commodities": "https://www.investing.com/rss/news_11.rss",
    "FXStreet-Analysis": "https://www.fxstreet.com/rss/analysis",
}

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ================= RULES =================

RULES = {
    "rate hike": {"usd": 3, "gold": -2},
    "hawkish": {"usd": 2, "gold": -1},
    "strong jobs": {"usd": 2, "gold": -1},
    "rate cut": {"usd": -3, "gold": 2},
    "dovish": {"usd": -2, "gold": 2},
    "recession": {"usd": -3, "gold": 3},
    "crisis": {"usd": -2, "gold": 3},
    "war": {"usd": -1, "gold": 3},
    "oil": {"usd": -1, "gold": 3},
    "fed": {"usd": 2, "gold": -1},
    "powell": {"usd": 2, "gold": -1},
    "safe haven": {"usd": -1, "gold": 3},
    "inflation": {"usd": 2, "gold": 2},
    "geopolitical": {"usd": -1, "gold": 3},
    "tariff": {"usd": 1, "gold": 2},
    "yields": {"usd": 2, "gold": -2},
}

SCALP_KEYWORDS = [
    "breaking", "just in", "urgent", "alert", "flash",
    "surges", "plunges", "crashes", "spikes", "tumbles",
    "soars", "slumps", "jumps", "falls sharply",
    "emergency", "unexpected", "shock",
]

TREND_KEYWORDS = [
    "outlook", "forecast", "long-term", "structural",
    "trend", "cycle", "policy", "strategy",
    "sustained", "continued", "persistent",
]


# ================= STATE =================

last_state = None
last_usd_score = 0
last_gold_score = 0
seen_news = deque(maxlen=NEWS_CACHE_SIZE)


# ================= MODELS =================

@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    age_seconds: int = 0

    def hash(self):
        raw = (self.title + self.summary).lower()
        return hashlib.md5(raw.encode()).hexdigest()

    def is_scalp(self):
        text = (self.title + " " + self.summary).lower()
        if self.age_seconds < SCALP_WINDOW:
            return True
        if any(k in text for k in SCALP_KEYWORDS):
            return True
        return False

    def is_trend(self):
        text = (self.title + " " + self.summary).lower()
        return any(k in text for k in TREND_KEYWORDS)


@dataclass
class TechnicalAnalysis:
    """Holds all technical indicators for a symbol"""
    symbol: str
    current_price: float = 0.0
    rsi: float = 50.0
    ma20: float = 0.0
    ma50: float = 0.0
    atr: float = 0.0
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    pivot: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    rsi_signal: str = "Neutral"
    ma_signal: str = "Neutral"
    is_valid: bool = False


# ================= HELPERS =================

def safe_get(url, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception as e:
            log.warning(f"Request failed (attempt {attempt+1}): {e}")
            if attempt < retries:
                time.sleep(1)
    return None


def get_age(entry):
    try:
        published = entry.get("published_parsed")
        if not published:
            return 999999
        news_time = datetime.fromtimestamp(time.mktime(published))
        return int((datetime.now() - news_time).total_seconds())
    except Exception:
        return 999999


def is_recent(age_seconds):
    return age_seconds < TREND_WINDOW


def classify_news(text):
    text = text.lower()
    analysis_words = ["forecast", "outlook", "analysis", "expects", "could", "may", "risks"]
    breaking_words = ["announces", "confirms", "says", "reports", "cancels", "halts"]
    if any(w in text for w in breaking_words):
        return "NEWS"
    if any(w in text for w in analysis_words):
        return "ANALYSIS"
    return "NEWS"


def get_confirmation(news_list):
    counts = {}
    for n in news_list:
        key = n.title[:60].lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


# ================= FETCH NEWS =================

def fetch_news():
    news_list = []

    for name, url in RSS_SOURCES.items():
        try:
            feed = feedparser.parse(url, request_headers=HTTP_HEADERS)
            if not feed.entries:
                r = safe_get(url)
                if r:
                    feed = feedparser.parse(r.content)
            if not feed.entries:
                continue

            count = 0
            for entry in feed.entries[:MAX_NEWS_PER_SOURCE]:
                age = get_age(entry)
                if not is_recent(age):
                    continue

                item = NewsItem(
                    title=entry.get("title", ""),
                    summary=entry.get("summary", ""),
                    source=name,
                    age_seconds=age,
                )

                if not item.title or item.hash() in seen_news:
                    continue

                seen_news.append(item.hash())
                news_list.append(item)
                count += 1

            log.info(f"{name}: {count} new items")
        except Exception as e:
            log.warning(f"Failed {name}: {e}")
            continue

    log.info(f"Total fresh news: {len(news_list)}")
    return news_list


# ================= NEWS ANALYSIS =================

def analyze(news_list):
    scalp_usd = 0
    scalp_gold = 0
    trend_usd = 0
    trend_gold = 0
    impactful = []

    confirmation = get_confirmation(news_list)

    for news in news_list:
        text = (news.title + " " + news.summary).lower()

        usd_score = 0
        gold_score = 0
        triggers = []

        for k, v in RULES.items():
            if k in text:
                usd_score += v["usd"]
                gold_score += v["gold"]
                triggers.append(k)

        news_type = classify_news(text)

        if news_type == "ANALYSIS":
            usd_score *= 0.5
            gold_score *= 0.5

        if confirmation.get(news.title[:60].lower(), 0) >= 2:
            usd_score *= 1.5
            gold_score *= 1.5

        is_scalp = news.is_scalp()
        is_trend = news.is_trend()

        if is_scalp:
            multiplier = 1.3 if news.age_seconds < SCALP_WINDOW else 1.0
            scalp_usd += usd_score * multiplier
            scalp_gold += gold_score * multiplier

        if is_trend or news.age_seconds > SHORT_WINDOW:
            trend_usd += usd_score * 0.8
            trend_gold += gold_score * 0.8

        if not is_scalp and not is_trend:
            scalp_usd += usd_score * 0.5
            scalp_gold += gold_score * 0.5

        if abs(usd_score) >= 2 or abs(gold_score) >= 2:
            tag = "🔴 SCALP" if is_scalp else ("🟢 TREND" if is_trend else "📰 NEWS")
            age_min = news.age_seconds // 60
            impactful.append(
                f"{tag} [{news.source}] ({age_min}min ago)\n"
                f"{news.title}\n"
                f"USD:{round(usd_score,1)} | Gold:{round(gold_score,1)}\n"
                f"Drivers: {', '.join(triggers)}"
            )

    return (
        round(scalp_usd, 1),
        round(scalp_gold, 1),
        round(trend_usd, 1),
        round(trend_gold, 1),
        impactful[:3],
    )


# ================= SIGNAL =================

def signal(score):
    if score >= 4:
        return "Strong Bullish"
    if score >= 1:
        return "Bullish"
    if score <= -4:
        return "Strong Bearish"
    if score <= -1:
        return "Bearish"
    return "Neutral"


def confidence(score):
    return min(int(abs(score) * CONFIDENCE_MULTIPLIER), CONFIDENCE_MAX)


def entry_bias(usd, gold):
    if usd > gold and usd > 0:
        return "Buy USD"
    if gold > usd and gold > 0:
        return "Buy Gold"
    if usd < 0 and gold < 0:
        return "No Clear Advantage"
    return "Neutral"


# ================= PRICE FETCHING =================

def get_price_yahoo(symbol, range_period="1d"):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_period}&interval=1d"
        r = safe_get(url)
        if not r:
            return None, None
        data = r.json()["chart"]["result"][0]
        price = data["meta"]["regularMarketPrice"]
        prev = data["meta"]["chartPreviousClose"]
        change = ((price - prev) / prev) * 100
        return round(price, 2), round(change, 2)
    except Exception as e:
        log.warning(f"Yahoo failed for {symbol}: {e}")
        return None, None


def get_price_stooq(symbol):
    try:
        url = f"https://stooq.com/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv"
        r = safe_get(url)
        if not r:
            return None, None
        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            return None, None
        fields = lines[1].split(",")
        if len(fields) < 7:
            return None, None
        open_p = float(fields[3])
        close_p = float(fields[6])
        if open_p == 0:
            return round(close_p, 2), 0.0
        change = ((close_p - open_p) / open_p) * 100
        return round(close_p, 2), round(change, 2)
    except Exception as e:
        log.warning(f"Stooq failed for {symbol}: {e}")
        return None, None


def get_gold_price_api():
    try:
        url = "https://api.gold-api.com/price/XAU"
        r = safe_get(url)
        if not r:
            return None, None
        data = r.json()
        price = data.get("price")
        if not price:
            return None, None
        return round(float(price), 2), None
    except Exception as e:
        log.warning(f"gold-api failed: {e}")
        return None, None


def get_dxy_price():
    price, change = get_price_yahoo("DX-Y.NYB")
    if price:
        return price, change
    price, change = get_price_yahoo("DX=F")
    if price:
        return price, change
    price, change = get_price_stooq("dx.f")
    if price:
        return price, change
    return None, None


def get_gold_price():
    price, change = get_price_yahoo("XAUUSD=X")
    if price:
        return price, change
    price, change = get_gold_price_api()
    if price:
        return price, change
    price, change = get_price_stooq("xauusd")
    if price:
        return price, change
    price, change = get_price_yahoo("GC=F")
    if price:
        return price, change
    return None, None


# ================= HISTORICAL DATA =================

def get_historical_yahoo(symbol, days=SR_LOOKBACK_DAYS):
    """Try Yahoo Finance for historical OHLC"""
    try:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?range={days}d&interval=1d"
        )
        r = safe_get(url)
        if not r:
            return None

        data = r.json()["chart"]["result"][0]
        quote = data["indicators"]["quote"][0]

        result = {
            "highs": [v for v in quote.get("high", []) if v is not None],
            "lows": [v for v in quote.get("low", []) if v is not None],
            "closes": [v for v in quote.get("close", []) if v is not None],
            "opens": [v for v in quote.get("open", []) if v is not None],
        }

        if len(result["closes"]) < 5:
            return None

        return result
    except Exception as e:
        log.warning(f"Yahoo historical failed for {symbol}: {e}")
        return None


def get_historical_stooq(symbol, days=SR_LOOKBACK_DAYS):
    """
    Stooq CSV historical OHLC (free, no API key).
    Symbol examples: xauusd, dx.f
    """
    try:
        from datetime import timedelta
        end = datetime.now()
        start = end - timedelta(days=days + 10)  # buffer for weekends

        d1 = start.strftime("%Y%m%d")
        d2 = end.strftime("%Y%m%d")

        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d&d1={d1}&d2={d2}"
        r = safe_get(url)
        if not r or not r.text:
            return None

        lines = r.text.strip().split("\n")
        if len(lines) < 5:  # Header + at least few rows
            return None

        # CSV header: Date,Open,High,Low,Close,Volume
        highs = []
        lows = []
        closes = []
        opens = []

        for line in lines[1:]:  # skip header
            fields = line.split(",")
            if len(fields) < 5:
                continue
            try:
                opens.append(float(fields[1]))
                highs.append(float(fields[2]))
                lows.append(float(fields[3]))
                closes.append(float(fields[4]))
            except (ValueError, IndexError):
                continue

        if len(closes) < 5:
            return None

        return {
            "highs": highs,
            "lows": lows,
            "closes": closes,
            "opens": opens,
        }

    except Exception as e:
        log.warning(f"Stooq historical failed for {symbol}: {e}")
        return None


def get_historical_data(symbol, days=SR_LOOKBACK_DAYS, asset_type="generic"):
    """
    Multi-source historical OHLC fetcher with smart fallback.

    asset_type: "gold", "dxy", or "generic"
    bach yKhtar les bons fallbacks 3la 7sab type dyal asset
    """
    # Try 1: Yahoo with given symbol
    data = get_historical_yahoo(symbol, days)
    if data:
        log.info(f"Historical from Yahoo ({symbol}): {len(data['closes'])} bars")
        return data

    # Asset-specific fallbacks
    if asset_type == "gold":
        # Try GC=F (gold futures) - 99% correlated, kayKhdem mzyan f Yahoo
        data = get_historical_yahoo("GC=F", days)
        if data:
            log.info(f"Historical from Yahoo (GC=F futures): {len(data['closes'])} bars")
            return data

        # Try Stooq xauusd
        data = get_historical_stooq("xauusd", days)
        if data:
            log.info(f"Historical from Stooq (xauusd): {len(data['closes'])} bars")
            return data

    elif asset_type == "dxy":
        # Try DXY future
        data = get_historical_yahoo("DX=F", days)
        if data:
            log.info(f"Historical from Yahoo (DX=F): {len(data['closes'])} bars")
            return data

        # Try Stooq
        data = get_historical_stooq("dx.f", days)
        if data:
            log.info(f"Historical from Stooq (dx.f): {len(data['closes'])} bars")
            return data

    log.warning(f"All historical sources failed for {symbol}")
    return None


# ================= TECHNICAL INDICATORS =================

def calculate_rsi(closes, period=RSI_PERIOD):
    """RSI calculation - shows overbought/oversold"""
    if len(closes) < period + 1:
        return 50.0

    gains = []
    losses = []

    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))

    # Simple average for first period
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder smoothing for the rest
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


def calculate_ma(closes, period):
    """Simple Moving Average"""
    if len(closes) < period:
        return closes[-1] if closes else 0
    return round(sum(closes[-period:]) / period, 2)


def calculate_atr(highs, lows, closes, period=ATR_PERIOD):
    """Average True Range - volatility measure for SL/TP sizing"""
    if len(closes) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        true_ranges.append(max(high_low, high_close, low_close))

    if len(true_ranges) < period:
        return round(sum(true_ranges) / len(true_ranges), 2)

    atr = sum(true_ranges[:period]) / period
    for i in range(period, len(true_ranges)):
        atr = (atr * (period - 1) + true_ranges[i]) / period

    return round(atr, 2)


def detect_support_resistance(highs, lows, closes, tolerance=SR_TOLERANCE):
    """
    Detect S/R zones using swing highs/lows.
    Groups nearby levels (within tolerance) into single zones.
    """
    if len(closes) < 5:
        return [], []

    # Find local swing highs and lows
    swing_highs = []
    swing_lows = []

    for i in range(2, len(closes) - 2):
        # Swing high: higher than 2 bars before and after
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])

        # Swing low: lower than 2 bars before and after
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])

    # Group nearby levels
    def group_levels(levels):
        if not levels:
            return []
        sorted_levels = sorted(levels)
        groups = [[sorted_levels[0]]]
        for level in sorted_levels[1:]:
            last_group_avg = sum(groups[-1]) / len(groups[-1])
            if abs(level - last_group_avg) / last_group_avg <= tolerance:
                groups[-1].append(level)
            else:
                groups.append([level])
        # Average each group, keep groups with 2+ touches (stronger levels)
        return sorted([round(sum(g) / len(g), 2) for g in groups], reverse=True)

    resistances = group_levels(swing_highs)
    supports = group_levels(swing_lows)

    return supports, resistances


def calculate_pivot_points(high, low, close):
    """Classic daily pivot points"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    return (
        round(pivot, 2),
        round(r1, 2),
        round(r2, 2),
        round(s1, 2),
        round(s2, 2),
    )


def find_nearest_levels(price, supports, resistances):
    """Find closest support below price + closest resistance above"""
    nearest_support = None
    nearest_resistance = None

    below = [s for s in supports if s < price]
    if below:
        nearest_support = max(below)

    above = [r for r in resistances if r > price]
    if above:
        nearest_resistance = min(above)

    return nearest_support, nearest_resistance


def rsi_signal(rsi):
    if rsi >= 70:
        return "🔴 Overbought (sell zone)"
    if rsi <= 30:
        return "🟢 Oversold (buy zone)"
    if rsi >= 60:
        return "🟡 Bullish momentum"
    if rsi <= 40:
        return "🟡 Bearish momentum"
    return "⚪ Neutral"


def ma_signal(price, ma20, ma50):
    if price > ma20 > ma50:
        return "🟢 Strong Bullish (price > MA20 > MA50)"
    if price < ma20 < ma50:
        return "🔴 Strong Bearish (price < MA20 < MA50)"
    if price > ma20 and ma20 < ma50:
        return "🟡 Mixed (potential reversal up)"
    if price < ma20 and ma20 > ma50:
        return "🟡 Mixed (potential reversal down)"
    return "⚪ Neutral"


# ================= TECHNICAL ANALYSIS BUILDER =================

def build_technical_analysis(symbol, asset_type="generic", spot_price=None):
    """
    Builds full TA snapshot for a symbol with smart fallback.

    spot_price: ila kayn, ghadi nadjuster levels bach ymatchu m3a spot price actuel
                (mohem khassa l gold: futures fih ~10-20$ premium 3la spot)
    """
    ta = TechnicalAnalysis(symbol=symbol)

    data = get_historical_data(symbol, SR_LOOKBACK_DAYS, asset_type)
    if not data:
        return ta

    highs = data["highs"]
    lows = data["lows"]
    closes = data["closes"]

    if len(closes) < 5:
        return ta

    historical_last = closes[-1]

    # Auto-adjust ila historical source mokhtalef men spot
    # Example: gold futures kayn 10-20$ above spot
    offset = 0.0
    if spot_price and historical_last > 0:
        diff_pct = abs(spot_price - historical_last) / historical_last
        # Ila l ftrq <2%, ghadi nadjuster (mab9inach m3a 99% correlation)
        if diff_pct < 0.02:
            offset = spot_price - historical_last
            if abs(offset) > 0.01:
                log.info(f"Adjusting {symbol} TA levels by {offset:+.2f} (futures->spot)")
                # Apply offset to all OHLC arrays
                highs = [h + offset for h in highs]
                lows = [l + offset for l in lows]
                closes = [c + offset for c in closes]

    # Use spot price ila 3titoha, sinon l last close (after adjustment)
    ta.current_price = round(spot_price if spot_price else closes[-1], 2)
    ta.rsi = calculate_rsi(closes)
    ta.ma20 = calculate_ma(closes, MA_SHORT)
    ta.ma50 = calculate_ma(closes, MA_LONG)
    ta.atr = calculate_atr(highs, lows, closes)

    # Support/Resistance
    supports, resistances = detect_support_resistance(highs, lows, closes)
    ta.support_levels = supports[:3]
    ta.resistance_levels = resistances[:3]

    # Pivot points (using last completed day)
    if len(closes) >= 2:
        ta.pivot, ta.r1, ta.r2, ta.s1, ta.s2 = calculate_pivot_points(
            highs[-2], lows[-2], closes[-2]
        )

    # Nearest levels
    all_supports = ta.support_levels + [ta.s1, ta.s2]
    all_resistances = ta.resistance_levels + [ta.r1, ta.r2]
    ta.nearest_support, ta.nearest_resistance = find_nearest_levels(
        ta.current_price, all_supports, all_resistances
    )

    # Signals
    ta.rsi_signal = rsi_signal(ta.rsi)
    ta.ma_signal = ma_signal(ta.current_price, ta.ma20, ta.ma50)

    ta.is_valid = True
    return ta


# ================= TRADE SUGGESTIONS =================

def suggest_trade(ta: TechnicalAnalysis, news_signal: str, direction: str):
    """
    Generate a trade suggestion with entry, SL, TP, R/R.
    direction: "BUY" or "SELL"
    """
    if not ta.is_valid or ta.atr == 0:
        return None

    price = ta.current_price
    atr = ta.atr

    if direction == "BUY":
        entry = price
        # SL below nearest support OR price - 1.5*ATR (whichever is closer)
        atr_sl = price - (atr * SL_ATR_MULTIPLIER)
        if ta.nearest_support and ta.nearest_support > atr_sl:
            sl = ta.nearest_support - (atr * 0.2)  # Small buffer below support
        else:
            sl = atr_sl

        # TP at nearest resistance OR price + 3*ATR
        atr_tp = price + (atr * TP_ATR_MULTIPLIER)
        if ta.nearest_resistance and ta.nearest_resistance < atr_tp:
            tp = ta.nearest_resistance - (atr * 0.1)
        else:
            tp = atr_tp

        risk = price - sl
        reward = tp - price

    else:  # SELL
        entry = price
        atr_sl = price + (atr * SL_ATR_MULTIPLIER)
        if ta.nearest_resistance and ta.nearest_resistance < atr_sl:
            sl = ta.nearest_resistance + (atr * 0.2)
        else:
            sl = atr_sl

        atr_tp = price - (atr * TP_ATR_MULTIPLIER)
        if ta.nearest_support and ta.nearest_support > atr_tp:
            tp = ta.nearest_support + (atr * 0.1)
        else:
            tp = atr_tp

        risk = sl - price
        reward = price - tp

    if risk <= 0:
        return None

    rr = round(reward / risk, 2)

    return {
        "direction": direction,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "rr_ratio": rr,
        "valid": rr >= 1.5,  # Minimum 1.5:1 R/R
    }


def determine_trade_direction(news_score, ta: TechnicalAnalysis):
    """
    Combine news + technicals to suggest BUY/SELL/HOLD
    """
    signals = []

    # News signal
    if news_score >= 2:
        signals.append("BUY")
    elif news_score <= -2:
        signals.append("SELL")

    # MA signal
    if "Strong Bullish" in ta.ma_signal:
        signals.append("BUY")
    elif "Strong Bearish" in ta.ma_signal:
        signals.append("SELL")

    # RSI signal (contrarian at extremes)
    if ta.rsi <= 30:
        signals.append("BUY")  # Oversold bounce
    elif ta.rsi >= 70:
        signals.append("SELL")  # Overbought reversal

    # Count
    buys = signals.count("BUY")
    sells = signals.count("SELL")

    if buys >= 2 and buys > sells:
        return "BUY"
    if sells >= 2 and sells > buys:
        return "SELL"
    return "HOLD"


# ================= TREND STRENGTH =================

def get_trend_strength(symbol, days=7):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={days}d&interval=1d"
        r = safe_get(url)
        if not r:
            return None, 0, "Unknown"
        data = r.json()["chart"]["result"][0]
        closes = [c for c in data["indicators"]["quote"][0]["close"] if c is not None]
        if len(closes) < 3:
            return None, 0, "Insufficient data"
        first = closes[0]
        last = closes[-1]
        change_pct = ((last - first) / first) * 100
        up_days = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        total = len(closes) - 1
        consistency = (up_days / total * 100) if total > 0 else 50

        if change_pct > 1.5 and consistency > 60:
            label = "🟢 Strong Uptrend"
            direction = "UP"
        elif change_pct > 0.3:
            label = "🟢 Mild Uptrend"
            direction = "UP"
        elif change_pct < -1.5 and consistency < 40:
            label = "🔴 Strong Downtrend"
            direction = "DOWN"
        elif change_pct < -0.3:
            label = "🔴 Mild Downtrend"
            direction = "DOWN"
        else:
            label = "⚪ Sideways/Range"
            direction = "FLAT"

        return direction, round(change_pct, 2), label
    except Exception as e:
        log.warning(f"Trend analysis failed for {symbol}: {e}")
        return None, 0, "Unknown"


def confluence_check(news_signal, trend_direction):
    if not trend_direction or trend_direction == "FLAT":
        return "⚠ No clear trend"
    if news_signal > 0 and trend_direction == "UP":
        return "✅ STRONG CONFLUENCE"
    if news_signal < 0 and trend_direction == "DOWN":
        return "✅ STRONG CONFLUENCE"
    if news_signal > 0 and trend_direction == "DOWN":
        return "⚠ DIVERGENCE"
    if news_signal < 0 and trend_direction == "UP":
        return "⚠ DIVERGENCE"
    return "⚪ Neutral"


def volatility(change):
    if change is None:
        return "Unknown"
    if abs(change) > 0.7:
        return "High"
    if abs(change) > 0.3:
        return "Medium"
    return "Low"


def fmt_price(price, change):
    if price is None:
        return "N/A"
    if change is None:
        return f"{price}"
    sign = "+" if change >= 0 else ""
    return f"{price} ({sign}{change}%)"


def fmt_levels(levels, max_n=3):
    """Format S/R levels for message"""
    if not levels:
        return "N/A"
    return ", ".join([str(l) for l in levels[:max_n]])


# ================= TELEGRAM =================

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": msg},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            log.info("Telegram sent ✅")
        else:
            log.error(f"Telegram error: {r.status_code} - {r.text}")
    except Exception as e:
        log.error(f"Telegram failed: {e}")


# ================= MESSAGE BUILDER =================

def format_ta_section(name, ta: TechnicalAnalysis):
    """Format technical analysis section for one asset"""
    if not ta.is_valid:
        return f"{name} TA: data unavailable\n"

    section = (
        f"📈 {name} TECHNICALS\n"
        f"Price: {ta.current_price}\n"
        f"RSI({RSI_PERIOD}): {ta.rsi} → {ta.rsi_signal}\n"
        f"MA20/MA50: {ta.ma20} / {ta.ma50}\n"
        f"   {ta.ma_signal}\n"
        f"ATR({ATR_PERIOD}): {ta.atr}\n\n"
        f"🎯 KEY LEVELS\n"
    )

    if ta.nearest_support:
        dist = round(((ta.current_price - ta.nearest_support) / ta.current_price) * 100, 2)
        section += f"Support: {ta.nearest_support} (-{abs(dist)}%)\n"
    if ta.nearest_resistance:
        dist = round(((ta.nearest_resistance - ta.current_price) / ta.current_price) * 100, 2)
        section += f"Resistance: {ta.nearest_resistance} (+{abs(dist)}%)\n"

    section += (
        f"All Supports: {fmt_levels(ta.support_levels)}\n"
        f"All Resistances: {fmt_levels(ta.resistance_levels)}\n"
        f"Pivot: {ta.pivot} | R1:{ta.r1} R2:{ta.r2} | S1:{ta.s1} S2:{ta.s2}\n"
    )

    return section


def format_trade_section(name, trade):
    """Format trade suggestion section"""
    if not trade:
        return f"💡 {name} TRADE: No clear setup\n"

    valid_tag = "✅ VALID" if trade["valid"] else "⚠ LOW R/R"

    return (
        f"💡 {name} TRADE SETUP {valid_tag}\n"
        f"Action: {trade['direction']}\n"
        f"Entry: {trade['entry']}\n"
        f"Stop Loss: {trade['sl']} (risk: {trade['risk']})\n"
        f"Take Profit: {trade['tp']} (reward: {trade['reward']})\n"
        f"Risk/Reward: 1:{trade['rr_ratio']}\n"
    )


# ================= MAIN JOB =================

def job():
    global last_state, last_usd_score, last_gold_score

    log.info("=" * 50)
    log.info("Running job...")

    # === NEWS ===
    news = fetch_news()
    if not news:
        log.info("No news, skipping")
        return

    scalp_usd, scalp_gold, trend_usd, trend_gold, impactful = analyze(news)
    total_usd = scalp_usd + trend_usd
    total_gold = scalp_gold + trend_gold

    scalp_usd_sig = signal(scalp_usd)
    scalp_gold_sig = signal(scalp_gold)
    trend_usd_sig = signal(trend_usd)
    trend_gold_sig = signal(trend_gold)

    state = f"{scalp_usd_sig}-{scalp_gold_sig}-{trend_usd_sig}-{trend_gold_sig}"

    if (
        state == last_state and
        abs(total_usd - last_usd_score) < SIGNAL_CHANGE_THRESHOLD and
        abs(total_gold - last_gold_score) < SIGNAL_CHANGE_THRESHOLD
    ):
        log.info("No significant change, skipping send")
        return

    last_state = state
    last_usd_score = total_usd
    last_gold_score = total_gold

    # === LIVE PRICES ===
    dxy_price, dxy_change = get_dxy_price()
    gold_price, gold_change = get_gold_price()

    # === TREND ===
    dxy_dir, dxy_7d, dxy_trend_label = get_trend_strength("DX-Y.NYB", 7)
    gold_dir, gold_7d, gold_trend_label = get_trend_strength("XAUUSD=X", 7)

    # === TECHNICAL ANALYSIS ===
    log.info("Computing technical analysis...")
    # N3titih spot price bach ila TA jabt men futures, yt ajusta l levels
    dxy_ta = build_technical_analysis(
        "DX-Y.NYB",
        asset_type="dxy",
        spot_price=dxy_price
    )
    gold_ta = build_technical_analysis(
        "XAUUSD=X",
        asset_type="gold",
        spot_price=gold_price
    )

    # === TRADE SUGGESTIONS ===
    usd_direction = determine_trade_direction(total_usd, dxy_ta)
    gold_direction = determine_trade_direction(total_gold, gold_ta)

    usd_trade = None
    gold_trade = None
    if usd_direction != "HOLD" and dxy_ta.is_valid:
        usd_trade = suggest_trade(dxy_ta, "USD", usd_direction)
    if gold_direction != "HOLD" and gold_ta.is_valid:
        gold_trade = suggest_trade(gold_ta, "GOLD", gold_direction)

    # === CONFLUENCE ===
    usd_confluence = confluence_check(trend_usd, dxy_dir)
    gold_confluence = confluence_check(trend_gold, gold_dir)

    vol = volatility(dxy_change)

    # === BUILD MESSAGE ===
    msg = (
        f"📊 PRO MARKET UPDATE v8\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🔴 SCALP (5-30min)\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"USD: {scalp_usd_sig} ({confidence(scalp_usd)}%)\n"
        f"GOLD: {scalp_gold_sig} ({confidence(scalp_gold)}%)\n"
        f"Bias: {entry_bias(scalp_usd, scalp_gold)}\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🟢 TREND (days/weeks)\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"USD: {trend_usd_sig} ({confidence(trend_usd)}%)\n"
        f"   7d: {dxy_trend_label} ({dxy_7d:+}%)\n"
        f"   {usd_confluence}\n"
        f"GOLD: {trend_gold_sig} ({confidence(trend_gold)}%)\n"
        f"   7d: {gold_trend_label} ({gold_7d:+}%)\n"
        f"   {gold_confluence}\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 LIVE PRICES\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"DXY: {fmt_price(dxy_price, dxy_change)}\n"
        f"GOLD: {fmt_price(gold_price, gold_change)}\n"
        f"Vol: {vol}\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"{format_ta_section('DXY', dxy_ta)}\n"
        f"{format_ta_section('GOLD', gold_ta)}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"{format_trade_section('USD', usd_trade)}\n"
        f"{format_trade_section('GOLD', gold_trade)}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📰 TOP NEWS\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"{chr(10).join(impactful) if impactful else 'No strong news'}"
    )

    send_telegram(msg)
    log.info("Job done ✅")


# ================= RUN (GitHub Mode) =================

if __name__ == "__main__":

    print("=" * 60)
    print("🔥 Smart Agent v8 PRO TRADER (GitHub Mode) ✅")
    print("=" * 60)

    try:
        job()
        print("✅ Execution finished")
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        sys.exit(1)
