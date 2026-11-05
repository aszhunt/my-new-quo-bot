import html
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from ta import add_all_ta_features

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AXIOM TRADER", page_icon="🤖", layout="wide")

SESSION_VER = "axiom-v3"   # old saved results from earlier versions are discarded automatically
FOLDS = 4                  # walk-forward validation folds
PAYOUT = 0.85              # payout used in the backtest equity curve

# =====================================================================
#  STYLE  (ultra-dark surfaces, bright text everywhere)
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');
:root { --txt:#f4fcff; --txt2:#c6ecfa; --cy:#00f0ff; --gr:#00ff9c; --rd:#ff5c7a; --bg:#02040a; --pn:#050c1c; }
html, body, [class*="css"], .stApp { font-family: 'Share Tech Mono', monospace; }
.stApp {
  background:
    linear-gradient(rgba(0,240,255,0.03) 1px, transparent 1px) 0 0 / 42px 42px,
    linear-gradient(90deg, rgba(0,240,255,0.03) 1px, transparent 1px) 0 0 / 42px 42px,
    radial-gradient(ellipse at 20% 0%, #06142a 0%, #02040a 55%, #010208 100%);
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { font-family: 'Orbitron', sans-serif !important; letter-spacing: 2px; }

.title-wrap { position: relative; padding: 14px 22px; margin-bottom: 14px;
  border: 1px solid rgba(0,240,255,0.4); border-radius: 14px; background: #050c1c;
  box-shadow: 0 0 28px rgba(0,240,255,0.12) inset, 0 0 22px rgba(0,240,255,0.12); overflow: hidden; }
.title-wrap:after { content:''; position:absolute; left:0; right:0; height:2px; top:0;
  background: linear-gradient(90deg, transparent, #00f0ff, transparent); animation: scan 3.2s linear infinite; }
@keyframes scan { 0% { top: 0; opacity: .9 } 100% { top: 100%; opacity: .1 } }
.title { font-family:'Orbitron',sans-serif; font-weight:900; font-size:32px; letter-spacing:8px;
  background: linear-gradient(90deg,#00f0ff,#8f7bff,#ff6be6); -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color: transparent !important; }
.sub { color:#c6ecfa !important; font-size:13px; letter-spacing:3px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#00ff9c;
  box-shadow:0 0 10px #00ff9c; margin-right:8px; animation: pulse 1.4s infinite; }
@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.3 } }

.hud { border:1px solid rgba(0,240,255,0.4); border-radius:12px; padding:12px 16px; margin-bottom:10px;
  background:#050c1c; box-shadow: 0 0 16px rgba(0,240,255,0.1); }
.hud .k { color:#c6ecfa !important; font-size:12px; letter-spacing:3px; }
.hud .v { font-family:'Orbitron',sans-serif; font-size:21px; color:#f4fcff !important; margin-top:4px; }
.hud .d { font-size:13px; margin-top:3px; color:#c6ecfa; }
.up { color:#00ff9c !important; } .dn { color:#ff5c7a !important; } .nt { color:#c6ecfa !important; }

.card { border:1px solid rgba(0,240,255,0.4); border-radius:14px; padding:14px 16px; margin-bottom:14px; background:#050c1c; }
.card.strong-buy { border-color:#00ff9c; box-shadow:0 0 22px rgba(0,255,156,0.3); }
.card.strong-sell { border-color:#ff5c7a; box-shadow:0 0 22px rgba(255,92,122,0.3); }
.card .tf { font-family:'Orbitron',sans-serif; font-size:13px; color:#c6ecfa !important; letter-spacing:3px; }
.card .sig { font-family:'Orbitron',sans-serif; font-size:20px; font-weight:700; margin:6px 0 10px 0; }
.sig.b { color:#00ff9c !important; } .sig.s { color:#ff5c7a !important; } .sig.w { color:#f4fcff !important; }
.bar { height:10px; border-radius:6px; background:#ff5c7a; overflow:hidden; }
.bar .fill { height:100%; background:linear-gradient(90deg,#00b36e,#00ff9c); }
.pl { display:flex; justify-content:space-between; font-size:13px; margin-top:5px; color:#f4fcff; }
.meta { margin-top:10px; font-size:13px; color:#c6ecfa; line-height:1.8; }
.meta b { color:#f4fcff; }
.meta .g { color:#00ff9c; } .meta .r { color:#ff5c7a; }
.news { border-left:3px solid #00f0ff; padding:8px 14px; margin:6px 0; background:#050c1c; font-size:14px; color:#f4fcff; }

.tblwrap { overflow-x:auto; }
.tbl { width:100%; border-collapse:collapse; margin:6px 0 16px; }
.tbl th { background:#071a33; color:#00f0ff !important; text-align:left; padding:9px 12px;
  font-family:'Orbitron',sans-serif; font-size:12px; letter-spacing:2px; border-bottom:1px solid #0e4468; }
.tbl td { background:#050c1c; color:#f4fcff !important; padding:9px 12px; border-bottom:1px solid #0e2a44; font-size:14px; }

/* ---------- BOOT ANIMATION ---------- */
.boot { position:relative; display:flex; flex-direction:column; align-items:center; padding:34px 10px 26px;
  margin:10px 0 20px; border:1px solid rgba(0,240,255,.45); border-radius:16px; overflow:hidden;
  background: radial-gradient(circle at 50% 28%, #07182f, #02040a 72%);
  box-shadow: 0 0 40px rgba(0,240,255,.18) inset, 0 0 30px rgba(0,240,255,.12); }
.radar { position:relative; width:190px; height:190px; border-radius:50%;
  border:2px solid rgba(0,240,255,.6); box-shadow:0 0 30px rgba(0,240,255,.35); }
.ring { position:absolute; border-radius:50%; border:1px solid rgba(0,240,255,.35); }
.r1 { inset:22px; } .r2 { inset:44px; } .r3 { inset:66px; }
.sweep { position:absolute; inset:0; border-radius:50%;
  background: conic-gradient(from 0deg, rgba(0,240,255,0) 0deg, rgba(0,240,255,.6) 70deg, rgba(0,240,255,0) 72deg);
  animation: spin 1.5s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.blip { position:absolute; width:8px; height:8px; border-radius:50%; background:#00ff9c;
  box-shadow:0 0 10px #00ff9c; animation: blink 1.2s infinite; }
@keyframes blink { 0%,100% { opacity:1; transform:scale(1) } 50% { opacity:.15; transform:scale(.5) } }
.core { position:absolute; inset:76px; border-radius:50%; background:#031427; border:1px solid #00f0ff;
  display:flex; align-items:center; justify-content:center; font-family:'Orbitron',sans-serif; font-weight:900;
  color:#00f0ff !important; box-shadow:0 0 18px rgba(0,240,255,.6); animation: pulse 1.2s infinite; }
.bt { margin-top:20px; font-family:'Orbitron',sans-serif; font-weight:900; font-size:30px; letter-spacing:10px;
  color:#00f0ff !important; text-shadow: 2px 0 #ff6be6, -2px 0 #8f7bff; animation: glitch 2.2s infinite; }
@keyframes glitch { 0%,92%,100% { transform:translateX(0) } 94% { transform:translateX(-4px) skewX(6deg) }
  96% { transform:translateX(4px) skewX(-6deg) } 98% { transform:translateX(-2px) } }
.bs { margin-top:8px; color:#e6f9ff !important; font-size:14px; letter-spacing:2px; }
.bs:after { content:'_'; animation: blink 0.8s infinite; }
.eq { display:flex; gap:4px; align-items:flex-end; height:34px; margin-top:16px; }
.eq i { display:block; width:6px; background:linear-gradient(180deg,#00f0ff,#8f7bff); border-radius:2px;
  animation: eqb 0.9s ease-in-out infinite; transform-origin: bottom; }
@keyframes eqb { 0%,100% { transform:scaleY(.25) } 50% { transform:scaleY(1) } }
.pbar { width:60%; max-width:520px; height:6px; border-radius:4px; background:#0a1a33; margin-top:16px; overflow:hidden; }
.pbar div { height:100%; background:linear-gradient(90deg,#00f0ff,#ff6be6); box-shadow:0 0 12px #00f0ff; }

/* ---------- ULTRA-DARK SURFACES + BRIGHT TEXT EVERYWHERE ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], .main, .block-container { background-color: #02040a !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: #03060f !important;
  border-right: 1px solid rgba(0,240,255,0.3); }
.stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp div[data-testid="stMarkdownContainer"], .stApp small,
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
  color: #f4fcff !important; }
.title { -webkit-text-fill-color: transparent !important; }
.up { color:#00ff9c !important; } .dn { color:#ff5c7a !important; } .nt { color:#c6ecfa !important; }
.sig.b { color:#00ff9c !important; } .sig.s { color:#ff5c7a !important; }
.meta .g { color:#00ff9c !important; } .meta .r { color:#ff5c7a !important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"], input, textarea {
  background-color: #050a18 !important; color: #f4fcff !important; border-color: rgba(0,240,255,0.45) !important; }
div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: #f4fcff !important; }
div[data-baseweb="select"] svg { fill: #00f0ff !important; }
div[data-baseweb="popover"], div[data-baseweb="popover"] > div, div[data-baseweb="menu"],
ul[role="listbox"], li[role="option"] { background-color: #050a18 !important; color: #f4fcff !important; }
li[role="option"]:hover, li[aria-selected="true"] { background-color: #0a2a4a !important; }
div[data-baseweb="tab-list"] { background: #02040a !important; border-bottom: 1px solid rgba(0,240,255,0.3); }
button[data-baseweb="tab"] { background: transparent !important; color: #c6ecfa !important;
  font-family:'Orbitron',sans-serif; letter-spacing:2px; }
button[data-baseweb="tab"][aria-selected="true"] { color: #00f0ff !important; }
.stButton button { font-family:'Orbitron',sans-serif; letter-spacing:3px; font-weight:700;
  border:1px solid #00f0ff; background:#06142a !important; color:#00f0ff !important; }
.stButton button p { color:#00f0ff !important; }
.stButton button:hover { box-shadow:0 0 20px rgba(0,240,255,.5); border-color:#ffffff; }
div[data-testid="stAlert"] { background: #07142a !important; border: 1px solid rgba(0,240,255,0.4); }
::-webkit-scrollbar { background: #02040a; width: 10px; }
::-webkit-scrollbar-thumb { background: #0e3a5c; border-radius: 6px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =====================================================================
#  MARKETS / TIMEFRAMES
# =====================================================================
MARKETS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X", "AUD/USD": "AUDUSD=X", "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X", "EUR/CHF": "EURCHF=X",
    "EUR/AUD": "EURAUD=X", "EUR/CAD": "EURCAD=X", "EUR/NZD": "EURNZD=X",
    "GBP/JPY": "GBPJPY=X", "GBP/CHF": "GBPCHF=X", "GBP/AUD": "GBPAUD=X",
    "GBP/CAD": "GBPCAD=X", "GBP/NZD": "GBPNZD=X", "AUD/JPY": "AUDJPY=X",
    "AUD/CHF": "AUDCHF=X", "AUD/CAD": "AUDCAD=X", "AUD/NZD": "AUDNZD=X",
    "CAD/JPY": "CADJPY=X", "CAD/CHF": "CADCHF=X", "CHF/JPY": "CHFJPY=X",
    "NZD/JPY": "NZDJPY=X", "NZD/CHF": "NZDCHF=X", "NZD/CAD": "NZDCAD=X",
    "USD/TRY": "USDTRY=X", "USD/ZAR": "USDZAR=X", "USD/MXN": "USDMXN=X",
    "USD/SGD": "USDSGD=X", "USD/HKD": "USDHKD=X", "USD/NOK": "USDNOK=X",
    "USD/SEK": "USDSEK=X", "USD/PLN": "USDPLN=X", "USD/INR": "USDINR=X",
    "USD/PKR": "USDPKR=X", "USD/CNH": "USDCNH=X",
    "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD", "BNB/USD": "BNB-USD",
    "XRP/USD": "XRP-USD", "SOL/USD": "SOL-USD", "DOGE/USD": "DOGE-USD",
    "Gold (XAU/USD)": "GC=F", "Silver (XAG/USD)": "SI=F",
}

# label -> (download interval, resample rule, period, minutes)
TF = {
    "1 MIN": ("1m", None, "7d", 1),
    "2 MIN": ("2m", None, "30d", 2),
    "3 MIN": ("1m", "3min", "7d", 3),
    "5 MIN": ("5m", None, "30d", 5),
    "10 MIN": ("5m", "10min", "30d", 10),
    "15 MIN": ("15m", None, "30d", 15),
}
RAW = ["Open", "High", "Low", "Close", "Volume"]


# =====================================================================
#  HELPERS
# =====================================================================
def to_utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def naive_day(idx):
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()


def flat_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fmt_price(x):
    x = float(x)
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 100:
        return f"{x:,.3f}"
    if x >= 10:
        return f"{x:,.4f}"
    return f"{x:,.5f}"


# =====================================================================
#  DATA
# =====================================================================
@st.cache_data(ttl=30, show_spinner=False)
def load_data(ticker, interval, rule, period):
    df = yf.download(ticker, interval=interval, period=period,
                     progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        return None
    df = flat_cols(df)[RAW].dropna(subset=["Close"])
    if rule:
        df = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    return df


@st.cache_data(ttl=15, show_spinner=False)
def load_live(ticker):
    """Latest traded price from the 1-minute feed."""
    try:
        d = yf.download(ticker, interval="1m", period="1d", progress=False,
                        auto_adjust=False, threads=False)
        if d is None or d.empty:
            return None
        c = flat_cols(d)["Close"].dropna()
        return float(c.iloc[-1]), str(to_utc(c.index[-1]))
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def load_macro():
    """Daily fundamental drivers: Dollar Index, US 10Y yield, VIX, S&P 500."""
    syms = {"dxy": "DX-Y.NYB", "tnx": "^TNX", "vix": "^VIX", "spx": "^GSPC"}
    out = {}
    for k, s in syms.items():
        try:
            d = yf.download(s, interval="1d", period="150d", progress=False,
                            auto_adjust=False, threads=False)
            if d is None or d.empty:
                continue
            c = flat_cols(d)["Close"].dropna()
            c.index = naive_day(c.index)
            c = c[~c.index.duplicated(keep="last")].sort_index()
            out[k] = c
        except Exception:
            continue
    return out or None


def macro_feature_frame(macro):
    if not macro:
        return None
    df = pd.DataFrame(macro).sort_index().ffill()
    f = pd.DataFrame(index=df.index)
    for k in df.columns:
        f[f"{k}_ret1"] = df[k].pct_change(1)
        f[f"{k}_ret5"] = df[k].pct_change(5)
        f[f"{k}_dev10"] = df[k] / df[k].rolling(10).mean() - 1
    if "vix" in df.columns:
        f["vix_level"] = df["vix"] / 20.0
    return f.shift(1)       # a day only sees the PREVIOUS day's data


@st.cache_data(ttl=600, show_spinner=False)
def load_news(ticker):
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    titles = []
    for it in items[:6]:
        try:
            c = it.get("content") or {}
            t = c.get("title") or it.get("title")
            if t:
                titles.append(str(t))
        except Exception:
            continue
    return titles


def macro_summary(macro):
    rows = {}
    for k, s in (macro or {}).items():
        if len(s) < 12:
            continue
        rows[k] = {"last": float(s.iloc[-1]),
                   "chg": float(s.iloc[-1] / s.iloc[-2] - 1) * 100,
                   "dev": float(s.iloc[-1] / s.iloc[-11:-1].mean() - 1) * 100}
    return rows


def fundamental_bias(label, ticker, ms):
    if not ms:
        return "NEUTRAL", "nt", "Macro data loading"
    dxy = ms.get("dxy", {}).get("dev", 0.0)
    tnx = ms.get("tnx", {}).get("dev", 0.0)
    spx = ms.get("spx", {}).get("dev", 0.0)
    vix = ms.get("vix", {}).get("dev", 0.0)
    if ticker.endswith("-USD"):
        score = (1 if spx > 0 else -1) + (1 if vix < 0 else -1)
        why = "Risk sentiment (S&P 500 + VIX)"
    elif ticker in ("GC=F", "SI=F"):
        score = (1 if dxy < 0 else -1) + (1 if tnx < 0 else -1)
        why = "Dollar Index + US 10Y yield"
    elif label.startswith("USD/"):
        score = 2 if dxy > 0 else -2
        why = "Dollar Index trend"
    elif label.endswith("/USD"):
        score = -2 if dxy > 0 else 2
        why = "Dollar Index trend"
    else:
        return "NEUTRAL", "nt", "Cross pair · USD-neutral"
    if score > 0:
        return "BULLISH", "up", why
    if score < 0:
        return "BEARISH", "dn", why
    return "NEUTRAL", "nt", why


# =====================================================================
#  INDICATOR VOTE BANK  (1500+ parameterised indicator variants)
#  every variant votes +1 (BUY), -1 (SELL) or 0 (NEUTRAL) on every candle
# =====================================================================
def sgn(x):
    a = np.asarray(x, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sign(a).astype(np.int8)


def true_range(df):
    c = df["Close"]
    return pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(),
                      (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)


def supertrend(df, p, m):
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    atr = true_range(df).ewm(alpha=1 / p, adjust=False).mean().values
    hl2 = (h + l) / 2
    ub, lb = hl2 + m * atr, hl2 - m * atr
    fub, flb = ub.copy(), lb.copy()
    trend = np.ones(len(c))
    for i in range(1, len(c)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if trend[i - 1] == 1 and c[i] < flb[i]:
            trend[i] = -1
        elif trend[i - 1] == -1 and c[i] > fub[i]:
            trend[i] = 1
        else:
            trend[i] = trend[i - 1]
    return trend


def indicator_bank(df):
    """Return (votes matrix n x m int8, family name per column)."""
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    n = len(df)
    cols, fams = [], []

    def add(fam, arr):
        cols.append(sgn(arr))
        fams.append(fam)

    sma_cache, ema_cache = {}, {}

    def S(p):
        if p not in sma_cache:
            sma_cache[p] = c.rolling(p).mean()
        return sma_cache[p]

    def E(p):
        if p not in ema_cache:
            ema_cache[p] = c.ewm(span=p, adjust=False).mean()
        return ema_cache[p]

    tr = true_range(df)
    tp = (h + l + c) / 3

    # --- moving averages -------------------------------------------------
    for p in range(3, 203, 2):
        add("SMA price", c - S(p))
        add("EMA price", c - E(p))
    for f in range(2, 22, 2):
        for s in range(20, 201, 20):
            if f < s:
                add("SMA cross", S(f) - S(s))
                add("EMA cross", E(f) - E(s))

    # --- oscillators -----------------------------------------------------
    for p in range(2, 51):
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
        rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        add("RSI trend", rsi - 50)
        add("RSI reversal", np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0)))
    for p in range(3, 41):
        ll, hh = l.rolling(p).min(), h.rolling(p).max()
        rng = (hh - ll).replace(0, np.nan)
        k = 100 * (c - ll) / rng
        add("Stochastic reversal", np.where(k < 20, 1, np.where(k > 80, -1, 0)))
        add("Stochastic cross", k - k.rolling(3).mean())
        wr = -100 * (hh - c) / rng
        add("Williams %R reversal", np.where(wr < -80, 1, np.where(wr > -20, -1, 0)))
        add("Williams %R trend", wr + 50)
    for p in range(5, 61):
        dev = tp.rolling(p).std().replace(0, np.nan)
        cci = (tp - tp.rolling(p).mean()) / (0.015 * 1.25 * dev)
        add("CCI trend", cci)
        add("CCI reversal", np.where(cci < -100, 1, np.where(cci > 100, -1, 0)))
        add("ROC", c.pct_change(p))
        z = (c - c.rolling(p).mean()) / c.rolling(p).std().replace(0, np.nan)
        add("Z-score reversal", np.where(z < -1.5, 1, np.where(z > 1.5, -1, 0)))

    # --- MACD / TRIX / Awesome -------------------------------------------
    for f in (3, 5, 8, 12, 16):
        for s in (13, 21, 26, 34, 50):
            if f >= s:
                continue
            m = E(f) - E(s)
            for sg in (5, 9, 13):
                add("MACD histogram", m - m.ewm(span=sg, adjust=False).mean())
            add("MACD line", m)
    for p in range(5, 31):
        e3 = c.ewm(span=p, adjust=False).mean().ewm(span=p, adjust=False).mean().ewm(span=p, adjust=False).mean()
        add("TRIX", e3.pct_change())
    mp = (h + l) / 2
    for f in (3, 5, 8):
        for s in (21, 34, 55):
            add("Awesome oscillator", mp.rolling(f).mean() - mp.rolling(s).mean())

    # --- bands / channels --------------------------------------------------
    for w in (10, 15, 20, 30, 40, 50):
        mid, sd = S(w), c.rolling(w).std()
        for k in (1.5, 2, 2.5, 3):
            add("Bollinger reversal", np.where(c < mid - k * sd, 1, np.where(c > mid + k * sd, -1, 0)))
    for p in range(5, 61):
        hh, ll = h.rolling(p).max(), l.rolling(p).min()
        add("Donchian breakout", np.where(c > hh.shift(1), 1, np.where(c < ll.shift(1), -1, 0)))
        add("Channel position", (c - ll) / (hh - ll).replace(0, np.nan) - 0.5)
    for p in (10, 20, 30):
        atr = tr.ewm(alpha=1 / p, adjust=False).mean()
        for m in (1, 1.5, 2):
            add("Keltner reversal", np.where(c < E(p) - m * atr, 1, np.where(c > E(p) + m * atr, -1, 0)))

    # --- directional movement --------------------------------------------
    up_m, dn_m = h.diff(), -l.diff()
    plus = pd.Series(np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0), index=df.index)
    minus = pd.Series(np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0), index=df.index)
    for p in range(5, 41):
        atr = tr.ewm(alpha=1 / p, adjust=False).mean().replace(0, np.nan)
        pdi = plus.ewm(alpha=1 / p, adjust=False).mean() / atr
        mdi = minus.ewm(alpha=1 / p, adjust=False).mean() / atr
        add("DI trend", pdi - mdi)

    # --- volume ------------------------------------------------------------
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    pos_flow = (tp * v).where(tp > tp.shift(), 0.0)
    neg_flow = (tp * v).where(tp < tp.shift(), 0.0)
    mfv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan) * v
    fi = c.diff() * v
    for p in range(3, 41):
        add("OBV slope", obv.diff(p))
    for p in range(5, 41):
        mfr = pos_flow.rolling(p).sum() / neg_flow.rolling(p).sum().replace(0, np.nan)
        mfi = 100 - 100 / (1 + mfr)
        add("MFI trend", mfi - 50)
        add("MFI reversal", np.where(mfi < 20, 1, np.where(mfi > 80, -1, 0)))
        add("Chaikin money flow", mfv.rolling(p).sum() / v.rolling(p).sum().replace(0, np.nan))
        add("Rolling VWAP", c - (tp * v).rolling(p).sum() / v.rolling(p).sum().replace(0, np.nan))
    for p in range(3, 31):
        add("Force index", fi.ewm(span=p, adjust=False).mean())

    # --- Ichimoku / Supertrend / regression / Heikin-Ashi / pivots ----------
    for t in (5, 7, 9, 12):
        ten = (h.rolling(t).max() + l.rolling(t).min()) / 2
        for k in (20, 26, 34, 52):
            kij = (h.rolling(k).max() + l.rolling(k).min()) / 2
            add("Ichimoku", ten - kij)
            add("Ichimoku", c - kij)
    for p in (7, 10, 14, 21):
        for m in (1.5, 2, 3):
            add("Supertrend", supertrend(df, p, m))
    cv = c.values
    for p in range(5, 61):
        x = np.arange(p, dtype=float)
        xc = x - x.mean()
        slope = np.convolve(cv, xc[::-1], mode="valid") / (xc ** 2).sum()
        add("Regression slope", np.r_[np.full(p - 1, np.nan), slope])
    ha_c = (o + h + l + c) / 4
    ha_o = (o.shift() + c.shift()) / 2
    add("Heikin-Ashi", ha_c - ha_o)
    for p in (2, 3, 5):
        add("Heikin-Ashi", np.sign(ha_c - ha_o).rolling(p).sum())
    add("Pivot points", c - (h.shift() + l.shift() + c.shift()) / 3)
    for p in (5, 10, 20, 50):
        add("Pivot points", c - (h.rolling(p).max().shift() + l.rolling(p).min().shift() + c.shift()) / 3)

    # --- candle patterns ---------------------------------------------------
    body, ab = c - o, (c - o).abs()
    rg = (h - l).replace(0, np.nan)
    uw, lw = h - np.maximum(o, c), np.minimum(o, c) - l
    b1, b2 = body.shift(), body.shift(2)
    add("Candle patterns", ((b1 < 0) & (body > 0) & (c >= o.shift()) & (o <= c.shift())).astype(int)
        - ((b1 > 0) & (body < 0) & (c <= o.shift()) & (o >= c.shift())).astype(int))
    add("Candle patterns", ((lw > 2 * ab) & (uw < 0.5 * ab) & (ab > 0)).astype(int)
        - ((uw > 2 * ab) & (lw < 0.5 * ab) & (ab > 0)).astype(int))
    add("Candle patterns", ((body > 0) & (b1 > 0) & (b2 > 0) & (c > c.shift()) & (c.shift() > c.shift(2))).astype(int)
        - ((body < 0) & (b1 < 0) & (b2 < 0) & (c < c.shift()) & (c.shift() < c.shift(2))).astype(int))
    add("Candle patterns", (ab / rg > 0.9) * np.sign(body))
    add("Candle patterns", ((b1 < 0) & (body > 0) & (c < o.shift()) & (o > c.shift()) & (ab < ab.shift() * 0.6)).astype(int)
        - ((b1 > 0) & (body < 0) & (c > o.shift()) & (o < c.shift()) & (ab < ab.shift() * 0.6)).astype(int))
    add("Candle patterns", ((b1 < 0) & (body > 0) & (o < c.shift()) & (c > (o.shift() + c.shift()) / 2) & (c < o.shift())).astype(int)
        - ((b1 > 0) & (body < 0) & (o > c.shift()) & (c < (o.shift() + c.shift()) / 2) & (c > o.shift())).astype(int))
    add("Candle patterns", ((b2 < 0) & (ab.shift() < ab.shift(2) * 0.4) & (body > 0) & (c > (o.shift(2) + c.shift(2)) / 2)).astype(int)
        - ((b2 > 0) & (ab.shift() < ab.shift(2) * 0.4) & (body < 0) & (c < (o.shift(2) + c.shift(2)) / 2)).astype(int))
    add("Candle patterns", (c - l) / rg - 0.5)

    V = np.column_stack(cols).astype(np.int8)
    return V, fams


def family_features(V, fams):
    """Aggregate the vote bank into compact ML features (one per family + totals)."""
    fams = np.array(fams)
    out = {}
    for f in sorted(set(fams)):
        out["fam_" + f.lower().replace(" ", "_").replace("%", "pct").replace("-", "_")] = V[:, fams == f].mean(axis=1)
    m = V.shape[1]
    out["vote_buy_share"] = (V > 0).sum(axis=1) / m
    out["vote_sell_share"] = (V < 0).sum(axis=1) / m
    out["vote_net"] = out["vote_buy_share"] - out["vote_sell_share"]
    return pd.DataFrame(out)


def vote_summary(V_last, fams):
    fams = np.array(fams)
    rows = []
    for f in sorted(set(fams)):
        seg = V_last[fams == f]
        rows.append({"family": f, "total": int(len(seg)), "buy": int((seg > 0).sum()),
                     "sell": int((seg < 0).sum()), "neutral": int((seg == 0).sum())})
    return {"buy": int((V_last > 0).sum()), "sell": int((V_last < 0).sum()),
            "neutral": int((V_last == 0).sum()), "total": int(len(V_last))}, rows


# =====================================================================
#  TRADING STRATEGIES  (rule based, each gives BUY / SELL / NEUTRAL)
# =====================================================================
def run_strategies(df):
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    tr = true_range(df)
    out = []

    def sig(buy, sell):
        return "BUY" if bool(buy) else ("SELL" if bool(sell) else "NEUTRAL")

    # 1 trend following
    e20, e50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    up_m, dn_m = h.diff(), -l.diff()
    plus = pd.Series(np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0), index=df.index)
    minus = pd.Series(np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0), index=df.index)
    pdi = 100 * plus.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    mdi = 100 * minus.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    adx = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).ewm(alpha=1 / 14, adjust=False).mean()
    out.append(("Trend Following", sig(
        e20.iloc[-1] > e50.iloc[-1] and c.iloc[-1] > e20.iloc[-1] and pdi.iloc[-1] > mdi.iloc[-1] and adx.iloc[-1] > 20,
        e20.iloc[-1] < e50.iloc[-1] and c.iloc[-1] < e20.iloc[-1] and pdi.iloc[-1] < mdi.iloc[-1] and adx.iloc[-1] > 20),
        "EMA20/50 + DI + ADX"))
    # 2 mean reversion
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    out.append(("Mean Reversion", sig(
        rsi.iloc[-1] < 35 and c.iloc[-1] < mid.iloc[-1] - 1.8 * sd.iloc[-1],
        rsi.iloc[-1] > 65 and c.iloc[-1] > mid.iloc[-1] + 1.8 * sd.iloc[-1]), "RSI14 + Bollinger"))
    # 3 breakout with volume
    hh, ll = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)
    rel = v / v.rolling(20).mean().replace(0, np.nan)
    out.append(("Breakout", sig(c.iloc[-1] > hh.iloc[-1] and rel.iloc[-1] > 1.1,
                                c.iloc[-1] < ll.iloc[-1] and rel.iloc[-1] > 1.1), "Donchian20 + volume"))
    # 4 momentum
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = m - m.ewm(span=9, adjust=False).mean()
    roc = c.pct_change(10)
    out.append(("Momentum", sig(hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2] and roc.iloc[-1] > 0,
                                hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2] and roc.iloc[-1] < 0),
                "MACD + ROC"))
    # 5 volume / VWAP
    tp = (h + l + c) / 3
    vwap = (tp * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    mfv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan) * v
    cmf = mfv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    ob10 = obv.diff(10).iloc[-1]
    out.append(("Volume & VWAP", sig(c.iloc[-1] > vwap.iloc[-1] and ob10 > 0 and cmf.iloc[-1] > 0,
                                     c.iloc[-1] < vwap.iloc[-1] and ob10 < 0 and cmf.iloc[-1] < 0),
                "VWAP + OBV + CMF"))
    # 6 supertrend + ichimoku
    st_dir = supertrend(df, 10, 3)[-1]
    ten = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kij = (h.rolling(26).max() + l.rolling(26).min()) / 2
    out.append(("Supertrend + Ichimoku", sig(st_dir > 0 and ten.iloc[-1] > kij.iloc[-1],
                                             st_dir < 0 and ten.iloc[-1] < kij.iloc[-1]), "ST(10,3) + Tenkan/Kijun"))
    return out


# =====================================================================
#  FEATURES + MODEL
# =====================================================================
def htf_features(df, mins):
    h = df.resample(f"{mins * 5}min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    c = h["Close"]
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    ema = c.ewm(span=10, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    f = pd.DataFrame({"htf_rsi": rsi, "htf_ema_dist": c / ema - 1,
                      "htf_macd": hist / c, "htf_ret": c.pct_change()}).shift(1)
    return f.reindex(df.index, method="ffill")


def build_features(df, macro_f, mins):
    d = df.copy()
    real_vol = bool((d["Volume"] > 0).mean() > 0.5)
    if real_vol:
        d["Volume"] = d["Volume"].replace(0, np.nan).ffill().fillna(1.0)
    else:   # forex has no real volume: tick-activity proxy
        d["Volume"] = ((d["High"] - d["Low"]) / d["Close"] * 1e6).clip(lower=1.0)
    vol = d["Volume"]

    ohlcv = d[RAW].copy()
    d = add_all_ta_features(d, open="Open", high="High", low="Low", close="Close",
                            volume="Volume", fillna=False)
    X = d.drop(columns=RAW).replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.isna().mean() < 0.3].ffill().fillna(0.0)

    close = d["Close"]
    for col in list(X.columns):
        s = X[col]
        if s.std() > 0 and abs(s.corr(close)) > 0.9:
            X[col] = s / close - 1.0

    for k in (1, 2, 3, 5, 8, 13):
        X[f"ret_{k}"] = close.pct_change(k)
    X["body"] = (d["Close"] - d["Open"]) / d["Close"]
    X["range"] = (d["High"] - d["Low"]) / d["Close"]
    X["upper_wick"] = (d["High"] - d[["Open", "Close"]].max(axis=1)) / d["Close"]
    X["lower_wick"] = (d[["Open", "Close"]].min(axis=1) - d["Low"]) / d["Close"]
    X["streak"] = np.sign(close.diff()).rolling(5).sum()

    mu, sd = vol.rolling(20).mean(), vol.rolling(20).std()
    X["vol_z"] = (vol - mu) / sd.replace(0, np.nan)
    X["rel_vol"] = vol / mu
    X["vol_trend"] = vol.rolling(5).mean() / mu
    X["up_vol_ratio"] = vol.where(d["Close"] > d["Open"], 0.0).rolling(10).sum() / vol.rolling(10).sum()
    X["pv_corr"] = close.pct_change().rolling(10).corr(vol.pct_change())
    day = naive_day(d.index)
    typ = (d["High"] + d["Low"] + d["Close"]) / 3
    X["vwap_dev"] = close / ((typ * vol).groupby(day).cumsum() / vol.groupby(day).cumsum()) - 1

    idx = pd.DatetimeIndex(d.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    hrs = idx.hour + idx.minute / 60.0
    X["hour_sin"] = np.sin(2 * np.pi * hrs / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hrs / 24)

    try:
        X = X.join(htf_features(df, mins))
    except Exception:
        pass

    if macro_f is not None:
        try:
            mf = macro_f[~macro_f.index.duplicated(keep="last")].sort_index()
            mf = mf.reindex(day, method="ffill")
            mf.index = X.index
            X = X.join(mf)
        except Exception:
            pass

    V, fams = indicator_bank(ohlcv)
    ff = family_features(V, fams)
    ff.index = X.index
    X = X.join(ff)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.loc[:, X.std() > 0], real_vol, V[-1], fams, ohlcv


def fit_ensemble(X, y, r):
    keep = r.abs() >= r.abs().quantile(0.2)
    Xt, yt = X[keep], y[keep]
    if yt.nunique() < 2:
        Xt, yt = X, y
    w = np.linspace(0.5, 1.0, len(Xt))
    hgb = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0, random_state=42)
    et = ExtraTreesClassifier(n_estimators=150, max_depth=7, min_samples_leaf=10,
                              n_jobs=-1, random_state=42)
    lr = make_pipeline(RobustScaler(), FunctionTransformer(lambda a: np.clip(a, -5, 5)),
                       LogisticRegression(C=0.05, max_iter=400))
    hgb.fit(Xt, yt, sample_weight=w)
    et.fit(Xt, yt, sample_weight=w)
    lr.fit(Xt, yt, logisticregression__sample_weight=w)
    return [hgb, et, lr]


def ens_proba(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


@st.cache_data(ttl=60, show_spinner=False)
def analyse(ticker, tf):
    interval, rule, period, mins = TF[tf]
    raw = load_data(ticker, interval, rule, period)
    if raw is None or len(raw) < 300:
        return {"ok": False, "reason": "Feed se is timeframe ka data nahi mila (market band ya feed delay)"}

    age_min = (datetime.now(timezone.utc) - to_utc(raw.index[-1])).total_seconds() / 60
    forming = age_min < mins
    stale = age_min > max(10, mins * 3)
    df = raw.iloc[:-1] if forming else raw

    macro_f = macro_feature_frame(load_macro())
    X, real_vol, v_last, fams, ohlcv = build_features(df, macro_f, mins)
    ret_next = df["Close"].shift(-1) / df["Close"] - 1.0
    y = (ret_next > 0).astype(int)

    Xl, yl, rl = X.iloc[:-1].iloc[60:], y.iloc[:-1].iloc[60:], ret_next.iloc[:-1].iloc[60:]
    flat = rl != 0
    Xl, yl, rl = Xl[flat], yl[flat], rl[flat]
    if len(Xl) < 250:
        return {"ok": False, "reason": "Is timeframe ke liye abhi candles jama ho rahi hain"}

    n = len(Xl)
    edges = np.linspace(int(n * 0.5), n, FOLDS + 1).astype(int)
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        models = fit_ensemble(Xl.iloc[:a], yl.iloc[:a], rl.iloc[:a])
        parts.append(pd.DataFrame({"p_up": ens_proba(models, Xl.iloc[a:b]),
                                   "y": yl.iloc[a:b].values, "ret": rl.iloc[a:b].values},
                                  index=Xl.index[a:b]))
    oos = pd.concat(parts)
    final = fit_ensemble(Xl, yl, rl)
    p_live = float(ens_proba(final, X.iloc[[-1]])[0])

    votes, fam_rows = vote_summary(v_last, fams)
    try:
        strategies = run_strategies(ohlcv)
    except Exception:
        strategies = []
    atr = float(true_range(ohlcv).rolling(14).mean().iloc[-1])

    return {
        "ok": True, "p_up": p_live, "oos": oos, "votes": votes, "fam_rows": fam_rows,
        "strategies": strategies, "atr": atr,
        "price": float(raw["Close"].iloc[-1]), "age": float(age_min), "stale": bool(stale),
        "chart": raw.tail(120), "real_vol": real_vol,
    }


# =====================================================================
#  SIGNAL + UI HELPERS
# =====================================================================
def signal_of(p, stale=False):
    """returns (text, side, css)"""
    if stale:
        return "STANDBY · MARKET QUIET", "NONE", "w"
    if p >= 0.60:
        return "STRONG BUY", "BUY", "b"
    if p >= 0.52:
        return "BUY", "BUY", "b"
    if p <= 0.40:
        return "STRONG SELL", "SELL", "s"
    if p <= 0.48:
        return "SELL", "SELL", "s"
    return "NEUTRAL", "NONE", "w"


def levels(side, live, atr):
    if side == "BUY":
        return live, live + 1.5 * atr, live - 1.0 * atr
    if side == "SELL":
        return live, live - 1.5 * atr, live + 1.0 * atr
    return None


def strat_counts(res):
    s = res.get("strategies", [])
    return (sum(1 for x in s if x[1] == "BUY"), sum(1 for x in s if x[1] == "SELL"),
            sum(1 for x in s if x[1] == "NEUTRAL"))


def card_html(tf, res, live):
    text, side, sc = signal_of(res["p_up"], res["stale"])
    cls = "strong-buy" if (side == "BUY" and "STRONG" in text) else ("strong-sell" if (side == "SELL" and "STRONG" in text) else "")
    icon = {"b": "▲", "s": "▼", "w": "◆"}[sc]
    buy = res["p_up"] * 100
    vt = res["votes"]
    sb, ss, sn = strat_counts(res)
    lv = levels(side, live, res["atr"])
    if lv:
        lvl = (f'ENTRY <b>{fmt_price(lv[0])}</b> · TP <b class="g">{fmt_price(lv[1])}</b> · '
               f'SL <b class="r">{fmt_price(lv[2])}</b><br>')
    else:
        lvl = "ENTRY <b>wait for a clear lean</b><br>"
    return (
        f'<div class="card {cls}"><div class="tf">{tf}</div>'
        f'<div class="sig {sc}">{icon} {text}</div>'
        f'<div class="bar"><div class="fill" style="width:{buy:.1f}%"></div></div>'
        f'<div class="pl"><span>BUY {buy:.1f}%</span><span>SELL {100 - buy:.1f}%</span></div>'
        f'<div class="meta">INDICATORS <b class="g">▲ {vt["buy"]} BUY</b> · <b class="r">▼ {vt["sell"]} SELL</b> · '
        f'◆ {vt["neutral"]} NEUTRAL (of {vt["total"]})<br>'
        f'STRATEGIES <b class="g">{sb} BUY</b> · <b class="r">{ss} SELL</b> · {sn} NEUTRAL<br>'
        f'{lvl}EXPIRY <b>next {tf.lower()} candle</b></div></div>'
    )


def hud(k, v, d="", dcls="nt"):
    extra = f'<div class="d {dcls}">{d}</div>' if d else ""
    return f'<div class="hud"><div class="k">{k}</div><div class="v">{v}</div>{extra}</div>'


def html_table(headers, rows):
    th = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>"
    return f'<div class="tblwrap"><table class="tbl"><tr>{th}</tr>{body}</table></div>'


def colored(text, kind):
    cls = {"b": "up", "s": "dn"}.get(kind, "nt")
    return f'<span class="{cls}">{html.escape(text)}</span>'


def boot_html(step, pct):
    blips = ('<div class="blip" style="top:38px;left:120px"></div>'
             '<div class="blip" style="top:112px;left:46px;animation-delay:.4s"></div>'
             '<div class="blip" style="top:140px;left:128px;animation-delay:.8s"></div>')
    bars = "".join(f'<i style="height:34px;animation-delay:{i * 0.08:.2f}s"></i>' for i in range(14))
    return (
        '<div class="boot"><div class="radar"><div class="ring r1"></div><div class="ring r2"></div>'
        '<div class="ring r3"></div><div class="sweep"></div>' + blips +
        '<div class="core">AX</div></div>'
        '<div class="bt">AXIOM TRADER</div>'
        f'<div class="bs">{html.escape(step)}</div>'
        f'<div class="eq">{bars}</div>'
        f'<div class="pbar"><div style="width:{pct:.0f}%"></div></div></div>'
    )


def style_fig(fig, h=420):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#030814", margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="Share Tech Mono", color="#f4fcff", size=13))
    fig.update_xaxes(gridcolor="rgba(0,240,255,0.12)", tickfont=dict(color="#f4fcff"))
    fig.update_yaxes(gridcolor="rgba(0,240,255,0.12)", tickfont=dict(color="#f4fcff"))
    return fig


# =====================================================================
#  SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### 🤖 AXIOM TRADER")
    pair = st.selectbox("ASSET", list(MARKETS.keys()))
    scan = st.button("⚡ INITIATE SIGNAL")
    st.caption("Sab kuch automatic hai: timeframes, indicators, strategies aur fundamentals.")

# =====================================================================
#  HEADER
# =====================================================================
st.markdown(
    '<div class="title-wrap"><div class="title">AXIOM TRADER</div>'
    '<div class="sub"><span class="dot"></span>AUTONOMOUS AI SIGNAL ENGINE · 1500+ INDICATORS · '
    'STRATEGIES · FUNDAMENTALS</div></div>',
    unsafe_allow_html=True,
)

# =====================================================================
#  SCAN  (boot animation)
# =====================================================================
if scan:
    stage = st.empty()
    for i, s in enumerate(["Connecting to live market feed",
                           "Reading volume and order-flow activity",
                           "Syncing macro and fundamental pulse"]):
        stage.markdown(boot_html(s, 4 + i * 6), unsafe_allow_html=True)
        time.sleep(0.7)
    try:
        load_macro()
    except Exception:
        pass

    results = {}
    tf_list = list(TF.keys())
    for i, tf in enumerate(tf_list):
        stage.markdown(boot_html(f"{tf} · computing 1500+ indicators · training neural ensemble",
                                 22 + 72 * i / len(tf_list)), unsafe_allow_html=True)
        try:
            results[tf] = analyse(MARKETS[pair], tf)
        except Exception as e:
            results[tf] = {"ok": False, "reason": f"Processing error: {type(e).__name__}"}
    live = load_live(MARKETS[pair])
    stage.markdown(boot_html("Signals locked · rendering dashboard", 100), unsafe_allow_html=True)
    time.sleep(0.8)
    stage.empty()
    st.session_state["scan"] = {"ver": SESSION_VER, "pair": pair, "results": results, "live": live}

state = st.session_state.get("scan")
if state is not None and state.get("ver") != SESSION_VER:
    state = None
    st.session_state.pop("scan", None)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["◉ COMMAND CENTER", "◉ INDICATORS & STRATEGIES", "◉ MACRO & NEWS", "◉ BACKTEST", "◉ CHARTS"])

if not state:
    with tab1:
        st.markdown(hud("STATUS", "READY", "Sidebar me asset chunein aur INITIATE SIGNAL dabayein.", "nt"),
                    unsafe_allow_html=True)
else:
    results = state["results"]
    sel_pair = state["pair"]
    sel_ticker = MARKETS[sel_pair]
    good = {tf: r for tf, r in results.items() if r.get("ok") and "votes" in r and "p_up" in r}
    any_res = next(iter(good.values())) if good else None
    live_info = state.get("live")
    live_price = live_info[0] if live_info else (any_res["price"] if any_res else None)

    # ---------------- COMMAND CENTER ----------------
    with tab1:
        if good:
            mean_p = float(np.mean([r["p_up"] for r in good.values()]))
            o_text, o_side, o_cls = signal_of(mean_p, any_res["stale"])
            tot_b = sum(r["votes"]["buy"] for r in good.values())
            tot_s = sum(r["votes"]["sell"] for r in good.values())
            age_txt = f"{any_res['age']:.1f} min"

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(hud("ASSET", sel_pair), unsafe_allow_html=True)
            c2.markdown(hud("LIVE PRICE", fmt_price(live_price),
                            "volume: " + ("real" if any_res["real_vol"] else "tick-activity"), "nt"),
                        unsafe_allow_html=True)
            c3.markdown(hud("LAST CANDLE AGE", age_txt, "feed: Yahoo Finance", "nt"), unsafe_allow_html=True)
            c4.markdown(hud("AXIOM SIGNAL", o_text, f"BUY {mean_p * 100:.1f}% · SELL {100 - mean_p * 100:.1f}%",
                            {"b": "up", "s": "dn", "w": "nt"}[o_cls]), unsafe_allow_html=True)

            g1, g2 = st.columns([1, 2])
            with g1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=mean_p * 100,
                    number={"suffix": "% BUY", "font": {"family": "Orbitron", "size": 26, "color": "#f4fcff"}},
                    title={"text": "ALL-TIMEFRAME PROBABILITY", "font": {"family": "Orbitron", "size": 13, "color": "#f4fcff"}},
                    gauge={"axis": {"range": [0, 100], "tickcolor": "#f4fcff"}, "bar": {"color": "#00f0ff"},
                           "steps": [{"range": [0, 48], "color": "rgba(255,92,122,0.45)"},
                                     {"range": [48, 52], "color": "rgba(198,236,250,0.30)"},
                                     {"range": [52, 100], "color": "rgba(0,255,156,0.40)"}]}))
                st.plotly_chart(style_fig(fig, 260), key="gauge")
            with g2:
                st.markdown(hud("ALL INDICATORS COMBINED",
                                f'<span class="up">▲ {tot_b}</span> &nbsp; <span class="dn">▼ {tot_s}</span>',
                                f"{sum(r['votes']['total'] for r in good.values())} indicator votes across "
                                f"{len(good)} timeframes", "nt"), unsafe_allow_html=True)
                ms0 = macro_summary(load_macro())
                bias, bcl, why = fundamental_bias(sel_pair, sel_ticker, ms0)
                st.markdown(hud("FUNDAMENTAL BIAS", bias, why, bcl), unsafe_allow_html=True)

        # all six timeframes in one table (always shown)
        rows = []
        for tf in TF:
            r = results.get(tf, {})
            if r.get("ok") and "votes" in r:
                text, side, sc = signal_of(r["p_up"], r["stale"])
                sb, ss, sn = strat_counts(r)
                rows.append([f"<b>{tf}</b>", colored(text, sc), f"{r['p_up'] * 100:.1f}%",
                             f"{100 - r['p_up'] * 100:.1f}%", f'<span class="up">{r["votes"]["buy"]}</span>',
                             f'<span class="dn">{r["votes"]["sell"]}</span>', f"{sb} / {ss} / {sn}"])
            else:
                rows.append([f"<b>{tf}</b>", '<span class="nt">Loading…</span>', "—", "—", "—", "—", "—"])
        st.markdown("#### ALL TIMEFRAMES")
        st.markdown(html_table(["Timeframe", "Signal", "Buy %", "Sell %", "Indicators BUY",
                                "Indicators SELL", "Strategies B / S / N"], rows), unsafe_allow_html=True)

        if good and live_price:
            st.markdown("#### NEXT-CANDLE SIGNALS (live price based)")
            cols = st.columns(3)
            for i, tf in enumerate(good):
                with cols[i % 3]:
                    st.markdown(card_html(tf, good[tf], live_price), unsafe_allow_html=True)
        for tf, r in results.items():
            if not r.get("ok"):
                st.markdown(f'<div class="news"><b>{tf}</b> · {html.escape(str(r.get("reason")))}</div>',
                            unsafe_allow_html=True)

    # ---------------- INDICATORS & STRATEGIES ----------------
    with tab2:
        if not good:
            st.markdown(hud("STATUS", "NO DATA YET", "Dobara INITIATE SIGNAL dabayein.", "nt"),
                        unsafe_allow_html=True)
        else:
            tf_i = st.selectbox("TIMEFRAME", list(good.keys()), key="ind_tf")
            r = good[tf_i]
            vt = r["votes"]
            a, b, c_, d_ = st.columns(4)
            a.markdown(hud("INDICATORS BUY", f'<span class="up">{vt["buy"]}</span>',
                           f'{vt["buy"] / vt["total"] * 100:.1f}% of {vt["total"]}', "up"), unsafe_allow_html=True)
            b.markdown(hud("INDICATORS SELL", f'<span class="dn">{vt["sell"]}</span>',
                           f'{vt["sell"] / vt["total"] * 100:.1f}% of {vt["total"]}', "dn"), unsafe_allow_html=True)
            c_.markdown(hud("NEUTRAL", f'{vt["neutral"]}', f'{vt["neutral"] / vt["total"] * 100:.1f}%', "nt"),
                        unsafe_allow_html=True)
            d_.markdown(hud("MODEL PROBABILITY", f'{r["p_up"] * 100:.1f}% BUY',
                            f'{100 - r["p_up"] * 100:.1f}% SELL', "nt"), unsafe_allow_html=True)

            fr = sorted(r["fam_rows"], key=lambda x: -(x["buy"] + x["sell"]))
            fig = go.Figure()
            fig.add_trace(go.Bar(y=[x["family"] for x in fr], x=[x["buy"] for x in fr], name="BUY",
                                 orientation="h", marker_color="#00ff9c"))
            fig.add_trace(go.Bar(y=[x["family"] for x in fr], x=[-x["sell"] for x in fr], name="SELL",
                                 orientation="h", marker_color="#ff5c7a"))
            fig.update_layout(barmode="relative", title="BUY vs SELL votes by indicator family",
                              yaxis=dict(autorange="reversed"), legend=dict(orientation="h"))
            st.plotly_chart(style_fig(fig, 80 + 22 * len(fr)), key="fam_chart")

            frows = []
            for x in fr:
                lean = "BUY" if x["buy"] > x["sell"] else ("SELL" if x["sell"] > x["buy"] else "NEUTRAL")
                kind = {"BUY": "b", "SELL": "s"}.get(lean, "w")
                frows.append([x["family"], x["total"], f'<span class="up">{x["buy"]}</span>',
                              f'<span class="dn">{x["sell"]}</span>', x["neutral"], colored(lean, kind)])
            st.markdown("#### INDICATOR FAMILIES")
            st.markdown(html_table(["Family", "Variants", "BUY", "SELL", "Neutral", "Lean"], frows),
                        unsafe_allow_html=True)

            st.markdown("#### TRADING STRATEGIES")
            srows = []
            for name, sg, rule in r["strategies"]:
                kind = {"BUY": "b", "SELL": "s"}.get(sg, "w")
                srows.append([f"<b>{html.escape(name)}</b>", colored(sg, kind), html.escape(rule)])
            st.markdown(html_table(["Strategy", "Signal", "Rules used"], srows), unsafe_allow_html=True)

    # ---------------- MACRO & NEWS ----------------
    with tab3:
        macro = None
        try:
            macro = load_macro()
        except Exception:
            pass
        ms = macro_summary(macro)
        bias, bcl, why = fundamental_bias(sel_pair, sel_ticker, ms)
        names = {"dxy": "DOLLAR INDEX", "tnx": "US 10Y YIELD", "vix": "VIX", "spx": "S&P 500"}
        cols = st.columns(5)
        for i, k in enumerate(["dxy", "tnx", "vix", "spx"]):
            if k in ms:
                chg = ms[k]["chg"]
                cols[i].markdown(hud(names[k], f"{ms[k]['last']:,.2f}", f"{chg:+.2f}% today",
                                     "up" if chg >= 0 else "dn"), unsafe_allow_html=True)
        cols[4].markdown(hud(f"{sel_pair} BIAS", bias, why, bcl), unsafe_allow_html=True)
        st.markdown("#### LATEST HEADLINES")
        news = load_news(sel_ticker)
        if news:
            for t in news:
                st.markdown(f'<div class="news">{html.escape(t)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="news">Is waqt headlines available nahi hain.</div>', unsafe_allow_html=True)

    # ---------------- BACKTEST ----------------
    with tab4:
        if not good:
            st.markdown(hud("STATUS", "NO DATA YET", "", "nt"), unsafe_allow_html=True)
        else:
            tf_b = st.selectbox("TIMEFRAME", list(good.keys()), key="bt_tf")
            oos = good[tf_b]["oos"]
            pred = (oos["p_up"] > 0.5).astype(int)
            acc_all = float((pred == oos["y"]).mean())
            lean = oos[(oos["p_up"] - 0.5).abs() >= 0.05]
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(hud("TESTED CANDLES", f"{len(oos)}", "walk-forward, unseen data", "nt"), unsafe_allow_html=True)
            m2.markdown(hud("OVERALL ACCURACY", f"{acc_all * 100:.1f}%", "all candles", "nt"), unsafe_allow_html=True)
            if len(lean) > 0:
                win = ((lean["p_up"] > 0.5).astype(int) == lean["y"])
                m3.markdown(hud("STRONG-LEAN SIGNALS", f"{len(lean)}", "BUY ≥ 55% or SELL ≥ 55%", "nt"),
                            unsafe_allow_html=True)
                m4.markdown(hud("STRONG-LEAN ACCURACY", f"{win.mean() * 100:.1f}%", "", "nt"), unsafe_allow_html=True)
                pnl = np.where(win.values, PAYOUT, -1.0)
                eq = pd.Series(pnl, index=lean.index).cumsum()
                fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, mode="lines",
                                           line=dict(color="#00f0ff", width=2),
                                           fill="tozeroy", fillcolor="rgba(0,240,255,0.10)"))
                fig.update_layout(title="EQUITY CURVE · strong-lean signals (1 unit stake)")
                st.plotly_chart(style_fig(fig, 340), key="equity")
            else:
                m3.markdown(hud("STRONG-LEAN SIGNALS", "0", "", "nt"), unsafe_allow_html=True)
                m4.markdown(hud("STRONG-LEAN ACCURACY", "—", "", "nt"), unsafe_allow_html=True)

    # ---------------- CHARTS ----------------
    with tab5:
        if not good:
            st.markdown(hud("STATUS", "NO DATA YET", "", "nt"), unsafe_allow_html=True)
        else:
            tf_c = st.selectbox("TIMEFRAME", list(good.keys()), key="ch_tf")
            r = good[tf_c]
            ch = r["chart"]
            text, side, sc = signal_of(r["p_up"], r["stale"])
            vol = ch["Volume"] if ch["Volume"].sum() > 0 else (ch["High"] - ch["Low"]) / ch["Close"] * 1e6
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(
                x=ch.index, open=ch["Open"], high=ch["High"], low=ch["Low"], close=ch["Close"],
                increasing_line_color="#00ff9c", decreasing_line_color="#ff5c7a", name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=ch.index, y=ch["Close"].ewm(span=20).mean(), name="EMA20",
                                     line=dict(color="#00f0ff", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=ch.index, y=ch["Close"].ewm(span=50).mean(), name="EMA50",
                                     line=dict(color="#ff6be6", width=1.5)), row=1, col=1)
            colors = np.where(ch["Close"] >= ch["Open"], "#00ff9c", "#ff5c7a")
            fig.add_trace(go.Bar(x=ch.index, y=vol, marker_color=colors, name="Volume"), row=2, col=1)
            lv = levels(side, live_price, r["atr"]) if live_price else None
            if lv:
                fig.add_hline(y=lv[0], line_dash="dot", line_color="#00f0ff", row=1, col=1,
                              annotation_text="ENTRY", annotation_font_color="#f4fcff")
                fig.add_hline(y=lv[1], line_dash="dot", line_color="#00ff9c", row=1, col=1,
                              annotation_text="TP", annotation_font_color="#f4fcff")
                fig.add_hline(y=lv[2], line_dash="dot", line_color="#ff5c7a", row=1, col=1,
                              annotation_text="SL", annotation_font_color="#f4fcff")
            fig.update_layout(title=f"{sel_pair} · {tf_c} · {text}", xaxis_rangeslider_visible=False,
                              showlegend=False)
            st.plotly_chart(style_fig(fig, 580), key="candles")

st.caption("AXIOM TRADER · analytics tool, not financial advice.")
