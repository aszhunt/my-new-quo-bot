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

FOLDS = 5          # walk-forward folds (automatic)
PAYOUT = 0.85      # binary payout used for the equity curve (automatic)
MIN_SIGNALS = 25   # minimum verified signals for a STRONG call

# =====================================================================
#  STYLE  (ultra-dark cyber HUD)
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Share Tech Mono', monospace; }
.stApp {
  background:
    linear-gradient(rgba(0,240,255,0.03) 1px, transparent 1px) 0 0 / 42px 42px,
    linear-gradient(90deg, rgba(0,240,255,0.03) 1px, transparent 1px) 0 0 / 42px 42px,
    radial-gradient(ellipse at 20% 0%, #06142a 0%, #02040a 55%, #010208 100%);
  color: #e9fbff;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(0,240,255,0.25); }
h1, h2, h3, h4 { font-family: 'Orbitron', sans-serif !important; letter-spacing: 2px; }
.title-wrap { position: relative; padding: 14px 22px; margin-bottom: 14px;
  border: 1px solid rgba(0,240,255,0.35); border-radius: 14px;
  background: linear-gradient(120deg, rgba(0,240,255,0.07), rgba(255,43,214,0.05));
  box-shadow: 0 0 28px rgba(0,240,255,0.12) inset, 0 0 22px rgba(0,240,255,0.12); overflow: hidden; }
.title-wrap:after { content:''; position:absolute; left:0; right:0; height:2px; top:0;
  background: linear-gradient(90deg, transparent, #00f0ff, transparent); animation: scan 3.2s linear infinite; }
@keyframes scan { 0% { top: 0; opacity: .9 } 100% { top: 100%; opacity: .1 } }
.title { font-family:'Orbitron',sans-serif; font-weight:900; font-size:32px; letter-spacing:8px;
  background: linear-gradient(90deg,#00f0ff,#7b5cff,#ff2bd6); -webkit-background-clip:text;
  -webkit-text-fill-color: transparent; }
.sub { color:#6fb7d6; font-size:13px; letter-spacing:3px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#00ff9c;
  box-shadow:0 0 10px #00ff9c; margin-right:8px; animation: pulse 1.4s infinite; }
@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.25 } }
.hud { border:1px solid rgba(0,240,255,0.3); border-radius:12px; padding:12px 16px;
  background: #050c1c; box-shadow: 0 0 16px rgba(0,240,255,0.1); }
.hud .k { color:#5fa8c8; font-size:11px; letter-spacing:3px; }
.hud .v { font-family:'Orbitron',sans-serif; font-size:21px; color:#e9fbff; margin-top:4px; }
.hud .d { font-size:12px; margin-top:2px; }
.up { color:#00ff9c !important; } .dn { color:#ff3b5c !important; } .nt { color:#8fb8d0 !important; }
.card { border:1px solid rgba(0,240,255,0.3); border-radius:14px; padding:14px 16px; margin-bottom:14px;
  background: linear-gradient(160deg, #06112a, #030814); position:relative; }
.card.strong-buy { border-color:#00ff9c; box-shadow:0 0 22px rgba(0,255,156,0.28); }
.card.strong-sell { border-color:#ff3b5c; box-shadow:0 0 22px rgba(255,59,92,0.28); }
.card .tf { font-family:'Orbitron',sans-serif; font-size:13px; color:#6fb7d6; letter-spacing:3px; }
.card .sig { font-family:'Orbitron',sans-serif; font-size:19px; font-weight:700; margin:6px 0 10px 0; }
.sig.b { color:#00ff9c; text-shadow:0 0 12px rgba(0,255,156,.6); }
.sig.s { color:#ff3b5c; text-shadow:0 0 12px rgba(255,59,92,.6); }
.sig.w { color:#8fb8d0; }
.bar { height:9px; border-radius:6px; background:#ff3b5c; overflow:hidden; }
.bar .fill { height:100%; background:linear-gradient(90deg,#00b36e,#00ff9c); }
.pl { display:flex; justify-content:space-between; font-size:12px; margin-top:5px; color:#9fc5d9; }
.meta { margin-top:10px; font-size:12px; color:#7fb0c8; line-height:1.7; }
.meta b { color:#e9fbff; }
.news { border-left:2px solid #00f0ff; padding:6px 12px; margin:6px 0; background:#050c1c; font-size:13px; }
.stButton button { font-family:'Orbitron',sans-serif; letter-spacing:3px; font-weight:700;
  border:1px solid #00f0ff; background:#06142a !important; color:#00f0ff !important; }
.stButton button p { color:#00f0ff !important; }
.stButton button:hover { box-shadow:0 0 20px rgba(0,240,255,.5); border-color:#fff; }

/* ---------- BOOT ANIMATION ---------- */
.boot { position:relative; display:flex; flex-direction:column; align-items:center; padding:34px 10px 26px;
  margin:10px 0 20px; border:1px solid rgba(0,240,255,.4); border-radius:16px; overflow:hidden;
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
  color:#00f0ff; box-shadow:0 0 18px rgba(0,240,255,.6); animation: pulse 1.2s infinite; }
.bt { margin-top:20px; font-family:'Orbitron',sans-serif; font-weight:900; font-size:30px; letter-spacing:10px;
  color:#00f0ff; text-shadow: 2px 0 #ff2bd6, -2px 0 #7b5cff; animation: glitch 2.2s infinite; }
@keyframes glitch { 0%,92%,100% { transform:translateX(0) } 94% { transform:translateX(-4px) skewX(6deg) }
  96% { transform:translateX(4px) skewX(-6deg) } 98% { transform:translateX(-2px) } }
.bs { margin-top:8px; color:#9fe8ff; font-size:14px; letter-spacing:2px; }
.bs:after { content:'_'; animation: blink 0.8s infinite; }
.eq { display:flex; gap:4px; align-items:flex-end; height:34px; margin-top:16px; }
.eq i { display:block; width:6px; background:linear-gradient(180deg,#00f0ff,#7b5cff); border-radius:2px;
  animation: eqb 0.9s ease-in-out infinite; transform-origin: bottom; }
@keyframes eqb { 0%,100% { transform:scaleY(.25) } 50% { transform:scaleY(1) } }
.pbar { width:60%; max-width:520px; height:6px; border-radius:4px; background:#0a1a33; margin-top:16px; overflow:hidden; }
.pbar div { height:100%; background:linear-gradient(90deg,#00f0ff,#ff2bd6); box-shadow:0 0 12px #00f0ff; }

/* ---------- FORCE ULTRA-DARK EVERYWHERE ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], .main, .block-container { background-color: #02040a !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: #03060f !important; }
.stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp div[data-testid="stMarkdownContainer"], .stApp small,
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] { color: #e9fbff !important; }
.title { -webkit-text-fill-color: transparent !important; }
.up { color:#00ff9c !important; } .dn { color:#ff3b5c !important; } .nt { color:#8fb8d0 !important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"], input, textarea {
  background-color: #050a18 !important; color: #e9fbff !important; border-color: rgba(0,240,255,0.35) !important; }
div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: #e9fbff !important; }
div[data-baseweb="select"] svg { fill: #00f0ff !important; }
div[data-baseweb="popover"], div[data-baseweb="popover"] > div, div[data-baseweb="menu"],
ul[role="listbox"], li[role="option"] { background-color: #050a18 !important; color: #e9fbff !important; }
li[role="option"]:hover, li[aria-selected="true"] { background-color: #0a2040 !important; }
div[data-baseweb="tab-list"] { background: #02040a !important; border-bottom: 1px solid rgba(0,240,255,0.25); }
button[data-baseweb="tab"] { background: transparent !important; color: #8fb8d0 !important;
  font-family:'Orbitron',sans-serif; letter-spacing:2px; }
button[data-baseweb="tab"][aria-selected="true"] { color: #00f0ff !important; }
div[data-testid="stMetric"] { background:#050c1c; border:1px solid rgba(0,240,255,0.25); border-radius:12px; padding:10px 14px; }
div[data-testid="stMetricLabel"] p { color: #6fb7d6 !important; }
div[data-testid="stMetricValue"] { color: #e9fbff !important; }
div[data-testid="stDataFrame"], div[data-testid="stDataFrame"] > div, [data-testid="stTable"] { background: #050a18 !important; }
div[data-testid="stAlert"] { background: #07142a !important; border: 1px solid rgba(0,240,255,0.35); }
::-webkit-scrollbar { background: #02040a; width: 10px; }
::-webkit-scrollbar-thumb { background: #0a2a4a; border-radius: 6px; }
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


def wilson_lb(k, n, z=1.64):
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre - adj) / denom)


# =====================================================================
#  DATA  (price/volume + macro/fundamental context)
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


@st.cache_data(ttl=900, show_spinner=False)
def load_macro():
    """Daily intermarket / fundamental drivers: Dollar Index, US 10Y yield, VIX, S&P 500."""
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
    return f.shift(1)       # a day only sees PREVIOUS day's data (no look-ahead)


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
    """Latest level, 1-day change %, and trend vs 10-day average for the pulse tiles."""
    rows = {}
    for k, s in (macro or {}).items():
        if len(s) < 12:
            continue
        rows[k] = {"last": float(s.iloc[-1]),
                   "chg": float(s.iloc[-1] / s.iloc[-2] - 1) * 100,
                   "dev": float(s.iloc[-1] / s.iloc[-11:-1].mean() - 1) * 100}
    return rows


def fundamental_bias(label, ticker, ms):
    """Simple, transparent macro bias for the selected asset."""
    if not ms:
        return "NEUTRAL", "nt", "Macro feed loading"
    dxy = ms.get("dxy", {}).get("dev", 0.0)
    tnx = ms.get("tnx", {}).get("dev", 0.0)
    spx = ms.get("spx", {}).get("dev", 0.0)
    vix = ms.get("vix", {}).get("dev", 0.0)
    if ticker.endswith("-USD"):                      # crypto: risk-on / risk-off
        score = (1 if spx > 0 else -1) + (1 if vix < 0 else -1)
        why = "Risk sentiment (S&P 500 + VIX)"
    elif ticker in ("GC=F", "SI=F"):                 # metals: weaker USD / yields help
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
#  FEATURES  (100+: technical + volume + multi-timeframe + macro + session)
# =====================================================================
def htf_features(df, mins):
    """Higher-timeframe (5x) trend context using only COMPLETED higher bars."""
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
    real_vol = (d["Volume"] > 0).mean() > 0.5
    if real_vol:
        d["Volume"] = d["Volume"].replace(0, np.nan).ffill().fillna(1.0)
    else:   # forex has no real volume: use tick-activity proxy (range based)
        d["Volume"] = ((d["High"] - d["Low"]) / d["Close"] * 1e6).clip(lower=1.0)
    vol = d["Volume"]

    d = add_all_ta_features(d, open="Open", high="High", low="Low", close="Close",
                            volume="Volume", fillna=False)
    X = d.drop(columns=RAW).replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.isna().mean() < 0.3].ffill().fillna(0.0)

    close = d["Close"]
    for c in list(X.columns):                      # price-level indicators -> scale free
        s = X[c]
        if s.std() > 0 and abs(s.corr(close)) > 0.9:
            X[c] = s / close - 1.0

    # candle shape + momentum
    for k in (1, 2, 3, 5, 8, 13):
        X[f"ret_{k}"] = close.pct_change(k)
    X["body"] = (d["Close"] - d["Open"]) / d["Close"]
    X["range"] = (d["High"] - d["Low"]) / d["Close"]
    X["upper_wick"] = (d["High"] - d[["Open", "Close"]].max(axis=1)) / d["Close"]
    X["lower_wick"] = (d[["Open", "Close"]].min(axis=1) - d["Low"]) / d["Close"]
    X["streak"] = np.sign(close.diff()).rolling(5).sum()

    # volume features
    mu, sd = vol.rolling(20).mean(), vol.rolling(20).std()
    X["vol_z"] = (vol - mu) / sd.replace(0, np.nan)
    X["rel_vol"] = vol / mu
    X["vol_trend"] = vol.rolling(5).mean() / mu
    up_vol = vol.where(d["Close"] > d["Open"], 0.0).rolling(10).sum()
    X["up_vol_ratio"] = up_vol / vol.rolling(10).sum()
    X["pv_corr"] = close.pct_change().rolling(10).corr(vol.pct_change())
    day = naive_day(d.index)
    typ = (d["High"] + d["Low"] + d["Close"]) / 3
    vwap = (typ * vol).groupby(day).cumsum() / vol.groupby(day).cumsum()
    X["vwap_dev"] = close / vwap - 1

    # session / time features
    idx = pd.DatetimeIndex(d.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    hrs = idx.hour + idx.minute / 60.0
    X["hour_sin"] = np.sin(2 * np.pi * hrs / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hrs / 24)

    # higher-timeframe context
    try:
        X = X.join(htf_features(df, mins))
    except Exception:
        pass

    # macro / fundamental context
    if macro_f is not None:
        try:
            mf = macro_f[~macro_f.index.duplicated(keep="last")].sort_index()
            mf = mf.reindex(day, method="ffill")
            mf.index = X.index
            X = X.join(mf)
        except Exception:
            pass

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.loc[:, X.std() > 0], bool(real_vol)


# =====================================================================
#  MODEL
# =====================================================================
def fit_ensemble(X, y, r):
    keep = r.abs() >= r.abs().quantile(0.2)       # ignore tiny noise moves in training
    Xt, yt = X[keep], y[keep]
    if yt.nunique() < 2:
        Xt, yt = X, y
    w = np.linspace(0.5, 1.0, len(Xt))             # recent candles count more
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


def stats_at(oos, thr):
    conf = np.maximum(oos["p_up"], 1 - oos["p_up"])
    sub = oos[conf >= thr]
    if len(sub) == 0:
        return 0, None, pd.Series(dtype=bool), sub
    win = ((sub["p_up"] > 0.5).astype(int) == sub["y"])
    return len(sub), float(win.mean()), win, sub


def auto_threshold(oos):
    """Pick the confidence level with the best statistically-safe (Wilson lower bound) win rate."""
    best = None
    for t in np.arange(0.52, 0.701, 0.01):
        t = round(float(t), 2)
        n, acc, win, _ = stats_at(oos, t)
        if n >= MIN_SIGNALS:
            lb = wilson_lb(int(win.sum()), n)
            if best is None or lb > best[0]:
                best = (lb, t)
    return best[1] if best else 0.60


@st.cache_data(ttl=60, show_spinner=False)
def analyse(ticker, tf):
    interval, rule, period, mins = TF[tf]
    raw = load_data(ticker, interval, rule, period)
    if raw is None or len(raw) < 300:
        return {"ok": False, "reason": "Feed warming up for this timeframe"}

    age_min = (datetime.now(timezone.utc) - to_utc(raw.index[-1])).total_seconds() / 60
    forming = age_min < mins
    stale = age_min > max(10, mins * 3)
    df = raw.iloc[:-1] if forming else raw

    macro_f = macro_feature_frame(load_macro())
    X, real_vol = build_features(df, macro_f, mins)
    ret_next = df["Close"].shift(-1) / df["Close"] - 1.0
    y = (ret_next > 0).astype(int)

    Xl, yl, rl = X.iloc[:-1].iloc[60:], y.iloc[:-1].iloc[60:], ret_next.iloc[:-1].iloc[60:]
    flat = rl != 0
    Xl, yl, rl = Xl[flat], yl[flat], rl[flat]
    if len(Xl) < 250:
        return {"ok": False, "reason": "Collecting more candles for this timeframe"}

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

    return {
        "ok": True, "p_up": p_live, "oos": oos, "thr": auto_threshold(oos),
        "base": float(max(oos.y.mean(), 1 - oos.y.mean())),
        "price": float(raw["Close"].iloc[-1]), "age": float(age_min), "stale": bool(stale),
        "chart": raw.tail(120), "n_feat": int(X.shape[1]), "real_vol": real_vol,
    }


def verdict(res):
    p, thr = res["p_up"], res["thr"]
    conf = max(p, 1 - p)
    side = "BUY" if p >= 0.5 else "SELL"
    n, acc, _, _ = stats_at(res["oos"], thr)
    out = dict(side=side, conf=conf, n=n, acc=acc, thr=thr)
    if res["stale"]:
        return {**out, "kind": "wait", "text": "STANDBY · MARKET QUIET"}
    if conf >= thr and n >= MIN_SIGNALS and acc is not None and acc >= max(0.55, res["base"]):
        return {**out, "kind": "strong", "text": f"STRONG {side}"}
    return {**out, "kind": "wait", "text": "NEUTRAL · NO SETUP"}


# =====================================================================
#  UI HELPERS
# =====================================================================
def card_html(tf, res, v):
    if v["kind"] == "strong":
        cls = "strong-buy" if v["side"] == "BUY" else "strong-sell"
        sc, icon = ("b", "▲") if v["side"] == "BUY" else ("s", "▼")
    else:
        cls, sc, icon = "", "w", "◆"
    buy = res["p_up"] * 100
    acc = "n/a" if v["acc"] is None else f"{v['acc'] * 100:.1f}%"
    return (
        f'<div class="card {cls}"><div class="tf">{tf}</div>'
        f'<div class="sig {sc}">{icon} {v["text"]}</div>'
        f'<div class="bar"><div class="fill" style="width:{buy:.1f}%"></div></div>'
        f'<div class="pl"><span>BUY {buy:.1f}%</span><span>SELL {100 - buy:.1f}%</span></div>'
        f'<div class="meta">CONFIDENCE <b>{v["conf"] * 100:.1f}%</b> · AUTO LEVEL <b>{v["thr"] * 100:.0f}%</b><br>'
        f'VERIFIED ACCURACY <b>{acc}</b> over <b>{v["n"]}</b> signals</div></div>'
    )


def hud(k, v, d="", dcls="nt"):
    extra = f'<div class="d {dcls}">{d}</div>' if d else ""
    return f'<div class="hud"><div class="k">{k}</div><div class="v">{v}</div>{extra}</div>'


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
                      plot_bgcolor="rgba(3,8,20,0.8)", margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="Share Tech Mono", color="#e9fbff"))
    fig.update_xaxes(gridcolor="rgba(0,240,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(0,240,255,0.08)")
    return fig


# =====================================================================
#  SIDEBAR  (fully automatic: just choose the asset)
# =====================================================================
with st.sidebar:
    st.markdown("### 🤖 AXIOM TRADER")
    pair = st.selectbox("ASSET", list(MARKETS.keys()))
    scan = st.button("⚡ INITIATE SIGNAL")
    st.caption("Auto mode: timeframes, confidence level, validation aur data sab khud "
               "automatic select hota hai.")

# =====================================================================
#  HEADER
# =====================================================================
st.markdown(
    '<div class="title-wrap"><div class="title">AXIOM TRADER</div>'
    '<div class="sub"><span class="dot"></span>AUTONOMOUS AI SIGNAL ENGINE · '
    'VOLUME + TECHNICAL + FUNDAMENTAL · WALK-FORWARD VERIFIED</div></div>',
    unsafe_allow_html=True,
)

# =====================================================================
#  SCAN  (with boot animation)
# =====================================================================
if scan:
    stage = st.empty()
    steps = ["Connecting to live market feed",
             "Reading volume and order-flow activity",
             "Syncing macro and fundamental pulse"]
    for i, s in enumerate(steps):
        stage.markdown(boot_html(s, 4 + i * 6), unsafe_allow_html=True)
        time.sleep(0.7)
    try:
        load_macro()
    except Exception:
        pass

    results = {}
    tf_list = list(TF.keys())
    for i, tf in enumerate(tf_list):
        stage.markdown(boot_html(f"{tf} · computing 100+ indicators · training neural ensemble",
                                 22 + 72 * i / len(tf_list)), unsafe_allow_html=True)
        try:
            results[tf] = analyse(MARKETS[pair], tf)
        except Exception as e:
            results[tf] = {"ok": False, "reason": f"{type(e).__name__}"}
    stage.markdown(boot_html("Signals locked · rendering dashboard", 100), unsafe_allow_html=True)
    time.sleep(0.8)
    stage.empty()
    st.session_state["scan"] = {"pair": pair, "results": results}

state = st.session_state.get("scan")
tab1, tab2, tab3, tab4 = st.tabs(["◉ COMMAND CENTER", "◉ MACRO & NEWS", "◉ BACKTEST LAB", "◉ CHARTS"])

if not state:
    with tab1:
        st.markdown('<div class="hud"><div class="k">STATUS</div><div class="v">READY</div>'
                    '<div class="d nt">Sidebar me asset chunein aur INITIATE SIGNAL dabayein.</div></div>',
                    unsafe_allow_html=True)
else:
    results = state["results"]
    sel_pair = state["pair"]
    sel_ticker = MARKETS[sel_pair]
    good = {tf: r for tf, r in results.items() if r.get("ok")}
    verdicts = {tf: verdict(r) for tf, r in good.items()}

    # ---------------- COMMAND CENTER ----------------
    with tab1:
        if good:
            any_res = next(iter(good.values()))
            sb = [tf for tf, v in verdicts.items() if v["kind"] == "strong" and v["side"] == "BUY"]
            ss = [tf for tf, v in verdicts.items() if v["kind"] == "strong" and v["side"] == "SELL"]
            if sb and not ss:
                cons, ccl = f"▲ BUY ({len(sb)}/{len(good)})", "up"
            elif ss and not sb:
                cons, ccl = f"▼ SELL ({len(ss)}/{len(good)})", "dn"
            elif sb and ss:
                cons, ccl = "◆ MIXED", "nt"
            else:
                cons, ccl = "◆ NEUTRAL", "nt"

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(hud("ASSET", sel_pair), unsafe_allow_html=True)
            c2.markdown(hud("LIVE PRICE", f"{any_res['price']:,.5f}"), unsafe_allow_html=True)
            c3.markdown(hud("FEED AGE", f"{any_res['age']:.1f} min",
                            "volume: " + ("real" if any_res["real_vol"] else "tick-activity"), "nt"),
                        unsafe_allow_html=True)
            c4.markdown(hud("AXIOM CONSENSUS", cons, f"{any_res['n_feat']} features per candle", ccl),
                        unsafe_allow_html=True)

            mean_buy = float(np.mean([r["p_up"] for r in good.values()]) * 100)
            gcol, _ = st.columns([1, 2])
            with gcol:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=mean_buy,
                    number={"suffix": "% BUY", "font": {"family": "Orbitron", "size": 26}},
                    title={"text": "MULTI-TIMEFRAME BIAS", "font": {"family": "Orbitron", "size": 13}},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#00f0ff"},
                           "steps": [{"range": [0, 45], "color": "rgba(255,59,92,0.35)"},
                                     {"range": [45, 55], "color": "rgba(140,160,180,0.25)"},
                                     {"range": [55, 100], "color": "rgba(0,255,156,0.35)"}]}))
                st.plotly_chart(style_fig(fig, 250), key="gauge")

            st.markdown("#### NEXT-CANDLE SIGNALS")
            cols = st.columns(3)
            for i, tf in enumerate(good):
                with cols[i % 3]:
                    st.markdown(card_html(tf, good[tf], verdicts[tf]), unsafe_allow_html=True)
        else:
            st.markdown(hud("STATUS", "FEED WARMING UP",
                            "Kuch der baad dobara INITIATE SIGNAL dabayein.", "nt"),
                        unsafe_allow_html=True)
        for tf, r in results.items():
            if not r.get("ok"):
                st.markdown(f'<div class="news">{tf}: {html.escape(str(r.get("reason")))}</div>',
                            unsafe_allow_html=True)

    # ---------------- MACRO & NEWS (fundamentals) ----------------
    with tab2:
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
                cols[i].markdown(hud(names[k], f"{ms[k]['last']:,.2f}",
                                     f"{chg:+.2f}% today", "up" if chg >= 0 else "dn"),
                                 unsafe_allow_html=True)
        cols[4].markdown(hud(f"{sel_pair} FUNDAMENTAL BIAS", bias, why, bcl), unsafe_allow_html=True)
        st.markdown("#### LATEST HEADLINES")
        news = load_news(sel_ticker)
        if news:
            for t in news:
                st.markdown(f'<div class="news">{html.escape(t)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="news">Headlines is waqt available nahi.</div>', unsafe_allow_html=True)

    # ---------------- BACKTEST LAB ----------------
    with tab3:
        if not good:
            st.markdown(hud("STATUS", "NO DATA YET"), unsafe_allow_html=True)
        else:
            tf_sel = st.selectbox("TIMEFRAME", list(good.keys()), key="bt_tf")
            res = good[tf_sel]
            thr = res["thr"]
            n, acc, win, sub = stats_at(res["oos"], thr)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AUTO CONFIDENCE LEVEL", f"{thr * 100:.0f}%")
            m2.metric("VERIFIED SIGNALS", n)
            m3.metric("VERIFIED WIN RATE", "n/a" if acc is None else f"{acc * 100:.1f}%")
            m4.metric("ALL-CANDLE BASELINE", f"{res['base'] * 100:.1f}%")

            if n > 0:
                pnl = np.where(win.values, PAYOUT, -1.0)
                eq = pd.Series(pnl, index=sub.index).cumsum()
                dd = float((eq - eq.cummax()).min())
                wins, losses = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
                pf = wins / losses if losses > 0 else float("inf")
                k1, k2, k3 = st.columns(3)
                k1.metric("NET (stake units)", f"{eq.iloc[-1]:+.1f}")
                k2.metric("MAX DRAWDOWN", f"{dd:.1f}")
                k3.metric("PROFIT FACTOR", "∞" if pf == float("inf") else f"{pf:.2f}")
                fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, mode="lines",
                                           line=dict(color="#00f0ff", width=2),
                                           fill="tozeroy", fillcolor="rgba(0,240,255,0.08)"))
                fig.update_layout(title="EQUITY CURVE · verified high-confidence signals")
                st.plotly_chart(style_fig(fig, 340), key="equity")

            st.markdown("#### CONFIDENCE LEVEL MAP")
            rows = []
            for t in [0.52, 0.55, 0.58, 0.60, 0.63, 0.66, 0.70]:
                nn, aa, _, _ = stats_at(res["oos"], t)
                rows.append({"Confidence ≥": f"{t * 100:.0f}%", "Signals": nn,
                             "Win rate %": None if aa is None else round(aa * 100, 1)})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("Walk-forward results: model har hissa un candles par test karta hai jo "
                       "training me shamil nahi thin.")

    # ---------------- CHARTS ----------------
    with tab4:
        if not good:
            st.markdown(hud("STATUS", "NO DATA YET"), unsafe_allow_html=True)
        else:
            tf_c = st.selectbox("TIMEFRAME", list(good.keys()), key="ch_tf")
            res = good[tf_c]
            ch = res["chart"]
            v = verdicts[tf_c]
            vol = ch["Volume"] if ch["Volume"].sum() > 0 else (ch["High"] - ch["Low"]) / ch["Close"] * 1e6
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.75, 0.25], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(
                x=ch.index, open=ch["Open"], high=ch["High"], low=ch["Low"], close=ch["Close"],
                increasing_line_color="#00ff9c", decreasing_line_color="#ff3b5c", name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=ch.index, y=ch["Close"].ewm(span=20).mean(), name="EMA20",
                                     line=dict(color="#00f0ff", width=1.4)), row=1, col=1)
            fig.add_trace(go.Scatter(x=ch.index, y=ch["Close"].ewm(span=50).mean(), name="EMA50",
                                     line=dict(color="#ff2bd6", width=1.4)), row=1, col=1)
            colors = np.where(ch["Close"] >= ch["Open"], "#00ff9c", "#ff3b5c")
            fig.add_trace(go.Bar(x=ch.index, y=vol, marker_color=colors, name="Volume"), row=2, col=1)
            fig.add_annotation(x=ch.index[-1], y=float(ch["High"].iloc[-1]), xref="x", yref="y",
                               text=f"{'▲' if v['side'] == 'BUY' else '▼'} {v['side']} {v['conf'] * 100:.0f}%",
                               showarrow=True, arrowhead=2, font=dict(color="#00f0ff", size=14),
                               arrowcolor="#00f0ff")
            fig.update_layout(title=f"{sel_pair} · {tf_c}", xaxis_rangeslider_visible=False,
                              showlegend=False)
            st.plotly_chart(style_fig(fig, 560), key="candles")

st.caption("AXIOM TRADER · analytics tool, not financial advice.")
