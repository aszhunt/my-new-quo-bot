import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from ta import add_all_ta_features

warnings.filterwarnings("ignore")

st.set_page_config(page_title="NEURAL TRADE CORE", page_icon="🤖", layout="wide")

# =====================================================================
#  STYLE  (cyber / robotic HUD theme)
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Share Tech Mono', monospace; }
.stApp {
  background:
    linear-gradient(rgba(0,240,255,0.035) 1px, transparent 1px) 0 0 / 42px 42px,
    linear-gradient(90deg, rgba(0,240,255,0.035) 1px, transparent 1px) 0 0 / 42px 42px,
    radial-gradient(ellipse at 20% 0%, #0b1d3a 0%, #050912 55%, #02040a 100%);
  color: #cfeaff;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060d1f 0%, #040813 100%);
  border-right: 1px solid rgba(0,240,255,0.25);
}
h1, h2, h3, h4 { font-family: 'Orbitron', sans-serif !important; letter-spacing: 2px; }
.title-wrap { position: relative; padding: 14px 22px; margin-bottom: 14px;
  border: 1px solid rgba(0,240,255,0.35); border-radius: 14px;
  background: linear-gradient(120deg, rgba(0,240,255,0.08), rgba(255,43,214,0.06));
  box-shadow: 0 0 28px rgba(0,240,255,0.15) inset, 0 0 22px rgba(0,240,255,0.12);
  overflow: hidden; }
.title-wrap:after { content:''; position:absolute; left:0; right:0; height:2px; top:0;
  background: linear-gradient(90deg, transparent, #00f0ff, transparent);
  animation: scan 3.2s linear infinite; }
@keyframes scan { 0% { top: 0; opacity: .9 } 100% { top: 100%; opacity: .1 } }
.title { font-family:'Orbitron',sans-serif; font-weight:900; font-size:30px; letter-spacing:6px;
  background: linear-gradient(90deg,#00f0ff,#7b5cff,#ff2bd6); -webkit-background-clip:text;
  -webkit-text-fill-color: transparent; }
.sub { color:#6fb7d6; font-size:13px; letter-spacing:3px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#00ff9c;
  box-shadow:0 0 10px #00ff9c; margin-right:8px; animation: pulse 1.4s infinite; }
@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.25 } }
.hud { border:1px solid rgba(0,240,255,0.3); border-radius:12px; padding:12px 16px;
  background: rgba(6,16,38,0.75); box-shadow: 0 0 16px rgba(0,240,255,0.1); }
.hud .k { color:#5fa8c8; font-size:11px; letter-spacing:3px; }
.hud .v { font-family:'Orbitron',sans-serif; font-size:22px; color:#e9fbff; margin-top:4px; }
.card { border:1px solid rgba(0,240,255,0.3); border-radius:14px; padding:14px 16px; margin-bottom:14px;
  background: linear-gradient(160deg, rgba(8,20,46,0.92), rgba(4,10,24,0.92)); position:relative; }
.card.strong-buy { border-color:#00ff9c; box-shadow:0 0 22px rgba(0,255,156,0.28); }
.card.strong-sell { border-color:#ff3b5c; box-shadow:0 0 22px rgba(255,59,92,0.28); }
.card .tf { font-family:'Orbitron',sans-serif; font-size:13px; color:#6fb7d6; letter-spacing:3px; }
.card .sig { font-family:'Orbitron',sans-serif; font-size:21px; font-weight:700; margin:6px 0 10px 0; }
.sig.b { color:#00ff9c; text-shadow:0 0 12px rgba(0,255,156,.6); }
.sig.s { color:#ff3b5c; text-shadow:0 0 12px rgba(255,59,92,.6); }
.sig.w { color:#8aa4b8; }
.bar { height:9px; border-radius:6px; background:#ff3b5c; overflow:hidden; }
.bar .fill { height:100%; background:linear-gradient(90deg,#00b36e,#00ff9c); }
.pl { display:flex; justify-content:space-between; font-size:12px; margin-top:5px; color:#9fc5d9; }
.meta { margin-top:10px; font-size:12px; color:#7fb0c8; line-height:1.7; }
.meta b { color:#e9fbff; }
.badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:11px; letter-spacing:2px;
  border:1px solid #00f0ff; color:#00f0ff; }
div[data-testid="stMetric"] { background: rgba(6,16,38,0.75); border:1px solid rgba(0,240,255,0.25);
  border-radius:12px; padding:10px 14px; }
button[data-baseweb="tab"] { font-family:'Orbitron',sans-serif; letter-spacing:2px; }
.stButton button { font-family:'Orbitron',sans-serif; letter-spacing:3px; font-weight:700;
  border:1px solid #00f0ff; background: linear-gradient(90deg, rgba(0,240,255,.18), rgba(123,92,255,.18));
  color:#00f0ff; }
.stButton button:hover { box-shadow:0 0 20px rgba(0,240,255,.5); color:#fff; border-color:#fff; }
.warn { border:1px solid #ffb020; color:#ffb020; border-radius:10px; padding:8px 14px; margin:8px 0;
  background: #1a1203; font-size:13px; }

/* ---------- FORCE ULTRA-DARK EVERYWHERE (no white / light surfaces) ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], .main, .block-container { background-color: #02040a !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: #03060f !important; }
.stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp div[data-testid="stMarkdownContainer"], .stApp small, .stApp caption,
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] { color: #e9fbff !important; }
/* inputs / selects / multiselect */
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"],
input, textarea { background-color: #050a18 !important; color: #e9fbff !important;
  border-color: rgba(0,240,255,0.35) !important; }
div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: #e9fbff !important; }
div[data-baseweb="select"] svg { fill: #00f0ff !important; }
span[data-baseweb="tag"] { background-color: #0a1a33 !important; border: 1px solid #00f0ff; }
span[data-baseweb="tag"] span { color: #00f0ff !important; }
/* dropdown popups */
div[data-baseweb="popover"], div[data-baseweb="popover"] > div, div[data-baseweb="menu"],
ul[role="listbox"], li[role="option"] { background-color: #050a18 !important; color: #e9fbff !important; }
li[role="option"]:hover, li[aria-selected="true"] { background-color: #0a2040 !important; }
/* radio / slider / tabs */
div[role="radiogroup"] label, div[role="radiogroup"] p { color: #e9fbff !important; }
div[data-baseweb="slider"] div { color: #e9fbff !important; }
div[data-testid="stSliderThumbValue"], div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] {
  color: #00f0ff !important; background: transparent !important; }
div[data-baseweb="tab-list"] { background: #02040a !important; border-bottom: 1px solid rgba(0,240,255,0.25); }
button[data-baseweb="tab"] { background: transparent !important; color: #8fb8d0 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #00f0ff !important; }
/* metrics, alerts, tables, progress */
div[data-testid="stMetric"] label, div[data-testid="stMetricLabel"] p { color: #6fb7d6 !important; }
div[data-testid="stMetricValue"] { color: #e9fbff !important; }
div[data-testid="stAlert"] { background: #07142a !important; border: 1px solid rgba(0,240,255,0.35);
  color: #e9fbff !important; }
div[data-testid="stDataFrame"], div[data-testid="stDataFrame"] > div, [data-testid="stTable"],
[data-testid="stTable"] table, [data-testid="stTable"] th, [data-testid="stTable"] td {
  background: #050a18 !important; color: #e9fbff !important; }
div[data-testid="stProgress"] p { color: #e9fbff !important; }
.stButton button, .stButton button p { background: #06142a !important; color: #00f0ff !important; }
::-webkit-scrollbar { background: #02040a; width: 10px; }
::-webkit-scrollbar-thumb { background: #0a2a4a; border-radius: 6px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =====================================================================
#  MARKETS / TIMEFRAMES
# =====================================================================
MARKETS = {
    # Majors
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X", "AUD/USD": "AUDUSD=X", "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    # Crosses
    "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X", "EUR/CHF": "EURCHF=X",
    "EUR/AUD": "EURAUD=X", "EUR/CAD": "EURCAD=X", "EUR/NZD": "EURNZD=X",
    "GBP/JPY": "GBPJPY=X", "GBP/CHF": "GBPCHF=X", "GBP/AUD": "GBPAUD=X",
    "GBP/CAD": "GBPCAD=X", "GBP/NZD": "GBPNZD=X", "AUD/JPY": "AUDJPY=X",
    "AUD/CHF": "AUDCHF=X", "AUD/CAD": "AUDCAD=X", "AUD/NZD": "AUDNZD=X",
    "CAD/JPY": "CADJPY=X", "CAD/CHF": "CADCHF=X", "CHF/JPY": "CHFJPY=X",
    "NZD/JPY": "NZDJPY=X", "NZD/CHF": "NZDCHF=X", "NZD/CAD": "NZDCAD=X",
    # Exotics
    "USD/TRY": "USDTRY=X", "USD/ZAR": "USDZAR=X", "USD/MXN": "USDMXN=X",
    "USD/SGD": "USDSGD=X", "USD/HKD": "USDHKD=X", "USD/NOK": "USDNOK=X",
    "USD/SEK": "USDSEK=X", "USD/PLN": "USDPLN=X", "USD/INR": "USDINR=X",
    "USD/PKR": "USDPKR=X", "USD/CNH": "USDCNH=X",
    # Crypto and metals (24/7 or near 24/5)
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
#  DATA
# =====================================================================
def to_utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


@st.cache_data(ttl=30, show_spinner=False)
def load_data(ticker, interval, rule, period):
    df = yf.download(ticker, interval=interval, period=period,
                     progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[RAW].dropna(subset=["Close"])
    if rule:
        df = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    return df


# =====================================================================
#  FEATURES  (~90 TA indicators + candle shape + lagged returns)
# =====================================================================
def build_features(df):
    d = df.copy()
    d["Volume"] = d["Volume"].replace(0, np.nan).fillna(1.0)
    d = add_all_ta_features(d, open="Open", high="High", low="Low", close="Close",
                            volume="Volume", fillna=False)
    X = d.drop(columns=RAW).replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.isna().mean() < 0.3].ffill().fillna(0.0)

    close = d["Close"]
    for c in list(X.columns):                      # make price-level indicators scale free
        s = X[c]
        if s.std() > 0 and abs(s.corr(close)) > 0.9:
            X[c] = s / close - 1.0

    for k in (1, 2, 3, 5, 8, 13):
        X[f"ret_{k}"] = close.pct_change(k)
    X["body"] = (d["Close"] - d["Open"]) / d["Close"]
    X["range"] = (d["High"] - d["Low"]) / d["Close"]
    X["upper_wick"] = (d["High"] - d[["Open", "Close"]].max(axis=1)) / d["Close"]
    X["lower_wick"] = (d[["Open", "Close"]].min(axis=1) - d["Low"]) / d["Close"]
    X["streak"] = np.sign(close.diff()).rolling(5).sum()

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.loc[:, X.std() > 0]


# =====================================================================
#  MODEL  (ensemble: boosting + extra-trees + logistic)
# =====================================================================
def fit_ensemble(X, y, r):
    keep = r.abs() >= r.abs().quantile(0.2)       # drop tiny noise moves from training
    Xt, yt = X[keep], y[keep]
    if yt.nunique() < 2:
        Xt, yt = X, y
    models = [
        HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_depth=4,
                                       l2_regularization=1.0, random_state=42),
        ExtraTreesClassifier(n_estimators=150, max_depth=7, min_samples_leaf=10,
                             n_jobs=-1, random_state=42),
        make_pipeline(RobustScaler(), FunctionTransformer(lambda a: np.clip(a, -5, 5)),
                      LogisticRegression(C=0.05, max_iter=400)),
    ]
    for m in models:
        m.fit(Xt, yt)
    return models


def ens_proba(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


@st.cache_data(ttl=60, show_spinner=False)
def analyse(ticker, tf, folds):
    """Walk-forward backtest + live next-candle probability for one timeframe."""
    interval, rule, period, mins = TF[tf]
    raw = load_data(ticker, interval, rule, period)
    if raw is None or len(raw) < 300:
        return {"ok": False, "reason": "Data unavailable / not enough candles"}

    age_min = (datetime.now(timezone.utc) - to_utc(raw.index[-1])).total_seconds() / 60
    forming = age_min < mins                       # last candle still open
    stale = age_min > max(10, mins * 3)            # feed old (market closed / delayed)
    df = raw.iloc[:-1] if forming else raw         # model only sees CLOSED candles

    X = build_features(df)
    ret_next = df["Close"].shift(-1) / df["Close"] - 1.0
    y = (ret_next > 0).astype(int)

    Xl, yl, rl = X.iloc[:-1], y.iloc[:-1], ret_next.iloc[:-1]
    Xl, yl, rl = Xl.iloc[60:], yl.iloc[60:], rl.iloc[60:]          # indicator warm-up
    flat = rl != 0
    Xl, yl, rl = Xl[flat], yl[flat], rl[flat]
    if len(Xl) < 250:
        return {"ok": False, "reason": "Not enough usable candles"}

    n = len(Xl)
    edges = np.linspace(int(n * 0.5), n, folds + 1).astype(int)
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        models = fit_ensemble(Xl.iloc[:a], yl.iloc[:a], rl.iloc[:a])
        p = ens_proba(models, Xl.iloc[a:b])
        parts.append(pd.DataFrame({"p_up": p, "y": yl.iloc[a:b].values,
                                   "ret": rl.iloc[a:b].values}, index=Xl.index[a:b]))
    oos = pd.concat(parts)

    final = fit_ensemble(Xl, yl, rl)
    p_live = float(ens_proba(final, X.iloc[[-1]])[0])

    return {
        "ok": True, "p_up": p_live, "oos": oos, "base": float(max(oos.y.mean(), 1 - oos.y.mean())),
        "price": float(raw["Close"].iloc[-1]), "age": float(age_min), "forming": bool(forming),
        "stale": bool(stale), "chart": raw.tail(120), "n_feat": int(X.shape[1]), "n_rows": int(n),
        "last_time": to_utc(raw.index[-1]),
    }


# =====================================================================
#  SIGNAL LOGIC
# =====================================================================
def stats_at(oos, thr):
    conf = np.maximum(oos["p_up"], 1 - oos["p_up"])
    sub = oos[conf >= thr]
    if len(sub) == 0:
        return 0, None, pd.Series(dtype=bool), sub
    win = ((sub["p_up"] > 0.5).astype(int) == sub["y"])
    return len(sub), float(win.mean()), win, sub


def verdict(res, thr, min_n=30):
    p = res["p_up"]
    conf = max(p, 1 - p)
    side = "BUY" if p >= 0.5 else "SELL"
    n, acc, _, _ = stats_at(res["oos"], thr)
    if res["stale"]:
        return dict(side=side, conf=conf, n=n, acc=acc, kind="wait", text="DATA STALE")
    if conf >= thr and n >= min_n and acc is not None and acc >= max(0.55, res["base"]):
        return dict(side=side, conf=conf, n=n, acc=acc, kind="strong", text=f"STRONG {side}")
    if conf >= thr:
        return dict(side=side, conf=conf, n=n, acc=acc, kind="wait", text="WAIT · UNVERIFIED")
    return dict(side=side, conf=conf, n=n, acc=acc, kind="wait", text="WAIT · LOW CONF")


def card_html(tf, res, v):
    if v["kind"] == "strong":
        cls = "strong-buy" if v["side"] == "BUY" else "strong-sell"
        sc = "b" if v["side"] == "BUY" else "s"
        icon = "▲" if v["side"] == "BUY" else "▼"
    else:
        cls, sc, icon = "", "w", "◆"
    buy = res["p_up"] * 100
    acc = "n/a" if v["acc"] is None else f"{v['acc'] * 100:.1f}%"
    return (
        f'<div class="card {cls}"><div class="tf">{tf}</div>'
        f'<div class="sig {sc}">{icon} {v["text"]}</div>'
        f'<div class="bar"><div class="fill" style="width:{buy:.1f}%"></div></div>'
        f'<div class="pl"><span>BUY {buy:.1f}%</span><span>SELL {100 - buy:.1f}%</span></div>'
        f'<div class="meta">CONFIDENCE <b>{v["conf"] * 100:.1f}%</b><br>'
        f'BACKTEST ACC @ THRESHOLD <b>{acc}</b> ({v["n"]} signals)<br>'
        f'BASELINE <b>{res["base"] * 100:.1f}%</b></div></div>'
    )


def hud(k, v):
    return f'<div class="hud"><div class="k">{k}</div><div class="v">{v}</div></div>'


def style_fig(fig, h=420):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(5,12,28,0.6)", margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="Share Tech Mono", color="#cfeaff"))
    fig.update_xaxes(gridcolor="rgba(0,240,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(0,240,255,0.08)")
    return fig


# =====================================================================
#  SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙ CONTROL PANEL")
    group = st.radio("MARKET", ["All", "Forex", "Crypto / Metals"], horizontal=True)
    names = list(MARKETS.keys())
    if group == "Forex":
        names = [m for m in names if MARKETS[m].endswith("=X")]
    elif group == "Crypto / Metals":
        names = [m for m in names if not MARKETS[m].endswith("=X")]
    pair = st.selectbox("PAIR", names)
    tfs = st.multiselect("TIMEFRAMES", list(TF.keys()), default=list(TF.keys()))
    thr_pct = st.slider("HIGH-CONFIDENCE THRESHOLD %", 55, 75, 60)
    folds = st.slider("WALK-FORWARD FOLDS", 3, 8, 5)
    payout_pct = st.slider("BINARY PAYOUT %", 70, 95, 85)
    scan = st.button("⚡ INITIATE SCAN")
    st.caption("Signal tabhi 'STRONG' banta hai jab live confidence threshold se upar ho "
               "AND backtest me us threshold par accuracy verified ho.")

thr = thr_pct / 100.0
payout = payout_pct / 100.0

# =====================================================================
#  HEADER
# =====================================================================
st.markdown(
    '<div class="title-wrap"><div class="title">NEURAL TRADE CORE</div>'
    '<div class="sub"><span class="dot"></span>AI CANDLE PROBABILITY ENGINE · '
    'WALK-FORWARD VERIFIED · ~90 INDICATORS</div></div>',
    unsafe_allow_html=True,
)

# =====================================================================
#  RUN SCAN
# =====================================================================
if scan:
    if not tfs:
        st.warning("Kam az kam aik timeframe select karein.")
    else:
        results = {}
        bar = st.progress(0.0, text="Booting neural core...")
        for i, tf in enumerate(tfs):
            bar.progress(i / len(tfs), text=f"Scanning {pair} · {tf} · training ensemble...")
            try:
                results[tf] = analyse(MARKETS[pair], tf, folds)
            except Exception as e:
                results[tf] = {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:80]}"}
        bar.empty()
        st.session_state["scan"] = {"pair": pair, "results": results,
                                    "time": datetime.now(timezone.utc)}

state = st.session_state.get("scan")

tab1, tab2, tab3, tab4 = st.tabs(["◉ COMMAND CENTER", "◉ BACKTEST LAB", "◉ CHARTS", "◉ SYSTEM INFO"])

if not state:
    with tab1:
        st.markdown('<div class="warn">👈 Sidebar me pair aur timeframes chunein, phir '
                    '<b>INITIATE SCAN</b> dabayein.</div>', unsafe_allow_html=True)
else:
    results = state["results"]
    good = {tf: r for tf, r in results.items() if r.get("ok")}
    verdicts = {tf: verdict(r, thr) for tf, r in good.items()}

    # ---------------- COMMAND CENTER ----------------
    with tab1:
        if good:
            any_res = next(iter(good.values()))
            strong_b = [tf for tf, v in verdicts.items() if v["kind"] == "strong" and v["side"] == "BUY"]
            strong_s = [tf for tf, v in verdicts.items() if v["kind"] == "strong" and v["side"] == "SELL"]
            if strong_b and not strong_s:
                cons = f"▲ BUY ({len(strong_b)}/{len(good)})"
            elif strong_s and not strong_b:
                cons = f"▼ SELL ({len(strong_s)}/{len(good)})"
            elif strong_b and strong_s:
                cons = "⚠ CONFLICT"
            else:
                cons = "◆ STAND BY"

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(hud("ASSET", state["pair"]), unsafe_allow_html=True)
            c2.markdown(hud("LIVE PRICE", f"{any_res['price']:,.5f}"), unsafe_allow_html=True)
            c3.markdown(hud("LAST CANDLE AGE", f"{any_res['age']:.1f} min"), unsafe_allow_html=True)
            c4.markdown(hud("CONSENSUS", cons), unsafe_allow_html=True)

            if any_res["stale"]:
                st.markdown('<div class="warn">⚠ Data purani hai (market band ya feed delayed). '
                            'Signals WAIT par force kiye gaye hain.</div>', unsafe_allow_html=True)

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

        for tf, r in results.items():
            if not r.get("ok"):
                st.markdown(f'<div class="warn">{tf}: {r.get("reason")}</div>', unsafe_allow_html=True)

    # ---------------- BACKTEST LAB ----------------
    with tab2:
        if not good:
            st.info("Pehle scan chalayein.")
        else:
            tf_sel = st.selectbox("TIMEFRAME", list(good.keys()), key="bt_tf")
            res = good[tf_sel]
            n, acc, win, sub = stats_at(res["oos"], thr)
            breakeven = 1 / (1 + payout)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("SIGNALS (OUT-OF-SAMPLE)", n)
            m2.metric("WIN RATE", "n/a" if acc is None else f"{acc * 100:.1f}%")
            m3.metric("BREAK-EVEN WIN RATE", f"{breakeven * 100:.1f}%")
            m4.metric("BASELINE (NO MODEL)", f"{res['base'] * 100:.1f}%")

            if n > 0:
                pnl = np.where(win.values, payout, -1.0)
                eq = pd.Series(pnl, index=sub.index).cumsum()
                dd = float((eq - eq.cummax()).min())
                wins = float(pnl[pnl > 0].sum())
                losses = float(-pnl[pnl < 0].sum())
                pf = wins / losses if losses > 0 else float("inf")
                k1, k2, k3 = st.columns(3)
                k1.metric("NET (units of 1 stake)", f"{eq.iloc[-1]:+.1f}")
                k2.metric("MAX DRAWDOWN", f"{dd:.1f}")
                k3.metric("PROFIT FACTOR", "∞" if pf == float("inf") else f"{pf:.2f}")

                fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, mode="lines",
                                           line=dict(color="#00f0ff", width=2),
                                           fill="tozeroy", fillcolor="rgba(0,240,255,0.08)"))
                fig.update_layout(title="EQUITY CURVE (high-confidence signals only)")
                st.plotly_chart(style_fig(fig, 340), key="equity")
                if acc is not None and acc < breakeven:
                    st.markdown('<div class="warn">⚠ Is threshold par win rate break-even se neeche hai: '
                                'in settings par trade karne se paisa kam hoga.</div>',
                                unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn">Is threshold par koi out-of-sample signal nahi bana. '
                            'Threshold kam karein.</div>', unsafe_allow_html=True)

            st.markdown("#### THRESHOLD SWEEP")
            rows = []
            for t in [0.52, 0.55, 0.58, 0.60, 0.63, 0.66, 0.70]:
                nn, aa, _, _ = stats_at(res["oos"], t)
                rows.append({"Confidence ≥": f"{t * 100:.0f}%", "Signals": nn,
                             "Win rate %": None if aa is None else round(aa * 100, 1)})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("Ye sab out-of-sample (walk-forward) results hain: model ne in candles ko "
                       "training me kabhi nahi dekha. Past result future ki guarantee nahi.")

    # ---------------- CHARTS ----------------
    with tab3:
        if not good:
            st.info("Pehle scan chalayein.")
        else:
            tf_c = st.selectbox("TIMEFRAME", list(good.keys()), key="ch_tf")
            res = good[tf_c]
            ch = res["chart"]
            fig = go.Figure(go.Candlestick(
                x=ch.index, open=ch["Open"], high=ch["High"], low=ch["Low"], close=ch["Close"],
                increasing_line_color="#00ff9c", decreasing_line_color="#ff3b5c", name="Price"))
            fig.add_trace(go.Scatter(x=ch.index, y=ch["Close"].ewm(span=20).mean(), name="EMA20",
                                     line=dict(color="#00f0ff", width=1.4)))
            fig.add_trace(go.Scatter(x=ch.index, y=ch["Close"].ewm(span=50).mean(), name="EMA50",
                                     line=dict(color="#ff2bd6", width=1.4)))
            v = verdicts[tf_c]
            fig.add_annotation(x=ch.index[-1], y=float(ch["High"].iloc[-1]),
                               text=f"{'▲' if v['side'] == 'BUY' else '▼'} {v['side']} "
                                    f"{v['conf'] * 100:.0f}%", showarrow=True, arrowhead=2,
                               font=dict(color="#00f0ff", size=14), arrowcolor="#00f0ff")
            fig.update_layout(title=f"{state['pair']} · {tf_c}", xaxis_rangeslider_visible=False)
            st.plotly_chart(style_fig(fig, 520), key="candles")

    # ---------------- SYSTEM INFO ----------------
    with tab4:
        any_res = next(iter(good.values())) if good else None
        st.markdown("#### ENGINE")
        st.write(
            "- **Features:** " + (f"{any_res['n_feat']} " if any_res else "~90 ")
            + "(ta library indicators + candle shape + lagged returns)\n"
            "- **Model:** ensemble of Gradient Boosting + Extra Trees + Logistic Regression\n"
            "- **Validation:** walk-forward (model hamesha sirf past data par train hota hai, "
            "agle chunk par test)\n"
            "- **Signal rule:** STRONG sirf tab jab live confidence ≥ threshold, backtest me kam az kam "
            "30 signals, aur accuracy ≥ max(55%, baseline)\n"
            "- **Candles:** model sirf CLOSED candles dekhta hai; forming candle ignore hoti hai\n"
            "- **Data:** Yahoo Finance (yfinance), cache 30-60 sec"
        )
        st.markdown('<div class="warn">Koi bot 100% accurate nahi ho sakta. Ye educational tool hai, '
                    'financial advice nahi. Pehle demo account par test karein.</div>',
                    unsafe_allow_html=True)
