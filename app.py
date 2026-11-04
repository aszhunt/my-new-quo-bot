import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from ta import add_all_ta_features

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Candle Probability Bot", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- symbols
FOREX = {
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
    # Crypto and metals
    "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD", "BNB/USD": "BNB-USD",
    "XRP/USD": "XRP-USD", "SOL/USD": "SOL-USD", "DOGE/USD": "DOGE-USD",
    "Gold (XAU/USD)": "GC=F", "Silver (XAG/USD)": "SI=F",
}

# timeframe -> (download interval, resample rule or None, period)
TF = {
    "1 min": ("1m", None, "7d"),
    "2 min": ("2m", None, "30d"),
    "3 min": ("1m", "3min", "7d"),
    "5 min": ("5m", None, "30d"),
    "10 min": ("5m", "10min", "30d"),
    "15 min": ("15m", None, "30d"),
}

RAW = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------- data
@st.cache_data(ttl=30, show_spinner=False)
def load_data(ticker: str, interval: str, rule, period: str):
    df = yf.download(
        ticker, interval=interval, period=period,
        progress=False, auto_adjust=False, threads=False,
    )
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[RAW].dropna(subset=["Close"])
    if rule:
        df = df.resample(rule).agg(
            {"Open": "first", "High": "max", "Low": "min",
             "Close": "last", "Volume": "sum"}
        ).dropna(subset=["Close"])
    return df


# ---------------------------------------------------------------- features
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Volume"] = d["Volume"].replace(0, np.nan).fillna(1.0)
    d = add_all_ta_features(
        d, open="Open", high="High", low="Low", close="Close",
        volume="Volume", fillna=False,
    )
    X = d.drop(columns=RAW)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.isna().mean() < 0.3]
    X = X.ffill().fillna(0.0)

    # make price-level indicators scale free (relative to close)
    close = d["Close"]
    for c in list(X.columns):
        s = X[c]
        if s.std() > 0 and abs(s.corr(close)) > 0.9:
            X[c] = s / close - 1.0

    for k in (1, 2, 3, 5, 8):
        X[f"ret_{k}"] = close.pct_change(k)
    X["body"] = (d["Close"] - d["Open"]) / d["Close"]
    X["range"] = (d["High"] - d["Low"]) / d["Close"]
    X["upper_wick"] = (d["High"] - d[["Open", "Close"]].max(axis=1)) / d["Close"]
    X["lower_wick"] = (d[["Open", "Close"]].min(axis=1) - d["Low"]) / d["Close"]

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.loc[:, X.std() > 0]
    return X


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, random_state=42,
    )


def predict_next(df: pd.DataFrame):
    X = build_features(df)
    y = (df["Close"].shift(-1) > df["Close"]).astype(int)

    X_lab, y_lab = X.iloc[:-1].iloc[50:], y.iloc[:-1].iloc[50:]  # skip warm-up rows
    x_last = X.iloc[[-1]]
    if len(X_lab) < 200:
        return None

    split = int(len(X_lab) * 0.8)
    m_test = make_model().fit(X_lab.iloc[:split], y_lab.iloc[:split])
    test_pred = m_test.predict(X_lab.iloc[split:])
    test_acc = accuracy_score(y_lab.iloc[split:], test_pred)
    base = max(y_lab.iloc[split:].mean(), 1 - y_lab.iloc[split:].mean())

    m_final = make_model().fit(X_lab, y_lab)
    p_up = float(m_final.predict_proba(x_last)[0][1])
    return {
        "p_up": p_up, "test_acc": float(test_acc), "baseline": float(base),
        "n_features": X.shape[1], "n_rows": len(X_lab),
    }


# ---------------------------------------------------------------- UI
st.title("📈 Candle Probability Bot")
st.caption(
    "Next-candle Buy/Sell probability using ~90 technical indicators + machine learning. "
    "Educational tool, not financial advice. No bot can be 100% accurate."
)

with st.sidebar:
    st.header("Settings")
    group = st.radio("Market", ["All", "Forex", "Crypto / Metals"], horizontal=True)
    names = list(FOREX.keys())
    if group == "Forex":
        names = [n for n in names if FOREX[n].endswith("=X")]
    elif group == "Crypto / Metals":
        names = [n for n in names if not FOREX[n].endswith("=X")]
    pair = st.selectbox("Pair", names)
    tfs = st.multiselect("Timeframes", list(TF.keys()), default=list(TF.keys()))
    run = st.button("🔄 Analyse / Refresh", type="primary", use_container_width=True)

ticker = FOREX[pair]

if run:
    rows = []
    last_price = None
    last_time = None
    prog = st.progress(0.0)
    for i, tf in enumerate(tfs):
        interval, rule, period = TF[tf]
        try:
            df = load_data(ticker, interval, rule, period)
            if df is None or len(df) < 300:
                rows.append({"Timeframe": tf, "Signal": "No data", "Buy %": None,
                             "Sell %": None, "Backtest acc %": None,
                             "Baseline %": None, "Candles": 0})
            else:
                if tf == tfs[0] or last_price is None:
                    last_price = float(df["Close"].iloc[-1])
                    last_time = df.index[-1]
                r = predict_next(df)
                if r is None:
                    rows.append({"Timeframe": tf, "Signal": "Not enough data",
                                 "Buy %": None, "Sell %": None,
                                 "Backtest acc %": None, "Baseline %": None,
                                 "Candles": len(df)})
                else:
                    buy, sell = r["p_up"] * 100, (1 - r["p_up"]) * 100
                    edge = max(buy, sell)
                    if edge < 55:
                        sig = "⚪ No clear edge"
                    else:
                        sig = "🟢 BUY" if buy > sell else "🔴 SELL"
                    rows.append({
                        "Timeframe": tf, "Signal": sig,
                        "Buy %": round(buy, 1), "Sell %": round(sell, 1),
                        "Backtest acc %": round(r["test_acc"] * 100, 1),
                        "Baseline %": round(r["baseline"] * 100, 1),
                        "Candles": r["n_rows"],
                    })
        except Exception as e:  # never crash the app
            rows.append({"Timeframe": tf, "Signal": f"Error: {str(e)[:60]}",
                         "Buy %": None, "Sell %": None, "Backtest acc %": None,
                         "Baseline %": None, "Candles": 0})
        prog.progress((i + 1) / max(len(tfs), 1))
    prog.empty()

    c1, c2 = st.columns(2)
    if last_price is not None:
        c1.metric(f"{pair} last price", f"{last_price:,.5f}")
        age = (datetime.now(timezone.utc) - pd.Timestamp(last_time).tz_convert("UTC")
               ).total_seconds() / 60 if pd.Timestamp(last_time).tzinfo else None
        if age is not None:
            c2.metric("Last candle age (min)", f"{age:,.1f}")
            if age > 15:
                st.warning("Data purani hai (market band ho sakti hai ya data delayed hai).")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        "**Backtest acc %** = model ki accuracy un candles par jo training me use nahi hui. "
        "**Baseline %** = bina model ke sirf majority direction guess karne ki accuracy. "
        "Agar backtest accuracy baseline se zyada nahi, to signal par bharosa na karein."
    )
else:
    st.write("👈 Pair aur timeframes select kar ke **Analyse / Refresh** dabayein.")

st.divider()
st.caption("Data: Yahoo Finance via yfinance (1-min data delayed ho sakta hai). "
           "Binary/forex trading me paisa doobne ka khatra hai.")
