import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime, timedelta
import pytz

# --- CORE FUNCTIONS ---
def get_data(symbol, period="1y", interval="1d"):
    data = yf.download(symbol, period=period, interval=interval)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    return data

def bsm_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0: T = 0.00001
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="Ultimate Pro Terminal")
st.title("🏛️ Institutional Trading Terminal (Nifty/BankNifty)")

# --- SIDEBAR: ASSET & INDICATORS ---
st.sidebar.header("🕹️ Global Controls")
index_map = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS"}
selected_index = st.sidebar.selectbox("Market Index", list(index_map.keys()))

# The "All 150+" Indicator Engine
st.sidebar.subheader("🔬 Technical Indicator Categories")
# Get all indicators from pandas_ta
all_indicators = pd.DataFrame().ta.indicators(as_list=True)
category = st.sidebar.selectbox("Filter Category", ["All", "Momentum", "Overlap", "Trend", "Volatility", "Volume", "Statistics"])

if category != "All":
    # Filter indicators based on category keywords (simplified logic)
    indicators_to_show = [i for i in all_indicators if category.lower() in i or category == "Overlap"] 
else:
    indicators_to_show = all_indicators

selected_ta = st.sidebar.multiselect("Select Indicators to Overlay", indicators_to_show, default=["sma", "rsi"])

# --- DATA FETCHING ---
df = get_data(index_map[selected_index])
spot = df['Close'].iloc[-1]

# --- CHARTING ---
st.subheader(f"📊 {selected_index} Live Analysis")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Candlestick"))

# Dynamically apply any selected indicator
for indicator in selected_ta:
    try:
        # Calls the function dynamically from pandas_ta
        method = getattr(df.ta, indicator)
        result = method()
        if isinstance(result, pd.Series):
            fig.add_trace(go.Scatter(x=df.index, y=result, name=indicator.upper()))
        elif isinstance(result, pd.DataFrame):
            for col in result.columns:
                fig.add_trace(go.Scatter(x=df.index, y=result[col], name=col))
    except Exception as e:
        st.sidebar.warning(f"Could not load {indicator}: {e}")

fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# --- OPTION STRATEGY ENGINE ---
st.divider()
st.header("🛠️ Advanced Strategy Builder")
col_opt1, col_opt2 = st.columns([1, 2])

with col_opt1:
    strategy = st.selectbox("Strategy Mode", ["Custom", "Iron Condor", "Straddle", "Bull Call Spread"])
    iv = st.slider("Implied Volatility (IV%)", 5, 60, 15) / 100
    expiry_days = st.number_input("Days to Expiry", 1, 30, 2)
    T = expiry_days / 365

with col_opt2:
    st.write(f"**Current Spot:** ₹{spot:.2f}")
    strike = st.number_input("Base Strike", value=int(round(spot, -2)), step=50)
    
    # Quick Calculation Table
    ce_price = bsm_price(spot, strike, T, 0.07, iv, "call")
    pe_price = bsm_price(spot, strike, T, 0.07, iv, "put")
    
    st.table(pd.DataFrame({
        "Option": ["Call (CE)", "Put (PE)"],
        "Strike": [strike, strike],
        "Theoretical Price": [f"₹{ce_price:.2f}", f"₹{pe_price:.2f}"]
    }))
