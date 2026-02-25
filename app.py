import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from scipy.stats import norm

# --- 1. QUANT MODELS ---
def black_scholes(S, K, T, r, sigma, type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def monte_carlo_sim(S, T, r, sigma, iterations=1000, days=30):
    dt = 1/252
    paths = np.zeros((days, iterations))
    paths[0] = S
    for t in range(1, days):
        z = np.random.standard_normal(iterations)
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths

# --- 2. TERMINAL UI ---
st.set_page_config(layout="wide", page_title="Institutional Terminal")
st.title("🏛️ Global Alpha Terminal")

# Sidebar - Asset Selection
st.sidebar.header("🌍 Market Selection")
market_type = st.sidebar.radio("Market", ["Indian Indices", "Global Indices"])

indices = {
    "Indian Indices": {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN"},
    "Global Indices": {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DAX": "^GDAXI", "NIKKEI 225": "^N225"}
}

symbol = st.sidebar.selectbox("Select Asset", list(indices[market_type].keys()))
ticker = indices[market_type][symbol]

# --- 3. DASHBOARD TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Technical Charts", "🎲 Monte Carlo Simulation", "🧮 BSM Options Lab"])

data = yf.download(ticker, period="1y")
# Clean multi-index columns if they exist
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

with tab1:
    st.subheader(f"{symbol} Advanced Charting")
    # All 150+ Indicators via multiselect
    all_indicators = [m for m in dir(pd.DataFrame().ta) if not m.startswith("_")]
    selected_ta = st.multiselect("Add Indicators", all_indicators, default=["sma", "rsi", "bbands"])
    
    fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Market")])
    for ind in selected_ta:
        try:
            res = getattr(data.ta, ind)()
            if isinstance(res, pd.DataFrame):
                for col in res.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=res[col], name=col))
            else:
                fig.add_trace(go.Scatter(x=data.index, y=res, name=ind))
        except: pass
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Future Price Projection (Monte Carlo)")
    n_sims = st.slider("Number of Simulations", 100, 5000, 1000)
    n_days = st.slider("Days into Future", 5, 252, 30)
    
    # Calculate daily volatility for the simulation
    returns = np.log(data['Close'] / data['Close'].shift(1))
    vol = returns.std() * np.sqrt(252)
    
    sim_paths = monte_carlo_sim(data['Close'].iloc[-1], n_days/252, 0.07, vol, n_sims, n_days)
    
    fig_sim = go.Figure()
    for i in range(min(n_sims, 100)): # Plot first 100 for performance
        fig_sim.add_trace(go.Scatter(y=sim_paths[:, i], mode='lines', line=dict(width=1), opacity=0.1, showlegend=False))
    st.plotly_chart(fig_sim, use_container_width=True)
    st.write(f"**Expected Range in {n_days} days:** ₹{np.percentile(sim_paths[-1], 5):.2f} - ₹{np.percentile(sim_paths[-1], 95):.2f}")

with tab3:
    st.subheader("Black-Scholes Pricing Model")
    col1, col2 = st.columns(2)
    with col1:
        strike = st.number_input("Strike Price", value=float(round(data['Close'].iloc[-1], -2)))
        expiry = st.slider("Days to Expiry", 1, 365, 30)
    with col2:
        risk_free = st.number_input("Risk Free Rate (%)", value=7.0) / 100
        vol_input = st.slider("Implied Volatility (%)", 5, 100, int(vol*100)) / 100
    
    c_price = black_scholes(data['Close'].iloc[-1], strike, expiry/365, risk_free, vol_input, "call")
    p_price = black_scholes(data['Close'].iloc[-1], strike, expiry/365, risk_free, vol_input, "put")
    
    st.metric("Theoretical Call Price", f"₹{c_price:.2f}")
    st.metric("Theoretical Put Price", f"₹{p_price:.2f}")
