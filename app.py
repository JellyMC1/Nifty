import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

# 1. Robust Price Fetcher
def get_nifty_spot():
    try:
        ticker = yf.Ticker("^NSEI")
        # Try fast info first
        price = ticker.fast_info['last_price']
        if price and price > 0:
            return price
        # Fallback to 1-day history if fast info fails
        df = ticker.history(period="1d", interval="1m")
        return df['Close'].iloc[-1]
    except:
        return None

# 2. Black-Scholes Formula
def bsm_calculation(S, K, T, r, sigma, option_type="call"):
    if T <= 0: T = 0.00001
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type=="call" else norm.cdf(-d2))) / 365
    return price, delta, gamma, vega, theta

# 3. Streamlit Interface
st.set_page_config(page_title="Nifty Live Greeks", layout="wide")
st.title("🇮🇳 Nifty 50 Options Analysis Dashboard")

# Sidebar
st.sidebar.header("User Inputs")
iv = st.sidebar.slider("Volatility (IV %)", 5.0, 50.0, 15.0) / 100
r_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 6.5) / 100

spot = get_nifty_spot()

if spot:
    st.metric("NIFTY 50 SPOT", f"₹{spot:.2f}")
    
    # Calculate Expiry (Next Thursday)
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    days_to_expiry = (3 - now.weekday()) % 7
    expiry_dt = (now + timedelta(days=days_to_expiry)).replace(hour=15, minute=30)
    T_years = max((expiry_dt - now).total_seconds() / (365 * 24 * 3600), 0.0001)

    strike = st.number_input("Strike Price", value=int(round(spot, -2)), step=50)

    # Compute Greeks
    c_p, c_d, gamma, vega, c_t = bsm_calculation(spot, strike, T_years, r_rate, iv, "call")
    p_p, p_d, _, _, p_t = bsm_calculation(spot, strike, T_years, r_rate, iv, "put")

    # Display Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Call Option (CE)**\nPrice: ₹{c_p:.2f} | Delta: {c_d:.3f} | Theta: {c_t:.2f}")
    with col2:
        st.error(f"**Put Option (PE)**\nPrice: ₹{p_p:.2f} | Delta: {p_d:.3f} | Theta: {p_t:.2f}")

    # Payoff Chart
    st.subheader("Strategy Payoff (Projected)")
    s_range = np.linspace(spot * 0.95, spot * 1.05, 100)
    payoff_vals = [bsm_calculation(s, strike, T_years, r_rate, iv, "call")[0] - c_p for s in s_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_range, y=payoff_vals, name="Call Profit/Loss", line=dict(color='lime')))
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Trying to fetch market data... please refresh in a moment.")