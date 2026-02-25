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

with tab3: # This is the BSM Options Lab tab
    st.subheader("Black-Scholes Pricing & Strategy Lab")
    
    # This creates two columns INSIDE the tab
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        # THE DROPDOWN: Ensure it is indented here!
        strategy = st.selectbox("Select Strategy Mode", 
                                ["Single Option", "Long Straddle", "Bull Call Spread", "Iron Condor"])
        
        iv = st.slider("Implied Volatility (%)", 5, 60, 15) / 100
        days_to_expiry = st.number_input("Days to Expiry", value=2)

    with col_opt2:
        # Strategy results will show here based on the dropdown choice
        st.write(f"Analyzing: **{strategy}**")
        # --- ADVANCED UI OVERLAY ---
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Global Market Pulse")
# Quick health check for US markets to see 'sentiment'
us_market = yf.Ticker("^GSPC").history(period="1d")
change = ((us_market['Close'].iloc[-1] - us_market['Open'].iloc[-1]) / us_market['Open'].iloc[-1]) * 100
st.sidebar.metric("S&P 500 (US)", f"{us_market['Close'].iloc[-1]:.2f}", f"{change:.2f}%")

# Simulation Precision Setting
precision = st.sidebar.select_slider("Simulation Rigor", options=["Standard", "High", "Institutional"], value="Standard")
iterations = {"Standard": 1000, "High": 5000, "Institutional": 10000}[precision]

# --- STEP 1: Define the price range for the chart ---
sT = np.linspace(strike * 0.85, strike * 1.15, 100)

# --- STEP 2: Calculate Payoff for EVERY strategy ---
if strategy == "Long Straddle":
    # Buy Call + Buy Put
    payoff = (np.maximum(sT - strike, 0) - ce_price) + (np.maximum(strike - sT, 0) - pe_price)

elif strategy == "Bull Call Spread":
    # Buy ATM Call, Sell OTM Call (assuming 100 point spread)
    strike_high = strike + 100
    ce_price_high = black_scholes(spot, strike_high, T, 0.07, iv, "call")
    payoff = (np.maximum(sT - strike, 0) - ce_price) - (np.maximum(sT - strike_high, 0) - ce_price_high)

elif strategy == "Iron Condor":
    # Simplified Iron Condor Payoff
    s1, s2, s3, s4 = strike-200, strike-100, strike+100, strike+200
    payoff = (np.maximum(sT-s1,0) - np.maximum(sT-s2,0) - np.maximum(s3-sT,0) + np.maximum(s4-sT,0))
    # --- ADD THIS SAFETY BLOCK ABOVE LINE 151 ---
# Ensure 'strike', 'T', and 'iv' also have fallback values
try: _ = strike
except NameError: strike = int(round(spot, -2))

try: _ = T
except NameError: T = 7/365 # Default 1 week

try: _ = iv
except NameError: iv = 0.15 # Default 15% IV
    _ = ce_price 
except NameError:
    # 0.07 is the risk-free rate (7%). Ensure 'spot', 'strike', 'T', and 'iv' exist above!
    ce_price = black_scholes(spot, strike, T, 0.07, iv, "call")
    pe_price = black_scholes(spot, strike, T, 0.07, iv, "put")

# --- YOUR LINE 151 SHOULD NOW WORK ---

else:
    # DEFAULT/FALLBACK: Single Option (Call)
    payoff = np.maximum(sT - strike, 0) - ce_price

# --- STEP 3: Now the chart will always find the 'payoff' variable ---
fig_payoff = go.Figure()
fig_payoff.add_trace(go.Scatter(x=sT, y=payoff, name="P&L at Expiry", fill='tozeroy'))
    
with tab2:
    st.subheader("🎲 Institutional Risk Projection")
    
    # 1. Ensure we have data to calculate volatility
    if not data.empty:
        # Calculate daily log returns
        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
        
        # Calculate annualized volatility
        vol_calc = returns.std() * np.sqrt(252)
        
        # User inputs
        n_sims = st.slider("Simulations", 100, 5000, 1000)
        n_days = st.slider("Days Ahead", 5, 252, 30)
        
        # 2. Run the actual simulation
        # Using .iloc[-1] to get the most recent price
        start_price = data['Close'].iloc[-1]
        
        # Call the simulation function (ensure this is defined at the top of your app.py)
        sim_paths = monte_carlo_sim(start_price, n_days/252, 0.07, vol_calc, n_sims, n_days)
        
        # 3. Plotting the results
        fig_mc = go.Figure()
        for i in range(min(n_sims, 50)): # Plotting fewer lines for performance
            fig_mc.add_trace(go.Scatter(y=sim_paths[:, i], mode='lines', 
                                      line=dict(width=1), opacity=0.1, showlegend=False))
        
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Show statistical floor (VaR)
        st.write(f"**95% Confidence Floor:** ₹{np.percentile(sim_paths[-1], 5):.2f}")
    else:
        st.error("No market data found to run simulation. Please check your ticker.")

st.divider()
st.header("🏢 Institutional Research & Greeks")

# 1. Strategy Greeks Table
st.subheader("📊 Position Greeks")
col_g1, col_g2, col_g3, col_g4 = st.columns(4)

# Function to calculate Delta (for the table)
def get_delta(S, K, T, r, sigma, type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if type == "call" else norm.cdf(d1) - 1

delta_val = get_delta(spot, strike, T, 0.07, iv, "call")
theta_val = -(spot * norm.pdf((np.log(spot/strike)+(0.07+0.5*iv**2)*T)/(iv*np.sqrt(T))) * iv) / (2 * np.sqrt(T))

col_g1.metric("Net Delta", f"{delta_val:.3f}", help="Directional Risk")
col_g2.metric("Net Theta", f"{theta_val/365:.2f}", help="Daily Time Decay")
col_g3.metric("Max Profit", "Calculated at Expiry")
col_g4.metric("Margin Required", "Approx ₹1.2L (Selling)")

# 2. Global Macro News Feed
st.sidebar.markdown("---")
st.sidebar.subheader("📰 Global Market News")
try:
    news = ticker.news[:3] # Gets latest 3 news items from yfinance
    for item in news:
        st.sidebar.write(f"**{item['title']}**")
        st.sidebar.caption(f"Source: {item['publisher']}")
except:
    st.sidebar.write("News feed temporarily unavailable.")

# 3. Interactive Scenario "What-If" Analysis
st.subheader("🧪 Stress Test: Price & Volatility Shift")
slide_price = st.select_slider("Simulate Price Move (%)", options=[-10, -5, -2, 0, 2, 5, 10], value=0)
slide_iv = st.select_slider("Simulate IV Spike (%)", options=[-5, 0, 5, 10, 20], value=0)

new_spot = spot * (1 + slide_price/100)
new_iv = iv + (slide_iv/100)
new_price = black_scholes(new_spot, strike, T, 0.07, new_iv, "call")

profit_change = (new_price - ce_price)
st.info(f"If {symbol} moves {slide_price}% and IV shifts {slide_iv}%, your P&L changes by: **₹{profit_change:.2f} per unit**")

try:
    st.write("### 🛡️ Extended Analytics")
    # Check for core variables
    current_val = spot if 'spot' in locals() else 0
    st.metric("Live Feed Check", f"Active: {symbol}", f"Price: {current_val}")
except Exception as e:
    st.info("Analytics will load once market data is fetched above.")











