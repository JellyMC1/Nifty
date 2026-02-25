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

with tab3:
    st.subheader("📊 Strategy Payoff Simulator")
    
    # Create a range of prices for the payoff chart
    sT = np.linspace(strike * 0.8, strike * 1.2, 100)
    
    if strategy == "Long Straddle":
        payoff = np.maximum(sT - strike, 0) + np.maximum(strike - sT, 0) - (ce_price + pe_price)
    elif strategy == "Bull Call Spread":
        payoff = np.maximum(sT - strike, 0) - np.maximum(sT - (strike + 100), 0) - (ce_price - 10) # 10 is dummy hedge cost
        
    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(x=sT, y=payoff, name="P&L at Expiry", fill='tozeroy'))
    fig_payoff.add_hline(y=0, line_dash="dash", line_color="red")
    fig_payoff.update_layout(title="Expected Profit/Loss Profile", xaxis_title="Spot Price at Expiry", yaxis_title="Profit / Loss")
    st.plotly_chart(fig_payoff, use_container_width=True)
    
with tab2:
    st.subheader("🎲 Institutional Risk Projection")
    
    # Advanced Monte Carlo with Confidence Intervals
    returns = np.log(data['Close'] / data['Close'].shift(1))
    mu, sigma_daily = returns.mean(), returns.std()
    
    # Run Simulation
    sim_results = monte_carlo_sim(data['Close'].iloc[-1], n_days/252, 0.07, vol, iterations, n_days)
    
    # UI: Metrics for Risk
    final_prices = sim_results[-1]
    var_95 = np.percentile(final_prices, 5)
    expected_val = np.mean(final_prices)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("95% VaR (Floor)", f"₹{var_95:.2f}", help="95% certainty price won't fall below this")
    col2.metric("Mean Projection", f"₹{expected_val:.2f}")
    col3.metric("Volatility (σ)", f"{vol*100:.1f}%")

    # Simulation Chart with Quantile Shading
    fig_mc = go.Figure()
    x_axis = list(range(n_days))
    fig_mc.add_trace(go.Scatter(y=np.percentile(sim_results, 95, axis=1), line=dict(width=0), name="95th Pctl"))
    fig_mc.add_trace(go.Scatter(y=np.percentile(sim_results, 5, axis=1), fill='tonexty', line=dict(width=0), name="5th Pctl", fillcolor='rgba(0,176,246,0.2)'))
    fig_mc.add_trace(go.Scatter(y=np.mean(sim_results, axis=1), line=dict(color='white', dash='dash'), name="Mean Path"))
    
    st.plotly_chart(fig_mc, use_container_width=True)
    # ... (rest of your calculation code)


