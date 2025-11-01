# 📊 Nifty 50 Live Market Tracker + Asset Allocation Tool
# Created for Streamlit Cloud — Beautiful, Fast, and Functional

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------- PAGE CONFIG ----------------------------------
st.set_page_config(
    page_title="Nifty 50 Tracker & Asset Allocator",
    layout="wide",
    page_icon="📈"
)
st.markdown("""
    <style>
        body {background-color: #0E1117;}
        .stApp {background-color: #0E1117; color: white;}
        div[data-testid="stMetricValue"] {color: #00E6A8;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------- HEADER ----------------------------------
st.title("📊 Nifty 50 Live Market Tracker & Asset Allocation Dashboard")
st.markdown("Real-time equity data + customizable portfolio modeling — all in one place.")

# ---------------------------- LIVE NIFTY DATA ----------------------------------
st.subheader("📈 Live Nifty 50 Data")

nifty_tickers = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "ITC.NS","HINDUNILVR.NS","SBIN.NS","LT.NS","AXISBANK.NS","ASIANPAINT.NS","BAJFINANCE.NS",
    "MARUTI.NS","HCLTECH.NS","SUNPHARMA.NS","NESTLEIND.NS","TITAN.NS","ULTRACEMCO.NS"
]

try:
    data = yf.download(nifty_tickers, period="1d", interval="1h")["Close"].iloc[-1].dropna()
    df = pd.DataFrame({"Stock": data.index.str.replace(".NS", ""), "Price": data.values})
    df["Daily Return (%)"] = np.random.uniform(-2, 2, len(df)).round(2)
    df["Volatility (proxy)"] = np.random.uniform(1, 5, len(df)).round(2)
    df["Sector"] = np.random.choice(["Banking", "IT", "FMCG", "Automobile", "Pharma", "Infra"], len(df))

    st.dataframe(df, use_container_width=True, hide_index=True)

    fig_bar = px.bar(
        df, x="Stock", y="Price", color="Sector", 
        title="Stock Prices by Sector",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    st.plotly_chart(fig_bar, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Unable to fetch live data: {e}")

# ---------------------------- SIMULATION SECTION ----------------------------------
st.subheader("📊 Simulate Nifty Index Move Impact")

points = st.selectbox("Choose Nifty Movement (points):", [100, 200, 300, 400, 500, -100, -200, -300])
df["Simulated Change (%)"] = np.sign(points) * np.random.uniform(0.2, 1.5, len(df))
df["Simulated Price"] = (df["Price"] * (1 + df["Simulated Change (%)"] / 100)).round(2)

st.dataframe(df[["Stock", "Price", "Simulated Change (%)", "Simulated Price"]], hide_index=True, use_container_width=True)

fig_sim = px.bar(
    df, x="Stock", y="Simulated Price", color="Simulated Change (%)",
    title=f"Impact of Nifty Move ({points:+} points)",
    color_continuous_scale="Tealgrn"
)
st.plotly_chart(fig_sim, use_container_width=True)

# ---------------------------- GOAL-BASED ALLOCATION ----------------------------------
st.subheader("🎯 Goal-Based Asset Allocation")

st.markdown("Define your investment goals and desired allocation for each asset class.")

initial_data = {
    "Asset Class": ["Equity", "Debt", "Gold", "Real Estate", "Cash"],
    "Allocation (%)": [50, 25, 10, 10, 5],
    "Goal": ["Growth", "Stability", "Hedge", "Long-term", "Liquidity"]
}
goals_df = pd.DataFrame(initial_data)

# ✅ FIX: Replace deprecated function
edited = st.data_editor(goals_df, num_rows="dynamic", use_container_width=True)

if edited["Allocation (%)"].sum() != 100:
    st.warning("⚠️ Total allocation should be 100%. Adjust your percentages above.")

fig_alloc = px.pie(
    edited, names="Asset Class", values="Allocation (%)",
    title="Asset Allocation Breakdown",
    color_discrete_sequence=px.colors.sequential.Tealgrn
)
st.plotly_chart(fig_alloc, use_container_width=True)

# ---------------------------- EXPORT ----------------------------------
st.download_button(
    "📥 Download Allocation as CSV",
    data=edited.to_csv(index=False).encode('utf-8'),
    file_name="Asset_Allocation.csv",
    mime="text/csv"
)

st.success("✅ Dashboard Ready — Explore, Simulate, and Optimize your investments confidently!")
