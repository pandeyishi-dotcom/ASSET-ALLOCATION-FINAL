import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Smart Asset Allocator", page_icon="💹", layout="wide")

st.title("💹 Smart Asset Allocation Planner")
st.caption("Diversify wisely based on your age, risk, and total investment.")

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Select Age", 20, 70, 30)
with col2:
    risk = st.selectbox("Select Risk Profile", ["High", "Moderate", "Low"])
with col3:
    total_investment = st.number_input("Enter Total Investment (₹)", min_value=10000, step=10000, value=500000)

st.divider()

# ---------------------------------------------------
# ALLOCATION LOGIC
# ---------------------------------------------------
def allocate_assets(age, risk):
    assets = [
        "Large Cap Equity", "Mid/Small Cap Equity", "Debt Funds", "Gold ETF",
        "REITs/InvITs", "International Equity", "Cash/Liquid", "Govt Bonds",
        "Commodities", "Crypto (Regulated)", "Index Funds", "Hybrid Funds",
        "Thematic Funds", "Corporate Bonds"
    ]

    base = np.array([20, 18, 10, 8, 5, 10, 2, 5, 4, 3, 5, 3, 4, 3])
    if risk == "Moderate":
        base = np.array([18, 12, 18, 8, 5, 8, 5, 8, 3, 0, 5, 5, 3, 2])
    elif risk == "Low":
        base = np.array([10, 5, 25, 10, 5, 5, 10, 15, 3, 0, 5, 5, 1, 1])

    age_factor = max(0, (age - 25) / 100)
    conservative = np.array([5, 3, 25, 10, 5, 3, 10, 20, 2, 0, 5, 4, 2, 6])
    weights = base * (1 - age_factor) + conservative * age_factor
    weights = np.round(weights / weights.sum() * 100, 2)

    df = pd.DataFrame({"Asset Class": assets, "Allocation (%)": weights})
    return df

# ---------------------------------------------------
# DATAFRAME + CALCULATIONS
# ---------------------------------------------------
df = allocate_assets(age, risk)
df["Amount (₹)"] = np.round(df["Allocation (%)"] / 100 * total_investment, 0)

# Expected annual returns (assumed averages)
returns = {
    "Large Cap Equity": 11, "Mid/Small Cap Equity": 13, "Debt Funds": 7,
    "Gold ETF": 8, "REITs/InvITs": 9, "International Equity": 10,
    "Cash/Liquid": 4, "Govt Bonds": 6, "Commodities": 8,
    "Crypto (Regulated)": 18, "Index Funds": 10, "Hybrid Funds": 8,
    "Thematic Funds": 12, "Corporate Bonds": 7
}

df["Exp. Annual Return (%)"] = df["Asset Class"].map(returns)
df["Exp. 1Y Gain (₹)"] = np.round(df["Amount (₹)"] * df["Exp. Annual Return (%)"] / 100, 0)
df["Exp. 5Y Value (₹)"] = np.round(df["Amount (₹)"] * (1 + df["Exp. Annual Return (%)"] / 100) ** 5, 0)

# ---------------------------------------------------
# DISPLAY
# ---------------------------------------------------
st.subheader("📊 Recommended Asset Allocation")

st.dataframe(df, hide_index=True, use_container_width=True)

# Built-in Streamlit chart
st.bar_chart(df.set_index("Asset Class")["Allocation (%)"])

# ---------------------------------------------------
# SUMMARY SECTION
# ---------------------------------------------------
st.divider()
st.subheader("📈 Portfolio Projection Summary")

total_return_1y = df["Exp. 1Y Gain (₹)"].sum()
total_value_5y = df["Exp. 5Y Value (₹)"].sum()

col1, col2 = st.columns(2)
col1.metric("Expected 1-Year Gain", f"₹{int(total_return_1y):,}")
col2.metric("Expected 5-Year Portfolio Value", f"₹{int(total_value_5y):,}")

st.caption("Returns are illustrative and not guaranteed. Based on average historical data.")

st.divider()
st.caption("© 2025 Smart Asset Allocator | Powered by Streamlit")
