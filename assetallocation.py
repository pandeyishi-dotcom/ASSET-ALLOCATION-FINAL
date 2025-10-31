import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="💹 Smart Asset Allocation Planner",
    page_icon="💰",
    layout="wide",
)

# ---------------------------------------------------
# STYLE
# ---------------------------------------------------
st.markdown("""
    <style>
        [data-testid="stMetricValue"] {
            font-size: 24px;
            font-weight: 600;
        }
        div.block-container {
            padding-top: 1rem;
        }
        h1, h2, h3 {
            color: #035E7B;
        }
        .big-font {
            font-size:18px !important;
        }
        .note-box {
            background-color: #f6f9fc;
            padding: 10px 15px;
            border-left: 5px solid #035E7B;
            border-radius: 6px;
            margin-bottom: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("💹 Smart Asset Allocation Planner")
st.caption("Plan your diversified portfolio with logic-backed allocation suggestions based on age and risk profile.")

st.divider()

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    age = st.slider("🎂 Age", 20, 70, 30)
with col2:
    risk = st.selectbox("⚖️ Risk Profile", ["High", "Moderate", "Low"])
with col3:
    total_investment = st.number_input("💰 Total Investment (₹)", min_value=10000, step=10000, value=500000)

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

    return pd.DataFrame({"Asset Class": assets, "Allocation (%)": weights})

# ---------------------------------------------------
# CALCULATIONS
# ---------------------------------------------------
df = allocate_assets(age, risk)
df["Amount (₹)"] = np.round(df["Allocation (%)"] / 100 * total_investment, 0)

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

# Donut Chart
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(df["Allocation (%)"], labels=df["Asset Class"], startangle=90, wedgeprops={"width": 0.4})
centre_circle = plt.Circle((0, 0), 0.6, fc="white")
fig.gca().add_artist(centre_circle)
ax.set_title("Portfolio Allocation Breakdown", fontsize=14, fontweight="bold", color="#035E7B")
st.pyplot(fig)

# ---------------------------------------------------
# LOGIC FOR SELECTION + NOTES
# ---------------------------------------------------
st.divider()
st.subheader("🧠 Logic Behind Allocation & Key Notes")

if risk == "High":
    st.markdown("""
    <div class='note-box'>
    🚀 **High Risk:** Portfolio tilted heavily towards equities, mid/small caps, and thematic funds.  
    Goal is to maximize long-term growth even with higher volatility. Age and time horizon allow recovery from drawdowns.
    </div>
    """, unsafe_allow_html=True)

elif risk == "Moderate":
    st.markdown("""
    <div class='note-box'>
    ⚖️ **Moderate Risk:** Balanced mix of equities, debt, and alternative assets.  
    Designed for steady growth with manageable volatility. Gold and bonds cushion market dips.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class='note-box'>
    🛡️ **Low Risk:** Focus on capital protection and stable income.  
    Higher debt and bond exposure reduce volatility while still providing modest returns.
    </div>
    """, unsafe_allow_html=True)

# Key asset logic
asset_logic = {
    "Large Cap Equity": "Core portfolio anchor — less volatile and provides steady compounding.",
    "Mid/Small Cap Equity": "Higher growth potential, suitable for long-term wealth creation.",
    "Debt Funds": "Provide stability and steady income; reduce portfolio volatility.",
    "Gold ETF": "Acts as an inflation hedge and safe haven during market uncertainty.",
    "REITs/InvITs": "Generate passive income through real estate or infrastructure projects.",
    "International Equity": "Adds global diversification, reducing country-specific risk.",
    "Cash/Liquid": "Provides liquidity and flexibility for emergencies or quick redeployment.",
    "Govt Bonds": "Low-risk component; protects principal and adds fixed income stability.",
    "Commodities": "Diversifies portfolio with inflation-sensitive exposure.",
    "Crypto (Regulated)": "Tiny speculative exposure for tech-driven growth opportunities.",
    "Index Funds": "Provide low-cost, passive participation in market performance.",
    "Hybrid Funds": "Balance between equity and debt, offering moderate returns.",
    "Thematic Funds": "Capture emerging trends (AI, EVs, renewable energy) for growth boost.",
    "Corporate Bonds": "Higher yield debt exposure, balancing between safety and returns."
}

for _, row in df.iterrows():
    st.markdown(
        f"<div class='note-box'><b>{row['Asset Class']}</b>: {asset_logic[row['Asset Class']]}</div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# PORTFOLIO METRICS
# ---------------------------------------------------
st.divider()
st.subheader("📈 Portfolio Projection Summary")

total_return_1y = df["Exp. 1Y Gain (₹)"].sum()
total_value_5y = df["Exp. 5Y Value (₹)"].sum()

col1, col2 = st.columns(2)
col1.metric("Expected 1-Year Gain", f"₹{int(total_return_1y):,}", "+Est.")
col2.metric("Expected 5-Year Portfolio Value", f"₹{int(total_value_5y):,}")

st.markdown(
    f"<p class='big-font'>🎯 Based on a total investment of <b>₹{total_investment:,}</b>, "
    f"with a <b>{risk}</b> risk profile at age <b>{age}</b>.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# DOWNLOAD OPTION
# ---------------------------------------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Allocation as CSV",
    data=csv,
    file_name="asset_allocation_plan.csv",
    mime="text/csv",
)

st.divider()
st.caption("🚀 Built with Streamlit | © 2025 Smart Asset Allocator | For Educational Use Only")
