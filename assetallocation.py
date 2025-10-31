import streamlit as st
import pandas as pd

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Smart Asset Allocation Dashboard",
    page_icon="💼",
    layout="wide"
)

# -------------------------------
# HEADER SECTION
# -------------------------------
st.title("💼 Smart Asset Allocation Dashboard")
st.markdown("### Diversify. Balance. Grow.")
st.write(
    "This dashboard illustrates asset allocation strategies across age groups and risk profiles. "
    "Allocation logic is built on diversification, inflation protection, and return-risk balance."
)

# -------------------------------
# SIDEBAR: USER INPUTS
# -------------------------------
st.sidebar.header("Customize Portfolio")
age_group = st.sidebar.selectbox(
    "Select Your Age Group",
    ["20–25 (High Risk)", "30–45 (Moderate Risk)", "45–65 (Low Risk)"]
)
investment_amount = st.sidebar.number_input(
    "Enter Total Investment Amount (₹)", min_value=10000, value=500000, step=10000
)

# -------------------------------
# ASSET ALLOCATION LOGIC
# -------------------------------
if "20–25" in age_group:
    data = {
        "Asset Class": ["Equity", "International Equity", "Mutual Funds", "Gold ETFs", "REITs", "Cryptocurrency"],
        "Allocation (%)": [40, 20, 15, 10, 10, 5],
        "Logic": [
            "High equity exposure for long-term compounding.",
            "Global diversification for currency and tech sector exposure.",
            "Professional management of diversified portfolios.",
            "Inflation hedge with long-term value.",
            "Passive income through real estate-backed assets.",
            "High-risk small allocation for innovation exposure."
        ],
        "Key Notes": [
            "Equities offer maximum growth potential in youth.",
            "Exposure to US and Asian markets improves balance.",
            "Mutual funds smooth volatility.",
            "Gold provides downside protection.",
            "REITs diversify income sources.",
            "Limit crypto to 5% for volatility control."
        ]
    }

elif "30–45" in age_group:
    data = {
        "Asset Class": ["Equity", "Debt Instruments", "Gold ETFs", "Mutual Funds", "Real Estate", "International Equity"],
        "Allocation (%)": [35, 25, 10, 15, 10, 5],
        "Logic": [
            "Balanced equity exposure for growth and stability.",
            "Debt adds predictable income and stability.",
            "Gold protects against inflation.",
            "Mutual funds enhance diversification.",
            "Real estate builds long-term wealth.",
            "Foreign markets provide global growth potential."
        ],
        "Key Notes": [
            "Ideal phase for wealth acceleration.",
            "Debt helps balance volatility.",
            "Maintain gold exposure around 10%.",
            "Active funds outperform in mid-term cycles.",
            "Real estate gives both rental and appreciation gains.",
            "5% in global equity maintains global hedge."
        ]
    }

else:  # 45–65 (Low Risk)
    data = {
        "Asset Class": ["Debt Instruments", "Equity", "Gold ETFs", "Fixed Deposits", "Mutual Funds", "Real Estate"],
        "Allocation (%)": [40, 20, 10, 15, 10, 5],
        "Logic": [
            "Preserve capital while earning moderate returns.",
            "Limited equity for inflation-beating growth.",
            "Gold maintains real value.",
            "Fixed deposits ensure liquidity and safety.",
            "Balanced mutual funds for mild growth.",
            "Real estate stabilizes portfolio value."
        ],
        "Key Notes": [
            "Focus on capital protection and steady income.",
            "Equity limited to avoid volatility.",
            "Gold guards purchasing power.",
            "FDs ensure liquidity for short-term needs.",
            "Hybrid funds help balance risk and reward.",
            "Small property exposure supports retirement income."
        ]
    }

# -------------------------------
# DISPLAY RESULTS
# -------------------------------
df = pd.DataFrame(data)
df["Allocation (₹)"] = (df["Allocation (%)"] / 100) * investment_amount

st.subheader("📊 Recommended Asset Allocation")
st.dataframe(df, use_container_width=True, hide_index=True)

# -------------------------------
# SIMPLE BAR CHART
# -------------------------------
st.subheader("📈 Portfolio Allocation Breakdown")
st.bar_chart(df.set_index("Asset Class")["Allocation (%)"])

# -------------------------------
# ADDITIONAL INSIGHTS
# -------------------------------
st.markdown("## 🧠 Investment Insights")
st.info(
    "Asset allocation is designed to balance **growth and protection**. "
    "Younger investors can handle volatility, while older investors benefit from stable income and capital preservation. "
    "The goal isn’t to chase returns — it’s to stay invested through every market cycle."
)

st.success(
    "💡 *Key takeaway:* Diversification doesn’t eliminate risk, but it dramatically reduces the chance of permanent loss."
)

# Footer
st.caption("Developed with ❤️ using Streamlit — for smarter financial planning.")
