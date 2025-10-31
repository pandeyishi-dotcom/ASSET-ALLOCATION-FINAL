import streamlit as st
import pandas as pd

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Smart Asset Allocation Planner",
    page_icon="💰",
    layout="wide"
)

# -------------------------------
# HEADER
# -------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #004e89;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 40px;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.05);
    }
    .bar {
        height: 12px;
        border-radius: 10px;
        margin-top: 6px;
        background: linear-gradient(90deg, #0099ff, #66ccff);
    }
    </style>
    <h1 class='main-title'>💼 Smart Asset Allocation Planner</h1>
    <p class='subtitle'>Balance growth, safety, and diversification — based on your age and risk appetite.</p>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# SIDEBAR - USER INPUT
# -------------------------------
st.sidebar.header("⚙️ Portfolio Customization")

age_group = st.sidebar.selectbox(
    "Select Your Age Group",
    ["20–25 (High Risk)", "30–45 (Moderate Risk)", "45–65 (Low Risk)"]
)

investment_amount = st.sidebar.number_input(
    "💵 Enter Total Investment (₹)",
    min_value=10000, value=500000, step=10000
)

# -------------------------------
# ALLOCATION LOGIC
# -------------------------------
if "20–25" in age_group:
    data = {
        "Asset Class": ["📈 Equity", "🌏 International Equity", "💹 Mutual Funds", "🪙 Gold ETFs", "🏢 REITs", "₿ Cryptocurrency"],
        "Allocation (%)": [40, 20, 15, 10, 10, 5],
        "Logic": [
            "Maximize growth potential with equity exposure.",
            "Diversify globally for tech and currency resilience.",
            "Benefit from professional fund management.",
            "Hedge against inflation and volatility.",
            "Get passive real estate income.",
            "Capture emerging digital asset trends."
        ],
        "Key Notes": [
            "Young investors can handle higher volatility.",
            "Global allocation improves long-term risk-adjusted returns.",
            "Mutual funds add stability within equities.",
            "Gold stabilizes your portfolio.",
            "REITs provide both income and appreciation.",
            "Crypto exposure should be limited to 5%."
        ]
    }

elif "30–45" in age_group:
    data = {
        "Asset Class": ["📈 Equity", "💰 Debt Instruments", "🪙 Gold ETFs", "💹 Mutual Funds", "🏠 Real Estate", "🌏 International Equity"],
        "Allocation (%)": [35, 25, 10, 15, 10, 5],
        "Logic": [
            "Balance growth with manageable volatility.",
            "Debt ensures predictable income.",
            "Gold protects purchasing power.",
            "Mutual funds enhance diversification.",
            "Real estate builds wealth stability.",
            "Foreign exposure hedges domestic slowdown."
        ],
        "Key Notes": [
            "Peak earning years — focus on growth + safety.",
            "Debt allocation reduces portfolio shocks.",
            "Gold allocation at 10% is ideal inflation hedge.",
            "Active funds can outperform in mid-term cycles.",
            "Real estate for dual benefits: rent + value.",
            "Small global exposure completes diversification."
        ]
    }

else:  # 45–65
    data = {
        "Asset Class": ["💰 Debt Instruments", "📈 Equity", "🪙 Gold ETFs", "🏦 Fixed Deposits", "💹 Mutual Funds", "🏠 Real Estate"],
        "Allocation (%)": [40, 20, 10, 15, 10, 5],
        "Logic": [
            "Focus on stable income and capital safety.",
            "Limited equity exposure for growth.",
            "Gold provides protection in downturns.",
            "FDs ensure liquidity and guaranteed returns.",
            "Balanced funds offer mild growth.",
            "Real estate preserves long-term value."
        ],
        "Key Notes": [
            "Shift towards steady income generation.",
            "Lower equity reduces risk of drawdowns.",
            "Gold guards purchasing power.",
            "Liquidity is key near retirement.",
            "Balanced funds smoothen returns.",
            "Real estate supports passive income."
        ]
    }

# -------------------------------
# DATAFRAME CREATION
# -------------------------------
df = pd.DataFrame(data)
df["Allocation (₹)"] = (df["Allocation (%)"] / 100) * investment_amount

# -------------------------------
# MAIN DISPLAY
# -------------------------------
st.subheader("📊 Portfolio Summary")
st.dataframe(df, use_container_width=True, hide_index=True)

# -------------------------------
# VISUAL - ALLOCATION PROGRESS BARS
# -------------------------------
st.markdown("## 🎨 Visual Allocation Breakdown")

for i, row in df.iterrows():
    pct = row["Allocation (%)"]
    st.markdown(
        f"""
        <div class='card'>
            <b>{row["Asset Class"]}</b> — {pct}% ({row["Allocation (₹)"]:,.0f} ₹)
            <div class='bar' style='width:{pct}%; background:linear-gradient(90deg, #0066cc {pct}%, #00ccff);'></div>
            <p style='font-size:13px; color:#444; margin-top:10px;'>
            <b>Logic:</b> {row["Logic"]}<br>
            <b>Key Note:</b> {row["Key Notes"]}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# INSIGHTS SECTION
# -------------------------------
st.markdown("## 💡 Investment Insights")
st.success(
    "Asset allocation aligns with your financial life stage. "
    "Younger investors can afford higher volatility, while mature investors must prioritize capital preservation and income."
)

st.info(
    "A diversified portfolio helps **reduce risk without lowering expected returns**. "
    "Each asset behaves differently across economic cycles — that’s the magic of diversification."
)

st.caption("Crafted with ❤️ using Streamlit — empowering smarter, data-driven financial planning.")
