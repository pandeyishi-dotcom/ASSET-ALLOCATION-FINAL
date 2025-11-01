import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------#
# APP CONFIGURATION
# ----------------------------#
st.set_page_config(page_title="Indian Asset Allocation Advisor", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #06122A;
        color: white;
    }
    .title {
        color: #00FFFF;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #A9CCE3;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------#
# HEADER
# ----------------------------#
st.markdown('<p class="title">💼 Smart Asset Allocation Advisor (India)</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Diversify your investments smartly based on your age, goals, and risk appetite</p>', unsafe_allow_html=True)

# ----------------------------#
# USER INPUTS
# ----------------------------#
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Your Age", 18, 70, 30)

with col2:
    risk = st.selectbox("Risk Appetite", ["Low", "Moderate", "High"])

with col3:
    goal = st.selectbox("Investment Goal", ["Wealth Creation", "Retirement", "Short-Term Stability"])

# ----------------------------#
# ALLOCATION LOGIC
# ----------------------------#
def get_allocation(age, risk, goal):
    if risk == "Low":
        return {
            "Equity": 20,
            "Debt": 50,
            "Gold": 20,
            "REITs": 5,
            "Cash/Liquid": 5
        }
    elif risk == "Moderate":
        return {
            "Equity": 40 if age > 45 else 50,
            "Debt": 30,
            "Gold": 10,
            "REITs": 10,
            "Cash/Liquid": 10
        }
    else:  # High risk
        return {
            "Equity": 60 if age < 35 else 50,
            "Debt": 15,
            "Gold": 10,
            "REITs": 10,
            "Cash/Liquid": 5
        }

allocation = get_allocation(age, risk, goal)
df = pd.DataFrame(list(allocation.items()), columns=["Asset Class", "Allocation (%)"])

# ----------------------------#
# PIE CHART
# ----------------------------#
fig = px.pie(
    df,
    names="Asset Class",
    values="Allocation (%)",
    color_discrete_sequence=px.colors.sequential.Tealgrn,
    hole=0.3,
)

fig.update_layout(
    title="Asset Allocation Breakdown",
    title_font=dict(size=22, color="#00FFFF"),
    paper_bgcolor="#06122A",
    font=dict(color="white"),
    showlegend=True,
    legend=dict(
        orientation="h",
        y=-0.2,
        x=0.5,
        xanchor="center",
        font=dict(color="white")
    ),
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------#
# LOGIC & NOTES
# ----------------------------#
st.markdown("---")
st.subheader("📘 Allocation Logic and Key Notes")

logic = """
- **Equity:** Chosen for capital appreciation and long-term growth. Exposure depends on age and risk appetite.
- **Debt:** Provides regular income and stability, essential for conservative investors.
- **Gold:** Acts as an inflation hedge and safe haven during market uncertainty.
- **REITs (Real Estate Investment Trusts):** Offer real estate exposure with liquidity.
- **Cash/Liquid Funds:** Ensure short-term liquidity and protect against volatility.
"""

st.markdown(logic)

st.info(
    "💡 *Younger investors or high-risk takers* can hold more equity for long-term growth. "
    "*Older investors or low-risk takers* should focus on debt and stability. "
    "Diversification across 4–5 asset classes reduces risk and improves consistency of returns."
)

# ----------------------------#
# OPTIONAL ANALYSIS SECTION
# ----------------------------#
st.markdown("---")
st.subheader("📊 Expected Return vs Risk (Illustrative)")

expected_returns = {
    "Equity": 12,
    "Debt": 7,
    "Gold": 8,
    "REITs": 9,
    "Cash/Liquid": 4
}

risk_levels = {
    "Equity": "High",
    "Debt": "Low",
    "Gold": "Medium",
    "REITs": "Medium",
    "Cash/Liquid": "Very Low"
}

df_analysis = pd.DataFrame({
    "Asset Class": expected_returns.keys(),
    "Expected Return (%)": expected_returns.values(),
    "Risk Level": risk_levels.values()
})

st.dataframe(df_analysis, use_container_width=True)

st.success("✅ Tip: Rebalance your portfolio once a year to maintain your target allocation.")

# ----------------------------#
# FOOTER
# ----------------------------#
st.markdown("---")
st.caption("Developed with ❤️ for Indian investors — powered by Streamlit & Plotly")
