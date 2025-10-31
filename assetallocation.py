# asset_allocation_final.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import html

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="AI-Driven Asset Allocation Dashboard", layout="wide")

# -------------------------
# Modern Executive Theme (Glassmorphic)
# -------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg1: #0E1F1E;
  --bg2: #193733;
  --accent: #E9C46A;
  --text: #FFFFFF;
  --muted: #D9E0DF;
}
[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at top left, var(--bg2), var(--bg1));
  font-family: 'Inter', sans-serif;
  color: var(--text);
}
[data-testid="stSidebar"] {
  background: rgba(12, 24, 22, 0.8);
  backdrop-filter: blur(8px);
}
.card {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(12px);
  border-radius: 12px;
  padding: 16px;
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 6px 18px rgba(0,0,0,0.4);
}
.title {
  font-size: 32px;
  font-weight: 700;
  color: var(--text);
}
.subtitle {
  font-size: 15px;
  color: var(--muted);
  margin-bottom: 20px;
}
.metric-box {
  background: rgba(255,255,255,0.12);
  border-radius: 8px;
  padding: 10px;
  text-align: center;
  box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}
.metric-title {
  font-size: 14px;
  color: var(--muted);
}
.metric-value {
  font-size: 20px;
  color: var(--accent);
  font-weight: 600;
}
.ai-box {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 10px;
  padding: 14px;
  color: var(--text);
  backdrop-filter: blur(10px);
}
.ai-title {
  color: var(--accent);
  font-weight: 600;
  margin-bottom: 4px;
}
hr {
  border: 0;
  border-top: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Title
# -------------------------
st.markdown("<div class='title'>AI-Driven Asset Allocation Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Interactive visual allocation engine — powered by logic, diversification science, and behavioral insight.</div>", unsafe_allow_html=True)

# -------------------------
# Data setup
# -------------------------
data = {
    "Low Risk (45–65)": pd.DataFrame({
        "Asset Class": ["Equity", "Debt", "Gold", "Real Estate", "Cash"],
        "Allocation (%)": [15, 50, 20, 10, 5],
        "Logic": [
            "Equity adds limited volatility; acts as inflation-beating component.",
            "Debt is core — ensures stability and predictable income.",
            "Gold protects capital during downturns and crises.",
            "Real estate is illiquid but steady; diversifies tangible assets.",
            "Cash adds liquidity buffer for emergencies."
        ],
        "Key Note": [
            "Stability-focused; wealth preservation with low drawdowns.",
            "Main anchor of safety; returns tied to interest cycle.",
            "Acts as portfolio insurance.",
            "Used for long-term security and inheritance planning.",
            "Emergency cushion and short-term tactical fund."
        ]
    }),
    "Moderate Risk (30–45)": pd.DataFrame({
        "Asset Class": ["Equity", "Debt", "Real Estate", "Gold", "Cash"],
        "Allocation (%)": [40, 35, 15, 7, 3],
        "Logic": [
            "Higher equity exposure to capture growth and compounding.",
            "Debt provides balance during equity volatility.",
            "Real estate builds stable long-term wealth.",
            "Gold hedges against inflation shocks.",
            "Cash ensures flexibility for rebalancing."
        ],
        "Key Note": [
            "Growth-focused; handles short-term volatility for long-term gain.",
            "Core stabilizer — keeps downside manageable.",
            "Slow compounding tangible asset.",
            "Diversification tool to counter market cycles.",
            "Provides liquidity safety net."
        ]
    }),
    "High Risk (25–30)": pd.DataFrame({
        "Asset Class": ["Equity", "Debt", "Gold", "Real Estate", "Cash"],
        "Allocation (%)": [60, 15, 10, 10, 5],
        "Logic": [
            "Equity dominates to maximize growth over long horizon.",
            "Debt adds minimal stability for downside control.",
            "Gold offers global hedge and stability.",
            "Real estate provides diversification and tangible exposure.",
            "Cash used tactically for re-entry after dips."
        ],
        "Key Note": [
            "Aggressive compounding strategy — volatility tolerated for higher CAGR.",
            "Small allocation for risk offset.",
            "Adds protection in global uncertainty.",
            "Inflation-beating and wealth-building.",
            "Maintains flexibility in turbulent phases."
        ]
    })
}

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Customize Portfolio")
profile = st.sidebar.selectbox("Select Risk Profile", list(data.keys()))
chart_type = st.sidebar.selectbox("Chart Style", ["Pie", "Bar", "3D Scatter"])
years = st.sidebar.slider("Projection Horizon (years)", 1, 20, 10)

# -------------------------
# Main Layout
# -------------------------
left, right = st.columns([2.4, 1])

df = data[profile]
base_returns = {"Low Risk (45–65)": 0.079, "Moderate Risk (30–45)": 0.09, "High Risk (25–30)": 0.108}
growth = [(1 + base_returns[profile]) ** i for i in range(years + 1)]

with left:
    st.markdown(f"### Portfolio — {profile}")
    st.dataframe(df, use_container_width=True)

    # --- Projection ---
    st.markdown("### Growth Projection")
    proj = pd.DataFrame({"Year": list(range(years + 1)), "Index (Base=100)": np.round(100 * np.array(growth), 2)})
    fig = px.line(proj, x="Year", y="Index (Base=100)", markers=True,
                  title="Wealth Projection Over Time",
                  color_discrete_sequence=["#E9C46A"])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#fff"))
    st.plotly_chart(fig, use_container_width=True)

    # --- Allocation Chart ---
    st.markdown("### Allocation Visualization")
    if chart_type == "Pie":
        fig = px.pie(df, names="Asset Class", values="Allocation (%)",
                     color="Asset Class", color_discrete_sequence=px.colors.qualitative.Pastel)
    elif chart_type == "Bar":
        fig = px.bar(df, x="Asset Class", y="Allocation (%)", text="Allocation (%)",
                     color="Asset Class", color_discrete_sequence=px.colors.qualitative.Safe)
    else:
        df3d = df.copy()
        df3d["Risk Score"] = [2, 1, 3, 2, 1]
        df3d["Expected Return"] = [9, 6, 8, 7, 3]
        fig = go.Figure(data=[go.Scatter3d(
            x=df3d["Allocation (%)"], y=df3d["Risk Score"], z=df3d["Expected Return"],
            text=df3d["Asset Class"], mode='markers+text', marker=dict(size=8, color=df3d["Allocation (%)"], colorscale='Viridis')
        )])
        fig.update_layout(scene=dict(xaxis_title='Allocation (%)', yaxis_title='Risk', zaxis_title='Return (%)'),
                          title="3D Risk–Return Space", font=dict(color="#fff"))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### AI Insight Panel")
    insight_text = f"""
    **{profile} Portfolio Overview**

    **Logic Behind Allocation:**  
    {', '.join(df['Logic'])}

    **Key Takeaways:**  
    {', '.join(df['Key Note'])}

    **Projection Summary:**  
    Over {years} years, this portfolio compounds to approximately **{proj['Index (Base=100)'].iloc[-1]:.2f}** on ₹100 invested — assuming disciplined holding and periodic rebalancing.
    """
    st.markdown(f"<div class='ai-box'>{insight_text}</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#B6C5C3;'>Created for portfolio design education — not investment advice.</div>", unsafe_allow_html=True)
