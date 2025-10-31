# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import html
from textwrap import shorten

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Asset Allocation Dashboard", layout="wide", initial_sidebar_state="auto")

# -------------------------
# Styling + Fonts (Modern Executive)
# -------------------------
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
    :root{
      --bg-top:#1a4b44;      /* deeper teal */
      --bg-bottom:#204f47;   /* softer teal/graphite */
      --ivory:#FFFFFF;       /* main text - pure white */
      --muted:#E6F0EE;       /* light muted (for subtle text) */
      --accent-gold:#E9C46A; /* gold accent */
      --card:#163933;        /* card panel */
    }
    [data-testid="stAppViewContainer"]{
      background: linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
      color: var(--ivory);
      font-family: 'Montserrat', sans-serif;
    }
    .title-large {
      font-size: 34px;
      font-weight: 700;
      color: var(--ivory);
      margin-bottom: 6px;
    }
    .subtitle {
      color: var(--muted);
      margin-top: 0px;
      margin-bottom: 14px;
      font-size: 13.5px;
    }
    [data-testid="stSidebar"]{
      background: linear-gradient(180deg, rgba(15,30,28,0.95), rgba(12,26,25,0.92));
      border-right: 1px solid rgba(255,255,255,0.04);
    }
    [data-testid="stDataFrame"] table {
      border-radius: 10px;
      background: rgba(255,255,255,0.02);
      border: 1px solid rgba(233,196,106,0.06);
      overflow: hidden;
    }
    [data-testid="stDataFrame"] th {
      background: rgba(233,196,106,0.12) !important;
      color: #071b19 !important;
      font-weight: 600;
      text-align:center;
      padding: 8px;
    }
    [data-testid="stDataFrame"] td {
      color: var(--ivory) !important;
      text-align:center;
      padding: 6px 8px;
      font-size: 14px;
    }
    .panel {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border-radius: 10px;
      padding: 12px;
      border: 1px solid rgba(255,255,255,0.03);
      box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    }
    .ai-box {
      background: rgba(8,12,11,0.65);
      border-radius: 10px;
      border: 1px solid rgba(233,196,106,0.10);
      padding: 14px;
      color: var(--ivory);
      box-shadow: 0 8px 26px rgba(0,0,0,0.5);
      font-size: 15px;
      line-height:1.5;
    }
    .ai-title {
      color: var(--accent-gold);
      font-weight: 700;
      margin-bottom: 6px;
      font-size: 15px;
    }
    .muted { color: var(--muted); font-size:13px; }
    .small { font-size:13px; color:var(--muted); }
    .metric-box {
      background: #fff;
      color: #043;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.06);
      box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Title
# -------------------------
st.markdown('<div class="title-large">Asset Allocation Portfolio Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Executive dashboard — clear tables, pro charts, and an AI-style summary module (typing + subtle sound).</div>', unsafe_allow_html=True)

# -------------------------
# Data (original logic & summary preserved)
# -------------------------
data = {
    "Low Risk (45–65)": pd.DataFrame({
        "Asset Class": ["Equity (Stocks / MFs)", "Debt / Fixed Income", "Gold / Commodities", "Real Estate", "Cash / Liquid Funds"],
        "Allocation (%)": [15, 50, 20, 10, 5],
        "Risk": ["Moderate", "Low", "Moderate", "Low moderate", "Very low"],
        "Reward (Expected)": ["9%–11%", "6%–7%", "8%–9%", "6%–8%", "3%–4%"],
        "Time Period": ["5–7 YRS", "3–5 YRS", "7–10 YRS", "3–5 YRS", "0–1 YRS"],
        "Portfolio Weight Impact (%)": [15, 50, 20, 10, 5],
        "LOGIC": [
            "Stock market fluctuates; you can lose 20–30% value in short term. But since only 15% is invested, overall portfolio impact is ~10%.",
            "Safer because returns are fixed. Risk comes only if interest rates change or company defaults (rare if chosen wisely).",
            "Prices move up/down with global economy, but gold always regains in long term; protects during crises.",
            "Property isn't easy to sell quickly; market cycles can take years. Physical asset = safe long-term.",
            "Almost risk-free, but inflation slowly reduces its real value."
        ],
        "SUMMARY": [
            "Keeps inflation protection and long-term growth, but limits volatility.",
            "Main stability pillar; ensures predictable cash flow and acts as shock absorber.",
            "Protects capital during downturns; hedge against equity risk.",
            "Illiquid but steady; useful as diversification for long-term wealth.",
            "For liquidity in emergencies or reinvestment when markets dip."
        ]
    }),

    "Moderate Risk (30–45)": pd.DataFrame({
        "Asset Class": ["Equity (Stocks / MFs)", "Debt / Fixed Income", "Real Estate", "Gold / Commodities", "Cash / Liquid Funds"],
        "Allocation (%)": [40, 35, 15, 7, 3],
        "Risk": ["High", "Low", "Moderate", "Moderate", "Very low"],
        "Reward (Expected)": ["11%–13%", "6%–7%", "8%–9%", "8%", "3%–4%"],
        "Time Period": ["7–10 YRS", "3–5 YRS", "7–10 YRS", "3–5 YRS", "0–1 YRS"],
        "Portfolio Weight Impact (%)": [40, 35, 15, 7, 3],
        "LOGIC": [
            "People in their 30s–40s can handle volatility since they have longer time horizons. Equity gives growth and beats inflation.",
            "Ensures portfolio stability and consistent returns. Balances market swings from equity.",
            "Diversifies portfolio; suitable for long-term wealth building and rental potential.",
            "Gold acts as a hedge against market downturns and inflation.",
            "Maintains liquidity for opportunities or emergencies."
        ],
        "SUMMARY": [
            "Main wealth generator; short-term volatility balanced by long-term compounding.",
            "Provides steady income, reduces portfolio shocks, keeps capital partly protected.",
            "Tangible asset that provides inflation-adjusted growth though less liquid.",
            "Keeps portfolio stable during uncertainty; small allocation is enough.",
            "Helps during emergencies or reinvestment opportunities; avoids cash drag."
        ]
    }),

    "High Risk (25–30)": pd.DataFrame({
        "Asset Class": ["Equity (Stocks / MFs)", "Debt / Fixed Income", "Gold / Commodities", "Real Estate", "Cash / Liquid Funds"],
        "Allocation (%)": [60, 15, 10, 10, 5],
        "Risk": ["High", "Low", "Moderate", "Moderate", "Very low"],
        "Reward (Expected)": ["12%–14%", "6%–7%", "8%–9%", "9%–10%", "3%–4%"],
        "Time Period": ["7–10 YRS", "1–3 YRS", "5–7 YRS", "7–10 YRS", "0–1 YRS"],
        "Portfolio Weight Impact (%)": [60, 15, 10, 10, 5],
        "LOGIC": [
            "Long-term horizon allows high stock exposure. Short-term volatility (−20% to +30%) is tolerable because recovery over years gives compounding power.",
            "Offers stability to offset market dips and provides liquidity for short-term needs.",
            "Hedge against inflation, currency weakening, and equity crashes.",
            "Provides tangible asset diversification and potential rental income.",
            "Used for emergency fund and quick opportunities during market dips."
        ],
        "SUMMARY": [
            "Core growth engine — maximizes wealth creation through market cycles.",
            "Acts as portfolio stabilizer to reduce panic during volatility.",
            "Adds diversification and protection in global uncertainty.",
            "Long-term capital appreciation; less liquid but inflation-beating.",
            "Ensures liquidity without disturbing long-term holdings."
        ]
    })
}

# -------------------------
# Risk metrics (kept)
# -------------------------
metrics = {
    "Low Risk (45–65)": {"Expected Return": "7.95%", "Risk (Std. Dev.)": "9.90%", "Worst (95%)": "-11%", "Best (95%)": "27%", "Sharpe-like": "0.80"},
    "Moderate Risk (30–45)": {"Expected Return": "9.00%", "Risk (Std. Dev.)": "11.30%", "Worst (95%)": "-13%", "Best (95%)": "31%", "Sharpe-like": "0.80"},
    "High Risk (25–30)": {"Expected Return": "10.75%", "Risk (Std. Dev.)": "19.60%", "Worst (95%)": "-28%", "Best (95%)": "50%", "Sharpe-like": "0.55"}
}

# -------------------------
# Sources table
# -------------------------
sources = pd.DataFrame({
    "Asset": ["Equity (Nifty/Sensex)", "Debt (Govt + Corporate)", "Gold", "Real Estate", "Cash/FD"],
    "Historical Return": ["11–13%", "6–8%", "7–9%", "8–10%", "4–5%"],
    "Volatility": ["12–18%", "3–5%", "10–12%", "8–10%", "<1%"],
    "Source": ["15-year CAGR of Indian Equity Market", "Average yield-to-maturity of debt (2010–2024)", "RBI & World Gold Council long-term data", "Knight Frank + RBI housing index", "Bank FD rates (SBI, HDFC, ICICI)"]
})

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Portfolio Controls")
profile = st.sidebar.selectbox("Select Risk Profile", list(data.keys()))
chart_type = st.sidebar.selectbox("Chart type", ["Bar", "Pie", "3D Scatter"])
animate = st.sidebar.checkbox("Enable animation (pie/bar live redraw)", value=True)
show_typing = st.sidebar.checkbox("Enable typing + sound for AI summary", value=True)
years = st.sidebar.slider("Projection horizon (years)", 1, 20, 10)

# -------------------------
# Prepare current DataFrame
# -------------------------
df = data[profile].copy()

# -------------------------
# Layout: left (main) + right (AI box)
# -------------------------
left, right = st.columns([2.4, 1])

with left:
    st.subheader(f"Portfolio Overview — {profile}")
    st.markdown("<div class='small'>Full table with logic & summary (kept as text fields). Use the right panel for AI-style insight.</div>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

    # Projection
    st.markdown("### Projection (simple compound growth)")
    base_return_map = {"Low Risk (45–65)": 0.0795, "Moderate Risk (30–45)": 0.09, "High Risk (25–30)": 0.1075}
    base_return = base_return_map[profile]
    yrs = list(range(0, years + 1))
    values = [(1 + base_return) ** y for y in yrs]
    proj_df = pd.DataFrame({"Year": yrs, "Index (Base=100)": np.round(100 * np.array(values), 2)})
    proj_fig = px.line(proj_df, x="Year", y="Index (Base=100)", markers=True)
    proj_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           font=dict(color="#FFFFFF"))
    st.plotly_chart(proj_fig, use_container_width=True)

    # Charts: show primary allocation chart
    st.markdown("### Allocation Visual")
    col_a, col_b = st.columns([1, 1])

    labels = df["Asset Class"].tolist()
    vals = df["Allocation (%)"].tolist()
    palette = {
        "Equity (Stocks / MFs)": "#D4A373",
        "Equity": "#D4A373",
        "Debt / Fixed Income": "#4CB5AE",
        "Debt": "#4CB5AE",
        "Gold / Commodities": "#C08B4E",
        "Gold": "#C08B4E",
        "Real Estate": "#6B7280",
        "Cash / Liquid Funds": "#CED4DA",
        "Cash": "#CED4DA"
    }

    if chart_type == "Pie":
        if animate:
            p = st.empty()
            for _ in range(3):
                jitter = np.random.randint(-2, 3, len(vals))
                cur = np.clip(np.array(vals) + jitter, 1, 100)
                fig = px.pie(names=labels, values=cur, hole=0.36)
                fig.update_traces(textinfo='percent+label', marker=dict(colors=[palette.get(l.split(' ')[0], "#8AA4A8") for l in labels]))
                fig.update_layout(showlegend=False, title=f"Allocation — {profile}", font=dict(color="#FFFFFF"))
                p.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.pie(names=labels, values=vals, hole=0.36)
            fig.update_traces(textinfo='percent+label', marker=dict(colors=[palette.get(l.split(' ')[0], "#8AA4A8") for l in labels]))
            fig.update_layout(showlegend=False, title=f"Allocation — {profile}", font=dict(color="#FFFFFF"))
            col_a.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Bar":
        fig = px.bar(df, x="Asset Class", y="Allocation (%)", text="Allocation (%)")
        fig.update_traces(marker_color=[palette.get(l.split(' ')[0], "#8AA4A8") for l in labels], showlegend=False)
        fig.update_layout(title=f"Allocation — {profile}", font=dict(color="#FFFFFF"),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        col_a.plotly_chart(fig, use_container_width=True)
    else:
        # 3D scatter
        risk_map = {"Very low": 1, "Very low":1, "Very Low":1, "Low": 2, "Low moderate":3, "Low-Moderate":3, "Moderate":4, "High":5}
        df_plot = df.copy()
        df_plot["Risk Score"] = df_plot["Risk"].map(risk_map).fillna(3)
        def reward_num(v):
            s = str(v)
            nums = ''.join(ch for ch in s if ch.isdigit())
            return float(nums) if nums else 0.0
        df_plot["Reward %"] = df_plot["Reward (Expected)"].apply(reward_num)
        fig3 = go.Figure(data=[go.Scatter3d(
            x=df_plot["Allocation (%)"],
            y=df_plot["Risk Score"],
            z=df_plot["Reward %"],
            text=df_plot["Asset Class"],
            mode='markers+text',
            marker=dict(size=8, color=df_plot["Allocation (%)"], colorscale='Viridis', opacity=0.95)
        )])
        fig3.update_layout(scene=dict(xaxis_title='Allocation (%)', yaxis_title='Risk Score', zaxis_title='Reward (%)'),
                           title=f"3D Risk-Reward — {profile}", font=dict(color="#FFFFFF"),
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        col_a.plotly_chart(fig3, use_container_width=True)

    # small breakdown table
    col_b.markdown("#### Allocation (summary)")
    col_b.table(df[["Asset Class", "Allocation (%)"]].set_index("Asset Class"))

with right:
    # Performance metrics in readable boxes
    st.subheader("Performance Metrics")
    mm = metrics[profile]
    # show metric cards with white boxes for readability
    metric_cols = st.columns(len(mm))
    for i, (k, v) in enumerate(mm.items()):
        metric_cols[i].markdown(f"<div class='metric-box'><div style='font-weight:600'>{k}</div><div style='font-size:18px; margin-top:6px'>{v}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # AI summary + typing + sound
    st.markdown("### System Insight")
    # build combined summary (preserve original SUMMARY strings)
    combined_paragraph = f"{profile} — " + " ".join([s.strip() for s in df["SUMMARY"].tolist() if isinstance(s, str)])
    # escape for JS
    escaped_text = html.escape(combined_paragraph).replace("\n", "<br>")

    # Prepare HTML + JS for typing + typing sound (Web Audio API)
    # The JS will type and play a subtle clicking oscillator tone for each character.
    typing_html = f"""
    <div class="ai-box" id="aiBox">
      <div class="ai-title">System Analysis</div>
      <div class="muted">Automated insight (read-only)</div>
      <div style="height:10px;"></div>
      <div id="typing" style="white-space:pre-wrap; font-size:14px; color: #FFFFFF;"></div>
    </div>

    <script>
    const text = {escaped_text};
    const el = document.getElementById('typing');

    const enable = {str(show_typing).lower()}; // boolean from Python
    if (!enable) {{
      el.innerHTML = text;
    }} else {{
      // WebAudio typing sound (subtle)
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioContext();
      function playClick(volume=0.02, freq=1000, duration=0.03) {{
        const o = audioCtx.createOscillator();
        const g = audioCtx.createGain();
        o.type = 'square';
        o.frequency.value = freq;
        g.gain.value = volume;
        o.connect(g);
        g.connect(audioCtx.destination);
        o.start();
        setTimeout(()=> {{ o.stop(); }}, duration*1000);
      }}

      // typing
      let i=0;
      function typeChar() {{
        if (i >= text.length) return;
        el.innerHTML += text.charAt(i);
        // play small click only for non-space chars
        if (text.charAt(i) !== ' ' && Math.random() > 0.25) {{
          playClick(0.012, 700+Math.random()*800, 0.02 + Math.random()*0.02);
        }}
        i++;
        // dynamic delay: short after commas/periods
        let delay = 12 + Math.random()*10;
        const ch = text.charAt(i-1);
        if (ch === ',' ) delay = 80;
        if (ch === '.' || ch === '\\n') delay = 160;
        setTimeout(typeChar, delay);
      }}
      // Ensure audio context is resumed after user gesture (some browsers require)
      document.addEventListener('click', function initAudio() {{
        if (audioCtx.state === 'suspended') {{
          audioCtx.resume();
        }}
        // remove listener once resumed
        document.removeEventListener('click', initAudio);
      }});
      // start typing after small delay
      setTimeout(typeChar, 300);
    }}
    </script>
    """
    st.components.v1.html(typing_html, height=220)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Data Sources")
    st.dataframe(sources, use_container_width=True)

# Footer
st.markdown("<hr style='border:1px solid rgba(255,255,255,0.04)'>", unsafe_allow_html=True)
st.markdown("<div style='color: #E6F0EE; font-size:13px;'>Made with care by <b>Ishani Pandey </b>. Models are illustrative — consult an advisor before investing.</div>", unsafe_allow_html=True)
