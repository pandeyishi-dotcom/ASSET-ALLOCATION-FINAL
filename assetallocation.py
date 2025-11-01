# app.py
"""
All-in-one Indian Asset Allocation Dashboard
Features:
- Premium visuals, multi-goal planner, live-data blending (yfinance),
- Monte Carlo, Efficient Frontier (random portfolios), Sharpe/Sortino/VAR,
- Correlation heatmap, rebalancing worksheet, downloads, SIP planner.
Deploy on Streamlit Cloud. After push: Clear cache & Rerun from scratch.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from math import ceil
from datetime import datetime, timedelta

# -------------------------
# PAGE & STYLES
# -------------------------
st.set_page_config(page_title="Pro Indian Asset Allocator", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    :root{
      --bg:#061322;
      --panel:#0b2a3a;
      --muted:#9FB4C8;
      --accent:#6FF0B0;
      --card: rgba(255,255,255,0.03);
      --glass: rgba(255,255,255,0.02);
    }
    .stApp { background: linear-gradient(180deg,var(--bg), #03222b); color: #E8F0F2;}
    .title { font-size:28px; font-weight:700; color:var(--accent); margin-bottom:0; }
    .subtitle { color:var(--muted); margin-top:3px; margin-bottom:12px; }
    .small { color:var(--muted); font-size:12px; }
    .card { background:var(--card); padding:12px; border-radius:10px; border:1px solid var(--glass); }
    .metric { background:#071425; padding:10px; border-radius:8px; text-align:center; }
    .kpi { font-size:16px; font-weight:700; color:var(--accent); }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="title">💼 Pro Indian Asset Allocator & Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Visuals · Live data · Monte Carlo · Efficient frontier · Multi-goal planner</div>', unsafe_allow_html=True)

# -------------------------
# Asset universe & defaults
# -------------------------
ALL_ASSETS = [
    "Large Cap Equity","Mid/Small Cap Equity","International Equity","Index ETFs","Active Equity Funds",
    "Sectoral/Thematic Funds","Debt Funds","Government Bonds","Corporate Bonds","PPF","NPS","EPF",
    "Gold ETF","Silver/Commodities","REITs/InvITs","Real Estate (Direct)","Liquid/Cash","Fixed Deposits",
    "Commodities(non-precious)","Crypto (speculative)","Insurance-linked (ULIPs/Annuities)"
]

DEFAULT_TICKERS = {
    "Large Cap Equity": "^NSEI",       # Nifty 50 (Yahoo)
    "Index ETFs": "NIFTYBEES.NS",      # sample
    "Gold ETF": "GOLDBEES.NS",         # sample
    "International Equity": "VTI",     # fallback global ETF
}

# -------------------------
# Sidebar: user inputs
# -------------------------
st.sidebar.header("Investor Profile & Settings")
age = st.sidebar.slider("Age", 18, 75, 35)
risk_profile = st.sidebar.selectbox("Risk profile", ["High (20–35)", "Moderate (30–50)", "Conservative (45–65)"])
st.sidebar.markdown("---")

# multi-goal support
st.sidebar.subheader("Goals (add multiple)")
if "goals" not in st.session_state:
    st.session_state.goals = [{"name":"Retirement","amount":5000000,"years":15},{"name":"Home","amount":2000000,"years":6}]
with st.sidebar.expander("View / Edit goals"):
    goals_df = pd.DataFrame(st.session_state.goals)
    st.write("Existing goals")
    edited = st.experimental_data_editor(goals_df, num_rows="dynamic")
    # update session
    st.session_state.goals = edited.to_dict("records")

st.sidebar.markdown("---")
selected_assets = st.sidebar.multiselect("Select asset classes to include", ALL_ASSETS,
                                         default=["Large Cap Equity","Index ETFs","Debt Funds","Gold ETF","Liquid/Cash"])
auto_normalize = st.sidebar.checkbox("Auto-normalize allocations to 100%", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Goal inputs (global)")
goal_amount_manual = st.sidebar.number_input("Goal override (₹) — optional", min_value=0, value=0, step=10000)
current_investment = st.sidebar.number_input("Current invested (lump-sum ₹)", min_value=0, value=500000, step=10000)
use_sip = st.sidebar.checkbox("Use monthly SIP", value=True)
monthly_sip = st.sidebar.number_input("Monthly SIP (₹)", min_value=0, value=10000, step=1000) if use_sip else 0
horizon_global = st.sidebar.slider("Default horizon (years)", 1, 40, 10)
inflation = st.sidebar.slider("Expected annual inflation (%)", 0.0, 8.0, 4.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Live tickers (optional)")
ticker_overrides = {}
for a in ["Large Cap Equity","Index ETFs","Gold ETF","International Equity"]:
    v = st.sidebar.text_input(f"{a} ticker", value=DEFAULT_TICKERS.get(a,""), key=f"t_{a}")
    ticker_overrides[a] = v.strip() if v.strip() else None

st.sidebar.markdown("---")
st.sidebar.subheader("Computation limits (Streamlit Cloud friendly)")
mc_sims = st.sidebar.slider("Monte Carlo sims", 200, 4000, 1500, step=100)
frontier_samples = st.sidebar.slider("Efficient frontier samples", 50, 2000, 500, step=50)
corr_lookback = st.sidebar.selectbox("Correlation/Returns lookback (years)", [1,3,5], index=2)

st.sidebar.markdown("---")
st.sidebar.write("Tip: increase sims/samples for more precision; watch CPU/timeout on Streamlit Cloud.")

# -------------------------
# Baseline weights per profile
# -------------------------
BASELINES = {
    "High (20–35)": { "Large Cap Equity":20,"Mid/Small Cap Equity":18,"International Equity":10,"Index ETFs":8,
                      "Active Equity Funds":7,"Sectoral/Thematic Funds":5,"Debt Funds":6,"Government Bonds":3,"Corporate Bonds":3,
                      "PPF":0,"NPS":2,"EPF":2,"Gold ETF":6,"Silver/Commodities":0,"REITs/InvITs":4,"Real Estate (Direct)":6,
                      "Liquid/Cash":2,"Fixed Deposits":0,"Commodities(non-precious)":2,"Crypto (speculative)":2,"Insurance-linked (ULIPs/Annuities)":0},
    "Moderate (30–50)": { "Large Cap Equity":18,"Mid/Small Cap Equity":12,"International Equity":8,"Index ETFs":8,
                         "Active Equity Funds":7,"Sectoral/Thematic Funds":3,"Debt Funds":15,"Government Bonds":6,"Corporate Bonds":4,
                         "PPF":2,"NPS":3,"EPF":2,"Gold ETF":6,"Silver/Commodities":1,"REITs/InvITs":4,"Real Estate (Direct":5,
                         "Real Estate (Direct)":5,"Liquid/Cash":2,"Fixed Deposits":1,"Commodities(non-precious)":1,"Crypto (speculative)":0,"Insurance-linked (ULIPs/Annuities)":0},
    "Conservative (45–65)": { "Large Cap Equity":12,"Mid/Small Cap Equity":5,"International Equity":4,"Index ETFs":6,
                              "Active Equity Funds":3,"Sectoral/Thematic Funds":0,"Debt Funds":28,"Government Bonds":12,"Corporate Bonds":8,
                              "PPF":6,"NPS":4,"EPF":3,"Gold ETF":8,"Silver/Commodities":0,"REITs/InvITs":0,"Real Estate (Direct)":3,
                              "Liquid/Cash":5,"Fixed Deposits":3,"Commodities(non-precious)":0,"Crypto (speculative)":0,"Insurance-linked (ULIPs/Annuities)":1}
}

# -------------------------
# Build allocation
# -------------------------
def build_allocation(profile, selected, normalize=True):
    if not selected:
        return {}
    base = BASELINES[profile]
    alloc = {a: base.get(a, 0.0) for a in selected}
    total = sum(alloc.values())
    if total == 0:
        n = len(alloc)
        return {k: round(100.0/n,2) for k in alloc}
    if normalize:
        scale = 100.0/total
        alloc = {k: round(v*scale,2) for k,v in alloc.items()}
    return alloc

allocation = build_allocation(risk_profile, selected_assets, normalize=auto_normalize)
if not allocation:
    st.warning("Select at least one asset class.")
    st.stop()

alloc_df = pd.DataFrame({"Asset Class": list(allocation.keys()), "Allocation (%)": list(allocation.values())})
alloc_df["Allocation (₹)"] = alloc_df["Allocation (%)"]/100.0 * current_investment

# -------------------------
# Live data helpers (cagr, vol, returns series)
# -------------------------
@st.cache_data(ttl=60*30)
def fetch_ticker_history(ticker, years=5):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{years}y", interval="1d")
        if hist is None or hist.empty:
            return None
        close = hist["Close"].dropna()
        if len(close) < 10:
            return None
        return close
    except Exception:
        return None

def compute_cagr_and_vol(series):
    # series: pd.Series close indexed by date
    start = series.iloc[0]
    end = series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0:
        return None, None
    cagr = (end/start)**(1.0/years)-1.0
    daily = series.pct_change().dropna()
    vol = daily.std() * np.sqrt(252)
    return float(cagr), float(vol)

# default estimates (fallback)
DEFAULT_RETURNS = { a:0.10 if "Equity" in a else 0.06 for a in ALL_ASSETS }
# fine tune some
DEFAULT_RETURNS.update({"Large Cap Equity":0.10,"Mid/Small Cap Equity":0.13,"Gold ETF":0.07,"Liquid/Cash":0.035,"PPF":0.07})
DEFAULT_VOL = { a:0.18 if "Equity" in a else 0.06 for a in ALL_ASSETS }
DEFAULT_VOL.update({"Mid/Small Cap Equity":0.30,"Gold ETF":0.20,"Liquid/Cash":0.01,"PPF":0.01})

# build asset -> ticker map
asset_tickers = {}
for a in allocation.keys():
    asset_tickers[a] = ticker_overrides.get(a) or DEFAULT_TICKERS.get(a)

# compute asset live/blended returns & vols and prepare return series for correlation if possible
asset_returns = {}
asset_vols = {}
asset_series = {}
lookback_years = corr_lookback
for a in allocation.keys():
    t = asset_tickers.get(a)
    if t:
        s = fetch_ticker_history(t, years=lookback_years)
        if s is not None:
            cagr, vol = compute_cagr_and_vol(s)
            if cagr is not None:
                asset_returns[a] = 0.7*cagr + 0.3*DEFAULT_RETURNS.get(a,0.06)
                asset_vols[a] = 0.7*vol + 0.3*DEFAULT_VOL.get(a,0.15)
                asset_series[a] = s.pct_change().dropna()
            else:
                asset_returns[a] = DEFAULT_RETURNS.get(a,0.06)
                asset_vols[a] = DEFAULT_VOL.get(a,0.15)
        else:
            asset_returns[a] = DEFAULT_RETURNS.get(a,0.06)
            asset_vols[a] = DEFAULT_VOL.get(a,0.15)
    else:
        asset_returns[a] = DEFAULT_RETURNS.get(a,0.06)
        asset_vols[a] = DEFAULT_VOL.get(a,0.15)

alloc_df["Exp Return (%)"] = alloc_df["Asset Class"].map(lambda x: asset_returns.get(x,0.06)*100)
alloc_df["Volatility (%)"] = alloc_df["Asset Class"].map(lambda x: asset_vols.get(x,0.15)*100)

# -------------------------
# Portfolio expected & cov
# -------------------------
weights = np.array(alloc_df["Allocation (%)"]/100.0)
means = np.array([asset_returns[a] for a in alloc_df["Asset Class"]])
vols = np.array([asset_vols[a] for a in alloc_df["Asset Class"]])

# estimate covariance using available series; otherwise use vols and base corr
if len(asset_series) >= 2:
    combined = pd.concat(asset_series, axis=1).dropna()
    combined.columns = list(asset_series.keys())
    cov_mat = combined.cov().values * 252  # annualized
else:
    base_corr = 0.25
    cov_mat = np.outer(vols, vols) * base_corr
    np.fill_diagonal(cov_mat, vols**2)

port_return = float(np.dot(weights, means))
port_vol = float(np.sqrt(weights @ cov_mat @ weights))

# -------------------------
# Risk metrics: Sharpe, Sortino, VaR
# -------------------------
def sharpe_ratio(returns, rf=0.04):
    # returns annual decimal
    try:
        excess = returns - rf
        return excess / (np.std(returns) if np.std(returns)>0 else 1e-9)
    except Exception:
        return None

def sortino_ratio(returns, rf=0.04):
    # returns as annual decimals array - here we approximate from means & vols (not instrument-level)
    # For simplicity, compute using annualized mean and downside deviation estimate
    mu = np.mean(returns)
    downside = np.sqrt(np.mean(np.minimum(0, returns - rf)**2))
    return (mu - rf) / (downside if downside>0 else 1e-9)

def var_95(sim_vals):
    return np.percentile(sim_vals, 5)

# -------------------------
# Monte Carlo simulation (annual correlated returns)
# -------------------------
@st.cache_data(ttl=60*10)
def monte_carlo(invest, monthly, w, mu, cov, years, sims):
    n = len(w)
    L = np.linalg.cholesky(cov)
    final_vals = np.zeros(sims)
    annual_sip = monthly*12.0
    base_alloc = w * invest
    for s in range(sims):
        vals = base_alloc.copy()
        for y in range(years):
            z = np.random.normal(size=n)
            ret = mu + L @ z
            vals = vals * (1.0 + ret)
            if annual_sip > 0:
                vals = vals + annual_sip * w
        final_vals[s] = vals.sum()
    return final_vals

with st.spinner("Running Monte Carlo..."):
    mc = monte_carlo(current_investment, monthly_sip, weights, means, cov_mat, horizon_global, mc_sims)

prob_goal = float((mc >= (goal_amount_manual if goal_amount_manual>0 else sum([g["amount"] for g in st.session_state.goals]))) .sum() / len(mc) * 100.0)
median_end = float(np.median(mc))
expected_end = float(np.mean(mc))
p10 = float(np.percentile(mc,10))
p90 = float(np.percentile(mc,90))

# -------------------------
# Deterministic projection helpers and SIP shortfall
# -------------------------
def fv_lumpsum(pv, r, years):
    return pv * ((1+r)**years)
def fv_sip(monthly, r, years):
    r_m = (1+r)**(1/12)-1
    n = years*12
    if r_m==0:
        return monthly*n
    return monthly * (((1+r_m)**n - 1)/r_m) * (1+r_m)
def deterministic_portfolio_fv(invest, monthly, weights_v, means_v, years):
    tot = 0.0
    for i,w_ in enumerate(weights_v):
        r = means_v[i]
        pv = invest * w_
        tot += fv_lumpsum(pv, r, years) + fv_sip(monthly*w_, r, years)
    return tot

def find_required_sip(invest, cur_sip, weights_v, means_v, years, target):
    if deterministic_portfolio_fv(invest, cur_sip, weights_v, means_v, years) >= target:
        return int(cur_sip)
    lo, hi = 0.0, 500000.0
    for _ in range(45):
        mid = (lo+hi)/2
        if deterministic_portfolio_fv(invest, mid, weights_v, means_v, years) >= target:
            hi = mid
        else:
            lo = mid
    return int(ceil(hi))

# compute required monthly to meet combined goals or manual goal
target_goal = goal_amount_manual if goal_amount_manual>0 else sum([g["amount"] for g in st.session_state.goals])
required_monthly = find_required_sip(current_investment, monthly_sip, weights, means, horizon_global, target_goal)

# -------------------------
# Efficient frontier (random portfolios)
# -------------------------
def random_portfolios(n_assets, samples=500):
    # returns weights array shape (samples,n_assets)
    r = np.random.random((samples, n_assets))
    r = r / r.sum(axis=1, keepdims=True)
    return r

ef_samples = min(max(50, frontier_samples), 3000)
with st.spinner("Approximating Efficient Frontier (random portfolios)..."):
    rand_w = random_portfolios(len(weights), samples=ef_samples)
    ef_returns = rand_w.dot(means)
    ef_vols = np.sqrt(np.einsum('ij,jk,ik->i', rand_w, cov_mat, rand_w))
    ef_sharpes = (ef_returns - 0.04) / (ef_vols + 1e-9)

# -------------------------
# UI Layout - Tabs
# -------------------------
tabs = st.tabs(["Overview","Visualization","Analytics","Goals & Planner","Rebalance & Download"])
# ---------- OVERVIEW ----------
with tabs[0]:
    st.header("Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Exp. annual return", f"{port_return*100:.2f}%")
    c2.metric("Est. volatility (σ)", f"{port_vol*100:.2f}%")
    c3.metric(f"Median MC end ({horizon_global}y)", f"₹{median_end:,.0f}")
    c4.metric("Prob. reach combined goal", f"{prob_goal:.1f}%")
    st.markdown("**Allocation table**")
    st.dataframe(alloc_df.style.format({"Allocation (%)":"{:.2f}","Allocation (₹)":"₹{:,.0f}","Exp Return (%)":"{:.2f}%","Volatility (%)":"{:.2f}%"}), use_container_width=True)
    st.markdown("**Notes**")
    st.info("Live data blended with defaults. Efficient frontier approximated via random portfolios. Monte Carlo uses annual correlated normal draws.")

# ---------- VISUALIZATION ----------
with tabs[1]:
    st.header("Visualization")
    colA, colB = st.columns([1,1])
    with colA:
        # Allocation donut
        fig_pie = px.pie(alloc_df, names="Asset Class", values="Allocation (%)", hole=0.35,
                         color_discrete_sequence=px.colors.sequential.Tealgrn)
        fig_pie.update_layout(title="Allocation", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_pie, use_container_width=True)
        # scatter risk-return individual assets
        fig_sc = px.scatter(alloc_df, x="Volatility (%)", y="Exp Return (%)", size="Allocation (%)", text="Asset Class")
        fig_sc.update_layout(title="Asset Risk vs Return", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_sc, use_container_width=True)
    with colB:
        # Efficient frontier + current portfolio marker
        ef_df = pd.DataFrame({"Return":ef_returns*100, "Volatility":ef_vols*100, "Sharpe":ef_sharpes})
        fig_ef = px.scatter(ef_df, x="Volatility", y="Return", color="Sharpe", color_continuous_scale="Viridis", title="Efficient frontier (random samples)")
        fig_ef.add_trace(go.Scatter(x=[port_vol*100], y=[port_return*100], mode="markers+text", marker=dict(size=14,color="gold"), text=["Your portfolio"], textposition="top center"))
        fig_ef.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_ef, use_container_width=True)
        # deterministic timeline
        st.markdown("**Deterministic projection (expected returns)**")
        years_range = list(range(0,horizon_global+1))
        det_vals = [deterministic_portfolio_fv(current_investment, monthly_sip, weights, means, y) for y in years_range]
        det_df = pd.DataFrame({"Year":years_range, "Value":det_vals})
        fig_det = px.line(det_df, x="Year", y="Value", markers=True)
        fig_det.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_det, use_container_width=True)

# ---------- ANALYTICS ----------
with tabs[2]:
    st.header("Analytics & Risk Metrics")
    # correlation heatmap if enough series
    if len(asset_series) >= 2:
        comb = pd.concat(asset_series, axis=1).dropna()
        comb.columns = list(asset_series.keys())
        corr = comb.corr()
        fig_heat = px.imshow(corr, text_auto=True, title=f"Correlation (last {corr_lookback}y)", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Not enough live series to compute correlation heatmap (provide tickers or include assets with defaults).")
    # Portfolio metrics
    st.markdown("### Portfolio risk metrics")
    sr = (port_return - 0.04) / (port_vol + 1e-9)
    st.metric("Sharpe-like (annual)", f"{sr:.2f}")
    # compute Sortino approx using downside assumption (approx)
    sortino = (port_return - 0.04) / (np.sqrt(np.mean(np.minimum(0, means-0.04)**2)) + 1e-9)
    st.metric("Sortino-like (approx)", f"{sortino:.2f}")
    # VaR (95%) from MC
    var95 = var_95(mc)
    st.metric("VaR (95%) — final portfolio", f"₹{var95:,.0f}")
    st.markdown("### Monte Carlo distribution")
    hist = px.histogram(mc, nbins=80, title="Monte Carlo final value distribution")
    hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(hist, use_container_width=True)

# ---------- GOALS & PLANNER ----------
with tabs[3]:
    st.header("Goals & SIP Planner")
    st.subheader("Goals summary")
    goals_df = pd.DataFrame(st.session_state.goals)
    goals_df["Present value (₹)"] = goals_df["amount"]  # user provided amount
    st.dataframe(goals_df, use_container_width=True)
    st.markdown("### Combined target & suggestion")
    st.write(f"Combined target (sum of goals): ₹{goals_df['amount'].sum():,.0f}")
    st.write(f"Deterministic future value (current SIP & expected returns): ₹{deterministic_portfolio_fv(current_investment, monthly_sip, weights, means, horizon_global):,.0f}")
    if required_monthly <= monthly_sip:
        st.success(f"Current SIP ₹{monthly_sip:,.0f} (approx) meets the combined target in {horizon_global} years.")
    else:
        st.warning(f"Suggested SIP to meet combined target: ₹{required_monthly:,.0f}/month (deterministic approx).")
    # per-goal Monte Carlo probability (approx - same portfolio used for all goals but with different horizons/amounts)
    st.markdown("### Per-goal probability (Monte Carlo)")
    for g in st.session_state.goals:
        # re-run quick MC for this goal horizon
        g_h = g.get("years", int(horizon_global))
        g_final = monte_carlo(current_investment, monthly_sip, weights, means, cov_mat, g_h, max(500, int(mc_sims/3)))
        p = float((g_final >= g["amount"]).sum() / len(g_final) * 100.0)
        st.write(f"- **{g['name']}** (target ₹{g['amount']:,} in {g_h}y): **{p:.1f}%** chance")
    st.markdown("---")
    st.markdown("**SIP sensitivity**")
    sip_test = st.slider("Test SIP amount (₹)", 0, 200000, monthly_sip, step=500)
    fv_test = deterministic_portfolio_fv(current_investment, sip_test, weights, means, horizon_global)
    st.write(f"With SIP ₹{sip_test:,}/mo, deterministic future value ≈ ₹{fv_test:,.0f} in {horizon_global} years.")

# ---------- REBALANCE & DOWNLOAD ----------
with tabs[4]:
    st.header("Rebalance worksheet & Download")
    st.subheader("If you have current holdings, paste them (Asset, Value) below")
    cur_df = st.experimental_data_editor(pd.DataFrame(columns=["Asset Class","Current Value (₹)"]), num_rows="dynamic")
    if not cur_df.empty:
        # compute rebalancing trades
        cur_df = cur_df[cur_df["Asset Class"].isin(alloc_df["Asset Class"])]
        if cur_df.empty:
            st.warning("No matching assets found in your holdings vs target allocation.")
        else:
            total_current = cur_df["Current Value (₹)"].sum()
            target_vals = total_current * (alloc_df["Allocation (%)"]/100.0).values
            need = target_vals - cur_df.set_index("Asset Class")["Current Value (₹)"].reindex(alloc_df["Asset Class"]).fillna(0).values
            rebalance_df = pd.DataFrame({
                "Asset Class": alloc_df["Asset Class"],
                "Target Value (₹)": target_vals,
                "Current Value (₹)": cur_df.set_index("Asset Class")["Current Value (₹)"].reindex(alloc_df["Asset Class"]).fillna(0).values,
                "Buy(+)/Sell(-) (₹)": need
            })
            st.dataframe(rebalance_df.style.format({"Target Value (₹)":"₹{:,.0f}","Current Value (₹)":"₹{:,.0f}","Buy(+)/Sell(-) (₹)":"₹{:,.0f}"}), use_container_width=True)
            csv = rebalance_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download rebalance worksheet (CSV)", csv, file_name="rebalance.csv", mime="text/csv")
    st.markdown("---")
    st.subheader("Download allocation & projections")
    out = alloc_df.copy()
    out["Ticker (attempted)"] = out["Asset Class"].map(lambda x: asset_tickers.get(x) or "")
    st.download_button("Download allocation CSV", out.to_csv(index=False).encode("utf-8"), file_name="allocation.csv", mime="text/csv")
    # small printable summary
    st.markdown("### Quick summary (copy/paste)")
    st.code(f"Profile: age {age}, {risk_profile}\nAllocation: {allocation}\nExpected return (wtd): {port_return*100:.2f}%\nVolatility: {port_vol*100:.2f}%\nMC prob meet combined goal: {prob_goal:.1f}%")

# Footer
st.markdown("---")
st.caption("Professional tool for planning & education. Not investment advice. Verify tickers and tax implications before acting.")
