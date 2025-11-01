# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime
from math import ceil

st.set_page_config(page_title="Indian Asset Allocator — Live", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
body { background: linear-gradient(180deg,#041425 0%, #062b3a 100%); color: #E8F0F2; }
h1 { color: #7BE0AD; }
.card { background: rgba(255,255,255,0.03); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.04); }
.small { color:#9FB4C8; font-size:13px; }
</style>
""", unsafe_allow_html=True)

st.title("💼 Indian Goal-based Asset Allocator (Live)")
st.write("Live market-data backed allocation, SIP & Monte Carlo simulation, rebalancing guidance.")

# ---------- Asset universe & default tickers ----------
ALL_ASSETS = [
    "Large Cap Equity", "Mid/Small Cap Equity", "International Equity",
    "Index ETFs", "Active Funds", "Sectoral Funds",
    "Debt Funds", "Government Bonds", "Corporate Bonds",
    "PPF", "NPS", "EPF",
    "Gold ETF", "Silver / Commodities",
    "REITs/InvITs", "Real Estate (Direct)",
    "Liquid/Cash", "Fixed Deposits",
    "Commodities (non-precious)", "Crypto (speculative)", "Insurance-linked"
]

# Representative tickers for live fetch (you can change if you prefer)
# Note: Yahoo tickers ending with .NS are NSE/India where available.
REP_TICKERS = {
    "Large Cap Equity": "NSEI",           # Nifty 50 index (use '^NSEI' on Yahoo; we'll try both)
    "Index ETFs": "NIFTYBEES.NS",        # example Nifty ETF
    "Gold ETF": "GOLDBEES.NS",           # Gold ETF (example)
    "International Equity": "VTI",       # fallback global ETF (US) — user may change
    "REITs/InvITs": "NA",                # no single ticker by default
    "Liquid/Cash": None,
    "Debt Funds": None,
    "PPF": None,
    "Crypto (speculative)": None
}
# convert some index names to Yahoo format commonly recognized:
REP_TICKERS["Large Cap Equity"] = "^NSEI"  # try Yahoo index

# ---------- Sidebar: user inputs ----------
st.sidebar.header("Profile & Goal")
age = st.sidebar.slider("Age", 18, 70, 34)
risk_profile = st.sidebar.selectbox("Risk profile", ["High (20–35)", "Moderate (30–50)", "Conservative (45–65)"])
selected_assets = st.sidebar.multiselect("Select asset classes (toggle)", ALL_ASSETS,
                                         default=["Large Cap Equity", "Index ETFs", "Debt Funds", "Gold ETF", "Liquid/Cash"])
goal_amount = st.sidebar.number_input("Goal amount (₹)", min_value=10000, value=5000000, step=10000)
current_investment = st.sidebar.number_input("Current invested (₹) — lump sum", min_value=0, value=500000, step=10000)
use_sip = st.sidebar.checkbox("Use monthly SIP", value=True)
monthly_sip = st.sidebar.number_input("Monthly SIP (₹)", min_value=0, value=10000, step=1000) if use_sip else 0
horizon = st.sidebar.slider("Horizon (years)", 1, 30, 10)
inflation = st.sidebar.slider("Expected annual inflation (%)", 0.0, 10.0, 4.5, 0.1)
mc_sims = st.sidebar.slider("Monte Carlo simulations", 500, 5000, 2000, step=500)
auto_normalize = st.sidebar.checkbox("Auto-normalize allocations to 100%", value=True)

st.sidebar.markdown("---")
st.sidebar.write("Tip: Toggle asset classes to include them in allocation. Tickers for live data can be edited below.")

# Allow user to override representative tickers (optional)
st.sidebar.markdown("**Optional: override default tickers (Yahoo)**")
ticker_overrides = {}
for asset in ["Large Cap Equity", "Index ETFs", "Gold ETF", "International Equity"]:
    t = st.sidebar.text_input(f"{asset} ticker", value=REP_TICKERS.get(asset) or "", key=f"t_{asset}")
    ticker_overrides[asset] = t.strip() if t.strip() else None

# ---------- Baseline weights per risk profile ----------
BASELINES = {
    "High (20–35)": {
        "Large Cap Equity": 20, "Mid/Small Cap Equity": 18, "International Equity": 10,
        "Index ETFs": 8, "Active Funds": 7, "Sectoral Funds": 5,
        "Debt Funds": 6, "Government Bonds": 3, "Corporate Bonds": 3,
        "PPF": 0, "NPS": 2, "EPF": 2,
        "Gold ETF": 6, "Silver / Commodities": 0, "REITs/InvITs": 4,
        "Real Estate (Direct)": 6, "Liquid/Cash": 2, "Fixed Deposits": 0,
        "Commodities (non-precious)": 2, "Crypto (speculative)": 2, "Insurance-linked": 0
    },
    "Moderate (30–50)": {
        "Large Cap Equity": 18, "Mid/Small Cap Equity": 12, "International Equity": 8,
        "Index ETFs": 8, "Active Funds": 7, "Sectoral Funds": 3,
        "Debt Funds": 15, "Government Bonds": 6, "Corporate Bonds": 4,
        "PPF": 2, "NPS": 3, "EPF": 2,
        "Gold ETF": 6, "Silver / Commodities": 1, "REITs/InvITs": 4,
        "Real Estate (Direct)": 5, "Liquid/Cash": 2, "Fixed Deposits": 1,
        "Commodities (non-precious)": 1, "Crypto (speculative)": 0, "Insurance-linked": 0
    },
    "Conservative (45–65)": {
        "Large Cap Equity": 12, "Mid/Small Cap Equity": 5, "International Equity": 4,
        "Index ETFs": 6, "Active Funds": 3, "Sectoral Funds": 0,
        "Debt Funds": 28, "Government Bonds": 12, "Corporate Bonds": 8,
        "PPF": 6, "NPS": 4, "EPF": 3,
        "Gold ETF": 8, "Silver / Commodities": 0, "REITs/InvITs": 0,
        "Real Estate (Direct)": 3, "Liquid/Cash": 5, "Fixed Deposits": 3,
        "Commodities (non-precious)": 0, "Crypto (speculative)": 0, "Insurance-linked": 1
    }
}

# ---------- Build allocation based on selected assets ----------
def build_allocation(profile_key, selected, normalize=True):
    base = BASELINES[profile_key]
    alloc = {a: base.get(a, 0.0) for a in selected}
    if len(alloc) == 0:
        return {}
    total = sum(alloc.values())
    if total == 0:
        n = len(alloc)
        return {k: round(100/n, 2) for k in alloc.keys()}
    if normalize:
        scale = 100.0 / total
        alloc = {k: round(v * scale, 2) for k, v in alloc.items()}
    return alloc

allocation = build_allocation(risk_profile, selected_assets, normalize=auto_normalize)
if not allocation:
    st.warning("Select at least one asset class in the sidebar to build an allocation.")
    st.stop()

alloc_df = pd.DataFrame({"Asset Class": list(allocation.keys()), "Allocation (%)": list(allocation.values())})
alloc_df["Allocation (₹)"] = alloc_df["Allocation (%)"] / 100.0 * current_investment

# ---------- Utility: compute CAGR from ticker history ----------
@st.cache_data(ttl=60*30)
def ticker_cagr_vol(ticker, years=5):
    """Return (cagr, vol_annual) using close prices over `years`. If fails, return (None,None)."""
    if not ticker:
        return None, None
    try:
        # yfinance uses periods like '5y' or we can use start/end
        hist = yf.Ticker(ticker).history(period=f"{years}y", interval="1d")
        close = hist["Close"].dropna()
        if len(close) < 2:
            return None, None
        start = close.iloc[0]
        end = close.iloc[-1]
        total_years = (close.index[-1] - close.index[0]).days / 365.25
        if total_years <= 0:
            return None, None
        cagr = (end / start) ** (1 / total_years) - 1
        # annualized volatility from daily returns
        daily_ret = close.pct_change().dropna()
        vol_annual = daily_ret.std() * np.sqrt(252)
        return float(cagr), float(vol_annual)
    except Exception:
        return None, None

# ---------- Get expected returns & volatilities for assets (blend live + defaults) ----------
# Default fallback estimates (if live data not available)
DEFAULT_RETURNS = {
    "Large Cap Equity": 0.10, "Mid/Small Cap Equity": 0.13, "International Equity": 0.09,
    "Index ETFs": 0.095, "Active Funds": 0.10, "Sectoral Funds": 0.12,
    "Debt Funds": 0.06, "Government Bonds": 0.05, "Corporate Bonds": 0.065,
    "PPF": 0.07, "NPS": 0.08, "EPF": 0.085,
    "Gold ETF": 0.07, "Silver / Commodities": 0.06, "REITs/InvITs": 0.08,
    "Real Estate (Direct)": 0.08, "Liquid/Cash": 0.035, "Fixed Deposits": 0.05,
    "Commodities (non-precious)": 0.06, "Crypto (speculative)": 0.18, "Insurance-linked": 0.055
}
DEFAULT_VOL = {
    "Large Cap Equity": 0.18, "Mid/Small Cap Equity": 0.30, "International Equity": 0.16,
    "Index ETFs": 0.17, "Active Funds": 0.18, "Sectoral Funds": 0.28,
    "Debt Funds": 0.06, "Government Bonds": 0.03, "Corporate Bonds": 0.04,
    "PPF": 0.01, "NPS": 0.12, "EPF": 0.02,
    "Gold ETF": 0.20, "Silver / Commodities": 0.25, "REITs/InvITs": 0.16,
    "Real Estate (Direct)": 0.12, "Liquid/Cash": 0.01, "Fixed Deposits": 0.01,
    "Commodities (non-precious)": 0.22, "Crypto (speculative)": 0.60, "Insurance-linked": 0.03
}

# Map asset -> ticker to attempt live fetch (merged from REP_TICKERS + overrides)
asset_ticker_map = {}
for asset in allocation.keys():
    t = ticker_overrides.get(asset) or REP_TICKERS.get(asset)
    asset_ticker_map[asset] = t

# Compute returns/vols (if ticker available use live CAGR but fallback to defaults)
asset_returns = {}
asset_vols = {}
for asset in allocation.keys():
    t = asset_ticker_map.get(asset)
    if t:
        cagr, vol = ticker_cagr_vol(t, years=5)
        # if we got robust numbers, use blended estimate (70% live, 30% default) to smooth
        if cagr is not None:
            asset_returns[asset] = 0.7 * cagr + 0.3 * DEFAULT_RETURNS.get(asset, 0.06)
            asset_vols[asset] = 0.7 * vol + 0.3 * DEFAULT_VOL.get(asset, 0.15)
        else:
            asset_returns[asset] = DEFAULT_RETURNS.get(asset, 0.06)
            asset_vols[asset] = DEFAULT_VOL.get(asset, 0.15)
    else:
        asset_returns[asset] = DEFAULT_RETURNS.get(asset, 0.06)
        asset_vols[asset] = DEFAULT_VOL.get(asset, 0.15)

alloc_df["Exp Return (%)"] = alloc_df["Asset Class"].map(lambda x: asset_returns.get(x, 0.06) * 100)
alloc_df["Volatility (%)"] = alloc_df["Asset Class"].map(lambda x: asset_vols.get(x, 0.15) * 100)

# ---------- Portfolio expected metrics ----------
weights = np.array(alloc_df["Allocation (%)"] / 100.0)
means = np.array([asset_returns[a] for a in alloc_df["Asset Class"]])
vols = np.array([asset_vols[a] for a in alloc_df["Asset Class"]])

# simple covariance: assume correlation 0.25 across assets (we could refine by fetching historical returns per ticker)
base_corr = 0.25
cov = np.outer(vols, vols) * base_corr
np.fill_diagonal(cov, vols ** 2)
port_expected_return = float(np.dot(weights, means))
port_volatility = float(np.sqrt(weights @ cov @ weights))

# ---------- Monte Carlo simulation (annual steps) ----------
@st.cache_data(ttl=60*10)
def run_monte_carlo(investment, monthly_sip, weights_vec, means_vec, cov_mat, years, sims):
    n_assets = len(weights_vec)
    L = np.linalg.cholesky(cov_mat)
    final_values = np.zeros(sims)
    annual_sip = monthly_sip * 12.0
    base_alloc = weights_vec * investment
    for s in range(sims):
        asset_vals = base_alloc.copy()
        for y in range(years):
            z = np.random.normal(size=n_assets)
            ret = means_vec + L @ z  # approx correlated returns
            asset_vals = asset_vals * (1 + ret)
            if annual_sip > 0:
                asset_vals = asset_vals + annual_sip * weights_vec
        final_values[s] = asset_vals.sum()
    return final_values

with st.spinner("Running allocation & simulation... this may take a moment"):
    mc_final = run_monte_carlo(current_investment, monthly_sip, weights, means, cov, horizon, mc_sims)

prob_meet_goal = float((mc_final >= goal_amount).sum() / len(mc_final) * 100.0)
median_end = float(np.median(mc_final))
expected_end = float(np.mean(mc_final))
p10 = float(np.percentile(mc_final, 10))
p90 = float(np.percentile(mc_final, 90))

# ---------- SIP shortfall (deterministic approximation) ----------
def deterministic_fv(invest, monthly_sip_val, weights_v, means_v, years):
    total = 0.0
    for i, w in enumerate(weights_v):
        r = means_v[i]
        pv = invest * w
        fv_pv = pv * ((1 + r) ** years)
        if monthly_sip_val > 0:
            r_m = (1 + r) ** (1/12) - 1
            n = years * 12
            fv_sip = monthly_sip_val * w * (((1 + r_m) ** n - 1) / r_m) * (1 + r_m) if r_m != 0 else monthly_sip_val * w * n
        else:
            fv_sip = 0.0
        total += fv_pv + fv_sip
    return total

def find_required_sip(invest, current_sip, weights_v, means_v, years, target):
    # bisection between 0 and a high bound
    if deterministic_fv(invest, current_sip, weights_v, means_v, years) >= target:
        return current_sip
    lo, hi = 0.0, 500000.0
    for _ in range(45):
        mid = (lo + hi) / 2
        if deterministic_fv(invest, mid, weights_v, means_v, years) >= target:
            hi = mid
        else:
            lo = mid
    return ceil(hi)

required_sip = find_required_sip(current_investment, monthly_sip, weights, means, horizon, goal_amount) if goal_amount > 0 else 0

# ---------- UI Layout ----------
st.header("Overview")
c1, c2, c3, c4 = st.columns([1.2,1.2,1.2,1])
c1.metric("Portfolio exp. annual return", f"{port_expected_return*100:.2f}%")
c2.metric("Est. volatility (σ)", f"{port_volatility*100:.2f}%")
c3.metric("Median end value", f"₹{median_end:,.0f}")
c4.metric("Prob. reach goal", f"{prob_meet_goal:.1f}%")

st.markdown("### Allocation table")
st.dataframe(alloc_df.style.format({"Allocation (%)":"{:.2f}", "Allocation (₹)":"₹{:,.0f}", "Exp Return (%)":"{:.2f}%", "Volatility (%)":"{:.2f}%"}), use_container_width=True)

# Allocation pie (plotly)
pie = px.pie(alloc_df, names="Asset Class", values="Allocation (%)", hole=0.35, color_discrete_sequence=px.colors.sequential.Tealgrn)
pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), title="Allocation")
st.plotly_chart(pie, use_container_width=True)

# Monte Carlo histogram
st.markdown("### Monte Carlo distribution of final portfolio value")
hist = px.histogram(mc_final, nbins=60, marginal="box", title="Monte Carlo final value distribution", labels={"value":"Final portfolio value (₹)"})
hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
st.plotly_chart(hist, use_container_width=True)

# Timeline median path (deterministic)
st.markdown("### Deterministic projection (expected returns)")
years_range = list(range(0, horizon+1))
deterministic_timeline = []
for y in years_range:
    deterministic_timeline.append(deterministic_fv(current_investment, monthly_sip, weights, means, y))
df_timeline = pd.DataFrame({"Year": years_range, "Portfolio value": deterministic_timeline})
line = px.line(df_timeline, x="Year", y="Portfolio value", title="Deterministic timeline (expected)", markers=True)
line.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
st.plotly_chart(line, use_container_width=True)

# Benchmarks: fetch live price series and show growth for each benchmark ticker (if user provided)
st.markdown("### Live benchmarks (optional)")
bench_tickers = {
    "Nifty 50": ticker_overrides.get("Large Cap Equity") or "^NSEI",
    "Gold ETF": ticker_overrides.get("Gold ETF") or "GOLDBEES.NS",
    "International ETF": ticker_overrides.get("International Equity") or "VTI"
}
bench_data = {}
for name, t in bench_tickers.items():
    if not t:
        continue
    try:
        hist = yf.Ticker(t).history(period=f"{horizon}y")["Close"].dropna()
        if len(hist) > 1:
            # normalize to 100
            growth = hist / hist.iloc[0] * 100
            bench_data[name] = growth
    except Exception:
        pass

if bench_data:
    # align into dataframe for plotting
    bench_df = pd.DataFrame(bench_data)
    bench_df["Date"] = bench_df.index
    bench_long = bench_df.reset_index(drop=True).melt(id_vars="Date", var_name="Benchmark", value_name="Index")
    fig_b = px.line(bench_long, x="Date", y="Index", color="Benchmark", title="Benchmarks (normalized to 100)")
    fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig_b, use_container_width=True)
else:
    st.info("No benchmark data available for selected tickers (you can override tickers in the sidebar).")

st.markdown("---")
st.header("Actionable outputs")
cola, colb = st.columns(2)
with cola:
    st.subheader("SIP shortfall")
    if required_sip <= monthly_sip:
        st.success(f"Current SIP ₹{monthly_sip:,.0f} should be sufficient (deterministic approx).")
    else:
        st.warning(f"Deterministic suggestion: increase SIP to ₹{required_sip:,.0f}/month to reach the goal in {horizon} years.")
with colb:
    st.subheader("Downloadable report")
    csv = alloc_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download allocation CSV", csv, file_name="allocation.csv", mime="text/csv")

st.markdown("---")
st.subheader("Notes & next steps")
st.markdown("""
- Monte Carlo uses simplified return generation (annual correlated normal draws). For production-grade advice, use instrument-level historical returns and bespoke tax assumptions.  
- Live tickers via Yahoo Finance are attempted where provided — please verify ticker symbols for Indian ETFs/funds.  
- Keep `mc_sims` moderate on Streamlit Cloud (1k–3k) to avoid heavy CPU time.
""")
st.caption("Developed for financial education and planning. Not investment advice.")
