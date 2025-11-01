# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor

# ---------- Page & theme ----------
st.set_page_config(page_title="Professional Goal-Based Asset Allocator", page_icon="💼", layout="wide")
st.markdown(
    """
    <style>
    :root {
      --bg:#081425;
      --card:#0f2230;
      --accent:#5EE7A6;
      --muted:#9FB4C8;
      --glass: rgba(255,255,255,0.03);
      --panel:#0b2a3a;
    }
    .stApp { background: linear-gradient(180deg, var(--bg), #06202b 100%); color: #E6F0F2; }
    .title { font-size:28px; font-weight:700; color:var(--accent); margin-bottom:0; }
    .subtitle { color:var(--muted); margin-top:2px; margin-bottom:18px; }
    .card { background: var(--card); padding:14px; border-radius:10px; border:1px solid var(--glass); }
    .muted { color:var(--muted); font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">💼 Professional Goal-Based Asset Allocator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Indian asset classes · goal & SIP planning · Monte Carlo probability · rebalancing guidance</div>', unsafe_allow_html=True)

# ---------- Asset universe ----------
ALL_ASSETS = [
    "Large Cap Equity",
    "Mid/Small Cap Equity",
    "International Equity (ETFs)",
    "Index Funds / ETFs",
    "Active Equity Mutual Funds",
    "Sectoral / Thematic Funds",
    "Debt Mutual Funds",
    "Government Bonds (G-Secs)",
    "Corporate Bonds",
    "PPF",
    "NPS",
    "EPF",
    "Gold (ETF/Physical)",
    "Silver / Commodities",
    "REITs/InvITs",
    "Real Estate (Direct)",
    "Liquid Funds / Cash",
    "Fixed Deposits",
    "Commodities (non-precious)",
    "Crypto (small allocation)",
    "Insurance-linked (ULIPs/Annuities)"
]

# ---------- Sidebar controls ----------
st.sidebar.header("Setup your plan")
risk_profile = st.sidebar.selectbox("Risk profile", ["High (20–35)", "Moderate (30–50)", "Conservative (45–65)"])
st.sidebar.markdown("**Pick asset classes to include**")
default = ["Large Cap Equity", "Index Funds / ETFs", "Debt Mutual Funds", "Gold (ETF/Physical)", "Liquid Funds / Cash"]
selected_assets = st.sidebar.multiselect("Assets", ALL_ASSETS, default=default)

st.sidebar.markdown("---")
goal_amount = st.sidebar.number_input("Goal amount (₹)", min_value=10000, value=5000000, step=10000, format="%d")
investment_amount = st.sidebar.number_input("Current invested (lump-sum) (₹)", min_value=0, value=500000, step=10000, format="%d")
use_sip = st.sidebar.checkbox("Use monthly SIP (instead of only lump-sum)", value=False)
monthly_sip = st.sidebar.number_input("Monthly SIP (₹)", min_value=0, value=10000, step=1000, format="%d") if use_sip else 0
horizon_years = st.sidebar.slider("Horizon (years)", 1, 30, 10)
inflation = st.sidebar.slider("Expected annual inflation (%)", 0.0, 10.0, 4.5, 0.1)
simulations = st.sidebar.slider("Monte Carlo sims", 1000, 10000, 3000, step=500)
auto_normalize = st.sidebar.checkbox("Normalize allocation to sum to 100%", value=True)

st.sidebar.markdown("---")
st.sidebar.write("Tip: select a wide mix of assets for better diversification.")

if len(selected_assets) == 0:
    st.sidebar.error("Select at least one asset class.")

# ---------- Baseline weights (canonical) ----------
BASELINE_WEIGHTS = {
    "High (20–35)": {
        "Large Cap Equity": 20, "Mid/Small Cap Equity": 18, "International Equity (ETFs)": 10,
        "Index Funds / ETFs": 8, "Active Equity Mutual Funds": 7, "Sectoral / Thematic Funds": 5,
        "Debt Mutual Funds": 6, "Government Bonds (G-Secs)": 3, "Corporate Bonds": 3,
        "PPF": 0, "NPS": 2, "EPF": 2,
        "Gold (ETF/Physical)": 6, "Silver / Commodities": 0, "REITs/InvITs": 4,
        "Real Estate (Direct)": 6, "Liquid Funds / Cash": 2, "Fixed Deposits": 0,
        "Commodities (non-precious)": 2, "Crypto (small allocation)": 2, "Insurance-linked (ULIPs/Annuities)": 0
    },
    "Moderate (30–50)": {
        "Large Cap Equity": 18, "Mid/Small Cap Equity": 12, "International Equity (ETFs)": 8,
        "Index Funds / ETFs": 8, "Active Equity Mutual Funds": 7, "Sectoral / Thematic Funds": 3,
        "Debt Mutual Funds": 15, "Government Bonds (G-Secs)": 6, "Corporate Bonds": 4,
        "PPF": 2, "NPS": 3, "EPF": 2,
        "Gold (ETF/Physical)": 6, "Silver / Commodities": 1, "REITs/InvITs": 4,
        "Real Estate (Direct)": 5, "Liquid Funds / Cash": 2, "Fixed Deposits": 1,
        "Commodities (non-precious)": 1, "Crypto (small allocation)": 0, "Insurance-linked (ULIPs/Annuities)": 0
    },
    "Conservative (45–65)": {
        "Large Cap Equity": 12, "Mid/Small Cap Equity": 5, "International Equity (ETFs)": 4,
        "Index Funds / ETFs": 6, "Active Equity Mutual Funds": 3, "Sectoral / Thematic Funds": 0,
        "Debt Mutual Funds": 28, "Government Bonds (G-Secs)": 12, "Corporate Bonds": 8,
        "PPF": 6, "NPS": 4, "EPF": 3,
        "Gold (ETF/Physical)": 8, "Silver / Commodities": 0, "REITs/InvITs": 0,
        "Real Estate (Direct)": 3, "Liquid Funds / Cash": 5, "Fixed Deposits": 3,
        "Commodities (non-precious)": 0, "Crypto (small allocation)": 0, "Insurance-linked (ULIPs/Annuities)": 1
    }
}

# ---------- Expected returns & volatilities ----------
EXPECTED_RETURNS = {
    "Large Cap Equity": 0.10, "Mid/Small Cap Equity": 0.13, "International Equity (ETFs)": 0.09,
    "Index Funds / ETFs": 0.095, "Active Equity Mutual Funds": 0.10, "Sectoral / Thematic Funds": 0.12,
    "Debt Mutual Funds": 0.06, "Government Bonds (G-Secs)": 0.05, "Corporate Bonds": 0.065,
    "PPF": 0.07, "NPS": 0.08, "EPF": 0.085,
    "Gold (ETF/Physical)": 0.07, "Silver / Commodities": 0.06, "REITs/InvITs": 0.08,
    "Real Estate (Direct)": 0.08, "Liquid Funds / Cash": 0.035, "Fixed Deposits": 0.05,
    "Commodities (non-precious)": 0.06, "Crypto (small allocation)": 0.18, "Insurance-linked (ULIPs/Annuities)": 0.055
}

# Annual vol estimates (stdev)
EXPECTED_VOL = {
    "Large Cap Equity": 0.18, "Mid/Small Cap Equity": 0.30, "International Equity (ETFs)": 0.16,
    "Index Funds / ETFs": 0.17, "Active Equity Mutual Funds": 0.18, "Sectoral / Thematic Funds": 0.28,
    "Debt Mutual Funds": 0.06, "Government Bonds (G-Secs)": 0.03, "Corporate Bonds": 0.04,
    "PPF": 0.01, "NPS": 0.12, "EPF": 0.02,
    "Gold (ETF/Physical)": 0.20, "Silver / Commodities": 0.25, "REITs/InvITs": 0.16,
    "Real Estate (Direct)": 0.12, "Liquid Funds / Cash": 0.01, "Fixed Deposits": 0.01,
    "Commodities (non-precious)": 0.22, "Crypto (small allocation)": 0.60, "Insurance-linked (ULIPs/Annuities)": 0.03
}

# ---------- Build allocation ----------
def build_allocation(profile, chosen, normalize=True):
    base = BASELINE_WEIGHTS[profile]
    alloc = {a: base.get(a, 0.0) for a in chosen}
    if len(alloc) == 0:
        return {}
    total = sum(alloc.values())
    if total == 0:
        n = len(alloc)
        return {k: round(100/n, 2) for k in alloc.keys()}
    if normalize:
        scale = 100.0 / total
        alloc = {k: round(v*scale, 2) for k, v in alloc.items()}
    return alloc

allocation = build_allocation(risk_profile, selected_assets, normalize=auto_normalize)

if not allocation:
    st.warning("Select at least one asset class on the sidebar.")
    st.stop()

alloc_df = pd.DataFrame({
    "Asset Class": list(allocation.keys()),
    "Allocation (%)": list(allocation.values())
})
alloc_df["Allocation (₹)"] = alloc_df["Allocation (%)"]/100.0 * investment_amount
alloc_df["Exp Return (%)"] = alloc_df["Asset Class"].map(lambda x: EXPECTED_RETURNS.get(x, 0.06)*100)
alloc_df["Volatility (%)"] = alloc_df["Asset Class"].map(lambda x: EXPECTED_VOL.get(x, 0.15)*100)

# ---------- Portfolio stats ----------
weights = np.array(alloc_df["Allocation (%)"]/100.0)
means = np.array([EXPECTED_RETURNS[a] for a in alloc_df["Asset Class"]])
vols = np.array([EXPECTED_VOL[a] for a in alloc_df["Asset Class"]])
# simple covariance: assume base correlation 0.25 between different asset classes, 1 on diag
corr = 0.25
cov = np.outer(vols, vols) * corr
np.fill_diagonal(cov, vols**2)
# portfolio expected return & volatility
port_exp_return = float(np.dot(weights, means))
port_vol = float(np.sqrt(weights @ cov @ weights))

# ---------- Monte Carlo simulation ----------
def simulate_portfolio(investment, monthly_sip, weights_vec, means_vec, cov_mat, years, sims):
    """
    Simulate portfolio final values using annual steps.
    - investment: one-time lump sum invested now (apportioned by weights)
    - monthly_sip: monthly SIP (apportioned by weights). If 0, only lump sum applied.
    - weights_vec: allocation weights per asset (sums to 1)
    - means_vec: expected annual returns per asset
    - cov_mat: covariance matrix (annual)
    - years: horizon in years
    - sims: number of simulation paths
    """
    n_assets = len(weights_vec)
    # Cholesky for correlated returns
    L = np.linalg.cholesky(cov_mat)
    final_vals = np.zeros(sims)
    annual_sip = monthly_sip * 12.0
    # precompute asset-level initial lumps
    base_alloc_amounts = (weights_vec * investment)
    for s in range(sims):
        # start with base allocation
        asset_vals = base_alloc_amounts.copy()
        for y in range(years):
            # draw correlated normal shocks for this year
            z = np.random.normal(size=n_assets)
            ret = means_vec + L @ z  # approx returns using normal as mean + vol*epsilon
            # apply growth on asset_vals
            asset_vals = asset_vals * (1 + ret)
            # add annual SIP apportioned to assets at year end
            if annual_sip > 0:
                asset_vals += annual_sip * weights_vec
        final_vals[s] = asset_vals.sum() + (0 if annual_sip==0 else 0)  # already included
    return final_vals

# run Monte Carlo
weights_vec = weights
means_vec = means
cov_mat = cov
sims = simulations
final_values = simulate_portfolio(investment_amount, monthly_sip, weights_vec, means_vec, cov_mat, horizon_years, sims)

# probability of reaching goal
prob_success = float((final_values >= goal_amount).sum() / sims * 100.0)
median_end = float(np.median(final_values))
p10 = float(np.percentile(final_values, 10))
p90 = float(np.percentile(final_values, 90))
expected_end = float(np.mean(final_values))

# ---------- SIP shortfall finder ----------
def required_monthly_sip_to_hit_goal(investment, current_monthly_sip, weights_vec, means_vec, cov_mat, years, target, max_search=200000):
    # find minimal monthly SIP (bisection) such that simulated mean of final >= target
    # We'll use deterministic expected growth (not full MC) for speed: use expected returns, not stoch
    # FV of lump sum per asset: pv*(1+r)^n
    # FV of monthly SIP per asset: monthly*( (1+r_month)^{n*12}-1)/r_month*(1+r_month)
    def fv_deterministic(investment, monthly_sip):
        total = 0.0
        for i, w in enumerate(weights_vec):
            r = means_vec[i]
            pv = investment * w
            fv_pv = pv * ((1 + r) ** years)
            if monthly_sip > 0:
                r_m = (1 + r) ** (1/12) - 1
                n = years*12
                fv_sip = monthly_sip * w * (((1 + r_m) ** n - 1) / r_m) * (1 + r_m)
            else:
                fv_sip = 0.0
            total += fv_pv + fv_sip
        return total
    # if deterministic FV with current SIP already above target, return current SIP
    if fv_deterministic(investment, current_monthly_sip) >= target:
        return current_monthly_sip
    lo, hi = 0.0, max_search
    for _ in range(40):
        mid = (lo + hi)/2
        val = fv_deterministic(investment, mid)
        if val >= target:
            hi = mid
        else:
            lo = mid
    return round(hi, 0)

required_sip = required_monthly_sip_to_hit_goal(investment_amount, monthly_sip, weights_vec, means_vec, cov_mat, horizon_years, goal_amount) if goal_amount>0 else 0

# ---------- Inflation-adjusted (real) projection ----------
def nominal_to_real(value, inflation_rate, years):
    return value / ((1 + inflation_rate) ** years)

median_real = nominal_to_real(median_end, inflation/100.0, horizon_years)
expected_real = nominal_to_real(expected_end, inflation/100.0, horizon_years)
goal_real = nominal_to_real(goal_amount, inflation/100.0, horizon_years)

# ---------- Rebalancing guidance ----------
if port_vol > 0.22:
    rebalance_suggest = "Quarterly / Tactical (High volatility — monitor closely)"
elif port_vol > 0.12:
    rebalance_suggest = "Bi-annual or Annual (Moderate volatility)"
else:
    rebalance_suggest = "Annual (Low volatility — keep long term)"

# ---------- Tabs / UI ----------
tab = st.tabs(["Overview", "Visualization", "Insights", "Simulator"])
# ---------- Overview ----------
with tab[0]:
    st.header("Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Exp. annual return (wtd)", f"{port_exp_return*100:.2f}%")
    c2.metric("Est. portfolio volatility (σ)", f"{port_vol*100:.2f}%")
    c3.metric("Probability of reaching goal", f"{prob_success:.1f}%")

    st.markdown("**Allocation table**")
    st.table(alloc_df.style.format({"Allocation (%)":"{:.2f}", "Allocation (₹)":"₹{:,.0f}", "Exp Return (%)":"{:.2f}%", "Volatility (%)":"{:.2f}%"}))

    st.markdown("**Projection summary (Monte Carlo)**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Median end value ({horizon_years}y)", f"₹{median_end:,.0f}")
    col2.metric("10th percentile", f"₹{p10:,.0f}")
    col3.metric("90th percentile", f"₹{p90:,.0f}")
    col4.metric("Expected end value", f"₹{expected_end:,.0f}")

    st.markdown(f"*Inflation-adjusted (real) median value:* ₹{median_real:,.0f} (assuming {inflation:.2f}% pa inflation)")
    st.markdown(f"*Goal (real terms):* ₹{goal_real:,.0f}")

    st.markdown("---")
    st.download_button("📥 Download allocation & projection CSV", data=alloc_df.to_csv(index=False).encode("utf-8"), file_name="allocation_projection.csv", mime="text/csv")

# ---------- Visualization ----------
with tab[1]:
    st.header("Visualization")
    # donut chart (matplotlib)
    fig1, ax1 = plt.subplots(figsize=(5,5), facecolor="#071425")
    wedges, texts = ax1.pie(alloc_df["Allocation (%)"], labels=alloc_df["Asset Class"], startangle=90, wedgeprops=dict(width=0.4), textprops={'color':'w','fontsize':8})
    ax1.set_title("Allocation (donut)", color="white")
    st.pyplot(fig1, clear_figure=True)

    st.markdown("Allocation bar chart")
    st.bar_chart(alloc_df.set_index("Asset Class")["Allocation (%)"])

    st.markdown("Risk–Return scatter (assets)")
    fig2, ax2 = plt.subplots(figsize=(7,4), facecolor="#071425")
    for i, r in alloc_df.iterrows():
        ax2.scatter(r["Volatility (%)"], r["Exp Return (%)"], s=max(40, r["Allocation (%)"]*6), alpha=0.9)
        ax2.text(r["Volatility (%)"]+0.01, r["Exp Return (%)"]+0.1, r["Asset Class"], color="white", fontsize=8)
    ax2.set_xlabel("Volatility (%)", color="white")
    ax2.set_ylabel("Expected Return (%)", color="white")
    ax2.set_facecolor("#06202b")
    ax2.tick_params(colors="white")
    st.pyplot(fig2, clear_figure=True)

# ---------- Insights ----------
with tab[2]:
    st.header("Insights & Rebalancing Guidance")
    st.markdown("### Why this allocation?")
    st.markdown("- The baseline allocation is derived from practical, age-aware rules of thumb tailored for Indian investors.")
    st.markdown("- We normalize to selected assets (if you chose fewer assets, weights scale to 100%).")
    st.markdown("- Expected returns & volatilities are educated estimates (use as starting point; tune for chosen instruments).")

    st.markdown("### Rebalancing suggestion")
    st.info(rebalance_suggest)
    st.markdown("### Actionable tips")
    st.markdown("- Rebalance annually for moderate portfolios; quarterly if high volatility or large inflows/outflows.")
    st.markdown("- Use low-cost index ETFs as the core (Index Funds / ETFs).")
    st.markdown("- Keep emergency allocation in Liquid Funds / Cash to avoid forced selling.")

    st.markdown("### Asset-specific notes")
    ASSET_NOTES = {
        "Large Cap Equity": "Core growth with lower volatility vs smallcaps.",
        "Mid/Small Cap Equity": "Higher long-term upside, needs longer horizon.",
        "International Equity (ETFs)": "Currency and sector diversification.",
        "Index Funds / ETFs": "Low-cost passive exposure.",
        "Active Equity Mutual Funds": "Potential alpha; pick experienced managers.",
        "Sectoral / Thematic Funds": "High conviction; use small allocation.",
        "Debt Mutual Funds": "Yield higher than cash; duration risk applies.",
        "Government Bonds (G-Secs)": "Low credit risk — capital preservation.",
        "Corporate Bonds": "Better yield than G-Secs — check credit rating.",
        "PPF": "Tax-efficient, long lock-in, government-backed.",
        "NPS": "Retirement-focused; tax benefits and equity/debt mix.",
        "EPF": "Employee retirement savings; consistent returns.",
        "Gold (ETF/Physical)": "Inflation hedge; crisis protection.",
        "Silver / Commodities": "Cyclical; use for commodity exposure.",
        "REITs/InvITs": "Income-generating real estate exposure; liquid.",
        "Real Estate (Direct)": "Illiquid; long-term asset for rental + appreciation.",
        "Liquid Funds / Cash": "Emergency buffer and liquidity management.",
        "Fixed Deposits": "Guaranteed, but less inflation-protective.",
        "Commodities (non-precious)": "Commodities cycle exposure.",
        "Crypto (small allocation)": "Highly speculative; keep tiny allocation only.",
        "Insurance-linked (ULIPs/Annuities)": "Consider for guaranteed income or specific tax use-cases."
    }
    for asset in alloc_df["Asset Class"]:
        st.markdown(f"**{asset}** — {ASSET_NOTES.get(asset, '—')}")

# ---------- Simulator ----------
with tab[3]:
    st.header("Simulator & SIP Planner")
    st.markdown("Use the simulator to understand SIP impact and shortfall to reach the goal.")

    st.subheader("Monte Carlo distribution (final portfolio values)")
    fig3, ax3 = plt.subplots(figsize=(8,3), facecolor="#071425")
    ax3.hist(final_values, bins=50, color="#5EE7A6", alpha=0.8)
    ax3.axvline(goal_amount, color="yellow", linestyle="--", label="Goal")
    ax3.set_xlabel("Portfolio value (₹)", color="white")
    ax3.set_ylabel("Frequency", color="white")
    ax3.tick_params(colors="white")
    st.pyplot(fig3, clear_figure=True)

    st.markdown(f"**Probability of meeting goal in {horizon_years} years:** **{prob_success:.1f}%**")
    st.markdown(f"**Required monthly SIP (deterministic approx) to hit goal:** ₹{required_sip:,.0f} per month (distributed across chosen assets)")

    st.markdown("### SIP suggestion per asset")
    if required_sip > 0:
        asset_sip = {a: required_sip * (w/100.0) for a, w in allocation.items()}
        sip_df = pd.DataFrame({"Asset Class": list(asset_sip.keys()), "Monthly SIP (₹)": list(asset_sip.values())})
        st.table(sip_df.style.format({"Monthly SIP (₹)": "₹{:,.0f}"}))
    else:
        st.success("Your current plan is expected to meet the goal based on deterministic projection.")

    st.markdown("---")
    st.markdown("### Quick sensitivity checks")
    col_a, col_b = st.columns(2)
    shock_eq = col_a.slider("Equity shock (reduce equity returns by %)", 0, 50, 20)
    shock_infl = col_b.slider("Higher inflation test (%)", 0.0, 10.0, 6.0, 0.1)

    # quick scenario: reduce equity returns then deterministic FV
    means_scenario = means_vec.copy()
    for i, a in enumerate(alloc_df["Asset Class"]):
        if "Equity" in a or "Thematic" in a or "Crypto" in a:
            means_scenario[i] = max(-0.5, means_scenario[i] * (1 - shock_eq / 100.0))
    # deterministic final
    def deterministic_final(invest, monthly_sip, weights_v, means_v, years):
        total = 0.0
        r_months = [(1+m)**(1/12)-1 for m in means_v]
        for i, w in enumerate(weights_v):
            pv = invest * w
            fv_pv = pv * ((1 + means_v[i]) ** years)
            if monthly_sip > 0:
                rm = r_months[i]
                n = years*12
                fv_sip = monthly_sip * w * (((1+rm)**n - 1)/rm) * (1+rm) if rm!=0 else monthly_sip*w*n
            else:
                fv_sip = 0.0
            total += fv_pv + fv_sip
        return total

    scenario_value = deterministic_final(investment_amount, monthly_sip, weights_vec, means_scenario, horizon_years)
    scenario_value_real = nominal_to_real(scenario_value, shock_infl/100.0, horizon_years)
    st.markdown(f"Deterministic scenario final (after equity shock): **₹{scenario_value:,.0f}** · real (inflation {shock_infl}%): **₹{scenario_value_real:,.0f}**")

st.markdown("---")
st.caption("Developed for professional planning; projections are model-based estimates and not investment advice.")
