import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.strategies import (
    TSE_EQUITY_BUY_FEE,
    TSE_EXERCISE_FEE,
    TSE_OPTION_FEE,
    analyze_options_chain,
    scan_covered_calls,
    scan_put_call_parity,
)
from src.tse_options_loader import fetch_options_chain

st.set_page_config(
    page_title="TSE Options Quant | Pricing, IV Surface & Arbitrage",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ TSE-Options-Quant: Derivatives Pricing, Volatility Surface & Arbitrage Engine")
st.markdown(
    "Real-time Black-Scholes pricing, Greeks calculation, **3D Implied Volatility Surface**, "
    "and fee-adjusted **Covered Call / Put-Call Parity** scanner for the Tehran Stock Exchange."
)

# --- سایدبار تنظیمات ---
st.sidebar.header("⚙️ Market & Model Settings")
underlying = st.sidebar.selectbox("Underlying Asset", options=["اهرم", "فولاد", "خودرو", "فملی"], index=0)
rf_annual = st.sidebar.slider("Risk-Free Rate (%)", min_value=15.0, max_value=45.0, value=30.0, step=1.0) / 100.0
only_liquid = st.sidebar.checkbox("Filter Illiquid / 1-Rial Contracts", value=True)
refresh_btn = st.sidebar.button("🔄 Refresh Live Data from TSETMC")

try:
    spot, raw_chain = fetch_options_chain(underlying, force_refresh=refresh_btn)
except Exception as exc:
    st.error(f"Could not load data for '{underlying}': {exc}")
    st.stop()

enriched_df = analyze_options_chain(raw_chain, spot=spot, rf_annual=rf_annual, only_liquid=only_liquid)

if enriched_df.empty:
    st.warning("No liquid contracts matched the filter criteria. Try unchecking the liquidity filter.")
    st.stop()

cc_df = scan_covered_calls(enriched_df, spot=spot, rf_annual=rf_annual)
arb_df = scan_put_call_parity(enriched_df, spot=spot, rf_annual=rf_annual)

# --- کارت‌های کلیدی ---
median_iv = enriched_df["iv"].dropna().median() * 100.0
top_cc_yield = cc_df["annualized_yield_pct"].iloc[0] if not cc_df.empty else 0.0
top_cc_sym = cc_df["symbol"].iloc[0] if not cc_df.empty else "-"

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Underlying Spot ({underlying})", f"{spot:,.0f} Rials")
c2.metric("Active Liquid Contracts", f"{len(enriched_df)} / {len(raw_chain)}")
c3.metric("Median Implied Volatility", f"{median_iv:.1f}%")
c4.metric("Top Covered Call Annual Yield", f"{top_cc_yield:.1f}%", f"Contract: {top_cc_sym}")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Option Chain & Greeks",
    "🌐 3D Volatility Surface",
    "🛡️ Covered Call & Payoff Simulator",
    "⚖️ Put-Call Parity Arbitrage",
])

with tab1:
    st.subheader("Enriched Option Chain (Black-Scholes Theoretical Price vs. Market)")
    display_cols = [
        "symbol", "option_type", "strike", "exp_jalali", "days_to_exp",
        "market_price", "bs_price", "mispricing_pct", "iv",
        "delta", "gamma", "vega", "theta_daily",
    ]
    st.dataframe(
        enriched_df[display_cols].style.format({
            "strike": "{:,.0f}",
            "market_price": "{:,.1f}",
            "bs_price": "{:,.1f}",
            "mispricing_pct": "{:+.1f}%",
            "iv": "{:.1%}",
            "delta": "{:.3f}",
            "gamma": "{:.6f}",
            "vega": "{:.2f}",
            "theta_daily": "{:.2f}",
        }),
        use_container_width=True,
    )

    iv_clean = enriched_df.dropna(subset=["iv"]).copy()
    iv_clean["IV (%)"] = iv_clean["iv"] * 100.0
    fig_smile = px.scatter(
        iv_clean,
        x="strike",
        y="IV (%)",
        color="exp_jalali",
        symbol="option_type",
        size="market_price",
        hover_data=["symbol", "days_to_exp", "delta"],
        title="Implied Volatility Smile Across Strikes & Expirations",
        template="plotly_white",
    )
    fig_smile.add_vline(x=spot, line_dash="dash", line_color="gray", annotation_text="Spot Price")
    st.plotly_chart(fig_smile, use_container_width=True)

with tab2:
    st.subheader("3D Implied Volatility Surface (Strike × Days to Expiration × IV)")
    surf_df = enriched_df.dropna(subset=["iv"]).copy()
    if len(surf_df) >= 4:
        fig_3d = go.Figure(
            data=[
                go.Mesh3d(
                    x=surf_df["strike"],
                    y=surf_df["days_to_exp"],
                    z=surf_df["iv"] * 100.0,
                    intensity=surf_df["iv"] * 100.0,
                    colorscale="Viridis",
                    opacity=0.85,
                    colorbar_title="IV (%)",
                ),
                go.Scatter3d(
                    x=surf_df["strike"],
                    y=surf_df["days_to_exp"],
                    z=surf_df["iv"] * 100.0,
                    mode="markers",
                    marker=dict(size=4, color="black"),
                    text=surf_df["symbol"],
                    name="Contracts",
                ),
            ]
        )
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="Strike Price (Rials)",
                yaxis_title="Days to Expiration",
                zaxis_title="Implied Volatility (%)",
            ),
            height=560,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("Not enough valid IV data points to construct a 3D surface.")

with tab3:
    st.subheader("Fee-Adjusted Covered Call Scanner")
    if not cc_df.empty:
        st.dataframe(
            cc_df.style.format({
                "strike": "{:,.0f}",
                "sell_premium": "{:,.0f}",
                "iv": "{:.1%}",
                "delta": "{:.2f}",
                "breakeven": "{:,.0f}",
                "downside_protection_pct": "{:.2f}%",
                "max_return_pct": "{:.2f}%",
                "annualized_yield_pct": "{:.1f}%",
                "excess_over_rf_pct": "{:+.1f}%",
            }),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("📐 Expiration Payoff Simulator (Covered Call vs. Naked Underlying)")
        selected_sym = st.selectbox("Select Call Contract for Payoff Diagram:", cc_df["symbol"].tolist())
        row = cc_df[cc_df["symbol"] == selected_sym].iloc[0]

        k_val = float(row["strike"])
        prem_val = float(row["sell_premium"])
        net_debit = spot * (1.0 + TSE_EQUITY_BUY_FEE) - prem_val * (1.0 - TSE_OPTION_FEE)

        s_grid = np.linspace(spot * 0.65, spot * 1.45, 200)
        naked_pnl = s_grid - spot * (1.0 + TSE_EQUITY_BUY_FEE)
        cc_pnl = np.where(
            s_grid <= k_val,
            s_grid - net_debit,
            k_val * (1.0 - TSE_EXERCISE_FEE) - net_debit,
        )

        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(x=s_grid, y=naked_pnl, name="Buy Underlying Only", line=dict(dash="dash", color="#636efa")))
        fig_payoff.add_trace(go.Scatter(x=s_grid, y=cc_pnl, name=f"Covered Call ({selected_sym})", line=dict(width=3, color="#00cc96")))
        fig_payoff.add_hline(y=0, line_color="black", line_width=1)
        fig_payoff.add_vline(x=row["breakeven"], line_dash="dot", line_color="red", annotation_text=f"Breakeven: {row['breakeven']:,.0f}")
        fig_payoff.update_layout(
            title=f"Expiration Profit / Loss per Share — {selected_sym} (Strike={k_val:,.0f}, Exp={row['exp_jalali']})",
            xaxis_title="Underlying Price at Expiration (Rials)",
            yaxis_title="Net Profit / Loss (Rials)",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(fig_payoff, use_container_width=True)

with tab4:
    st.subheader("Put-Call Parity Arbitrage Monitor (C - P vs. S - K·e⁻ʳᵀ)")
    if not arb_df.empty:
        st.dataframe(
            arb_df.style.format({
                "strike": "{:,.0f}",
                "call_mid": "{:,.1f}",
                "put_mid": "{:,.1f}",
                "parity_spread": "{:+,.1f}",
                "executable_edge": "{:+,.1f}",
            }),
            use_container_width=True,
        )
    else:
        st.info("No matched Call/Put pairs with identical strike and expiration found in liquid contracts.")