# -*- coding: utf-8 -*-
"""
Smart ETF Portfolio Optimizer - Streamlit Cloud Version
No external API required - runs standalone
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
from datetime import datetime

# Import backend modules directly
from backend.forecaster import get_portfolio_data
from backend.optimizer import optimize_portfolio

st.set_page_config(
    page_title="Smart ETF Portfolio Optimizer",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Available ETFs with metadata
ETF_DATA = {
    "PSI": {"name": "Semiconductors", "region": "North America"},
    "IYW": {"name": "US Technology", "region": "North America"},
    "RING": {"name": "Gold Miners", "region": "Developed Markets"},
    "PICK": {"name": "Metals & Mining", "region": "Developed Markets"},
    "NLR": {"name": "Nuclear Energy", "region": "Developed Markets"},
    "UTES": {"name": "Utilities", "region": "North America"},
    "LIT": {"name": "Lithium & Battery", "region": "Developed Markets"},
    "NANR": {"name": "Natural Resources", "region": "North America"},
    "GUNR": {"name": "Global Resources", "region": "Developed Markets"},
    "XCEM": {"name": "Emerging Markets", "region": "Emerging Markets"},
    "PTLC": {"name": "Large Cap", "region": "North America"},
    "FXU": {"name": "Utilities Alpha", "region": "North America"},
}

DEFAULT_ETFS = ["IYW", "PSI", "NLR", "UTES", "PTLC", "FXU"]

RISK_STRATEGIES = {
    "Low Risk": "min_volatility",
    "Balanced": "risk_parity",
    "High Return": "max_sharpe"
}


def get_normalized_prices(tickers, period="6mo"):
    """Get normalized prices (starting at 100) for comparison."""
    try:
        data = yf.download(tickers, period=period, progress=False)
        if data.empty:
            return {}
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data[['Close']]
            close.columns = tickers
        normalized = (close / close.iloc[0]) * 100
        result = {
            "dates": [d.strftime('%Y-%m-%d') for d in normalized.index],
            "prices": {}
        }
        for ticker in tickers:
            if ticker in normalized.columns:
                prices = normalized[ticker].ffill().tolist()
                result["prices"][ticker] = [round(p, 2) if not pd.isna(p) else None for p in prices]
        return result
    except:
        return {}


def calculate_ytd_returns(tickers):
    """Calculate YTD returns for each ticker."""
    try:
        start_of_year = datetime(datetime.now().year, 1, 1).strftime('%Y-%m-%d')
        data = yf.download(tickers, start=start_of_year, progress=False)
        if data.empty:
            return {}
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data[['Close']]
            close.columns = tickers
        ytd_returns = {}
        for ticker in tickers:
            if ticker in close.columns:
                first_price = close[ticker].iloc[0]
                last_price = close[ticker].iloc[-1]
                if first_price > 0:
                    ytd_returns[ticker] = round(((last_price - first_price) / first_price) * 100, 2)
        return ytd_returns
    except:
        return {}


# Title
st.title(":material/query_stats: Smart ETF Portfolio Optimizer")
st.write("Determine optimal ETF allocation using **ML-forecasted returns** and **Modern Portfolio Theory**.")
st.write("")

cols = st.columns([1, 3])

# LEFT PANEL
top_left = cols[0].container(border=True)

with top_left:
    tickers = st.multiselect(
        "Select ETFs",
        options=sorted(ETF_DATA.keys()),
        default=DEFAULT_ETFS,
        format_func=lambda x: f"{x} - {ETF_DATA[x]['name']}",
        placeholder="Choose ETFs to include",
    )
    st.write("")
    risk_level = st.pills(
        "Risk Profile",
        options=list(RISK_STRATEGIES.keys()),
        default="Balanced",
        help="Low = Min Volatility | Balanced = Risk Parity | High = Max Sharpe"
    )
    st.write("")
    investment_amount = st.number_input(
        "Investment Amount ($)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000,
        format="%d"
    )
    st.write("")
    optimize_clicked = st.button(
        "Optimize Portfolio",
        type="primary",
        use_container_width=True
    )

if len(tickers) < 2:
    cols[1].container(border=True).info("Pick at least 2 ETFs to compare", icon=":material/info:")
    st.stop()

if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None

if optimize_clicked:
    with st.spinner("Fetching data, forecasting returns, and optimizing portfolio..."):
        try:
            strategy = RISK_STRATEGIES[risk_level]
            risk_free_rate = 0.05

            # Get portfolio data directly from backend
            portfolio_data = get_portfolio_data(tickers=tickers, model_path="models")
            mu = portfolio_data["expected_returns"]
            cov = portfolio_data["covariance_matrix"]
            hist_vol = portfolio_data["historical_volatility"]

            ytd_returns = calculate_ytd_returns(tickers)

            weights, metrics = optimize_portfolio(
                mu=mu, cov=cov, risk_free=risk_free_rate, strategy=strategy
            )

            allocations = []
            geo_allocation = {}
            portfolio_ytd = 0.0

            for i, ticker in enumerate(tickers):
                weight = float(weights[i])
                weight_percent = round(weight * 100, 2)
                amount = round(weight * investment_amount, 2)
                pred_return = float(mu[i])
                pred_return_capped = min(pred_return, 1.0)
                ticker_vol = hist_vol.get(ticker, 0.0)
                ytd_ret = ytd_returns.get(ticker, 0.0)
                portfolio_ytd += weight * ytd_ret
                metadata = ETF_DATA.get(ticker, {"name": ticker, "region": "Other"})
                region = metadata["region"]
                geo_allocation[region] = geo_allocation.get(region, 0) + weight_percent

                if weight_percent >= 15:
                    recommendation = "Strong Buy"
                elif weight_percent >= 10:
                    recommendation = "Buy"
                elif weight_percent >= 5:
                    recommendation = "Hold"
                elif weight_percent > 0:
                    recommendation = "Light"
                else:
                    recommendation = "Avoid"

                allocations.append({
                    "ticker": ticker, "name": metadata["name"], "region": region,
                    "weight_percent": weight_percent, "amount": amount,
                    "predicted_return": round(pred_return, 4),
                    "predicted_return_capped": round(pred_return_capped * 100, 1),
                    "historical_volatility": round(ticker_vol, 4),
                    "ytd_return": round(ytd_ret, 2), "recommendation": recommendation
                })

            allocations.sort(key=lambda x: x["weight_percent"], reverse=True)
            top_picks = [a["ticker"] for a in allocations[:3] if a["weight_percent"] > 0]
            weights_array = np.array([a["weight_percent"] / 100 for a in allocations])
            diversification_score = round(1 - np.sum(weights_array ** 2), 2)
            expected_return_capped = min(metrics["expected_return"], 1.0)
            normalized_prices = get_normalized_prices(tickers, period="6mo")

            st.session_state.optimization_result = {
                "success": True, "strategy_used": strategy, "allocations": allocations,
                "portfolio_metrics": {
                    "expected_annual_return": round(metrics["expected_return"], 4),
                    "expected_annual_return_percent": round(metrics["expected_return"] * 100, 2),
                    "expected_annual_return_capped": round(expected_return_capped * 100, 1),
                    "annual_volatility": round(metrics["volatility"], 4),
                    "annual_volatility_percent": round(metrics["volatility"] * 100, 2),
                    "sharpe_ratio": round(metrics["sharpe"], 2),
                    "diversification_score": diversification_score,
                    "portfolio_ytd_return": round(portfolio_ytd, 2)
                },
                "geographic_allocation": [
                    {"region": r, "allocation_percent": round(p, 1)}
                    for r, p in sorted(geo_allocation.items(), key=lambda x: -x[1]) if p > 0
                ],
                "investment_amount": investment_amount, "top_picks": top_picks,
                "normalized_prices": normalized_prices
            }
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

if st.session_state.optimization_result:
    result = st.session_state.optimization_result
    metrics = result["portfolio_metrics"]
    allocations = result["allocations"]

    top_right = cols[1].container(border=True)
    with top_right:
        st.subheader("Portfolio Allocation")
        st.write("Optimal allocation based on ML-forecasted returns and your risk profile.")

        top_row = st.columns(2)
        with top_row[0].container(border=True):
            st.markdown("**Portfolio Allocation**")
            pie_data = pd.DataFrame([
                {"ETF": a["ticker"], "Weight": a["weight_percent"]}
                for a in allocations if a["weight_percent"] > 0
            ])
            if not pie_data.empty:
                pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=40).encode(
                    theta=alt.Theta(field="Weight", type="quantitative"),
                    color=alt.Color(field="ETF", type="nominal", legend=alt.Legend(orient="bottom")),
                    tooltip=[alt.Tooltip("ETF:N"), alt.Tooltip("Weight:Q", format=".1f", title="Weight (%)")]
                ).properties(height=250)
                st.altair_chart(pie_chart, use_container_width=True)

        with top_row[1].container(border=True):
            st.markdown("**Key Metrics**")
            st.metric("Expected Return", f"{metrics['expected_annual_return_capped']:.1f}%",
                      delta=f"Sharpe: {metrics['sharpe_ratio']:.2f}", delta_color="off")
            st.metric("Volatility", f"{metrics['annual_volatility_percent']:.1f}%",
                      delta=f"YTD: {metrics['portfolio_ytd_return']:+.1f}%", delta_color="off")

        second_row = st.columns(2)
        with second_row[0].container(border=True):
            st.markdown("**Investment Summary**")
            expected_value = investment_amount * (1 + min(metrics["expected_annual_return"], 1.0))
            expected_gain = expected_value - investment_amount
            st.metric("Initial", f"${investment_amount:,.0f}")
            st.metric("Expected (1Y)", f"${expected_value:,.0f}")
            st.metric("Gain", f"${expected_gain:,.0f}", delta=f"{metrics['expected_annual_return_capped']:.1f}%")

        with second_row[1].container(border=True):
            active_allocs = [a for a in allocations if a['weight_percent'] > 0]
            if active_allocs:
                best_etf = max(active_allocs, key=lambda x: x['predicted_return_capped'])
                worst_etf = min(active_allocs, key=lambda x: x['predicted_return_capped'])
                mc = st.columns(2)
                mc[0].metric("Top Pick", best_etf['ticker'], delta=f"+{best_etf['predicted_return_capped']:.0f}%")
                mc[1].metric("Lowest", worst_etf['ticker'], delta=f"+{worst_etf['predicted_return_capped']:.0f}%")

    section_2 = cols[1].container(border=True)
    with section_2:
        st.subheader("Portfolio Analysis")
        if result.get("normalized_prices") and result["normalized_prices"].get("dates"):
            norm_data = result["normalized_prices"]
            chart_df = pd.DataFrame({"Date": pd.to_datetime(norm_data["dates"])})
            for ticker, prices in norm_data["prices"].items():
                chart_df[ticker] = prices
            melted = chart_df.melt(id_vars=["Date"], var_name="ETF", value_name="Normalized price")
            price_chart = alt.Chart(melted).mark_line().encode(
                alt.X("Date:T"), alt.Y("Normalized price:Q", scale=alt.Scale(zero=False)),
                alt.Color("ETF:N"), tooltip=["Date:T", "ETF:N", alt.Tooltip("Normalized price:Q", format=".1f")]
            ).properties(height=350)
            st.altair_chart(price_chart, use_container_width=True)

        bottom_row = st.columns(2)
        with bottom_row[0].container(border=True):
            if result.get("geographic_allocation"):
                geo_data = pd.DataFrame(result["geographic_allocation"])
                geo_chart = alt.Chart(geo_data).mark_bar().encode(
                    x=alt.X("allocation_percent:Q", title="Allocation (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("region:N", sort="-x", title=""), color=alt.Color("region:N", legend=None),
                    tooltip=["region:N", alt.Tooltip("allocation_percent:Q", format=".1f")]
                ).properties(height=150, title="Geographic Exposure")
                st.altair_chart(geo_chart, use_container_width=True)

        with bottom_row[1].container(border=True):
            forecast_data = pd.DataFrame(
                [{"ETF": a["ticker"], "Metric": "YTD", "Value": a["ytd_return"]} for a in allocations if a["weight_percent"] > 0] +
                [{"ETF": a["ticker"], "Metric": "Forecast", "Value": a["predicted_return_capped"]} for a in allocations if a["weight_percent"] > 0]
            )
            if not forecast_data.empty:
                fc = alt.Chart(forecast_data).mark_bar().encode(
                    x=alt.X("ETF:N", title=""), y=alt.Y("Value:Q", title="Return (%)"),
                    color=alt.Color("Metric:N"), xOffset="Metric:N",
                    tooltip=["ETF:N", "Metric:N", alt.Tooltip("Value:Q", format=".1f")]
                ).properties(height=150, title="YTD vs Forecast")
                st.altair_chart(fc, use_container_width=True)

    section_3 = cols[1].container(border=True)
    with section_3:
        st.subheader("ETF Details")
        table_data = [{
            "ETF": a["ticker"], "Sector": a["name"], "Region": a["region"],
            "Weight (%)": f"{a['weight_percent']:.1f}",
            "Amount ($)": f"{a['amount']:,.0f}" if a['amount'] else "-",
            "YTD (%)": f"{a['ytd_return']:+.1f}",
            "Forecast (%)": f"{a['predicted_return_capped']:.1f}",
            "Signal": a["recommendation"]
        } for a in allocations]
        st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
        if result.get("top_picks"):
            st.success(f"**Top Recommendations:** {', '.join(result['top_picks'])}")

    bottom_left = cols[0].container(border=True)
    with bottom_left:
        st.markdown(f"**Strategy:** {result['strategy_used'].replace('_', ' ').title()}")
        st.markdown(f"**Diversification:** {metrics['diversification_score']:.0%}")

st.write("")
st.divider()
st.caption("Smart ETF Portfolio Optimizer | Powered by ML & Modern Portfolio Theory")
