import streamlit as st
import pandas as pd
from analysis import (
    fetch_data,
    fit_garch,
    fit_egarch,
    get_conditional_volatility,
    forecast_volatility,
    model_comparison_table,
)

st.set_page_config(page_title="GARCH/EGARCH Volatility Explorer", layout="wide")
st.title("GARCH/EGARCH Volatility Explorer")
st.caption("Compare symmetric vs. asymmetric volatility models on any ticker.")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", "SPY")
    start_date = st.date_input("Start date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End date", pd.to_datetime("2026-01-01"))
    horizon = st.slider("Forecast horizon (days)", 5, 30, 10)
    run_button = st.button("Run Models", type="primary")

# --- Main logic ---
if run_button:
    with st.spinner(f"Fetching {ticker} data and fitting models..."):
        try:
            returns = fetch_data(ticker, str(start_date), str(end_date))

            if len(returns) < 50:
                st.error("Not enough data returned. Try a wider date range or check the ticker symbol.")
                st.stop()

            garch_res = fit_garch(returns)
            egarch_res = fit_egarch(returns)

        except Exception as e:
            st.error(f"Something went wrong fetching or fitting: {e}")
            st.stop()

    st.success(f"Fitted GARCH(1,1) and EGARCH(1,1) on {len(returns)} daily returns for {ticker}.")

    # --- Model comparison table ---
    st.subheader("Model Comparison")
    comparison = model_comparison_table(garch_res, egarch_res)
    st.dataframe(comparison.style.format("{:.2f}"), use_container_width=True)

    lower_aic = "EGARCH" if egarch_res.aic < garch_res.aic else "GARCH"
    st.caption(f"Lower AIC/BIC indicates better fit — **{lower_aic}** fits better here.")

    # --- Conditional volatility chart ---
    st.subheader("In-Sample Conditional Volatility (Annualized)")
    vol_df = pd.DataFrame({
        "GARCH": get_conditional_volatility(garch_res),
        "EGARCH": get_conditional_volatility(egarch_res),
    })
    st.line_chart(vol_df)

    # --- Leverage effect callout ---
    st.subheader("Leverage Effect (EGARCH gamma term)")
    gamma = egarch_res.params.get("gamma[1]", None)
    gamma_pval = egarch_res.pvalues.get("gamma[1]", None)
    if gamma is not None:
        col1, col2 = st.columns(2)
        col1.metric("Gamma coefficient", f"{gamma:.4f}")
        col2.metric("P-value", f"{gamma_pval:.2e}")
        if gamma < 0 and gamma_pval < 0.05:
            st.info(
                "Gamma is negative and statistically significant, confirming a leverage effect: "
                "negative return shocks increase future volatility more than positive shocks of "
                "equal size."
            )
        else:
            st.info("No statistically significant leverage effect detected in this sample.")

    # --- Forecast chart ---
    st.subheader(f"{horizon}-Day Volatility Forecast (Annualized)")
    garch_fc = forecast_volatility(garch_res, horizon=horizon, method='analytic')
    egarch_fc = forecast_volatility(egarch_res, horizon=horizon, method='simulation')
    forecast_df = pd.DataFrame({"GARCH": garch_fc, "EGARCH": egarch_fc})
    forecast_df.index.name = "Days Ahead"
    st.line_chart(forecast_df)
    st.caption("EGARCH forecasts use simulation (seeded for reproducibility); GARCH uses the analytic solution.")

else:
    st.info("Set your parameters in the sidebar and click **Run Models** to begin.")