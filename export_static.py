import matplotlib.pyplot as plt
import json
import os
from analysis import (
    fetch_data,
    fit_garch,
    fit_egarch,
    get_conditional_volatility,
    forecast_volatility,
    model_comparison_table,
)

# --- Config: set your "default" ticker/window for the portfolio showcase ---
TICKER = "SPY"
START = "2020-01-01"
END = "2026-01-01"
HORIZON = 10

CHART_DIR = "outputs/charts"
JSON_DIR = "outputs/json"

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)


def save_conditional_volatility_chart(garch_res, egarch_res):
    garch_vol = get_conditional_volatility(garch_res)
    egarch_vol = get_conditional_volatility(egarch_res)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(garch_vol.index, garch_vol.values, label="GARCH(1,1)", linewidth=1.2)
    ax.plot(egarch_vol.index, egarch_vol.values, label="EGARCH(1,1)", linewidth=1.2)
    ax.set_title(f"{TICKER} Conditional Volatility (Annualized)")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = f"{CHART_DIR}/conditional_volatility.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def save_forecast_chart(garch_res, egarch_res):
    garch_fc = forecast_volatility(garch_res, horizon=HORIZON, method='analytic')
    egarch_fc = forecast_volatility(egarch_res, horizon=HORIZON, method='simulation')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(garch_fc.index, garch_fc.values, marker='o', label="GARCH(1,1)")
    ax.plot(egarch_fc.index, egarch_fc.values, marker='o', label="EGARCH(1,1)")
    ax.set_title(f"{TICKER}: {HORIZON}-Day Volatility Forecast")
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = f"{CHART_DIR}/volatility_forecast.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def save_json_summary(garch_res, egarch_res):
    comparison = model_comparison_table(garch_res, egarch_res)
    gamma = float(egarch_res.params.get("gamma[1]"))
    gamma_pval = float(egarch_res.pvalues.get("gamma[1]"))

    summary = {
        "ticker": TICKER,
        "window": {"start": START, "end": END},
        "model_comparison": {
            "GARCH": {
                "AIC": float(comparison.loc["AIC", "GARCH(1,1)"]),
                "BIC": float(comparison.loc["BIC", "GARCH(1,1)"]),
                "LogLikelihood": float(comparison.loc["Log-Likelihood", "GARCH(1,1)"]),
            },
            "EGARCH": {
                "AIC": float(comparison.loc["AIC", "EGARCH(1,1)"]),
                "BIC": float(comparison.loc["BIC", "EGARCH(1,1)"]),
                "LogLikelihood": float(comparison.loc["Log-Likelihood", "EGARCH(1,1)"]),
            },
        },
        "leverage_effect": {
            "gamma": gamma,
            "p_value": gamma_pval,
            "significant": bool(gamma < 0 and gamma_pval < 0.05),
        },
        "forecast_horizon_days": HORIZON,
    }

    path = f"{JSON_DIR}/summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print(f"Fetching {TICKER} data and fitting models...")
    returns = fetch_data(TICKER, START, END)
    garch_res = fit_garch(returns)
    egarch_res = fit_egarch(returns)

    print("Generating exports...")
    save_conditional_volatility_chart(garch_res, egarch_res)
    save_forecast_chart(garch_res, egarch_res)
    save_json_summary(garch_res, egarch_res)

    print("\nDone. Check outputs/charts/ and outputs/json/")