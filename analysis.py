import yfinance as yf
from arch import arch_model
import numpy as np
import pandas as pd


def fetch_data(ticker: str, start: str, end: str) -> pd.Series:
    """Pull adjusted close prices and return daily log returns (in %)."""
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        raise ValueError(
            f"No data returned for '{ticker}' between {start} and {end}. "
            "This is usually a temporary Yahoo Finance API issue — try again in a minute, "
            "or double-check the ticker symbol."
        )
    returns = 100 * data['Close'].pct_change().dropna()
    return returns


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1):
    model = arch_model(returns, vol='Garch', p=p, q=q, dist='normal')
    return model.fit(disp='off')


def fit_egarch(returns: pd.Series, p: int = 1, o: int = 1, q: int = 1):
    """o=1 enables the leverage (asymmetry) term — this is what makes it EGARCH
    rather than just a differently-parameterized symmetric GARCH."""
    model = arch_model(returns, vol='EGarch', p=p, o=o, q=q, dist='normal')
    return model.fit(disp='off')


def get_conditional_volatility(fitted_result) -> pd.Series:
    """Extract the in-sample fitted volatility series (annualized)."""
    return fitted_result.conditional_volatility * (252 ** 0.5)


def forecast_volatility(fitted_result, horizon: int = 10, method: str = 'analytic', seed: int = 42) -> pd.Series:
    """Forecast future volatility, annualized, for `horizon` days ahead.
    EGARCH requires simulation-based forecasting for horizon > 1.
    `seed` ensures reproducible output when method='simulation'."""
    rng = np.random.default_rng(seed).standard_normal if method == 'simulation' else None
    forecast = fitted_result.forecast(horizon=horizon, reindex=False, method=method, rng=rng)
    variance_forecast = forecast.variance.values[-1]
    vol_forecast = (variance_forecast ** 0.5) * (252 ** 0.5)
    return pd.Series(vol_forecast, index=range(1, horizon + 1))


def model_comparison_table(garch_result, egarch_result) -> pd.DataFrame:
    """Side-by-side AIC/BIC/log-likelihood comparison."""
    return pd.DataFrame({
        "GARCH(1,1)": [garch_result.aic, garch_result.bic, garch_result.loglikelihood],
        "EGARCH(1,1)": [egarch_result.aic, egarch_result.bic, egarch_result.loglikelihood],
    }, index=["AIC", "BIC", "Log-Likelihood"])


if __name__ == "__main__":
    returns = fetch_data("SPY", "2020-01-01", "2026-01-01")

    garch_res = fit_garch(returns)
    egarch_res = fit_egarch(returns)

    print("\n--- GARCH(1,1) Summary ---")
    print(garch_res.summary())

    print("\n--- EGARCH(1,1) Summary ---")
    print(egarch_res.summary())

    print("\n--- Model Comparison (lower AIC/BIC = better fit) ---")
    print(model_comparison_table(garch_res, egarch_res))

    print("\n--- 10-Day Volatility Forecast (annualized %) ---")
    print("GARCH:\n", forecast_volatility(garch_res))
    print("EGARCH:\n", forecast_volatility(egarch_res, method='simulation'))