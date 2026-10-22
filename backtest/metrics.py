"""Performance metrics computed from daily returns."""

import numpy as np
import pandas as pd

from .engine import TRADING_DAYS


def equity_curve(returns: pd.Series) -> pd.Series:
    """Growth of 1 unit of capital."""
    return (1 + returns).cumprod()


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate, using calendar time between first and last day."""
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    return equity_curve(returns).iloc[-1] ** (1 / years) - 1


def annual_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate."""
    std = returns.std()
    return float("nan") if std == 0 else returns.mean() / std * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall of the equity curve, as a negative fraction."""
    equity = equity_curve(returns)
    return (equity / equity.cummax() - 1).min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = equity_curve(returns)
    return equity / equity.cummax() - 1


def summarize(result: pd.DataFrame) -> dict:
    """Headline numbers for one backtest (a run_backtest frame, possibly sliced by date)."""
    r = result["net_ret"]
    return {
        "CAGR": cagr(r),
        "Volatility": annual_volatility(r),
        "Sharpe": sharpe(r),
        "Max drawdown": max_drawdown(r),
        "Time invested": result["position"].mean(),
        "Trades": int((result["position"].diff().abs() > 0).sum()),
    }
