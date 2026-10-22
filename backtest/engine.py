"""Turn a signal into daily strategy returns, with execution delay, costs and cash yield."""

import pandas as pd

TRADING_DAYS = 252


def run_backtest(close: pd.Series, signal: pd.Series, cost_bps: float = 10.0,
                 delay: int = 1, cash_rate: float = 0.0) -> pd.DataFrame:
    """Simulate holding `signal` (0..1 of capital in the asset) over `close`.

    Timing: signal[t] is known after the close on day t. The trade is made at
    the close of day t + delay, so the new position first earns the return of
    day t + delay + 1.
      delay=0  trade at the same close the signal was computed from (optimistic)
      delay=1  trade at the next day's close (default; leaves a day to act)

    cost_bps is charged on every change in position, as basis points of
    capital: going from cash to fully invested costs cost_bps, and selling
    back to cash costs cost_bps again.

    cash_rate is the annual return earned on the part of capital not invested.

    Returns one row per day with the position held over that day's return,
    the asset return, the trading cost and the net strategy return.
    """
    if delay < 0:
        raise ValueError("delay must be >= 0")
    signal = signal.reindex(close.index)

    asset_ret = close.pct_change().fillna(0.0)
    # position[t] is what we hold from the close of t-1 to the close of t.
    position = signal.shift(delay + 1).fillna(0.0)
    turnover = position.diff().fillna(position).abs()
    cost = turnover * cost_bps / 10_000
    daily_cash = (1 + cash_rate) ** (1 / TRADING_DAYS) - 1

    net = position * asset_ret + (1 - position) * daily_cash - cost
    return pd.DataFrame({
        "position": position,
        "asset_ret": asset_ret,
        "cost": cost,
        "net_ret": net,
    })
