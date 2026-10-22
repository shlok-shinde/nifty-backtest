"""Trading signals. A signal is 1 (hold the asset) or 0 (hold cash) per day."""

import pandas as pd


def sma_crossover(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Hold the asset while the fast moving average is above the slow one.

    The value on day t uses closes up to and including day t only. During the
    warm-up, before the slow average exists, the signal is 0 (cash).
    """
    if not 0 < fast < slow:
        raise ValueError(f"need 0 < fast < slow, got fast={fast}, slow={slow}")
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    # Comparisons with NaN are False, so the warm-up period comes out as 0.
    return (fast_ma > slow_ma).astype(float)


def buy_and_hold(close: pd.Series) -> pd.Series:
    """Always hold the asset. Used as the benchmark."""
    return pd.Series(1.0, index=close.index)
