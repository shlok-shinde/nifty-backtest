"""Download, cache and validate daily price data."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class DataError(ValueError):
    """Raised when price data fails a sanity check."""


def load_prices(ticker: str, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """Return daily OHLCV for `ticker` from `start` up to (not including) `end`.

    The first download is cached to data/ so later runs are reproducible and
    work offline. Pass refresh=True to download again.
    """
    name = ticker.lstrip("^").replace(".", "_")
    path = DATA_DIR / f"{name}_{start}_{end}.csv"

    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
    else:
        import yfinance as yf

        df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                         progress=False, multi_level_index=False)
        if df.empty:
            raise DataError(f"No data returned for {ticker}")
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index.name = "Date"
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(path)

    validate_prices(df, ticker)
    return df


def validate_prices(df: pd.DataFrame, label: str = "prices", max_daily_move: float = 0.20) -> None:
    """Raise DataError if the data has gaps, bad values or implausible jumps.

    A daily close-to-close move above `max_daily_move` almost always means a
    bad print or an unadjusted split. The largest real one-day move in the
    Nifty 50 since 2007 is +17.7% (18 May 2009), so 20% leaves headroom.
    """
    problems = []
    if df.empty:
        raise DataError(f"{label}: no rows")
    if not df.index.is_unique:
        problems.append("duplicate dates")
    if not df.index.is_monotonic_increasing:
        problems.append("dates out of order")

    close = df["Close"]
    if close.isna().any():
        problems.append(f"{close.isna().sum()} missing closes")
    if (close <= 0).any():
        problems.append("non-positive closes")
    if (df["High"] < df["Low"]).any():
        problems.append("rows with High below Low")

    moves = close.pct_change()
    jumps = moves[moves.abs() > max_daily_move]
    if not jumps.empty:
        listed = ", ".join(f"{d:%Y-%m-%d} ({m:+.0%})" for d, m in jumps.items())
        problems.append(f"daily moves above {max_daily_move:.0%}: {listed}")

    if problems:
        raise DataError(f"{label}: " + "; ".join(problems))
