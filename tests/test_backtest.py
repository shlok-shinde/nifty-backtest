import numpy as np
import pandas as pd
import pytest

from backtest.data import DataError, validate_prices
from backtest.engine import run_backtest
from backtest.metrics import cagr, max_drawdown
from backtest.strategy import buy_and_hold, sma_crossover


def make_close(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def random_walk(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return make_close(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)))


def ohlc(close):
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close})


# --- no look-ahead -------------------------------------------------------------

@pytest.mark.parametrize("delay", [0, 1, 2])
def test_changing_prices_from_day_k_does_not_change_earlier_decisions(delay):
    close = random_walk()
    k = 400
    tampered = close.copy()
    tampered.iloc[k:] *= np.linspace(0.3, 3.0, len(close) - k)

    def sensitive_signal(c):
        # Moves with every close, so any use of a too-recent price shows up.
        return (c.pct_change().fillna(0) * 50 + 0.5).clip(0, 1)

    before = run_backtest(close, sensitive_signal(close), delay=delay)
    after = run_backtest(tampered, sensitive_signal(tampered), delay=delay)

    # The position held over day k was decided before day k's close existed.
    pd.testing.assert_series_equal(before["position"].iloc[:k + 1], after["position"].iloc[:k + 1])
    pd.testing.assert_frame_equal(before.iloc[:k], after.iloc[:k])


def test_signal_only_uses_past_and_current_closes():
    close = random_walk()
    full = sma_crossover(close, 20, 100)
    for t in [150, 300, 450]:
        assert sma_crossover(close.iloc[:t + 1], 20, 100).iloc[-1] == full.iloc[t]


# --- timing ----------------------------------------------------------------------

@pytest.mark.parametrize("delay", [0, 1, 2])
def test_position_starts_delay_plus_one_days_after_signal(delay):
    close = make_close(range(100, 110))
    signal = pd.Series(0.0, index=close.index)
    signal.iloc[3:] = 1.0  # decided at the close of day 3

    position = run_backtest(close, signal, cost_bps=0, delay=delay)["position"]

    first_day_held = position.to_numpy().nonzero()[0][0]
    assert first_day_held == 3 + delay + 1


# --- costs and cash ----------------------------------------------------------------

def test_round_trip_costs_twice_cost_bps_on_flat_prices():
    close = make_close([100.0] * 10)
    signal = make_close([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])

    result = run_backtest(close, signal, cost_bps=10, delay=0)

    final = (1 + result["net_ret"]).prod()
    assert final == pytest.approx(0.999 ** 2)
    assert (result["cost"] > 0).sum() == 2


def test_buy_and_hold_without_costs_matches_price_return():
    close = random_walk()
    result = run_backtest(close, buy_and_hold(close), cost_bps=0, delay=0)
    # Bought at the first close, so it earns every return after day 0.
    assert (1 + result["net_ret"]).prod() == pytest.approx(close.iloc[-1] / close.iloc[0])


def test_cash_earns_cash_rate():
    close = random_walk(n=253)
    always_cash = pd.Series(0.0, index=close.index)
    result = run_backtest(close, always_cash, cash_rate=0.06)
    # 252 daily returns after the first day add up to one year of cash yield.
    assert (1 + result["net_ret"].iloc[1:]).prod() == pytest.approx(1.06)


# --- strategy ------------------------------------------------------------------------

def test_crossover_is_cash_during_warm_up_and_rejects_bad_windows():
    close = random_walk()
    signal = sma_crossover(close, 20, 100)
    assert (signal.iloc[:99] == 0).all()
    with pytest.raises(ValueError):
        sma_crossover(close, 100, 50)


def test_crossover_holds_in_uptrend_and_exits_in_downtrend():
    close = make_close(list(range(100, 200)) + list(range(200, 100, -1)))
    signal = sma_crossover(close, 5, 20)
    assert signal.iloc[50] == 1
    assert signal.iloc[-1] == 0


# --- metrics ---------------------------------------------------------------------------

def test_max_drawdown():
    returns = make_close([100, 120, 90, 130]).pct_change().fillna(0)
    assert max_drawdown(returns) == pytest.approx(-0.25)


def test_cagr_uses_calendar_time():
    idx = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
    returns = pd.Series([0.0, 0.10, 0.10], index=idx)
    assert cagr(returns) == pytest.approx(0.10, abs=1e-3)


# --- data validation -------------------------------------------------------------------

def test_clean_data_passes_validation():
    validate_prices(ohlc(random_walk()))


def test_unadjusted_split_is_caught():
    close = random_walk()
    close.iloc[300:] *= 10  # like a 1:10 split that was never adjusted
    with pytest.raises(DataError, match=f"{close.index[300]:%Y-%m-%d}"):
        validate_prices(ohlc(close))


def test_missing_close_is_caught():
    close = random_walk()
    close.iloc[50] = np.nan
    with pytest.raises(DataError, match="missing closes"):
        validate_prices(ohlc(close))
