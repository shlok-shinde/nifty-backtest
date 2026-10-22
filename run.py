"""Backtest a moving-average crossover on the Nifty 50 and write results/.

    uv run python run.py            # uses the cached CSV in data/
    uv run python run.py --refresh  # downloads the data again
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest.data import load_prices
from backtest.engine import run_backtest
from backtest.metrics import drawdown_series, equity_curve, summarize
from backtest.strategy import buy_and_hold, sma_crossover

TICKER = "^NSEI"
DATA_START, DATA_END = "2007-09-17", "2026-10-01"  # end is exclusive: last bar is 2026-09-30
# A fixed end date keeps results reproducible; Yahoo's most recent bars are sometimes revised.

# The 250-day average needs about a year of history, so evaluation starts in 2009.
FULL = ("2009-01-01", "2026-09-30")
IN_SAMPLE = ("2009-01-01", "2017-12-31")
OUT_OF_SAMPLE = ("2018-01-01", "2026-09-30")

BASE = dict(cost_bps=10.0, delay=1, cash_rate=0.0)
FAST_GRID = [10, 20, 50, 100]
SLOW_GRID = [100, 150, 200, 250]

RESULTS = Path(__file__).resolve().parent / "results"

# Chart styling.
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STRATEGY_COLOR, BENCHMARK_COLOR = "#2a78d6", "#eb6834"


def window(result: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    return result.loc[period[0]:period[1]]


def fmt_row(name: str, stats: dict) -> str:
    return (f"| {name} | {stats['CAGR']:.1%} | {stats['Volatility']:.1%} | {stats['Sharpe']:.2f} "
            f"| {stats['Max drawdown']:.1%} | {stats['Time invested']:.0%} | {stats['Trades']} |")


TABLE_HEAD = ("| | CAGR | Volatility | Sharpe | Max drawdown | Time invested | Trades |\n"
              "|---|---:|---:|---:|---:|---:|---:|")


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_equity(strategy: pd.DataFrame, benchmark: pd.DataFrame, path: Path):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1]}, facecolor=SURFACE)
    for result, label, color in [(benchmark, "Buy and hold", BENCHMARK_COLOR),
                                 (strategy, "50/200 crossover", STRATEGY_COLOR)]:
        r = result["net_ret"]
        equity = equity_curve(r) * 100_000
        top.plot(equity, color=color, linewidth=1.6, label=label)
        top.annotate(f"₹{equity.iloc[-1] / 1e5:.1f}L", (equity.index[-1], equity.iloc[-1]),
                     xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
        bottom.plot(drawdown_series(r) * 100, color=color, linewidth=1.2, label=label)

    top.set_yscale("log")
    top.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1e5, 2e5, 4e5, 8e5]))
    top.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"₹{v / 1e5:.0f}L"))
    top.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    top.set_title("Growth of ₹1 lakh, Nifty 50, 2009–2026 (log scale, after costs)",
                  loc="left", color=INK, fontsize=12)
    top.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left")
    bottom.set_title("Drawdown from previous peak (%)", loc="left", color=INK, fontsize=10)
    for ax in (top, bottom):
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(sharpes: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=SURFACE)
    values = sharpes.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(sharpes.columns)), sharpes.columns)
    ax.set_yticks(range(len(sharpes.index)), sharpes.index)
    ax.set_xlabel("Slow window (days)", color=INK_MUTED)
    ax.set_ylabel("Fast window (days)", color=INK_MUTED)
    ax.set_title("In-sample Sharpe (2009–2017) by window pair", loc="left", color=INK, fontsize=11)
    low, high = pd.Series(values.ravel()).min(), pd.Series(values.ravel()).max()
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if pd.notna(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if (values[i, j] - low) / (high - low) > 0.6 else INK)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.8).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="download the data again")
    args = parser.parse_args()

    close = load_prices(TICKER, DATA_START, DATA_END, refresh=args.refresh)["Close"]
    RESULTS.mkdir(exist_ok=True)
    lines = [f"# Results\n\nNifty 50 (`{TICKER}`), daily closes {close.index[0]:%Y-%m-%d} to "
             f"{close.index[-1]:%Y-%m-%d}. Generated by `run.py`.\n"]

    def backtest(fast=50, slow=200, **overrides):
        return run_backtest(close, sma_crossover(close, fast, slow), **{**BASE, **overrides})

    benchmark = run_backtest(close, buy_and_hold(close), **BASE)
    strategy = backtest()

    # 1. Headline: 50/200 crossover vs buy and hold over the full period.
    lines += [f"## 1. Full period ({FULL[0]} to {FULL[1]})\n",
              "10 bps cost per buy or sell, trade one day after the signal, no interest on cash.\n",
              TABLE_HEAD,
              fmt_row("Buy and hold", summarize(window(benchmark, FULL))),
              fmt_row("50/200 crossover", summarize(window(strategy, FULL))), ""]
    plot_equity(window(strategy, FULL), window(benchmark, FULL), RESULTS / "equity_drawdown.png")

    # 2. How much the result depends on the assumptions.
    variants = [
        ("Base case", {}),
        ("No trading costs", {"cost_bps": 0.0}),
        ("25 bps per trade", {"cost_bps": 25.0}),
        ("Trade at the signal's own close", {"delay": 0}),
        ("6% a year on cash", {"cash_rate": 0.06}),
    ]
    lines += ["## 2. Sensitivity to assumptions (50/200, full period)\n", TABLE_HEAD]
    lines += [fmt_row(name, summarize(window(backtest(**kw), FULL))) for name, kw in variants]
    lines.append("")

    # 3. Choose windows on 2009-2017 only, then check them on 2018-2026.
    sharpes = pd.DataFrame(index=pd.Index(FAST_GRID, name="fast"),
                           columns=pd.Index(SLOW_GRID, name="slow"), dtype=float)
    for fast in FAST_GRID:
        for slow in SLOW_GRID:
            if fast < slow:
                sharpes.loc[fast, slow] = summarize(window(backtest(fast, slow), IN_SAMPLE))["Sharpe"]
    best_fast, best_slow = sharpes.stack().idxmax()
    plot_sweep(sharpes, RESULTS / "in_sample_sweep.png")

    best = backtest(best_fast, best_slow)
    lines += [f"## 3. In-sample choice vs out-of-sample result\n",
              f"Best window pair by Sharpe on {IN_SAMPLE[0]} to {IN_SAMPLE[1]}: "
              f"**{best_fast}/{best_slow}**.\n"]
    for label, period in [("In-sample", IN_SAMPLE), ("Out-of-sample", OUT_OF_SAMPLE)]:
        lines += [f"**{label} ({period[0]} to {period[1]})**\n", TABLE_HEAD,
                  fmt_row("Buy and hold", summarize(window(benchmark, period))),
                  fmt_row("50/200 crossover", summarize(window(strategy, period))),
                  fmt_row(f"{best_fast}/{best_slow} (best in-sample)", summarize(window(best, period))), ""]

    (RESULTS / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote {RESULTS / 'summary.md'}, equity_drawdown.png, in_sample_sweep.png")


if __name__ == "__main__":
    main()
