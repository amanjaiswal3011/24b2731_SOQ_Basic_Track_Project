# BTC/USD Trading Strategy Backtest

A trend-following BTC/USD trading strategy, backtested end-to-end with a custom
event-driven backtesting engine. Built for the BTC/USD Trading Strategy
Backtesting Challenge.

## Strategy Overview

**EMA Trend-Cross with ADX/RSI Filters + ATR Trailing Stop**

The strategy trades BTC/USD (long and short) based on the idea that Bitcoin's
daily price action is dominated by extended directional regimes punctuated by
choppy, range-bound periods. It only acts when a trend is confirmed, and
protects gains with a volatility-adaptive trailing stop rather than a fixed one.

| Component | Role |
|---|---|
| EMA(20) / EMA(50) crossover | Direction signal — marks a shift in the prevailing trend |
| ADX(14) > 20 | Strength filter — only trades crossovers that occur during a genuine trend, filtering out whipsaws |
| RSI(14), 30/70 bounds | Avoids opening fresh longs when overbought or fresh shorts when oversold |
| ATR(14) × 3 trailing stop | Volatility-adaptive risk management; only ever ratchets in the trade's favor |

Parameters were deliberately kept at conventional values (rather than the
single best combination found via grid search) to avoid overfitting to the
training window — see `BTC_Strategy_Report.pdf` for the full writeup and a
robustness check.

## Results (Training Data: BTC 2018–2022, Daily)

| Metric | Strategy | Buy & Hold |
|---|---|---|
| Total Return | +49.0% | +23.6% |
| Net Profit ($1,000 start) | $490.16 | $236.35 |
| Sharpe Ratio (annualized) | 0.43 | — |
| Win Rate | 50.0% (8W / 8L) | — |
| Total Trades | 16 | 1 |
| Max Drawdown | 23.0% | ~70%+ |

Full metrics, methodology, and charts are in [`BTC_Strategy_Report.pdf`](./BTC_Strategy_Report.pdf).

**No lookahead bias**: verified via the built-in check in `main.py`, which
recomputes indicators/signals on truncated data at every signaled index and
confirms the result matches the full-dataset run.

## Project Structure

```
.
├── main.py                     # Strategy implementation (process_data + strat)
├── backtester.py                # Event-driven backtesting engine (do not modify)
├── btc_18_22_1d.csv             # Training data: BTC/USD daily OHLCV, 2018-2022
├── final_data.csv               # Output: processed data with signals (generated)
├── BTC_Strategy_Report.pdf      # Strategy writeup, methodology, results, charts
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
plotly
```

Indicators (EMA, RSI, ATR, ADX) are implemented from scratch in `main.py`
using pandas rolling/`ewm` operations — no `talib` / `pandas_ta` dependency.

Install:
```bash
pip install pandas numpy matplotlib plotly
```

## Usage

```bash
python main.py
```

This will:
1. Load `btc_18_22_1d.csv` and compute indicators (`process_data`)
2. Generate trading signals (`strat`)
3. Save results to `final_data.csv`
4. Run the backtest via `BackTester` ($1,000 initial capital, compounding, 0.15% fee per trade)
5. Print performance statistics (Sharpe, win rate, drawdown, etc.)
6. Run the lookahead-bias check
7. Display the PnL / equity graph

## Signal Schema

| Signal | Action | Position After |
|---|---|---|
| `1` | Go long (if flat) / exit short (if short) | +1 or 0 |
| `-1` | Go short (if flat) / exit long (if long) | -1 or 0 |
| `2` | Reverse short → long | +1 |
| `-2` | Reverse long → short | -1 |
| `0` | Hold — no change | Same as before |

## Notes

- `backtester.py` is provided as-is and must not be modified.
- Only `process_data()` and `strat()` in `main.py` implement the strategy logic.
- The training set is provided for development only; the strategy is evaluated
  on unseen out-of-sample data, so parameters were chosen for robustness over
  peak backtest performance.
