import pandas as pd
import numpy as np
from backtester import BackTester

# -----------------------------------------------------------------------
# Strategy: EMA Trend-Cross + ADX Strength Filter + RSI Filter,
#           with an ATR-based trailing stop for risk management.
#
# Rationale (see report for full write-up):
#   - EMA(12)/EMA(26) crossover identifies trend direction changes.
#   - ADX(14) > 20 confirms the move has real trend strength, filtering
#     out the whipsaws that a plain EMA cross produces in choppy markets.
#   - RSI(14) keeps us from chasing entries that are already extended
#     (skip longs above 70 / shorts below 30).
#   - ATR(14) sizes a trailing stop that adapts to current volatility,
#     used to reverse/close positions when the trend gives back gains.
#
# All indicators below are computed with pandas rolling / ewm operations,
# which by construction only use data up to and including the current row
# (no centering, no shifting forward) -> no lookahead bias.
# -----------------------------------------------------------------------

EMA_FAST = 20
EMA_SLOW = 50
RSI_LEN = 14
ATR_LEN = 14
ADX_LEN = 14
ADX_THRESHOLD = 20
ATR_STOP_MULT = 3.0
WARMUP = 60  # bars needed for slowest indicator (EMA_SLOW, ADX ~ 2x length) to stabilize


def _ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def _rsi(close, length):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing (recursive, causal)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral during warmup / zero-loss periods
    return rsi


def _atr(high, low, close, length):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return atr


def _adx(high, low, close, length):
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr = _atr(high, low, close, length)

    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return adx.fillna(0)


def process_data(data):
    """
    Process the input data and return a dataframe with all the necessary indicators and data for making signals.

    Parameters:
    data (pandas.DataFrame): The input data to be processed.

    Returns:
    pandas.DataFrame: The processed dataframe with all the necessary indicators and data.
    """
    data = data.copy()
    data['EMA_fast'] = _ema(data['close'], EMA_FAST)
    data['EMA_slow'] = _ema(data['close'], EMA_SLOW)
    data['RSI'] = _rsi(data['close'], RSI_LEN)
    data['ATR'] = _atr(data['high'], data['low'], data['close'], ATR_LEN)
    data['ADX'] = _adx(data['high'], data['low'], data['close'], ADX_LEN)
    return data


def strat(data):
    """
    EMA trend-cross strategy with ADX/RSI filters and an ATR trailing stop.

    Parameters:
    - data: DataFrame containing OHLCV + indicator columns.

    Returns:
    - DataFrame with 'signals' and 'trade_type' columns populated.
    """
    data = data.copy()
    data['trade_type'] = "HOLD"
    data['signals'] = 0

    position = 0          # 0 = flat, 1 = long, -1 = short
    trailing_stop = 0.0

    for i in range(WARMUP, len(data)):
        close = data.loc[i, 'close']
        atr = data.loc[i, 'ATR']
        adx = data.loc[i, 'ADX']
        rsi = data.loc[i, 'RSI']
        ema_f, ema_s = data.loc[i, 'EMA_fast'], data.loc[i, 'EMA_slow']
        ema_f_prev, ema_s_prev = data.loc[i - 1, 'EMA_fast'], data.loc[i - 1, 'EMA_slow']

        bullish_cross = (ema_f > ema_s) and (ema_f_prev <= ema_s_prev)
        bearish_cross = (ema_f < ema_s) and (ema_f_prev >= ema_s_prev)

        trend_ok = adx > ADX_THRESHOLD

        if position == 0:
            if bullish_cross and trend_ok and rsi < 70:
                data.loc[i, 'signals'] = 1
                data.loc[i, 'trade_type'] = "LONG"
                position = 1
                trailing_stop = close - ATR_STOP_MULT * atr

            elif bearish_cross and trend_ok and rsi > 30:
                data.loc[i, 'signals'] = -1
                data.loc[i, 'trade_type'] = "SHORT"
                position = -1
                trailing_stop = close + ATR_STOP_MULT * atr

        elif position == 1:  # currently long
            if bearish_cross and trend_ok:
                data.loc[i, 'signals'] = -2
                data.loc[i, 'trade_type'] = "REVERSE_LONG_TO_SHORT"
                position = -1
                trailing_stop = close + ATR_STOP_MULT * atr

            elif close < trailing_stop:
                data.loc[i, 'signals'] = -1
                data.loc[i, 'trade_type'] = "CLOSE"
                position = 0

            else:
                trailing_stop = max(trailing_stop, close - ATR_STOP_MULT * atr)

        elif position == -1:  # currently short
            if bullish_cross and trend_ok:
                data.loc[i, 'signals'] = 2
                data.loc[i, 'trade_type'] = "REVERSE_SHORT_TO_LONG"
                position = 1
                trailing_stop = close - ATR_STOP_MULT * atr

            elif close > trailing_stop:
                data.loc[i, 'signals'] = 1
                data.loc[i, 'trade_type'] = "CLOSE"
                position = 0

            else:
                trailing_stop = min(trailing_stop, close + ATR_STOP_MULT * atr)

    return data


def main():
    data = pd.read_csv("btc_18_22_1d.csv")
    processed_data = process_data(data)
    result_data = strat(processed_data)
    csv_file_path = "final_data.csv"
    result_data.to_csv(csv_file_path, index=False)

    bt = BackTester("BTC", signal_data_path="final_data.csv", master_file_path="final_data.csv", compound_flag=1)
    bt.get_trades(1000)

    stats = bt.get_statistics()
    print("=== Performance Statistics ===")
    for key, val in stats.items():
        print(key, ":", val)

    # Check for lookahead bias
    print("\nChecking for lookahead bias...")
    lookahead_bias = False
    for i in range(len(result_data)):
        if result_data.loc[i, 'signals'] != 0:
            temp_data = data.iloc[:i + 1].copy()
            temp_data = process_data(temp_data)
            temp_data = strat(temp_data)
            if temp_data.loc[i, 'signals'] != result_data.loc[i, 'signals']:
                print(f"Lookahead bias detected at index {i}")
                lookahead_bias = True

    if not lookahead_bias:
        print("No lookahead bias detected.")

    # NOTE: bt.make_trade_graph() is referenced in the original template but is
    # not implemented in the provided backtester.py (DO NOT MODIFY), so it is
    # omitted here to avoid a crash. Only make_pnl_graph() exists in the class.
    bt.make_pnl_graph()


if __name__ == "__main__":
    main()
