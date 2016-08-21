"""MLT: Utility code."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def normalize_date(df):
    """Normalize stock prices using the first row of the dataframe."""
    return df / df.ix[0, :]


def compute_daily_risk_free_rate(rfr, sf):
    """Compute and return the daily risk free rate."""
    drfr = np.power(1 + rfr, 1 / sf) - 1
    return drfr


def compute_sharpe_ratio(daily_rets, rfr, sf):
    """Compute and return sharpe ratio."""
    daily_rf = compute_daily_risk_free_rate(rfr=rfr, sf=sf)
    sharpe_ratio = np.power(sf, 1 / 2) * (daily_rets - daily_rf).mean() / daily_rets.std()
    return sharpe_ratio


def compute_daily_returns(port_val):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    daily_returns = (port_val / port_val.shift(1)) - 1
    daily_returns = daily_returns[1:]
    return daily_returns


def compute_portfolio_statistics(port_val, rfr, sf):
    """Compute and return portfolio statistics (Cumulative Return, Average Daily Return, Volatility, Sharpe Ratio)."""
    dr = compute_daily_returns(port_val)
    cr = (port_val[-1] / port_val[0]) - 1
    adr = dr.mean()
    sddr = dr.std()
    sr = compute_sharpe_ratio(daily_rets=dr, rfr=rfr, sf=sf)
    return [cr, adr, sddr, sr]


def compute_portfolio_value(prices, allocs, sv=1):
    """Compute and return the portfolio value."""
    normed = normalize_date(prices)
    alloced = normed * allocs
    pos_vals = alloced * sv
    port_val = pos_vals.sum(axis=1)
    return port_val
