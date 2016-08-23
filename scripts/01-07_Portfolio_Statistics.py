"""Sharpe Ratio and other portfolio statistics."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def symbol_to_path(symbol, base_dir="../data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def normalize_date(df):
    """Normalize stock prices using the first row of the dataframe."""
    return df / df.ix[0,:]


def compute_portfolio_value(df, allocs, start_val=1000000):
    """Compute and return the portfolio value."""
    df_normed = normalize_date(df)
    df_alloced = df_normed * allocs
    df_pos_vals = df_alloced * start_val
    df_port_val = df_pos_vals.sum(axis=1)
    return df_port_val


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.ix[0, :] = 0
    return daily_returns


def compute_portfolio_daily_returns(df):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    daily_returns = (df / df.shift(1)) - 1
    # daily_returns.ix[0] = 0
    daily_returns = daily_returns[1:]
    return daily_returns


def compute_daily_risk_free_rate(intrest):
    """Compute and return the daily risk free rate."""
    daily_risk_free_rate = np.power(1+intrest, 1/252) - 1
    return daily_risk_free_rate


def compute_sharpe_ratio(daily_rets, intrest=0):
    """Compute and return sharpe ratio."""
    daily_rf = compute_daily_risk_free_rate(intrest=intrest)
    sharpe_ratio = np.power(252, 1/2) * (daily_rets - daily_rf).mean() / daily_rets.std()
    return sharpe_ratio



def test_run():
    # Read data
    dates = pd.date_range('2009-01-01', '2010-01-01')  # one month only
    symbols = ['SPY', 'XOM', 'GOOG', 'GLD']
    allocs = [0.4, 0.4, 0.1, 0.1]
    df = get_data(symbols, dates)

    # Compute portfolio value
    port_val = compute_portfolio_value(df, allocs=allocs)
    #plot_data(port_val)

    # Compute daily returns
    daily_returns = compute_portfolio_daily_returns(port_val)
    #plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")

    # Compute portfolio statistics
    cum_ret = (port_val[-1] / port_val[0]) - 1
    print("cum_ret =", cum_ret)

    avg_daily_ret = daily_returns.mean()
    print("avg_daily_ret =", avg_daily_ret)

    std_daily_ret = daily_returns.std()
    print("std_daily_ret =", std_daily_ret)

    sharpe_ratio = compute_sharpe_ratio(daily_rets=daily_returns)
    print("sharpe_ratio =", sharpe_ratio)


if __name__ == "__main__":
    test_run()
