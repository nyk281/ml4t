"""MLT: Utility code."""

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt


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


def read_csv(filename, base_dir="Orders-files"):
    """Return CSV file path given orders"""
    filepath = os.path.join(base_dir, filename)
    reader = csv.reader(open(filepath, 'rU'), delimiter=',')
    return reader


def get_base(orders):
    """Return two lists for all dates and symbols"""
    start_date = None
    end_date = None
    symbols = []

    for row in orders:
        date = dt.date(year=int(row[0]),month=int(row[1]),day=int(row[2]))
        if start_date is None or date < start_date:
            start_date = date
        if end_date is None or date > end_date:
            end_date = date
        symbols.append(row[3])

    #End date should be offset-ed by 1 day to read the close for the last date.
    end_date = end_date+ dt.timedelta(days=1)
    dates = pd.date_range(start_date, end_date)

    #Remove duplicates
    symbols = list(set(symbols))

    return [dates, symbols]

