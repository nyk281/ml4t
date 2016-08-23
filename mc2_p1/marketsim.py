"""MC2-P1: Build a market simulator."""

import numpy as np
import pandas as pd
import scipy.optimize as spo
import datetime as dt

from util import symbol_to_path, get_data, plot_data, read_csv, get_base


def test_code():
    # Read in adjusted closing prices for given symbols, date range
    orders = read_csv('orders.csv')
    dates, symbols = get_base(orders)
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Create a dataframe which has all values as zero with index as dates and columns as symbols.
    trade_matrix = pd.DataFrame(0, index=dates, columns=symbols)
    print trade_matrix

if __name__ == "__main__":
    test_code()