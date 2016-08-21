"""MC1-P2: Optimize a portfolio."""

import numpy as np
import pandas as pd
import scipy.optimize as spo
import datetime as dt

from util import symbol_to_path, get_data, plot_data


def normalize_date(df):
    """Normalize stock prices using the first row of the dataframe."""
    return df / df.ix[0,:]


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


def compute_daily_risk_free_rate(rfr, sf):
    """Compute and return the daily risk free rate."""
    drfr = np.power(1+rfr, 1/sf) - 1
    return drfr


def compute_sharpe_ratio(daily_rets, rfr=0.0, sf=252):
    """Compute and return sharpe ratio."""
    daily_rf = compute_daily_risk_free_rate(rfr=rfr, sf=sf)
    sharpe_ratio = np.power(sf, 0.5) * (daily_rets - daily_rf).mean() / daily_rets.std()
    return sharpe_ratio


def compute_daily_returns(port_val):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    daily_returns = (port_val / port_val.shift(1)) - 1
    daily_returns = daily_returns[1:]
    return daily_returns


def f(X, prices, rfr, sf):
    """given scalar X, return some value (a real number)."""
    port_val = compute_portfolio_value(prices=prices, allocs=X)
    daily_rets = compute_daily_returns(port_val=port_val)
    Y = compute_sharpe_ratio(daily_rets=daily_rets, rfr=rfr, sf=sf) * -1
    return Y


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    sf=252
    rfr=0.0
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    Xguess = np.asarray([.25 for _ in range(4)])
    bounds = [(0., 1.) for _ in range(len(Xguess))]
    constraint = { 'type': 'eq', 'fun': lambda X: sum(abs(X)) - 1.}
    min_result = spo.minimize(f,
                              Xguess,
                              method='SLSQP',
                              bounds = bounds,
                              constraints = constraint,
                              args=(prices, rfr, sf,),
                              options={'disp': True})

    allocs = np.asarray(min_result.x)

    # Get daily portfolio value
    port_val = compute_portfolio_value(prices=prices, allocs=allocs)
    cr, adr, sddr, sr = compute_portfolio_statistics(port_val=port_val, rfr=rfr, sf=sf)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp = normalize_date(df_temp)
        plot_data(df_temp, title="Portfolio vs SPY")

    return allocs, cr, adr, sddr, sr


def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2010, 12, 31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date,
                                                        syms=symbols,
                                                        gen_plot=False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
