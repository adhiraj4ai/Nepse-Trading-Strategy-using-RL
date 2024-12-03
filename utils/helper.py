import datetime

import math
from matplotlib import pyplot as plt


# prints formatted price
def format_price(n):
    return ("-Rs." if n < 0 else "Rs.") + "{0:.2f}".format(abs(n))


def sigmoid(x):
    """
    Compute the sigmoid of x

    Parameters
    ----------
    x : float or array_like
        The value(s) to compute the sigmoid for.

    Returns
    -------
    sigmoid : float or array_like
        The sigmoid computed for x.
    """
    return 1 / (1 + math.exp(-x))


def plot_behavior(data_input, states_buy, states_sell, profit):
    """
    Plot the graph of the total value of the portfolio against time.

    Parameters
    ----------
    data_input : array_like
        The values of the total portfolio at each time step.
    states_buy : array_like
        The time steps at which the model is buying.
    states_sell : array_like
        The time steps at which the model is selling.
    profit : float
        The total profit made by the model.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(15, 5))
    plt.plot(data_input, color='r', lw=2.)
    plt.plot(data_input, '^', markersize=10, color='m', label='Buying signal', markevery=states_buy)
    plt.plot(data_input, 'v', markersize=10, color='k', label='Selling signal', markevery=states_sell)
    plt.title('Total gains: %f' % (profit))
    plt.legend()
    plt.savefig('../output/' + str(datetime.datetime.now().strftime('%Y-%m-%d')) + '.png')
    plt.show()
