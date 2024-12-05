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


def plot_behavior(data_input, states_buy, states_sell, profit, rewards):
    """
    Plot the graph of the total value of the portfolio against time, and display reward/profit evolution.

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
    rewards : list
        List of total rewards per episode.

    Returns
    -------
    None
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot portfolio value with buy and sell signals
    axs[0].plot(data_input, color='r', lw=2.)
    axs[0].plot(data_input, '^', markersize=10, color='m', label='Buying signal', markevery=states_buy)
    axs[0].plot(data_input, 'v', markersize=10, color='k', label='Selling signal', markevery=states_sell)
    axs[0].set_title('Total Gains: %f' % profit)
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Portfolio Value')
    axs[0].legend()

    # Plot total reward and profit per episode
    axs[1].plot(rewards, label="Total Reward", color='blue')
    axs[1].plot(profit, label="Total Profit", color='green')
    axs[1].set_title('Reward and Profit per Episode')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Value')
    axs[1].legend()

    # Save and display the plot
    plt.tight_layout()
    plt.savefig('./output/' + str(datetime.datetime.now().strftime('%Y-%m-%d')) + '.png')
    plt.show()
