import numpy as np
import pandas as pd
from typing import Any, Tuple, List

import torch

from environment.base import Environment


class Nepse(Environment):
    def __init__(self, csv_file: str, window_size: int = 10):
        """
        Initialize the Nepse environment with stock price data.

        :param csv_file: Path to the CSV file containing stock price data.
        :param window_size: The size of the window used to generate the state.
        """
        super().__init__()
        self.csv_file = csv_file
        self.window_size = window_size
        self.data = None
        self.current_step = 0
        self.states_buy = []
        self.states_sell = []
        self.done = False

        self._load_data()

    def _load_data(self) -> None:
        """
        Load the stock price data from the given CSV file and store it in the `data` attribute.

        This method is called automatically in the constructor.
        """
        # Load CSV data into a DataFrame
        df = pd.read_csv(self.csv_file)

        # Ensure the "Date" column is parsed as datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

        # Drop any rows with missing or invalid "Close" values
        df = df.dropna(subset=['Close'])
        self.data = df['Close'].values

        # Check if there is enough data to create states with the window size
        if len(self.data) < self.window_size:
            raise ValueError("The data size is smaller than the window size.")

    def reset(self) -> np.ndarray:
        """
        Reset the environment to the initial state.

        :return: The initial state.
        """
        self.current_step = 0
        self.states_buy = []
        self.states_sell = []
        self.done = False
        return self.get_State(self.data, self.current_step, self.window_size)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take an action in the environment.

        :param action: The action to take (0 = Hold, 1 = Buy, 2 = Sell).
        :return: A tuple (state, reward, done, info).
        """
        if self.done:
            raise ValueError("The environment is already done. Please reset it.")

        reward = self.compute_reward(action, self.data, self.current_step, self.states_buy, self.states_sell)

        # Update to the next step
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1

        next_state = self.get_State(self.data, self.current_step, self.window_size)
        info = {
            "step": self.current_step,
            "states_buy": self.states_buy,
            "states_sell": self.states_sell,
        }
        return next_state, reward, self.done, info

    @staticmethod
    def get_State(data, t, window_size) -> torch.Tensor:
        """
        Generate the state representation for the RL agent.

        :param data: A list of stock closing prices.
        :param t: The current time step.
        :param window_size: The size of the window used to generate the state.
        :return: A PyTorch tensor representing the state.
        """
        # Ensure the start index does not go below 0
        start_idx = max(0, t - window_size + 1)

        # Extract the window of prices from data, starting from start_idx to t+1
        window = data[start_idx:t + 1]

        # Ensure the window has the correct length
        if len(window) < window_size:
            # Pad the window with the first value of the data (data[0])
            padding_length = window_size - len(window)
            window = [data[0]] * padding_length + list(window)

        # Convert window prices to percentage changes relative to the first value of the window
        state = [(price / window[0]) - 1 for price in window]

        # Convert the list to a PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)

        return state_tensor

    @staticmethod
    def compute_reward(action: int, data: List[float], t: int, states_buy: List[int], states_sell: List[int]) -> float:
        """
        Compute the reward for a given action in the stock trading environment.

        :param action: The action taken (0 = Hold, 1 = Buy, 2 = Sell).
        :param data: The list of stock closing prices.
        :param t: The current time step.
        :param states_buy: A list of indices where stocks were bought.
        :param states_sell: A list of indices where stocks were sold.
        :return: The reward based on the action and market conditions.
        """
        current_price = data[t]
        reward = 0

        if action == 0:  # Hold
            reward = 0  # No immediate reward for holding

        elif action == 1:  # Buy
            if t in states_buy:
                reward = -1  # Penalty for redundant buy
            else:
                states_buy.append(t)
                reward = 0  # Neutral reward for buying

        elif action == 2:  # Sell
            if len(states_buy) == 0:  # No stock to sell
                reward = -1  # Penalize trying to sell without owning stock
            else:
                buy_price_index = states_buy.pop(0)  # FIFO selling strategy
                buy_price = data[buy_price_index]
                profit = current_price - buy_price
                states_sell.append(t)
                reward = profit  # Reward is the profit or loss from the trade

        return reward
