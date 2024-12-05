import random

import torch
from torch import optim, Tensor

from algorithm.network.dqn import DeepQNetwork
from algorithm.rlmodel import RLModel
from algorithm.tools.replay_buffer import ReplayBuffer
from environment.base import Environment
from utils.helper import format_price


class DeepQLearning(RLModel):
    """
    Class used to represent a Deep Q-Learning model for stock trading in the Nepse environment.
    """
    def __init__(self, env: Environment, nn_params: dict, rl_params: dict):
        super().__init__()
        self.env = env  # The Nepse environment
        self.state_size = env['state_size']  # State size is determined by the window size
        self.action_size = 3  # 0 = Hold, 1 = Buy, 2 = Sell

        self.learning_rate = nn_params['learning_rate']
        self.batch_size = nn_params['batch_size']

        self.discount_rate = rl_params['discount_rate']
        self.epsilon = rl_params['epsilon']
        self.epsilon_min = rl_params['epsilon_min']
        self.epsilon_decay = rl_params['epsilon_decay']
        self.buffer_size = rl_params['buffer_size']
        self.target_network_update_frequency = rl_params['target_network_update_frequency']

        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")

        # Initialize networks
        self.q_network = DeepQNetwork(self.state_size, self.action_size).to(self.device)
        self.target_q_network = DeepQNetwork(self.state_size, self.action_size).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)

    def act(self, state: Tensor) -> int:
        """
        Return an action according to the epsilon-greedy policy.
        """
        # Ensure the state tensor is on the correct device
        state = state.to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            # print(f"Exploration: Choose random action {action}")
        else:
            with torch.no_grad():
                action = self.q_network(state).max(1)[1].item()
                # print(f"Exploitation: Chose action {action} with Q-value {self.q_network(state).max(1)[0].item()}")
        return action

    def update_target_network(self) -> None:
        """
        Update the target Q-network by copying the weights from the current Q-network.
        """
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train_network(self) -> None:
        """
        Train the Q-network using experiences from the replay buffer.
        """
        minibatch = self.replay_buffer.extract_samples()

        # Move data to device
        state_batch = minibatch[0].to(self.device).float()  # minibatch[0] is already a tensor
        action_batch = torch.tensor(minibatch[1], dtype=torch.long).to(self.device)  # minibatch[1] is a tuple
        reward_batch = torch.tensor(minibatch[2], dtype=torch.float).to(self.device)
        next_state_batch = minibatch[3].to(self.device).float()
        done_batch = torch.tensor(minibatch[4], dtype=torch.float).to(self.device)

        # Compute Q-values for current and next states
        q_values = self.q_network(state_batch)
        next_q_values = self.target_q_network(next_state_batch)

        # Select Q-values corresponding to the actions taken
        predicted_q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        target_q_values = reward_batch + (1 - done_batch) * self.discount_rate * next_q_values.max(1)[0]

        # Compute loss and backpropagate
        loss = torch.nn.functional.mse_loss(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
