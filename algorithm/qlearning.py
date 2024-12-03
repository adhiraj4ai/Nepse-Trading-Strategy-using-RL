import random

import torch
from torch import optim, Tensor

from algorithm.network.dqn import DeepQNetwork
from algorithm.tools.replay_buffer import ReplayBuffer
from rlmodel import RLModel
from utils.helper import plot_behavior


class DeepQLearning(RLModel):
    def __init__(self,
                 # environment parameters
                 state_size,
                 action_size,

                 # neural network parameters
                 learning_rate=0.001,
                 batch_size=32,

                 # RL Model parameters
                 discount_rate=0.9,
                 epsilon=0.9,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 target_network_update_frequency=100,
                 buffer_size=1000,
                 ):
        super().__init__()

        # environment parameters
        self.state_size = state_size
        self.action_size = action_size

        # neural network parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # RL Model parameters
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.target_network_update_frequency = target_network_update_frequency

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.q_network = DeepQNetwork(self.state_size, self.action_size).to(self.device)
        self.target_q_network = DeepQNetwork(self.state_size, self.action_size).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)

    def epsilon_greedy_policy(self, state: Tensor) -> int:
        """
        Return an action according to the epsilon-greedy policy.

        With probability epsilon, a random action is selected. Otherwise, the
        action that maximizes the Q-value is selected.

        :param state: The state of the environment.
        :return: The action to be taken.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                return self.q_network(state).max(1)[1].item()

    def update_target_network(self) -> None:
        """
        Update the target Q-network by copying the weights from the current Q-network.

        This function is called every `target_network_update_frequency` episodes to
        update the target Q-network. The target Q-network is used to compute the
        expected Q-values for the next state.
        """
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train_network(self) -> None:
        """
        Train the Q-network using experiences from the replay buffer.

        This function samples a batch of experiences from the replay buffer,
        computes the Q-value predictions, calculates the expected Q-values,
        computes the loss, and performs backpropagation to update the Q-network's
        weights.

        :returns: None
        """
        minibatch = self.replay_buffer.extract_samples()

        # Predict Q values from the network
        q_values = self.q_network(minibatch[0])

        # Compute Q values for the next state
        next_q_values = self.target_q_network(minibatch[3])

        # Expected Q values
        expected_q_values = minibatch[2] + self.discount_rate * next_q_values.max(1)[0]

        # Compute loss
        loss = torch.nn.functional.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        # Back propagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_policy(self, num_of_episodes, max_steps=1000):
        total_rewards = 0
        length_of_data = len(data)

        for episode in range(num_of_episodes+1):
            print(f"Running episode {e}/{num_of_episodes}")
            state = get_State(data, time, window_size)
            episodic_reward = 0     # total_profit

            next_state = get_State(data, time+1, window_size+1)
            reward = 0
            episodic_reward += reward
            done = t == length_of_data-1

            self.replay_buffer.push((state, action, reward, next_state, done))
            state = next_state

            if done:
                print(f"--------------------------------\nTotal Profit: {format_price(total_profit)}\n--------------------------------")
                plot_behavior(data, states_buy, states_sell, total_rewards)


