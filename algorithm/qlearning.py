import random
import torch
from torch import optim, Tensor
from algorithm.network.dqn import DeepQNetwork
from algorithm.tools.replay_buffer import ReplayBuffer
from environment.base import Environment
from algorithm.rlmodel import RLModel
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            state = state.unsqueeze(0)  # Add batch dimension if missing

        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                return self.q_network(state).max(1)[1].item()

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
        state_batch = torch.tensor(minibatch[0], dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(minibatch[1], dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(minibatch[2], dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(minibatch[3], dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(minibatch[4], dtype=torch.float32).to(self.device)

        # Compute Q-values for current and next states
        q_values = self.q_network(state_batch)
        next_q_values = self.target_q_network(next_state_batch)

        # Select Q-values corresponding to the actions taken
        predicted_q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        target_q_values = reward_batch + (1 - done_batch) * self.discount_rate * next_q_values.max(1)[0]

        # Compute loss and backpropagate
        loss = torch.nn.functional.smooth_l1_loss(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_policy(self, num_of_episodes: int, max_steps: int = 1000):
        """
        Compute the policy using deep Q-learning and apply over the given number of episodes.
        """
        total_rewards = []

        for episode in range(1, num_of_episodes + 1):
            print(f"Running episode {episode}/{num_of_episodes}")

            state = self.env.reset()
            total_profit = 0
            episodic_reward = 0
            self.env.states_buy, self.env.states_sell = [], []

            for t in range(max_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                action = self.act(state_tensor)

                next_state, reward, done, info = self.env.step(action)
                episodic_reward += reward

                # Track profit and manage buy/sell states
                if action == 1:
                    total_profit -= self.env.data[self.env.current_step]
                elif action == 2 and len(self.env.states_buy) > 0:
                    buy_price = self.env.data[self.env.states_buy.pop(0)]
                    total_profit += self.env.data[self.env.current_step] - buy_price

                # Store experience in replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)

                # Train network if replay buffer is large enough
                if len(self.replay_buffer) >= self.batch_size:
                    self.train_network()

                state = next_state

                if done:
                    break

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            total_rewards.append(episodic_reward)

            if episode % self.target_network_update_frequency == 0:
                self.update_target_network()

            print(f"Episode {episode} finished. Total profit: {format_price(total_profit)}")

        avg_reward = sum(total_rewards) / num_of_episodes
        print(f"Training complete. Average reward: {avg_reward:.2f}")
