import pandas as pd
from algorithm.qlearning import DeepQLearning
from environment.nepse import Nepse
from utils.helper import plot_behavior


def main():
    # Load the stock data from a CSV file
    csv_file = "./data/sapdbl.csv"  # Provide the file path, not the DataFrame
    env_params = {
        "state_size": 10,  # Window size for state representation
        "action_size": 3,  # Actions: 0 (Hold), 1 (Buy), 2 (Sell)
    }

    # Define neural network parameters
    nn_params = {
        "learning_rate": 0.001,
        "batch_size": 32,
    }

    # Define RL parameters
    rl_params = {
        "discount_rate": 0.95,  # Discount factor
        "epsilon": 1.0,         # Initial exploration rate
        "epsilon_min": 0.01,    # Minimum exploration rate
        "epsilon_decay": 0.995, # Exploration rate decay
        "buffer_size": 1000,    # Size of the replay buffer
        "target_network_update_frequency": 10,  # Frequency of target network updates
        "replay_batch_size": 32,  # Size of the replay batch
    }

    # Initialize the Deep Q-Learning agent
    dql_agent = DeepQLearning(env_params, nn_params, rl_params)

    # Define environment (pass the CSV file path instead of the DataFrame)
    env = Nepse(csv_file, window_size=env_params["state_size"])

    # Train the model
    num_episodes = 100  # Number of episodes for training
    total_rewards = []  # List to store total rewards per episode
    profit_per_episode = []  # List to store profit per episode

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        profit = 0

        while not done:
            # Select action based on the current policy
            action = dql_agent.act(state)

            # Take action and get the next state, reward, and done flag
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            profit += reward  # Reward is the profit from the action

            # Train the model with the experience
            dql_agent.replay_buffer.push((state, action, reward, next_state, done))
            dql_agent.train_network()

            state = next_state

        # Record the total reward and profit for the episode
        total_rewards.append(episode_reward)
        profit_per_episode.append(profit)

        # Optionally, decay epsilon after each episode for exploration vs exploitation
        dql_agent.epsilon = max(dql_agent.epsilon_min, dql_agent.epsilon * dql_agent.epsilon_decay)

        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {episode_reward}, Total Profit: {profit}")

    # After training, plot the behavior (currently commented out, but can be used for visualization)
    # plot_behavior(closing_prices, env.states_buy, env.states_sell, profit, total_rewards, profit_per_episode)


if __name__ == "__main__":
    main()
