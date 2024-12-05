import pandas as pd
from algorithm.qlearning import DeepQLearning
from environment.nepse import Nepse
from utils.helper import plot_behavior


def main():
    # Load the stock data from a CSV file
    csv_file = "./data/sapdbl.csv"  # Provide the file path, not the DataFrame
    env_params = {
        "state_size": 32,  # Window size for state representation
        "action_size": 3,  # Actions: 0 (Hold), 1 (Buy), 2 (Sell)
    }

    # Define neural network parameters
    nn_params = {
        "learning_rate": 0.01,
        "batch_size": 32,
    }

    # Define RL parameters
    rl_params = {
        "discount_rate": 0.9,                      # Discount factor
        "epsilon": 1.0,                             # Initial exploration rate
        "epsilon_min": 0.01,                        # Minimum exploration rate
        "epsilon_decay": 0.995,                     # Exploration rate decay
        "buffer_size": 1000,                        # Size of the replay buffer
        "target_network_update_frequency": 10,      # Frequency of target network updates
        "replay_batch_size": 32,                    # Size of the replay batch
    }

    # Initialize the Deep Q-Learning agent
    dql_agent = DeepQLearning(env_params, nn_params, rl_params)

    # Define environment (pass the CSV file path instead of the DataFrame)
    env = Nepse(csv_file, window_size=env_params["state_size"])

    # Train the model
    num_of_episodes = 1000  # Number of episodes for training
    total_rewards = []  # List to store total rewards per episode
    profit_per_episode = []  # List to store profit per episode

    for episode in range(num_of_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_profit = 0

        while not done:
            # Select action based on the current policy
            action = dql_agent.act(state)

            # Take action and get the next state, reward, and done flag
            next_state, reward, profit, done, info = env.step(action)
            episode_reward += reward
            episode_profit += profit

            # Train the model with the experience
            dql_agent.replay_buffer.push((state, action, reward, next_state, done))

            # Train network if replay buffer is large enough
            if len(dql_agent.replay_buffer) > rl_params["replay_batch_size"]:
                dql_agent.train_network()

            state = next_state

        # Record the total reward and profit for the episode
        total_rewards.append(episode_reward)
        profit_per_episode.append(episode_profit)

        # plot_behavior(env.data, env.states_buy, env.states_sell, sum(profit_per_episode), total_rewards)

        # decay epsilon after each episode for exploration vs exploitation
        dql_agent.epsilon = max(dql_agent.epsilon_min, dql_agent.epsilon * dql_agent.epsilon_decay)

        print(f"Episode {episode + 1}/{num_of_episodes} - Total Reward: {episode_reward}, Total Profit: {episode_profit}")

        # if episode % 5 == 0:
        #     dql_agent.target_q_network.save_weights(f"./models/sapdbl_{episode}.h5")

    avg_reward = sum(total_rewards) / num_of_episodes
    avg_profit = sum(profit_per_episode) / num_of_episodes
    print(f"Training complete. Average reward: {avg_reward:.2f}")
    print(f"Average profit: {avg_profit:.2f}")


if __name__ == "__main__":
    main()
