class RLModel:
    def __init__(self):
        pass

    def epsilon_greedy_policy(self, state):
        ...

    def compute_policy(self, num_of_episodes, max_steps=1000):
        ...

    def execute_policy(self, max_steps=1000):
        ...
