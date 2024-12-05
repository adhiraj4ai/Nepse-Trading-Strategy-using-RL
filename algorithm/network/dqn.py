import torch
from torch import nn
from torch.nn import functional as F

class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DeepQNetwork, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(state_size, 128)  # Input layer to hidden layer 1
        self.fc2 = nn.Linear(128, 256)         # Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(256, 256)         # Hidden layer 2 to hidden layer 3
        self.fc4 = nn.Linear(256, action_size) # Output layer (Q-values for each action)

        # Layer normalization layers
        self.ln1 = nn.LayerNorm(128)  # Normalizing the output of fc1
        self.ln2 = nn.LayerNorm(256)  # Normalizing the output of fc2
        self.ln3 = nn.LayerNorm(256)  # Normalizing the output of fc3

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))  # Apply fc1 + LayerNorm + ReLU
        x = F.relu(self.ln2(self.fc2(x)))     # Apply fc2 + LayerNorm + ReLU
        x = F.relu(self.ln3(self.fc3(x)))     # Apply fc3 + LayerNorm + ReLU
        x = self.dropout(x)                   # Apply Dropout for regularization
        return self.fc4(x)                    # Output the Q-values for each action
