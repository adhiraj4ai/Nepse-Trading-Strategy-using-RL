from torch import relu
from torch import nn

from algorithm.network.model import Network


class DeepQNetwork(Network):
    """
    Deep Q-Network class which is a subclass of the Network class.

    Attributes
    ----------
    fc1 : torch.nn.Linear
        The first fully connected layer.
    fc2 : torch.nn.Linear
        The second fully connected layer.
    fc3 : torch.nn.Linear
        The third fully connected layer.

    Methods
    -------
    __init__(state_size, action_size)
        Initializes the DeepQNetwork object.
    forward(state)
        The forward pass of the network.
    """
    def __init__(self, state_size, action_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = relu(x)
        x = self.fc2(x)
        x = relu(x)

        return self.fc3(x)

