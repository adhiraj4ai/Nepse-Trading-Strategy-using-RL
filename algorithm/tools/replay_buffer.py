import random
from collections import deque

import torch
import torch.nn


class ReplayBuffer:
    """
    A class used to represent a Replay Buffer for storing experiences in reinforcement learning.

    Attributes
    ----------
    storage : deque
        A double-ended queue to store experience tuples with a maximum length of buffer_size.
    batch_size : int
        The number of experiences to sample in each batch.

    Methods
    -------
    __init__(buffer_size, batch_size)
        Initializes the replay buffer with a specific size and batch size.
    push(params: tuple)
        Adds an experience tuple to the storage.
    extract_samples()
        Randomly samples a batch of experiences from the storage and returns them as torch tensors.
    """
    def __init__(self, buffer_size, batch_size):
        self.storage = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def push(self, params: tuple) -> None:
        """
        Adds an experience tuple to the storage.

        Parameters
        ----------
        params : tuple
            A tuple containing the experience in the format of (state, action, reward, next_state, done).

        Returns
        -------
        None
        """
        self.storage.append(params)

    def extract_samples(self):
        """
        Randomly samples a batch of experiences from the storage and returns them as torch tensors.

        Returns
        -------
        tuple
            A tuple containing batches of states, actions, rewards, next_states, and dones as torch tensors.
        """
        sample_size = min(len(self.storage), self.batch_size)
        experiences = random.sample(self.storage, k=sample_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        return (
            torch.stack(states).float(),
            actions,
            rewards,
            torch.stack(next_states).float(),
            dones
        )

    def __len__(self):
        """
        Returns the number of experiences currently stored in the replay buffer.

        Returns
        -------
        int
            The number of experiences in the storage.
        """
        return len(self.storage)

    def __repr__(self):
        """Returns a string representation of the ReplayBuffer object.

        Returns:
            str: The string representation of the ReplayBuffer object.
        """
        return f"ReplayBuffer(storage={self.storage}, batch_size={self.batch_size})"



