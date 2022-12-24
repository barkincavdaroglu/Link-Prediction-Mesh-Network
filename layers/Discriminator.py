import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Args:
            input_size: Dimension of input
            hidden_size: Dimension of hidden layer
        """
        super(Discriminator, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Sigmoid())
        self.fc2 = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, input) -> torch.Tensor:
        """
        Args:
            input: Output of Generator, represented via row-wise adjacency matrix
        """
        output = self.fc2(self.fc1(input)).squeeze(-1)
        return output
