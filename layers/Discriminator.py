import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, discriminator_config):
        """
        Args:
            input_size: Dimension of input
            hidden_size: Dimension of hidden layer
        """
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(
                discriminator_config.input_size, discriminator_config.hidden_size
            ),
            nn.LeakyReLU(0.2),
            nn.Linear(
                discriminator_config.hidden_size,
                int(discriminator_config.hidden_size / 2),
            ),
            nn.LeakyReLU(0.2),
            nn.Linear(int(discriminator_config.hidden_size / 2), 1),
            nn.Sigmoid(),
        )

    def forward(self, input) -> torch.Tensor:
        """
        Args:
            input: Output of Generator, represented via row-wise adjacency matrix
        """
        return self.fc(input)
