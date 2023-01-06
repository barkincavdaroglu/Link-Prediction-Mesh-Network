import torch
import torch.nn as nn


class DiscriminatorLossModule(nn.Module):
    def __init__(self, loss):
        super(DiscriminatorLossModule, self).__init__()
        self.loss = loss

    def forward(self, pred, target) -> torch.Tensor:
        """
        Args:
            input: Output of Generator, represented via row-wise adjacency matrix (batch_size x num_nodes x num_nodes)
            target: Ground truth adjacency matrix (batch_size x num_nodes x num_nodes)
        """
        return self.loss(pred, target)
