import torch
import torch.nn as nn


class GeneratorLossModule(nn.Module):
    def __init__(self, loss):
        super(GeneratorLossModule, self).__init__()
        self.loss = loss

    def forward(self, pred, target) -> torch.Tensor:
        """
        Args:
            input: Output of Generator, represented via row-wise adjacency matrix (batch_size x num_nodes x num_nodes)
            target: Ground truth adjacency matrix (batch_size x num_nodes x num_nodes)
        """
        # take only the first horizon predictions and target

        batched_target_adj = target.reshape(-1, target.shape[0] * target.shape[1])
        batched_predicted_adj = pred.reshape(-1, target.shape[0] * target.shape[1])

        return torch.sqrt(self.loss(batched_predicted_adj, batched_target_adj))
