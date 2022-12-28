import torch
import torch.nn as nn
import numpy as np


class DiffCenAGG(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(in_dim, out_dim)

    def order_nodes_by_diffcen(
        self, node_embeds: torch.tensor, adj: torch.tensor, t: int
    ) -> torch.tensor:
        """
        Order nodes by decreasing diffusion centrality.
        Args:
            adj: Adjacency matrix.
        Returns:
            Ordered node indices.
        """
        lambda_1 = np.linalg.eigvals(adj)[0]
        q = 1 / lambda_1
        eye = torch.eye(adj.shape[0], dtype=torch.float)

        H = torch.zeros(adj.shape)
        for i in range(1, t + 1):
            H = torch.add(H, torch.matrix_power(q * adj, i))

        H = torch.matmul(H, eye)
        embeds_ordered = node_embeds[torch.argsort(H.sum(axis=1), descending=True)]

        return embeds_ordered

    def forward(self, inputs):
        """
        Args:
            node_embeds: Node embeddings.
        Returns:
        """
        node_embeds, adj = inputs

        adj = adj.squeeze()
        embeds_ordered = self.order_nodes_by_diffcen(node_embeds, adj, 2)
        _, hn = self.gru(embeds_ordered)

        return hn
