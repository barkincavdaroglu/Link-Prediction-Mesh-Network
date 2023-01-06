import torch
import torch.nn as nn
import numpy as np


class DiffCenAGG(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, recurrent_model="lstm") -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        if recurrent_model == "lstm":
            self.rnn = nn.LSTM(
                in_dim,  # config.node_num * node_in_w_head,
                hidden_dim,  # gru_hidden,
                batch_first=True,
            )
        elif recurrent_model == "gru":
            self.rnn = nn.GRU(
                in_dim,
                hidden_dim,
                batch_first=True,
            )
        else:
            self.rnn = nn.RNN(
                in_dim,
                hidden_dim,
                batch_first=True,
            )

    def order_nodes_by_diffcen(
        self, node_embeds: torch.Tensor, adj: torch.Tensor, t: int
    ) -> torch.Tensor:
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
        embeds_ordered = node_embeds[torch.argsort(H.sum(dim=1), descending=True)]

        return embeds_ordered

    def forward(self, inputs):
        """
        Args:
            inputs: (node_embeds, edge_fts, edges)
        Returns:
            hn: Hidden state of the last step.
        """
        node_embeds, edge_fts, edges = inputs

        adj = torch.zeros((node_embeds.shape[0], node_embeds.shape[0]))
        adj[edges[0], edges[1]] = edge_fts[:, 0]

        embeds_ordered = self.order_nodes_by_diffcen(node_embeds, adj, 2)
        if self.recurrent_model == "lstm":
            _, (hn, _) = self.rnn(embeds_ordered)
        else:
            _, hn = self.rnn(embeds_ordered)

        return hn
