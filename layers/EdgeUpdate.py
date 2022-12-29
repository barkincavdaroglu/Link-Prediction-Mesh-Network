import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeUpdate(torch.nn.Module):
    def __init__(self, in_fts: int, edge_in_fts: int, out_dim: int) -> None:
        super().__init__()
        self.dim = out_dim
        self.initial_update = torch.nn.Linear(
            in_features=edge_in_fts, out_features=out_dim
        )
        self.mlp = torch.nn.Linear(
            in_features=2 * in_fts + out_dim, out_features=out_dim
        )
        self.alpha_mlp = torch.nn.Linear(
            in_features=2 * in_fts + out_dim, out_features=1
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: tuple of (node_fts, edge_fts, edges)
        Returns:
            edge_fts: updated edge features
        """
        node_fts, edge_fts, edges = inputs

        node_fts, edge_fts, edges = (
            node_fts.squeeze(),
            edge_fts.squeeze(),
            edges.squeeze(),
        )

        edge_fts = self.initial_update(edge_fts)
        heads, tails = edges[0], edges[1]

        head_vectors = torch.index_select(node_fts, index=heads, dim=0)
        tail_vectors = torch.index_select(node_fts, index=tails, dim=0)

        combined = torch.cat([edge_fts, head_vectors, tail_vectors], dim=1)
        alpha = self.alpha_mlp(combined)
        alpha = torch.sigmoid(alpha)
        processed = self.mlp(combined)
        processed.tanh_()
        right_side = torch.mul(F.layer_norm(processed, [self.dim]), alpha)
        output = F.layer_norm(
            edge_fts + right_side,
            [self.dim],
        )
        return output
