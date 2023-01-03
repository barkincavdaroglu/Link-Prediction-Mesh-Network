import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeUpdate(torch.nn.Module):
    def __init__(self, node_in_fts: int, edge_in_fts: int, edge_out_fts: int) -> None:
        super().__init__()
        self.dim = edge_out_fts
        self.initial_update = torch.nn.Linear(
            in_features=edge_in_fts, out_features=edge_out_fts
        )
        self.mlp = torch.nn.Linear(
            in_features=2 * node_in_fts + edge_out_fts, out_features=edge_out_fts
        )
        self.alpha_mlp = torch.nn.Linear(
            in_features=2 * node_in_fts + edge_out_fts, out_features=1
        )

    def forward(
        self,
        node_fts: torch.Tensor,
        edge_fts: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: tuple of (node_fts, edge_fts, edges)
        Returns:
            edge_fts: updated edge features
        """
        inputs = (node_fts, edge_fts, edges)
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
