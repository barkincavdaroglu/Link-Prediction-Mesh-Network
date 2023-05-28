from ctypes import c_int
import torch
import torch.nn as nn
from .EdgeUpdate import EdgeUpdate
import torch.nn as nn
from .GATv2Conv import GATv2Conv


class GNBlock(nn.Module):
    def __init__(
        self,
        specs,
        model,
    ):
        super(GNBlock, self).__init__()
        node_in_w_head = 0
        if specs.head_agg_mode == "concat":
            node_in_w_head = (
                (
                    specs.num_heads_node * specs.node_out_fts
                    + specs.num_heads_node * specs.edge_out_fts
                )
                * 2
                if specs.node_agg_mode == "concat"
                else specs.num_heads_node * specs.node_out_fts
                + specs.num_heads_node * specs.edge_out_fts
            )
        else:
            node_in_w_head = (
                specs.node_out_fts * 2
                if specs.node_agg_mode == "concat"
                else specs.node_out_fts
            )

        self.edge_update = (
            EdgeUpdate(
                node_in_fts=specs.node_in_fts,
                edge_in_fts=specs.edge_in_fts,
                edge_out_fts=specs.edge_out_fts,
            )
            if specs.update_edge_first
            else None
        )

        self.nr_of_hops = specs.nr_of_hops
        self.residual_mode = specs.residual_mode

        hops = []
        updated_dim = specs.node_in_fts

        for _ in range(specs.nr_of_hops):
            hops.append(
                GATv2Conv(
                    in_channels=updated_dim,
                    out_channels=specs.node_out_fts,
                    head_agg_mode=specs.head_agg_mode,
                    heads=specs.num_heads_node,
                    edge_dim=specs.edge_out_fts,
                )
            )
            updated_dim = node_in_w_head

        self.hops = nn.ModuleList(hops)
        self.lin_prior = (
            nn.Linear(specs.node_out_fts, specs.node_out_fts)
            if specs.residual_mode == "gated"
            else None
        )
        self.lin_curr = (
            nn.Linear(specs.node_out_fts, specs.node_out_fts)
            if specs.residual_mode == "gated"
            else None
        )

    def forward(
        self,
        node_fts: torch.Tensor,
        edge_fts: torch.Tensor,
        edges: torch.Tensor,
    ):
        if self.edge_update is not None:
            edge_fts = self.edge_update(node_fts, edge_fts, edges)

        node_fts_prior = None

        for layer in self.hops:
            node_fts_curr = layer(node_fts, edges, edge_fts)

            if node_fts_prior is not None:
                if self.residual_mode == "gated":
                    gate = torch.sigmoid(
                        self.lin_prior(node_fts_prior) + self.lin_curr(node_fts_curr)
                    )
                    node_fts = gate * node_fts_curr + (1 - gate) * node_fts_prior
                elif self.residual_mode == "add":
                    node_fts = node_fts_curr + node_fts_prior
                else:
                    node_fts = torch.cat((node_fts_curr, node_fts_prior), dim=1)
            else:
                node_fts = torch.clone(node_fts_curr)
            node_fts_prior = torch.clone(node_fts)

        return None, None, node_fts, edge_fts, edges
