import copy
from ctypes import c_int
import torch
import torch.nn as nn
from .MultiHeadGraphAttention import MultiHeadGraphAttention
from .EdgeUpdate import EdgeUpdate
from .DiffCenAGG import DiffCenAGG
from torch_geometric.nn import GATConv


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
                in_fts=specs.node_in_fts,
                edge_in_fts=specs.edge_in_fts,
                out_dim=specs.edge_out_fts,
            )
            if specs.update_edge_first
            else None
        )

        self.nr_of_hops = specs.nr_of_hops
        self.residual_mode = specs.residual_mode

        hops = []
        updated_dim = specs.node_in_fts
        # TODO: Is deepcopy the best way to do this?
        for _ in range(specs.nr_of_hops):
            hops.append(
                GATConv(
                    in_channels=updated_dim,
                    out_channels=specs.node_out_fts,
                    heads=specs.num_heads_node,
                )
            )
            # hops.append(copy.deepcopy(model))
            # model.node_in_fts = updated_dim
            updated_dim = node_in_w_head

        self.hops = nn.ModuleList(hops)
        self.W_prior = (
            torch.nn.Parameter(torch.Tensor(specs.node_out_fts, specs.node_out_fts))
            if specs.residual_mode == "gated"
            else None
        )
        self.W_curr = (
            torch.nn.Parameter(torch.Tensor(specs.node_out_fts, specs.node_out_fts))
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
            edge_fts = self.edge_update([node_fts, edge_fts, edges])

        node_fts_prior = None

        for layer in self.hops:
            node_fts_curr = layer(node_fts, edges, edge_fts)

            if node_fts_prior is not None:
                if self.residual_mode == "gated":
                    gate = torch.sigmoid(
                        torch.matmul(node_fts_prior, self.W_prior)
                        + torch.matmul(node_fts_curr, self.W_curr)
                    )
                    node_fts = gate * node_fts_curr + (1 - gate) * node_fts_prior
                elif self.residual_mode == "add":
                    node_fts = node_fts_curr + node_fts_prior
                else:  # concat
                    node_fts = torch.cat((node_fts_curr, node_fts_prior), dim=1)
            else:
                node_fts = torch.clone(node_fts_curr)
            node_fts_prior = torch.clone(node_fts)

        # agg_node_fts = self.node_agg([node_fts, adj])
        # agg_edge_fts = torch.mean(edge_fts, dim=0)

        return None, None, node_fts, edge_fts, edges
