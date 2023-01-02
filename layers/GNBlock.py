from ctypes import c_int
import torch
import torch.nn as nn
from .MultiHeadNodeAttention import MultiHeadNodeAttention
from .MultiHeadGraphAttention import MultiHeadGraphAttention
from .EdgeUpdate import EdgeUpdate
from .DiffCenAGG import DiffCenAGG


class GNBlock(nn.Module):
    def __init__(
        self,
        graph_in_fts,
        graph_out_fts,
        node_in_fts,
        node_out_fts,
        edge_in_fts,
        edge_out_fts,
        num_heads_node,
        num_heads_graph,
        node_agg_mode,
        residual_mode,
        messagenorm_learn_scale,
        update_edge_first=False,
        head_agg_mode="mean",
        nr_of_hops=2,
    ):
        super(GNBlock, self).__init__()
        node_in_w_head = 0
        if head_agg_mode == "concat":
            node_in_w_head = (
                (num_heads_node * node_out_fts + num_heads_node * edge_out_fts) * 2
                if node_agg_mode == "concat"
                else num_heads_node * node_out_fts + num_heads_node * edge_out_fts
            )
        else:
            node_in_w_head = (
                node_out_fts * 2 if node_agg_mode == "concat" else node_out_fts
            )

        self.edge_update = (
            EdgeUpdate(
                in_fts=node_in_fts,
                edge_in_fts=edge_in_fts,
                out_dim=edge_out_fts,
            )
            if update_edge_first
            else None
        )

        self.nr_of_hops = nr_of_hops
        self.residual_mode = residual_mode

        hops = []
        updated_dim = node_in_fts
        for _ in range(nr_of_hops):
            hops.append(
                MultiHeadNodeAttention(
                    node_in_fts=updated_dim,
                    node_out_fts=node_out_fts,
                    edge_in_fts=edge_in_fts,
                    num_heads=num_heads_node,
                    head_agg_mode=head_agg_mode,
                    node_agg_mode=node_agg_mode,
                    messagenorm_learn_scale=messagenorm_learn_scale,
                )
            )
            updated_dim = node_in_w_head

        self.hops = nn.ModuleList(hops)
        self.W_prior = (
            torch.nn.Parameter(torch.Tensor(node_out_fts, node_out_fts))
            if residual_mode == "gated"
            else None
        )
        self.W_curr = (
            torch.nn.Parameter(torch.Tensor(node_out_fts, node_out_fts))
            if residual_mode == "gated"
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
            node_fts_curr = layer([node_fts, edge_fts, edges])

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
