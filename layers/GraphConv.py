from cmath import isnan
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import *
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
import torch_scatter


class GraphConvolution(nn.Module):
    def __init__(
        self,
        node_in_fts: int,
        node_out_fts: int,
        neigh_agg_mode: str,
        dest_transform: str,
        normalize: bool,
        update_mode: str,
        hidden_dim: int,
        activation_update: str,
    ):
        super().__init__()
        self.node_in_fts = node_in_fts
        self.node_out_fts = node_out_fts
        self.node_agg_mode = neigh_agg_mode
        self.dest_transform = dest_transform
        self.normalize = normalize
        self.update_mode = update_mode

        self.lin_self = nn.Linear(node_in_fts, node_out_fts, bias=False)
        self.lin_neigh = nn.Linear(node_in_fts, node_out_fts, bias=False)
        self.lin_pool = nn.Linear(node_in_fts, node_in_fts, bias=True)

        if neigh_agg_mode in ["gru", "lstm", "rnn"]:
            if neigh_agg_mode == "lstm":
                self.rnn = nn.LSTM(
                    node_in_fts,  # config.node_num * node_in_w_head,
                    hidden_dim,  # gru_hidden,
                    batch_first=True,
                )
            elif neigh_agg_mode == "gru":
                self.rnn = nn.GRU(
                    node_in_fts,
                    hidden_dim,
                    batch_first=True,
                )
            else:
                self.rnn = nn.RNN(
                    node_in_fts,
                    hidden_dim,
                    batch_first=True,
                )

        self.normalize = normalize
        self.update_activation = getattr(nn, activation_update)(0.2)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.lin_self.weight, gain=1.414)
        nn.init.xavier_uniform_(self.lin_neigh.weight, gain=1.414)
        nn.init.xavier_uniform_(self.lin_pool.weight, gain=1.414)

    def forward(self, node_fts, edges, edge_fts):
        # TODO: Move around activation to test performance

        # Get the source and destination nodes of the edges
        src_node_fts = self.lin_self(node_fts)
        dst_node_fts = node_fts[edges[1]]

        # Neighborhood Aggregation Step
        if self.node_agg_mode == "mean":
            # MEAN over neighbors, apply linear transformation and finally the activation
            dst_node_fts_neigh_agg = scatter(
                dst_node_fts,
                edges[0],
                dim=0,
                reduce="mean",
            )
            dst_node_fts_neigh_agg_final = self.update_activation(
                self.lin_neigh(dst_node_fts_neigh_agg)
            )
        elif self.node_agg_mode == "max":
            # Apply linear transformation, MAX over neighbors and finally the activation
            dst_node_fts_neigh_transformed = self.lin_neigh(dst_node_fts)
            dst_node_fts_neigh_agg_final = scatter(
                dst_node_fts_neigh_transformed,
                edges[0],
                dim=0,
                reduce="max",
            )
        else:
            # We form sequences where each sequence is the neighborhood of a node
            # ordered by edge weight. Then each sequence is padded to the max
            # neighborhood size and passed through an RNN.
            lengths = torch_scatter.scatter_add(
                torch.ones(edges.shape[1]),
                edges[0],
                dim=0,
                dim_size=node_fts[0],
            )
            max_neighborhood_size = int(lengths.max().item())

            neighbs, _ = to_dense_batch(
                node_fts[edges[1]], edges[0], 0, max_neighborhood_size
            )

            edg, _ = to_dense_batch(edge_fts, edges[0], 0, max_neighborhood_size)

            sorted_edg = torch.argsort(edg, dim=1, descending=True)
            sorted_edg = sorted_edg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 12)

            res = torch.zeros_like(neighbs)
            torch.gather(neighbs, 1, sorted_edg, out=res)

            packed = nn.utils.rnn.pack_padded_sequence(
                res, lengths.to("cpu"), batch_first=True, enforce_sorted=False
            )
            _, (dst_node_fts_neigh_agg_final, _) = self.rnn(packed)

            dst_node_fts_neigh_agg_final = dst_node_fts_neigh_agg_final.squeeze(0)

        if self.update_mode == "sum":
            node_fts_updated = src_node_fts + dst_node_fts_neigh_agg_final
        else:
            # concat
            node_fts_updated = torch.cat(
                [src_node_fts, dst_node_fts_neigh_agg_final], dim=1
            )

        node_fts_updated = self.update_activation(node_fts_updated)
        if self.normalize:
            node_fts_updated = F.normalize(node_fts_updated, p=2, dim=1)

        return node_fts_updated
