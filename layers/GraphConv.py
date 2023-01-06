from cmath import isnan
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import *
from torch_scatter import scatter
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(
        self,
        specs,
    ):
        super().__init__()
        self.node_in_fts = specs.node_in_fts
        self.node_out_fts = specs.node_out_fts
        self.node_agg_mode = specs.neigh_agg_mode
        self.dest_transform = specs.dest_transform
        self.normalize = specs.normalize
        self.update_mode = specs.update_mode

        self.kernel_init = specs.kernel_init
        self.kernel_reg = specs.kernel_reg

        self.lin_self = nn.Linear(specs.node_in_fts, specs.node_out_fts, bias=False)
        self.lin_neigh = nn.Linear(specs.node_in_fts, specs.node_out_fts, bias=False)
        self.lin_pool = nn.Linear(specs.node_in_fts, specs.node_in_fts, bias=True)

        if specs.neigh_agg_mode in ["gru", "lstm", "rnn"]:
            if specs.neigh_agg_mode == "lstm":
                self.rnn = nn.LSTM(
                    specs.node_in_fts,  # config.node_num * node_in_w_head,
                    specs.hidden_dim,  # gru_hidden,
                    batch_first=True,
                )
            elif specs.neigh_agg_mode == "gru":
                self.rnn = nn.GRU(
                    specs.node_in_fts,
                    specs.hidden_dim,
                    batch_first=True,
                )
            else:
                self.rnn = nn.RNN(
                    specs.node_in_fts,
                    specs.hidden_dim,
                    batch_first=True,
                )

        self.normalize = specs.normalize
        self.update_activation = getattr(nn, specs.activation_update)(0.2)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.lin_self.weight, gain=1.414)
        nn.init.xavier_uniform_(self.lin_neigh.weight, gain=1.414)
        nn.init.xavier_uniform_(self.lin_pool.weight, gain=1.414)

    def forward(self, inputs: List[torch.Tensor]):
        # TODO: Move around activation to test performance
        node_fts, edge_fts, edges = inputs
        # node_fts: [total_num_nodes_in_batch, node_in_fts]
        # edge_fts: [total_num_edges_in_batch, edge_in_fts]
        # edges: [2, total_num_edges_in_batch]

        # node_fts = torch.squeeze(node_fts)
        # edge_fts = torch.squeeze(edge_fts)
        edge_fts_undirected = torch.cat([edge_fts, edge_fts], dim=0)

        # edges = torch.squeeze(edges)
        edges_undirected = torch.cat([edges, edges.flip(0)], dim=1)

        # Get the source and destination nodes of the edges
        src_node_fts = self.lin_self(
            node_fts
        )  # self.update_activation(self.lin_self(node_fts))
        dst_node_fts = node_fts[edges_undirected[1]]

        # src_node_fts_transformed = self.W_root(src_node_fts)
        # dst_node_fts_transformed = self.activation_proj(self.W_proj(dst_node_fts))

        # Neighborhood Aggregation Step
        if self.node_agg_mode == "mean":
            # MEAN over neighbors, apply linear transformation and finally the activation
            dst_node_fts_neigh_agg = scatter(
                dst_node_fts,
                edges_undirected[0],
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
                edges_undirected[0],
                dim=0,
                reduce="max",
            )
        else:
            # We form sequences where each sequence is the neighborhood of a node
            # ordered by edge weight. Then each sequence is padded to the max
            # neighborhood size and passed through an RNN.
            seqs, lenghts = [], []
            for node_idx in range(node_fts.shape[0]):
                seqs.append(
                    dst_node_fts[
                        torch.argsort(
                            edge_fts_undirected[:, 0][edges_undirected[0] == node_idx],
                            descending=True,
                        )
                    ]
                )
                lenghts.append(len(seqs[-1]))
            lenghts = torch.tensor(lenghts)
            seqs = pad_sequence(seqs, batch_first=True)
            packed = nn.utils.rnn.pack_padded_sequence(
                seqs, lenghts.to("cpu"), batch_first=True, enforce_sorted=False
            )
            # h_n: (1, N, hidden_dim)
            # where N = batch_size * seq_len
            # where seq_len = max neighborhood size
            _, (dst_node_fts_neigh_agg_final, _) = self.rnn(packed)
            # dst_node_fts_neigh_agg_final = self.update_activation(
            #    dst_node_fts_neigh_agg
            # )

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
