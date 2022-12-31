from cmath import isnan
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import *
from .MessageNorm import MessageNorm


class NodeAttentionHead(nn.Module):
    def __init__(
        self,
        node_in_fts,
        node_out_fts,
        edge_in_fts,
        node_agg_mode,
        alpha=0.2,
        kernel_init=nn.init.xavier_uniform_,
        kernel_reg=None,
    ):
        """
        Aggregates and updates node features using attention mechanism.

        Args:
            node_in_fts: Dimensionality of node input features
            node_out_fts: Dimensionality of node output features
            edge_in_fts: Dimensionality of edge input features
            edge_out_fts: Dimensionality of edge output features
            alpha: Negative slope of the LeakyReLU activation
            kernel_init: Kernel Initializer function, default is xavier_uniform_
            kernel_reg: Kernel Regularizer function, default is None
        """
        super().__init__()
        self.node_in_fts = node_in_fts
        self.node_out_fts = node_out_fts
        self.edge_in_fts = edge_in_fts
        self.node_agg_mode = node_agg_mode

        self.message_norm = MessageNorm(learn_scale=True)

        self.kernel_init = kernel_init
        self.kernel_reg = kernel_reg

        self.W_node = nn.Linear(node_in_fts, node_out_fts)
        self.a_node = nn.Parameter(torch.Tensor(2 * node_out_fts + edge_in_fts, 1))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.W_node.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_node, gain=1.414)

    def forward(self, inputs: List[torch.Tensor]):
        """
        Args:
            inputs: List of inputs [node_fts, edge_fts, edges]
        """
        node_fts, edge_fts, edges = inputs

        node_fts = torch.squeeze(node_fts)
        edge_fts = torch.squeeze(edge_fts)
        edge_fts_undirected = torch.cat([edge_fts, edge_fts], dim=0)

        edges = torch.squeeze(edges)
        edges = edges.reshape(edges.shape[1], 2)
        edges_undirected = torch.cat([edges, edges.flip(1)], dim=0)

        # Transform initial node features
        h_v = self.W_node(node_fts)  # torch.mm(node_fts, self.W_node)

        # Each row will contain the tensor:
        # [node_fts[edge[0]], node_fts[edge[1]], edge_fts for edge]
        # with dimension (2 * node_out_fts + edge_in_fts)
        h_v_expanded = torch.cat(
            (
                h_v[edges_undirected].view(-1, 2 * self.node_out_fts),
                edge_fts_undirected,
            ),
            dim=1,
        )

        # Compute attention scores
        node_attention = self.leakyrelu(torch.matmul(h_v_expanded, self.a_node))

        # normalize attention scores using softmax
        node_attention = torch.exp(node_attention)

        # For each node, sum the attention scores of all its neighbors
        node_attention_sum = unsorted_segment_sum(
            node_attention,
            edges_undirected[:, 0],
            torch.unique(edges_undirected[:, 0]).shape[0],
        )
        # Repeat the sum for each neighbor of the node
        node_attention_sum = node_attention_sum[edges_undirected[:, 0]]

        # Normalize attention scores by dividing each by neighborhood sum
        node_attention_norm = torch.log(node_attention / node_attention_sum)

        # print("NORMALIZED ATT: ", node_attention_norm, "\n")
        node_attention_var = torch.var(node_attention_norm)

        # Get the node features of neighbors for all nodes
        node_ft_neighbors = h_v[edges_undirected[:, 1]]

        # Sum the node features of neighbors weighted by attention scores
        agg_message = unsorted_segment_sum(
            node_ft_neighbors * node_attention_norm,  # .unsqueeze(1),
            edges_undirected[:, 0],
            torch.unique(edges_undirected[:, 0]).shape[0],
        )

        agg_message_normalized = self.message_norm(h_v, agg_message)

        final_embed = (
            torch.cat((h_v, agg_message_normalized), dim=1)
            if self.node_agg_mode == "concat"
            else h_v + agg_message_normalized
        )

        return (
            final_embed,
            node_attention_var,
        )  # self.message_norm(h_v, agg_message), node_attention_var
