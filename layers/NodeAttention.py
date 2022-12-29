from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import *


class NodeAttentionHead(nn.Module):
    def __init__(
        self,
        node_in_fts,
        node_out_fts,
        edge_in_fts,
        edge_out_fts,
        alpha=0.2,
        kernel_init=nn.init.xavier_uniform_,
        kernel_reg=None,
    ):
        """
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
        self.edge_out_fts = edge_out_fts

        self.kernel_init = kernel_init
        self.kernel_reg = kernel_reg

        self.W_node = nn.Parameter(torch.Tensor(node_in_fts, node_out_fts))
        self.W_edge = nn.Parameter(torch.Tensor(edge_in_fts, edge_out_fts))
        self.kernel_init(self.W_node)
        self.kernel_init(self.W_edge)

        self.a_node = nn.Parameter(torch.Tensor(2 * node_out_fts, 1))
        self.a_edge = nn.Parameter(torch.Tensor(node_out_fts + edge_out_fts, 1))
        self.kernel_init(self.a_node, gain=1.414)
        self.kernel_init(self.a_edge, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

        self.node_agg = nn.Linear(node_out_fts, node_out_fts)
        self.edge_agg = nn.Linear(edge_out_fts, edge_out_fts)
        self.node_edge_agg = nn.Linear(node_out_fts + edge_out_fts, node_out_fts)

    def reset_parameters(self):
        self.kernel_init(self.W_node)
        self.kernel_init(self.W_edge)
        self.kernel_init(self.a_node)
        self.kernel_init(self.a_edge)

    def forward(self, inputs: List[torch.Tensor]):
        """
        Args:
            inputs: List of inputs [node_fts, edge_fts, edges]
        """
        node_fts, edge_fts, edges = inputs

        node_fts = torch.squeeze(node_fts)
        edge_fts = torch.squeeze(edge_fts)
        edge_fts = torch.cat([edge_fts, edge_fts], dim=0)

        edges = torch.squeeze(edges)
        edges = edges.reshape(edges.shape[1], 2)
        edges = torch.cat([edges, edges.flip(1)], dim=0)

        h_v = torch.mm(node_fts, self.W_node)
        e_v = torch.mm(edge_fts, self.W_edge)

        h_v_pairs = h_v[
            edges[
                :,
            ]
        ]

        h_v_expanded = torch.reshape(h_v_pairs, (-1, 2 * self.node_out_fts))

        e_v_expanded = torch.cat([h_v[edges[:, 0]], e_v], dim=1)

        node_attention = self.leakyrelu(torch.matmul(h_v_expanded, self.a_node))
        node_attention = torch.squeeze(node_attention, -1)

        edge_attention = self.leakyrelu(torch.matmul(e_v_expanded, self.a_edge))
        edge_attention = torch.squeeze(edge_attention, -1)

        # normalize attention scores using softmax
        node_attention = torch.exp(torch.clamp(node_attention, -2, 2))
        node_attention_sum = unsorted_segment_sum(
            node_attention, edges[:, 0], torch.unique(edges[:, 0]).shape[0]
        )
        node_attention_sum = node_attention_sum.repeat_interleave(
            torch.bincount(edges[:, 0])
        )
        node_attention_norm = node_attention / node_attention_sum

        node_attention_var = torch.var(node_attention_norm)

        edge_attention = torch.exp(torch.clamp(edge_attention, -2, 2))
        edge_attention_sum = unsorted_segment_sum(
            edge_attention, edges[:, 0], torch.unique(edges[:, 0]).shape[0]
        )
        edge_attention_sum = edge_attention_sum.repeat_interleave(
            torch.bincount(edges[:, 0])
        )
        edge_attention_norm = edge_attention / edge_attention_sum

        edge_attention_var = torch.var(edge_attention_norm)

        # aggregate
        node_ft_neighbors = h_v[edges[:, 1]]
        node_out = unsorted_segment_sum(
            node_ft_neighbors * node_attention_norm.unsqueeze(1),
            edges[:, 0],
            torch.unique(edges[:, 0]).shape[0],
        )

        edge_ft_neighbors = e_v[edges[:, 1]]
        edge_out = unsorted_segment_sum(
            edge_ft_neighbors * edge_attention_norm.unsqueeze(1),
            edges[:, 0],
            torch.unique(edges[:, 0]).shape[0],
        )

        return node_out, edge_out, node_attention_var, edge_attention_var
