import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import unsorted_segment_sum


class NodeAttentionHead(nn.Module):
    def __init__(
        self,
        node_in_fts,
        node_out_fts,
        edge_in_fts,
        edge_out_fts,
        kernel_init,
        kernel_reg,
    ):
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
        self.kernel_init(self.a_node)
        self.kernel_init(self.a_edge)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def reset_parameters(self):
        self.kernel_init(self.W_node)
        self.kernel_init(self.W_edge)
        self.kernel_init(self.a_node)
        self.kernel_init(self.a_edge)

    def forward(self, inputs):
        node_fts, edge_fts, edges = inputs

        h_v = torch.mm(node_fts, self.W_node)
        e_v = torch.mm(edge_fts, self.W_edge)

        h_v_expanded = h_v[
            edges[
                :,
            ]
        ].reshape(-1, 2 * self.node_out_fts)
        e_v_expanded = e_v[
            edges[
                :,
            ]
        ].reshape(-1, self.node_out_fts + self.edge_out_fts)

        node_attention = self.leakyrelu(torch.matmul(h_v_expanded, self.a_node))
        node_attention = torch.squeeze(node_attention, -1)

        edge_attention = self.leakyrelu(torch.matmul(e_v_expanded, self.a_edge))
        edge_attention = torch.squeeze(edge_attention, -1)

        # normalize attention scores using softmax
        node_attention = torch.exp(torch.clamp(node_attention, -2, 2))
        node_attention_sum = unsorted_segment_sum(
            node_attention, edges[:, 0], torch.unique(edges[:, 0]).shape[0]
        )
        node_attention_sum = node_attention_sum.repeat(torch.bincount(edges[:, 0]))
        node_attention_norm = node_attention / node_attention_sum

        node_attention_var = torch.var(node_attention_norm)

        edge_attention = torch.exp(torch.clamp(edge_attention, -2, 2))
        edge_attention_sum = unsorted_segment_sum(
            edge_attention, edges[:, 0], torch.unique(edges[:, 0]).shape[0]
        )
        edge_attention_sum = edge_attention_sum.repeat(torch.bincount(edges[:, 0]))
        edge_attention_norm = edge_attention / edge_attention_sum

        edge_attention_var = torch.var(edge_attention_norm)

        # aggregate
        node_ft_neighbors = h_v[
            edges[
                :,
            ]
        ].reshape(-1, 2 * self.node_out_fts)
        node_out = unsorted_segment_sum(
            node_ft_neighbors * node_attention_norm.unsqueeze(1),
            edges[:, 0],
            torch.unique(edges[:, 0]).shape[0],
        )

        edge_ft_neighbors = e_v[
            edges[
                :,
            ]
        ].reshape(-1, self.node_out_fts + self.edge_out_fts)
        edge_out = unsorted_segment_sum(
            edge_ft_neighbors * edge_attention_norm.unsqueeze(1),
            edges[:, 0],
            torch.unique(edges[:, 0]).shape[0],
        )

        return node_out, edge_out, node_attention_var, edge_attention_var
