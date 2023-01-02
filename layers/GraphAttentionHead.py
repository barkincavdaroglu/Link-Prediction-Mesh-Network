import torch
import torch.nn as nn
from utils.helpers import *


class GraphAttentionHead(nn.Module):
    def __init__(
        self,
        graph_in_fts,
        graph_out_fts,
        node_in_fts,
        node_out_fts,
        edge_in_fts,
        edge_out_fts,
        alpha=0.2,
        kernel_init=nn.init.xavier_uniform_,
        kernel_reg=None,
    ):
        super().__init__()
        self.graph_in_fts = graph_in_fts
        self.graph_out_fts = graph_out_fts

        self.node_in_fts = node_in_fts
        self.node_out_fts = node_out_fts

        self.edge_in_fts = edge_in_fts
        self.edge_out_fts = edge_out_fts

        self.kernel_init = kernel_init
        self.kernel_reg = kernel_reg

        self.W_graph = nn.Parameter(torch.tensor(graph_in_fts, graph_out_fts))
        self.W_node = nn.Parameter(torch.Tensor(node_in_fts, node_out_fts))
        self.W_edge = nn.Parameter(torch.Tensor(edge_in_fts, edge_out_fts))
        self.kernel_init(self.W_graph)
        self.kernel_init(self.W_node)
        self.kernel_init(self.W_edge)

        self.a_node = nn.Parameter(torch.Tensor(node_out_fts + graph_out_fts, 1))
        self.a_edge = nn.Parameter(torch.Tensor(edge_out_fts + graph_out_fts, 1))
        self.kernel_init(self.a_node, gain=1.414)
        self.kernel_init(self.a_edge, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def reset_parameters(self):
        self.kernel_init(self.W_graph)
        self.kernel_init(self.W_node)
        self.kernel_init(self.W_edge)

        self.kernel_init(self.a_graph)
        self.kernel_init(self.a_node)
        self.kernel_init(self.a_edge)

    def forward(self, inputs):
        node_fts, edge_fts, graph_fts, edges = inputs

        h_v = torch.mm(node_fts, self.W_node)
        e_v = torch.mm(edges, self.W_edge)
        g_v = torch.mm(graph_fts, self.W_graph)

        # create expanded version of g_v for node and edge where we append each h_v and e_v to g_v
        g_h_v_expanded = g_v.repeat(node_fts.shape[0], 1)
        g_h_v_expanded = torch.cat([g_h_v_expanded, h_v], dim=1)

        g_e_v_expanded = g_v.repeat(edge_fts.shape[0], 1)
        g_e_v_expanded = torch.cat([g_e_v_expanded, e_v], dim=1)

        node_attention = self.leakyrelu(
            torch.matmul(g_h_v_expanded, self.a_node)
        ).squeeze(-1)

        edge_attention = self.leakyrelu(
            torch.matmul(g_e_v_expanded, self.a_edge)
        ).squeeze(-1)

        # normalize
        node_attention = torch.exp(torch.clamp(node_attention, -2, 2))
        node_attention_sum = unsorted_segment_sum(
            node_attention, edges[:, 0], torch.unique(edges[:, 0]).shape[0]
        )
        node_attention_sum = node_attention_sum.repeat(torch.bincount(edges[:, 0]))
        node_attention_norm = node_attention / node_attention_sum

        edge_attention = torch.exp(torch.clamp(edge_attention, -2, 2))
        edge_attention_sum = unsorted_segment_sum(
            edge_attention, edges[:, 0], torch.unique(edges[:, 0]).shape[0]
        )
        edge_attention_sum = edge_attention_sum.repeat(torch.bincount(edges[:, 0]))
        edge_attention_norm = edge_attention / edge_attention_sum

        return node_attention_norm, edge_attention_norm
