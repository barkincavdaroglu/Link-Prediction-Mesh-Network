from typing import Tuple
import torch
import torch.nn as nn


class GraphGRU(nn.Module):
    def __init__(
        self,
        input_size_1,
        hidden_size_1,
        input_size_2,
        hidden_size_2,
        input_size_3,
        hidden_size_3,
    ):
        """
        Args:
            input_size_1: Dimension of node_fts
            hidden_size_1: Dimension of GRU hidden state for node_fts
            input_size_2: Dimension of edge_fts
            hidden_size_2: Dimension of GRU hidden state for edge_fts
            input_size_3: Dimension of graph_fts
            hidden_size_3: Dimension of GRU hidden state for graph_fts
        """
        super(GraphGRU, self).__init__()
        self.gru_node = nn.GRU(input_size_1, hidden_size_1)
        self.gru_edge = nn.GRU(input_size_2, hidden_size_2)
        self.gru_graph = nn.GRU(input_size_3, hidden_size_3)

    def forward(self, edges, gn_output) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            edges: LongTensor (# of edges, 2)
            gn_output: Tuple of (node_fts, edge_fts, graph_fts)
        """
        node_fts, edge_fts, graph_ft = gn_output

        # convert node_fts (node_num, node_num) to (node_num * node_num, )
        node_fts = node_fts.view(-1, node_fts.size(1))

        node_fts, node_hn = self.gru_node(node_fts)

        # convert node_fts (node_num * node_num, ) to (node_num, node_num)
        node_fts = node_fts.view(-1, node_fts.size(1))

        # append node_fts of edge endpoints to each edge_ft
        edge_fts = torch.cat(
            [edge_fts, node_fts[edges[:, 0]]], dim=1
        )  # (edge_num, node_fts_dim + edge_fts_dim)

        # convert edge_fts (edge_num, edge_num) to (edge_num * edge_num, )
        edge_fts = edge_fts.view(-1, edge_fts.size(1))

        edge_fts, _ = self.gru_edge(edge_fts)

        # append edge_fts to graph_fts
        graph_ft = torch.cat([graph_ft, edge_fts], dim=1)

        _, graph_hn = self.gru_graph(graph_ft)

        return node_hn, graph_hn
