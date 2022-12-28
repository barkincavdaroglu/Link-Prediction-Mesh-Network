import torch
import torch.nn as nn
from layers.GraphGRU import GraphGRU
from layers.GNBlock import GNBlock


class Generator(nn.Module):
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
        node_num,
        in_features=0,
        out_features=0,
        gru_hidden=128,
        head_agg_mode="mean",
    ):
        """ """
        super(Generator, self).__init__()
        node_in_w_head = 0
        if head_agg_mode == "concat":
            node_in_w_head = (
                num_heads_node * node_out_fts + num_heads_node * edge_out_fts
            )
        else:
            node_in_w_head = node_out_fts + edge_out_fts

        self.node_num = node_num

        self.in_features = in_features
        self.out_features = node_in_w_head

        self.gn = GNBlock(
            graph_in_fts=graph_in_fts,
            graph_out_fts=graph_out_fts,
            node_in_fts=node_in_fts,
            node_out_fts=node_out_fts,
            edge_in_fts=edge_in_fts,
            edge_out_fts=edge_out_fts,
            num_heads_node=num_heads_node,
            num_heads_graph=num_heads_graph,
        )

        self.gru = nn.GRU(
            node_num * node_in_w_head,
            gru_hidden,
        )

        """self.graph_gru = GraphGRU(
            input_size_1=in_features,
            hidden_size_1=in_features,
            input_size_2=in_features * 2,
            hidden_size_2=in_features,
            input_size_3=in_features,
            hidden_size_3=in_features,
        )"""

        self.ffn = nn.Sequential(
            nn.Linear(gru_hidden, node_num * node_num), nn.Sigmoid()
        )

    def forward(self, inputs):
        """
        :param input: FloatTensor ()
        :return output: FloatTensor (node_num * node_num)
        """
        all_node_fts = torch.tensor([])
        for input_ in inputs:
            edges, node_fts, edge_fts, graph_fts, adj = input_

            agg_node_fts, agg_edge_fts, node_fts, edge_fts, edges = self.gn(
                (node_fts, edge_fts, edges, adj)
            )

            all_node_fts = torch.cat((all_node_fts, node_fts.unsqueeze(0)), dim=0)

        all_node_fts = all_node_fts.view(-1, self.node_num * self.out_features)
        _, hn = self.gru(all_node_fts)

        output = self.ffn(hn)
        return output
