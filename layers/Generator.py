import torch
import torch.nn as nn
from layers.GraphGRU import GraphGRU
from layers.GNBlock import GNBlock
from configs.GeneratorConfig import GeneratorConfig


class Generator(nn.Module):
    def __init__(self, config: GeneratorConfig):
        """ """
        super(Generator, self).__init__()
        node_in_w_head = (
            (
                config.num_heads_node * config.node_out_fts
                + config.num_heads_node * config.edge_out_fts
            )
            if config.head_agg_mode == "concat"
            else config.node_out_fts
        )

        self.node_num = config.node_num

        self.in_features = config.in_features
        self.out_features = node_in_w_head

        self.gn = GNBlock(
            graph_in_fts=config.graph_in_fts,
            graph_out_fts=config.graph_out_fts,
            node_in_fts=config.node_in_fts,
            node_out_fts=config.node_out_fts,
            edge_in_fts=config.edge_in_fts,
            edge_out_fts=config.edge_out_fts,
            num_heads_node=config.num_heads_node,
            num_heads_graph=config.num_heads_graph,
            head_agg_mode=config.head_agg_mode,
        )

        self.gru = nn.GRU(
            config.node_num * node_in_w_head,
            config.gru_hidden,
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
            nn.Linear(
                config.gru_hidden,
                int((config.node_num * config.node_num - config.node_num) / 2),
            ),
            nn.Sigmoid(),
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
