import torch
import torch.nn as nn
from layers.GraphGRU import GraphGRU
from layers.GNBlock import GNBlock
from configs.GeneratorConfig import GeneratorConfig


class Generator(nn.Module):
    def __init__(self, config: GeneratorConfig):
        """ """
        super(Generator, self).__init__()

        if config.head_agg_mode == "concat":
            node_in_w_head = (
                (
                    config.num_heads_node * config.node_out_fts
                    + config.num_heads_node * config.edge_out_fts
                )
                * 2
                if config.node_agg_mode == "concat"
                else config.num_heads_node * config.node_out_fts
                + config.num_heads_node * config.edge_out_fts
            )
        else:
            node_in_w_head = (
                config.node_out_fts * 2
                if config.node_agg_mode == "concat"
                else config.node_out_fts
            )

        self.node_num = config.node_num

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
            node_agg_mode=config.node_agg_mode,
            residual_mode=config.residual_mode,
            messagenorm_learn_scale=config.messagenorm_learn_scale,
            head_agg_mode=config.head_agg_mode,
            nr_of_hops=config.nr_of_hops,
        )
        self.hidden_dim = config.gru_hidden

        # TODO: Add BatchNorm1d before passing to self.rnn

        """self.rnn = nn.LSTM(
            config.node_num * node_in_w_head,
            config.gru_hidden,
        )"""

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.rnn = nn.GRU(
            config.node_num * node_in_w_head,
            config.gru_hidden,
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                config.gru_hidden,
                # int((config.node_num * config.node_num - config.node_num) / 2),
                int(config.node_num * config.node_num),
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

            _, _, node_fts, edge_fts, edges = self.gn((node_fts, edge_fts, edges, adj))

            all_node_fts = torch.cat((all_node_fts, node_fts.unsqueeze(0)), dim=0)

        all_node_fts = all_node_fts.view(-1, self.node_num * self.out_features)
        h0 = torch.zeros(1, self.hidden_dim).requires_grad_()

        _, hn = self.rnn(all_node_fts, h0.detach())

        # hn = self.leaky_relu(hn)

        output = self.ffn(hn)
        return output
