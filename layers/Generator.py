import torch
import torch.nn as nn
from layers.GraphGRU import GraphGRU
from layers.GNBlock import GNBlock
from configs.GeneratorConfig import GeneratorConfig


class Generator(nn.Module):
    def __init__(self, config: GeneratorConfig, gnn: nn.Module):
        super(Generator, self).__init__()
        node_in_w_head = config.node_out_fts

        if config.model == "attention_heads":
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

        self.gnn = GNBlock(
            config,
            gnn,
        )
        self.hidden_dim = config.gru_hidden

        # TODO: Add BatchNorm1d before passing to self.rnn
        self.batchnorm1d = nn.BatchNorm1d(config.node_num * node_in_w_head)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.rnn = nn.GRU(
            config.node_num * node_in_w_head,
            config.gru_hidden,
            batch_first=True,
        )

        self.sequence_length = config.sequence_length
        self.batch_size = config.batch_size

        self.ffn = nn.Sequential(
            nn.Linear(
                config.gru_hidden,
                int(config.node_num * config.node_num),
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        edges: torch.Tensor,
        node_fts: torch.Tensor,
        edge_fts: torch.Tensor,
        graph_fts: torch.Tensor,
    ):
        """
        :param edges: LongTensor (2, batch_size * sequence_length * edge_num)
        :param node_fts: FloatTensor (batch_size * sequence_length * node_num, node_in_fts)
        :param edge_fts: FloatTensor (batch_size * sequence_length * edge_num, edge_in_fts)
        :param graph_fts: FloatTensor (batch_size * sequence_length, graph_in_fts)
        :return output: FloatTensor (batch_size, node_num * node_num)
        """
        _, _, all_node_fts, edge_fts, edges = self.gnn(node_fts, edge_fts, edges)

        all_node_fts = all_node_fts.view(
            -1, self.sequence_length, self.node_num * self.out_features
        )
        h0 = torch.zeros(1, self.batch_size, self.hidden_dim).requires_grad_()

        _, hn = self.rnn(all_node_fts, h0.detach())
        hn = hn.squeeze()

        # TODO: Investigate whether this would explode gradients
        # hn = self.leaky_relu(hn)

        output = self.ffn(hn)
        return output
