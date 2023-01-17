from dataclasses import dataclass, asdict
from typing import Tuple
import torch.nn as nn


@dataclass
class GeneratorConfig:
    horizon: int = 3
    batch_size: int = 64
    sequence_length: int = 12
    # Dimension of input graph features
    graph_in_fts: int = 2
    # Dimension of output graph features
    graph_out_fts: int = 32
    # Dimension of input node features
    node_in_fts: int = 2
    # Dimension of output node features
    node_out_fts: int = 32
    # Dimension of input edge features
    edge_in_fts: int = 1
    # Dimension of output edge features
    edge_out_fts: int = 4
    # Number of attention heads for graph update layer
    num_heads_graph: int = 5
    # Number of nodes in the graph
    node_num: int = 207
    # Dimension of the hidden state of the GRU
    gru_hidden: int = 207
    # Number of hops we want to aggregate information for
    nr_of_hops: int = 2
    # The mode with which to aggregate each node: "concat" or "sum"
    node_agg_mode: str = "sum"
    # Negative slope of the LeakyReLU activation function.
    alpha: float = 0.2
    # If True, pass edges through a linear layer before node attention aggregation.
    update_edge_first: bool = True
    # Mode for residual connections: "add", "concat", "gated"
    residual_mode: str = "add"
    #
    loss_module: Tuple[str, str] = ("MSELoss", "sum")
    model: str = "attention_heads"  # one of attention_heads, graph_conv

    ### Attention Config ###
    # Number of attention heads for node update layer
    num_heads_node: int = 4
    # The mode with which to aggregate the heads: "var", "sum", "concat"
    head_agg_mode: str = "var"
    # If True, learn the scale of the message passing norm
    messagenorm_learn_scale: bool = False
    attention_head_type: str = "v1"

    ### GConv Config ###
    # one of mean, max, lstm, gru, rnn
    neigh_agg_mode: str = "lstm"
    # One of "add", "concat
    update_mode: str = "sum"
    # One of the activation functions listed here:
    # https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
    activation_update: str = "LeakyReLU"
    # Hidden dimension for RNN if neigh_agg_mode is "lstm", "gru", or "rnn"
    hidden_dim: int = 64
    # If True, normalize the layer output
    normalize = True
    # If True, pass edges through a linear layer before node attention aggregation.
    dest_transform = False

    kernel_init = nn.init.xavier_uniform_
    kernel_reg = None

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
