from dataclasses import dataclass, asdict


@dataclass
class GeneratorConfig:
    # Dimension of input graph features
    graph_in_fts: int = 9
    # Dimension of output graph features
    graph_out_fts: int = 64
    # Dimension of input node features
    node_in_fts: int = 6
    # Dimension of output node features
    node_out_fts: int = 64
    # Dimension of input edge features
    edge_in_fts: int = 2
    # Dimension of output edge features
    edge_out_fts: int = 32
    # Number of attention heads for node update layer
    num_heads_node: int = 1
    # Number of attention heads for graph update layer
    num_heads_graph: int = 5
    # Number of nodes in the graph
    node_num: int = 19
    # Dimension of the hidden state of the GRU
    gru_hidden: int = 128
    # The mode with which to aggregate the heads: "weighted_mean", "sum", "concat"
    head_agg_mode: str = "weighted_mean"
    # Number of hops we want to aggregate information for
    nr_of_hops: int = 1
    # The mode with which to aggregate each node: "concat" or "sum"
    node_agg_mode: str = "concat"
    # Negative slope of the LeakyReLU activation function.
    alpha: float = 0.2
    # If True, pass edges through a linear layer before node attention aggregation.
    update_edge_first: bool = False
    # Mode for residual connections: "add", "concat", "gated"
    residual_mode: str = "add"
    # If True, learn the scale of the message passing norm
    messagenorm_learn_scale: bool = False

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
