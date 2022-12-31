from dataclasses import dataclass
import string


@dataclass
class GeneratorConfig:
    graph_in_fts: int = 9
    graph_out_fts: int = 64
    node_in_fts: int = 6
    node_out_fts: int = 64
    edge_in_fts: int = 2
    edge_out_fts: int = 32
    num_heads_node: int = 1
    num_heads_graph: int = 5
    node_num: int = 19
    in_features: int = 0
    out_features: int = 0
    gru_hidden: int = 64
    head_agg_mode: str = "weighted_mean"
    nr_of_hops: int = 1
    # The mode with which to aggregate each node: "concat" or "sum"
    node_agg_mode: str = "concat"
    # Negative slope of the LeakyReLU activation function.
    alpha: float = 0.2
    # If True, pass edges through a linear layer before node attention aggregation.
    update_edge_first: bool = False
