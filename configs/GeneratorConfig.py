from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    graph_in_fts = 9
    graph_out_fts = 64
    node_in_fts = 7
    node_out_fts = 64
    edge_in_fts = 2
    edge_out_fts = 32
    num_heads_node = 5
    num_heads_graph = 5
    node_num = 19
    in_features = 0
    out_features = 0
    gru_hidden = 128
    head_agg_mode = "mean"
