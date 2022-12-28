from dataclasses import dataclass


@dataclass
class GNConfig:
    num_classes: int = 2
    num_layers: int = 3
    num_channels: int = 64
    num_heads: int = 8
    num_edge_features: int = 1
    num_node_features: int = 1
    num_global_features: int = 1
    dropout: float = 0.1
    edge_dim: int = 1
    node_dim: int = 1
    global_dim: int = 1
    edge_hidden_dim: int = 64
    node_hidden_dim: int = 64
    global_hidden_dim: int = 64
    edge_output_dim: int = 64
    node_output_dim: int = 64
    global_output_dim: int = 64
    edge_activation: str = "relu"
    node_activation: str = ""
