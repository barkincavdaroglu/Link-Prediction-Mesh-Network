import torch
import torch.nn as nn

from layers.Discriminator import Discriminator
from layers.Generator import Generator
from layers.GraphConv import GraphConvolution
from configs.GeneratorConfig import GeneratorConfig
from configs.DiscriminatorConfig import DiscriminatorConfig
from configs.GANConfig import GANConfig
from torch_geometric.nn import GATConv


def model_adapter(model_string: str) -> nn.Module:
    """
    Args:
        model_string: Name of model to be used
    """
    if model_string == "attention_heads":
        return GATConv
    elif model_string == "graph_conv":
        return GraphConvolution
    else:
        raise ValueError("Model not found")


def create_generator_model(config: GeneratorConfig) -> nn.Module:
    """
    Args:
        config: GeneratorConfig object
    """
    model_string = config.model
    if model_string == "attention_heads":
        sub_model = GATConv(
            config.node_in_fts, config.node_out_fts, config.num_heads_node
        )
    elif model_string == "graph_conv":
        sub_model = GraphConvolution(config)
    else:
        raise ValueError("Model not found")
    model = Generator(config, sub_model)
    return model


def create_discriminator_model(config: DiscriminatorConfig) -> nn.Module:
    """
    Args:
        config: DiscriminatorConfig object
    """
    model = Discriminator(config)

    return model
