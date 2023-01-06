import torch
import torch.nn as nn

from layers.Discriminator import Discriminator
from layers.Generator import Generator
from layers.MultiHeadNodeAttention import MultiHeadNodeAttention
from layers.GraphConv import GraphConvolution
from configs.GeneratorConfig import GeneratorConfig
from configs.DiscriminatorConfig import DiscriminatorConfig
from configs.GANConfig import GANConfig
from layers.NodeAttention import NodeAttentionHead
from layers.NodeAttentionv2 import NodeAttentionHeadv2


def model_adapter(model_string: str) -> nn.Module:
    """
    Args:
        model_string: Name of model to be used
    """
    if model_string == "attention_heads":
        return MultiHeadNodeAttention
    elif model_string == "graph_conv":
        return GraphConvolution
    else:
        raise ValueError("Model not found")


def create_generator_model(config: GeneratorConfig) -> nn.Module:
    """
    Args:
        model_string: Name of model to be imported
    """
    model_string = config.model
    if model_string == "attention_heads":
        attention_head_type = config.attention_head_type
        if attention_head_type == "v1":
            attention_head = NodeAttentionHead
        elif attention_head_type == "v2":
            attention_head = NodeAttentionHeadv2
        else:
            raise ValueError("Attention head type not found")
        sub_model = MultiHeadNodeAttention(config, attention_head)
    elif model_string == "graph_conv":
        sub_model = GraphConvolution(config)
    else:
        raise ValueError("Model not found")
    model = Generator(config, sub_model)
    return model


def create_discriminator_model(config: DiscriminatorConfig) -> nn.Module:
    """
    Args:
        model_string: Name of model to be imported
    """
    model = Discriminator(config)

    return model
