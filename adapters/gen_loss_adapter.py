import torch
import torch.nn as nn
from configs.GeneratorConfig import GeneratorConfig
from layers.GeneratorLossModule import GeneratorLossModule


def create_gen_loss_module(config: GeneratorConfig) -> nn.Module:
    """
    Args:
        model_string: Name of model to be imported
    """
    gen_loss_func_, gen_reduction = config.loss_module

    try:
        gen_loss_func = getattr(nn, gen_loss_func_)(reduction=gen_reduction)
    except AttributeError:
        raise AttributeError(
            f"Loss function {gen_loss_func_} not found in torch.nn. Please use one of the function names in https://pytorch.org/docs/stable/nn.html#loss-functions"
        )

    return GeneratorLossModule(gen_loss_func)
