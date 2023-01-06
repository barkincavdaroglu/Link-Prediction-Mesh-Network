import torch
import torch.nn as nn
from configs.DiscriminatorConfig import DiscriminatorConfig
from layers.DiscriminatorLossModule import DiscriminatorLossModule


def create_disc_loss_module(config: DiscriminatorConfig) -> nn.Module:
    """
    Args:
        config: DiscriminatorConfig object
    """
    disc_loss_func_, disc_reduction = config.loss_module

    try:
        disc_loss_func = getattr(nn, disc_loss_func_)(reduction=disc_reduction)
    except AttributeError:
        raise AttributeError(
            f"Loss function {disc_loss_func_} not found in torch.nn. Please use one of the function names in https://pytorch.org/docs/stable/nn.html#loss-functions"
        )

    return DiscriminatorLossModule(disc_loss_func)
