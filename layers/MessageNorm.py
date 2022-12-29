import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


class MessageNorm(torch.nn.Module):
    """
    "DeeperGCNs: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>
    """

    def __init__(self, learn_scale: bool):
        super().__init__()

        self.scale = Parameter(torch.Tensor([1.0]), requires_grad=learn_scale)

        self.reset_parameters()

    def reset_parameters(self):
        self.scale.data.fill_(1.0)

    def forward(self, x: Tensor, msg: Tensor, p: float = 2.0) -> Tensor:
        msg = F.normalize(msg, p=p, dim=-1)
        x_norm = x.norm(p=p, dim=-1, keepdim=True)
        return msg * x_norm * self.scale
