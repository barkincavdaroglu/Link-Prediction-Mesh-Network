from dataclasses import dataclass, asdict


@dataclass
class TrainerConfig:
    # Learning rate
    lr: float = 0.002  # 0.0025  # previously 0.0005
    # https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html#gradient-clipping
    is_clip_grads: bool = False
    gradient_clip_val: float = 6.0
    gradient_clip_algorithm: str = "norm"

    track_grad_norm: int = 2
    # Weight decay / L2 Regularization
    weight_decay: float = 0.0005
    # num_epochs: int = 20
    pretrain_epochs: int = 30
    gan_epochs: int = 5

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
