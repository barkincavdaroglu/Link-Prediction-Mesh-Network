from dataclasses import dataclass, asdict
from typing import Tuple


@dataclass
class DiscriminatorConfig:
    input_size: int = 19 * 19
    hidden_size: int = 256
    loss_module: Tuple[str, str] = ("BCELoss", "mean")

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
