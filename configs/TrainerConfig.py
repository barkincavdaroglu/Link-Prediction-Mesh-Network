from dataclasses import dataclass


@dataclass
class TrainerConfig:
    lr: float = 0.002
    weight_decay: float = None
    num_epochs: int = 10
    batch_size: int = 1
    validation_split: float = 0.2
    test_split: float = 0.1
    shuffle_dataset: bool = True
    random_seed: int = 42
