from dataclasses import dataclass


@dataclass
class DiscriminatorConfig:
    input_size = 19 * 19
    hidden_size = 256
