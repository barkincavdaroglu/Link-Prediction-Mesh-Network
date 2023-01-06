from dataclasses import dataclass, asdict

from .GeneratorConfig import GeneratorConfig
from .DiscriminatorConfig import DiscriminatorConfig


@dataclass
class GANConfig:
    #
    generator_config: GeneratorConfig = GeneratorConfig()
    discriminator_config: DiscriminatorConfig = DiscriminatorConfig()

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
