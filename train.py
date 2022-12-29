from torch import nn
from layers.Generator import Generator
import neptune.new as neptune
from configs.GeneratorConfig import GeneratorConfig
from configs.TrainerConfig import TrainerConfig
from pytorch_lightning import Trainer
from LightningModule import GraphLightningModule
from data_modules.data_module import GraphDataModule
from pytorch_lightning.loggers import NeptuneLogger
from dotenv import load_dotenv
import os

load_dotenv()

NEPTUNE_TOKEN = os.getenv("NEPTUNE_API_TOKEN")
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")

neptune_logger = NeptuneLogger(
    api_key=NEPTUNE_TOKEN,
    project=NEPTUNE_PROJECT,
    tags=["simple", "showcase"],
    log_model_checkpoints=False,
)

mse = nn.MSELoss(reduction="sum")

model_config = GeneratorConfig()
trainer_config = TrainerConfig()

model = Generator(model_config)

gd_module = GraphDataModule()
pl_model = GraphLightningModule(model, mse)

# (neptune) initialize a trainer and pass neptune_logger
trainer = Trainer(
    logger=neptune_logger,
    max_epochs=trainer_config.num_epochs,
)

# (neptune) log hyper-parameters
# neptune_logger.log_hyperparams(params=trainer_config)

# train the model log metadata to the Neptune run
trainer.fit(pl_model, datamodule=gd_module)
