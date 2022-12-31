from torch import nn
from layers.Generator import Generator
from configs.GeneratorConfig import GeneratorConfig
from configs.TrainerConfig import TrainerConfig
from pytorch_lightning import Trainer
from LightningModule import GraphLightningModule
from data_modules.data_module import GraphDataModule
from pytorch_lightning.loggers import NeptuneLogger
from dotenv import load_dotenv
import os
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

load_dotenv()

NEPTUNE_TOKEN = os.getenv("NEPTUNE_API_TOKEN")
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")

neptune_logger = NeptuneLogger(
    api_key=NEPTUNE_TOKEN,
    project=NEPTUNE_PROJECT,
    tags=["simple", "showcase"],
    log_model_checkpoints=True,
)

mse = nn.MSELoss(reduction="sum")

model_config = GeneratorConfig()
trainer_config = TrainerConfig()

model = Generator(model_config)

gd_module = GraphDataModule()
pl_model = GraphLightningModule(model, mse)

lr_logger = LearningRateMonitor(logging_interval="epoch")

# create model checkpointing object
model_checkpoint = ModelCheckpoint(
    dirpath="model_ckpts/checkpoints/",
    filename="{epoch:02d}",
    save_weights_only=True,
    save_top_k=2,
    save_last=True,
    monitor="val/loss",
    every_n_epochs=1,
)

trainer = Trainer(
    logger=neptune_logger,
    callbacks=[lr_logger, model_checkpoint],
    max_epochs=trainer_config.num_epochs,
    track_grad_norm=2,  # track gradient norm
    gradient_clip_val=3,
)

# neptune_logger.log_hyperparams(params=trainer_config)
neptune_logger.log_model_summary(model=pl_model, max_depth=-1)

# train the model log metadata to the Neptune run
trainer.fit(pl_model, datamodule=gd_module)
