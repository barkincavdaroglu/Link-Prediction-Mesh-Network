from torch import nn
from layers.Generator import Generator
from layers.LossModule import LossModule
from layers.Discriminator import Discriminator
from configs.GeneratorConfig import GeneratorConfig
from configs.TrainerConfig import TrainerConfig
from configs.DataConfig import DataConfig
from configs.DiscriminatorConfig import DiscriminatorConfig
from pytorch_lightning import Trainer
from LightningModule import GraphLightningModule
from data_modules.data_module import GraphDataModule
from pytorch_lightning.loggers import NeptuneLogger, WandbLogger
from dotenv import load_dotenv
import os

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

wandb_logger = WandbLogger(project="test-project")

pretrain_loss_mse = LossModule(
    nn.MSELoss(reduction="sum")  # nn.MSELoss(reduction="sum")
)  # nn.L1Loss(reduction="sum")  # nn.MSELoss(reduction="sum")
gan_loss_bce = nn.BCELoss()

generator_config = GeneratorConfig()
discriminator_config = DiscriminatorConfig()
trainer_config = TrainerConfig()
data_config = DataConfig()

PARAMS = {
    "model_config": generator_config.dict(),
    "trainer_config": trainer_config.dict(),
    "data_config": data_config.dict(),
}

generator = Generator(generator_config)
discriminator = Discriminator(discriminator_config)

gd_module = GraphDataModule(data_config=data_config)
pl_model = GraphLightningModule(
    generator,
    discriminator,
    pretrain_loss_mse,
    gan_loss_bce,
    trainer_config.lr,
    trainer_config.is_clip_grads,
    trainer_config.gradient_clip_val,
    trainer_config.gradient_clip_algorithm,
    trainer_config.pretrain_epochs,
    trainer_config.pretrain_epochs + trainer_config.gan_epochs,
)

wandb_logger.watch(pl_model, log="all", log_freq=100)


lr_logger = LearningRateMonitor(logging_interval="epoch")

# create model checkpointing object
model_checkpoint = ModelCheckpoint(
    dirpath="model_ckpts1/checkpoints/",
    filename="{epoch:02d}",
    save_weights_only=True,
    save_top_k=2,
    save_last=True,
    monitor="val/loss",
    every_n_epochs=1,
)

trainer = Trainer(
    logger=wandb_logger,
    callbacks=[lr_logger, model_checkpoint],
    max_epochs=trainer_config.pretrain_epochs + trainer_config.gan_epochs,
    track_grad_norm=trainer_config.track_grad_norm,  # track gradient norm
    # gradient_clip_val=trainer_config.gradient_clip_val,
    # gradient_clip_algorithm=trainer_config.gradient_clip_algorithm,
)

neptune_logger.log_model_summary(model=pl_model, max_depth=-1)
neptune_logger.log_hyperparams(params=PARAMS)

trainer.fit(pl_model, datamodule=gd_module)
