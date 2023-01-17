from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric.data import LightningDataset
from configs.GeneratorConfig import GeneratorConfig
from configs.GANConfig import GANConfig
from configs.TrainerConfig import TrainerConfig
from LightningModule import GraphLightningModule
from pytorch_lightning import Trainer
import torch_geometric.transforms as T

from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="test-project")

loader = METRLADatasetLoader()

diff_transform = T.GDC(
    self_loop_weight=1,
    normalization_in="sym",
    normalization_out="col",
    diffusion_kwargs=dict(method="ppr", alpha=0.05),
    sparsification_kwargs=dict(method="topk", k=128, dim=0),
    exact=True,
)
dataset = loader.get_dataset()

gen_config = GeneratorConfig()

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
train_dataset, val_dataset = temporal_signal_split(train_dataset, train_ratio=0.8)

train_samples = [x for x in train_dataset]
val_samples = [x for x in val_dataset]
test_samples = [x for x in test_dataset]

lightning_datamodel = LightningDataset(
    train_samples, val_samples, batch_size=gen_config.batch_size
)

gan_config = GANConfig()
trainer_config = TrainerConfig()
PARAMS = {
    "model_config": gan_config.dict(),
    "trainer_config": trainer_config.dict(),
}

pl_model = GraphLightningModule(
    gan_config,
    trainer_config.lr,
    trainer_config.is_clip_grads,
    trainer_config.gradient_clip_val,
    trainer_config.gradient_clip_algorithm,
    trainer_config.pretrain_epochs,
)

wandb_logger.watch(pl_model, log="all", log_freq=10)

trainer = Trainer(
    logger=wandb_logger,
    max_epochs=trainer_config.pretrain_epochs + trainer_config.gan_epochs,
    track_grad_norm=trainer_config.track_grad_norm,  # track gradient norm
)

trainer.fit(pl_model, datamodule=lightning_datamodel)
