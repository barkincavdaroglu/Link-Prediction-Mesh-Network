import torch
import pytorch_lightning as pl
import numpy as np


class GraphLightningModule(pl.LightningModule):
    def __init__(self, model, loss_module):
        super().__init__()
        self.model = model
        self.loss_module = loss_module

    def forward(self, x):
        """
        Use for inference only (separate from training_step)
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        adj_predicted = self(x)
        _, _, _, _, target_adj = y

        target_adj = target_adj.squeeze()
        adj_predicted = adj_predicted.squeeze()

        upper_target_adj = target_adj[np.triu_indices(19, 1)]

        # target_adj = target_adj.view(1, -1)
        loss = self.loss_module(adj_predicted, upper_target_adj)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        adj_predicted = self(x)
        _, _, _, _, target_adj = y
        target_adj = target_adj.squeeze()
        adj_predicted = adj_predicted.squeeze()
        upper_target_adj = target_adj[np.triu_indices(19, 1)]
        # target_adj = target_adj.view(1, -1)
        loss = self.loss_module(adj_predicted, upper_target_adj)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        sequence, target_adj = batch
        predicted_adj = self(sequence)
        loss = self.loss_module(predicted_adj, target_adj)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optimizer = torch.optim.RMSprop(self.parameters(), lr=0.002)
        return optimizer
