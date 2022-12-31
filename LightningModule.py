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

        # upper_target_adj = target_adj[np.triu_indices(19, 1)]

        target_adj = target_adj.view(1, -1).squeeze()
        loss = self.loss_module(adj_predicted, target_adj)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = np.array([])
        for result in outputs:
            loss = np.append(loss, result["loss"])

        self.log("train/epoch/loss", loss.mean())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        adj_predicted = self(x)
        _, _, _, _, target_adj = y
        target_adj = target_adj.squeeze()
        adj_predicted = adj_predicted.squeeze()
        # upper_target_adj = target_adj[np.triu_indices(19, 1)]
        target_adj = target_adj.view(1, -1).squeeze()
        loss = self.loss_module(adj_predicted, target_adj)
        # self.log("val/loss", loss)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        loss = np.array([])
        for result in outputs:
            loss = np.append(loss, result["loss"])

        self.log("val/loss", loss.mean())

    def test_step(self, batch, batch_idx):
        sequence, target_adj = batch
        predicted_adj = self(sequence)
        loss = self.loss_module(predicted_adj, target_adj)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=0.0005)
        return optimizer
