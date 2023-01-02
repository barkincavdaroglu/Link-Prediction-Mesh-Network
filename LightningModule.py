from asyncio.log import logger
from email import generator
import torch
import pytorch_lightning as pl
import numpy as np
import torch.nn as nn


class GraphLightningModule(pl.LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        pretrain_loss_module,
        gan_loss_module,
        lr,
        is_clip_grads,
        gradient_clip_val,
        gradient_clip_algorithm,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.pretrain_loss_module = pretrain_loss_module
        self.gan_loss_module = gan_loss_module
        self.automatic_optimization = False
        self.lr = lr
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.counter = 0
        self.is_clip_grads = is_clip_grads
        self.pretrain_epochs = 5

    def forward(self, x):
        """
        Use for inference only (separate from training_step)
        """
        return self.generator(x)

    def clip_grads(self, model: str):
        if self.is_clip_grads:
            model = getattr(self, model)
            if self.gradient_clip_algorithm == "value":
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), self.gradient_clip_val
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.gradient_clip_val
                )

    def pretrain_step(self, batch, batch_idx):
        x, y = batch

        _, _, pretrain_gen_opt = self.optimizers()
        logger_dict = {}

        pretrain_gen_opt.zero_grad()
        adj_predicted = self.generator(x)
        _, _, _, _, target_adj = y

        target_adj = target_adj.squeeze()
        adj_predicted = adj_predicted.squeeze()

        target_adj = target_adj.view(1, -1).squeeze()
        loss = self.pretrain_loss_module(adj_predicted, target_adj)

        self.manual_backward(loss)
        self.clip_grads("generator")

        pretrain_gen_opt.step()

        logger_dict["pretrain/loss"] = loss

        return logger_dict

    def gan_step(self, batch, batch_idx):
        x, y = batch

        generator_opt, discriminator_opt, _ = self.optimizers()
        logger_dict = {}

        discriminator_opt.zero_grad()
        generator_opt.zero_grad()

        # real data
        _, _, _, _, target_adj = y
        target_adj = target_adj.squeeze().view(1, -1).squeeze()

        # fake data
        adj_predicted = self.generator(x)  # .squeeze()

        # Optimize discriminator
        disc_output_real = self.discriminator(target_adj)
        real_logit = self.gan_loss_module(disc_output_real, torch.ones(1))

        disc_outout_fake = self.discriminator(adj_predicted.detach().squeeze())
        fake_logit = self.gan_loss_module(disc_outout_fake, torch.zeros(1))

        disc_loss = fake_logit + real_logit

        self.manual_backward(disc_loss)
        self.clip_grads("discriminator")
        discriminator_opt.step()

        # Optimize generator
        d_g = self.discriminator(adj_predicted.squeeze())
        d_g_loss = self.crit(d_g, torch.ones(1))

        self.manual_backward(d_g_loss)
        self.clip_grads("generator")
        generator_opt.step()

        pretrain_loss = self.pretrain_loss_module(adj_predicted.squeeze(), target_adj)

        logger_dict["train/generator_loss"] = d_g_loss
        logger_dict["train/discriminator_loss"] = disc_loss
        logger_dict["train/mse_loss"] = pretrain_loss

        return logger_dict

    def training_step(self, batch, batch_idx):
        # torch.autograd.set_detect_anomaly(True)

        if self.counter <= self.pretrain_epochs:
            logger_dict, log_mode = (
                self.pretrain_step(batch, batch_idx),
                "pretrain/loss",
            )
        else:
            logger_dict, log_mode = self.gan_step(batch, batch_idx), "gan"

        self.log(
            log_mode,
            logger_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return logger_dict

    def training_epoch_end(self, outputs):
        if self.counter <= self.pretrain_epochs:
            loss = np.array([])
            for result in outputs:
                loss = np.append(loss, result["loss"])

            self.log("pretrain/epoch/loss", loss.mean())
        else:
            mse_loss, gen_loss, disc_loss = np.array([]), np.array([]), np.array([])
            for result in outputs:
                mse_loss = np.append(
                    mse_loss, result["train/mse_loss"].detach().numpy()
                )
                gen_loss = np.append(
                    gen_loss, result["train/generator_loss"].detach().numpy()
                )
                disc_loss = np.append(
                    disc_loss, result["train/discriminator_loss"].detach().numpy()
                )

            self.log(
                "epoch/gan",
                {
                    "train/epoch/mse_loss": mse_loss.mean(),
                    "train/epoch/generator_loss": gen_loss.mean(),
                    "train/epoch/discriminator_loss": disc_loss.mean(),
                },
            )
        self.counter += 1

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.counter <= self.pretrain_epochs:
            adj_predicted = self.generator(x)
            _, _, _, _, target_adj = y
            target_adj = target_adj.squeeze()
            adj_predicted = adj_predicted.squeeze()
            # upper_target_adj = target_adj[np.triu_indices(19, 1)]
            target_adj = target_adj.view(1, -1).squeeze()
            loss = self.loss_module(adj_predicted, target_adj)
            return {"loss": loss}
        else:
            adj_predicted = self.generator(x)

            _, _, _, _, target_adj = y

            target_adj = target_adj.squeeze()
            adj_predicted = adj_predicted.squeeze()
            # upper_target_adj = target_adj[np.triu_indices(19, 1)]
            target_adj = target_adj.view(1, -1).squeeze()

            loss = self.loss_module(adj_predicted, target_adj)
            self.log(
                "val/step/mse_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return {
                "val/mse_loss": loss,
                "adj_predicted": adj_predicted,
                "target_adj": target_adj,
            }

    def validation_epoch_end(self, outputs):
        if self.counter <= self.pretrain_epochs:
            loss = np.array([])
            for result in outputs:
                loss = np.append(loss, result["loss"])

            self.log("val/loss", loss.mean())
        else:
            loss = np.array([])
            last_pred, last_target = (
                outputs[-1]["adj_predicted"],
                outputs[-1]["target_adj"],
            )
            for result in outputs:
                loss = np.append(loss, result["val/mse_loss"])
            print("last_pred", last_pred)
            print("last_target", last_target)
            self.log("val/loss", loss.mean())

    def test_step(self, batch, batch_idx):
        sequence, target_adj = batch
        predicted_adj = self(sequence)
        loss = self.loss_module(predicted_adj, target_adj)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        preoptimizer_gen = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        optimizer_gen = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        optimizer_disc = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=self.lr
        )
        return optimizer_gen, optimizer_disc, preoptimizer_gen
