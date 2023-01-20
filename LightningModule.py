from asyncio.log import logger
from email import generator
import torch
import pytorch_lightning as pl
import numpy as np
from adapters.gen_loss_adapter import create_gen_loss_module
from adapters.disc_loss_adapter import create_disc_loss_module
from configs.GANConfig import GANConfig
from adapters.model_adapters import create_generator_model, create_discriminator_model


class GraphLightningModule(pl.LightningModule):
    def __init__(
        self,
        gan_config: GANConfig,
        lr,
        is_clip_grads,
        gradient_clip_val,
        gradient_clip_algorithm,
        pretrain_epochs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.generator = create_generator_model(gan_config.generator_config)
        self.discriminator = create_discriminator_model(gan_config.discriminator_config)

        self.pretrain_loss_module = create_gen_loss_module(gan_config.generator_config)
        self.gan_loss_module = create_disc_loss_module(gan_config.discriminator_config)

        self.automatic_optimization = False
        self.lr = lr
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.counter = 0
        self.is_clip_grads = is_clip_grads
        self.horizon = gan_config.generator_config.horizon
        self.pretrain_epochs = pretrain_epochs

        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), gradient_clip_val)

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
        """
        Pretrain generator
        :param batch
        :param batch_idx:
        """
        node_fts, edges, edge_fts, target = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.y[:, : self.horizon],
        )

        edge_fts = edge_fts.unsqueeze(dim=1)

        # * node_fts has shape (num_nodes, num_node_fts, time_steps)
        # * edges has shape (2, num_edges) where num_edges is the
        # total number of edges in the batch
        # * edge_fts has shape (num_edges,)
        # * target has shape (num_nodes, time_steps)

        _, _, pretrain_gen_opt = self.optimizers()
        logger_dict = {}

        pretrain_gen_opt.zero_grad()
        adj_predicted = self.generator(edges, node_fts, edge_fts, None)

        loss = self.pretrain_loss_module(adj_predicted, target)  #

        self.manual_backward(loss)
        self.clip_grads("generator")

        pretrain_gen_opt.step()

        logger_dict["pretrain/loss"] = loss

        return logger_dict

    def gan_step(self, batch, batch_idx):
        edges, node_fts, edge_fts, graph_fts, target_adj = batch

        generator_opt, discriminator_opt, _ = self.optimizers()
        logger_dict = {}

        discriminator_opt.zero_grad()
        generator_opt.zero_grad()

        adj_predicted = self.generator(edges, node_fts, edge_fts, graph_fts)

        # Optimize discriminator
        disc_output_real = self.discriminator(target_adj)
        real_logit = self.gan_loss_module(
            disc_output_real, torch.ones(target_adj.shape[0], 1)
        ).mean()

        disc_outout_fake = self.discriminator(adj_predicted.detach().squeeze()).mean()
        fake_logit = self.gan_loss_module(
            disc_outout_fake, torch.zeros(target_adj.shape[0], 1)
        )

        disc_loss = fake_logit + real_logit

        self.manual_backward(disc_loss)
        self.clip_grads("discriminator")
        discriminator_opt.step()

        # Optimize generator
        d_g = self.discriminator(adj_predicted.squeeze())
        d_g_loss = self.crit(d_g, torch.ones(target_adj.shape[0], 1))

        self.manual_backward(d_g_loss)
        self.clip_grads("generator")
        generator_opt.step()

        pretrain_loss = self.pretrain_loss_module(adj_predicted.squeeze(), target_adj)

        logger_dict["train/generator_loss"] = d_g_loss
        logger_dict["train/discriminator_loss"] = disc_loss
        logger_dict["train/mse_loss"] = pretrain_loss

        return logger_dict

    def training_step(self, batch, batch_idx):
        if self.counter < self.pretrain_epochs:
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
        if self.counter < self.pretrain_epochs:
            loss = np.array([])
            for result in outputs:
                loss = np.append(loss, result["pretrain/loss"].detach().numpy())

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
        node_fts, edges, edge_fts, target = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.y[:, : self.horizon],
        )

        edge_fts = edge_fts.unsqueeze(dim=1)

        if self.counter < self.pretrain_epochs:
            adj_predicted = self.generator(edges, node_fts, edge_fts, target)
            loss = self.pretrain_loss_module(adj_predicted, target)
            return {"loss": loss.detach().numpy()}
        else:
            adj_predicted = self.generator(edges, node_fts, edge_fts, target)
            loss = self.loss_module(adj_predicted, target)
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
            }

    def validation_epoch_end(self, outputs):
        # TODO: Add visualization of predicted and actual graphs
        if self.counter < self.pretrain_epochs:
            loss = np.array([])
            for result in outputs:
                loss = np.append(loss, result["loss"])

            self.log("val/loss", loss.mean())
        else:
            loss = np.array([])

            for result in outputs:
                loss = np.append(loss, result["val/mse_loss"])

            self.log("val/loss", loss.mean())

    def test_step(self, batch, batch_idx):
        node_fts, edges, edge_fts, target = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.y[:, : self.horizon],
        )

        edge_fts = edge_fts.unsqueeze(dim=1)

        adj_predicted = self.generator(edges, node_fts, edge_fts, target)
        loss = self.pretrain_loss_module(adj_predicted, target)

        self.log("test_loss", loss)

    def configure_optimizers(self):
        preoptimizer_gen = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, amsgrad=True
        )  # torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        optimizer_gen = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        optimizer_disc = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=self.lr
        )
        return optimizer_gen, optimizer_disc, preoptimizer_gen
