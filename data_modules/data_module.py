from typing import Optional
import pytorch_lightning as pl
from .dataset import GraphDataset
from torch.utils.data import random_split, DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from configs.DataConfig import DataConfig
from .data_helpers import *


class GraphDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(
        self,
        data_config: DataConfig,
    ):
        super().__init__()
        self.data_dir = data_config.data_dir
        self.mode = data_config.mode
        self.batch_size = data_config.batch_size
        self.validation_split = data_config.validation_split
        self.test_split = data_config.test_split
        self.shuffle_dataset = data_config.shuffle_dataset
        self.random_seed = data_config.random_seed
        self.dataset = None
        self.test_sampler = None
        self.train_sampler = None
        self.valid_sampler = None

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here we use the dataset class and prepare the data
        in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """
        dataset = GraphDataset(data_dir=self.data_dir, mode=self.mode)

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        test_split = int(np.floor(self.test_split * dataset_size))

        if self.shuffle_dataset:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        test_indices = indices[:test_split]
        self.test_sampler = SubsetRandomSampler(test_indices)

        indices = indices[test_split:]
        training_dataset_size = len(dataset) - len(test_indices)
        train_split = int(
            np.floor((1.0 - self.validation_split) * training_dataset_size)
        )
        train_indices, val_indices = indices[:train_split], indices[train_split:]

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(val_indices)

        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.valid_sampler,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            collate_fn=collate_fn,
            drop_last=True,
        )
