from typing import Optional
import pytorch_lightning as pl
from .dataset import GraphDataset
from torch.utils.data import random_split, DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch


class GraphDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(
        self,
    ):
        super().__init__()

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """
        dataset = GraphDataset()
        batch_size = 1
        validation_split = 0.2
        test_split = 0.1
        shuffle_dataset = True
        random_seed = 42

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        test_indices = indices[:split]

        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=test_sampler
        )

        training_dataset_size = len(dataset) - len(test_indices)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * training_dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, sampler=train_sampler
        )
        validation_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, sampler=valid_sampler
        )

        self.train = train_loader
        self.validate = validation_loader
        self.test = test_loader

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.validate

    def test_dataloader(self):
        return self.test
