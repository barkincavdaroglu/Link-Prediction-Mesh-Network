from typing import Optional
import pytorch_lightning as pl
from dataset import GraphDataset
from torch.utils.data import random_split, DataLoader


class GraphDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(
        self,
    ):
        super(GraphDataModule).__init__()

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally.
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here

        dataset = GraphDataset()
        train_set_size = int(len(dataset) * 0.7)
        valid_set_size = int(len(dataset) * 0.1)
        test_set_size = len(dataset) - train_set_size - valid_set_size

        self.train, self.validate, self.test = random_split(
            dataset, [train_set_size, valid_set_size, test_set_size]
        )
        """if stage == "fit" or stage is None:
            train_set_full =  GraphDataset()
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = YourCustomDataset(
                root_path="/Users/yourusername/path/to/data/test_set/",
                ipt="input/",
                tgt="target/",
                tgt_scale=25, 
                train_transform=False)"""

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=1, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=1, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, num_workers=4)
