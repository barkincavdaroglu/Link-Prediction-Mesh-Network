from dataclasses import dataclass, asdict


@dataclass
class DataConfig:
    # Path to the dataset
    data_dir: str = "dataset_all_processed"
    # How to load each sample
    mode: str = "pickle"
    batch_size: int = 1
    # Size of the validation set (as a fraction of the training set)
    validation_split: float = 0.1
    # Size of the test set (as a fraction of the whole dataset, remaining is used for training)
    test_split: float = 0.05
    shuffle_dataset: bool = True
    random_seed: int = 42

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
