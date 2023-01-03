#!/usr/bin/env python3

import torch
import pickle
from .data_helpers import *
from utils.helpers import *


class GraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        mode,
        load_all=True,
        transform=None,
        target_transform=None,
    ):
        self.dir = data_dir
        self.mode = mode
        if mode == "pickle":
            if load_all:
                self.data = load_all_pickle(data_dir)
            else:
                self.data = data_dir
        else:
            self.data = load_all_data(data_dir)
        self.transform = transform
        self.loaded = load_all
        self.seq_length = 9
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.dir)) - self.seq_length - 1

    def get_pickle(self, name):
        with open(name, "rb") as pickle_file:
            return pickle.load(pickle_file)

    def __getitem__(self, idx):
        if self.mode == "pickle":
            timeline = (
                [
                    self.get_pickle(
                        self.data
                        + "/"
                        + str(idx + i)
                        + ".pickle"  # str(idx * self.seq_length + i)
                    )
                    for i in range(self.seq_length)
                ]
                if not self.loaded
                else self.data[idx : idx + self.seq_length]
            )
        else:
            timeline = self.data[idx]
        sample = timeline[:-1]
        label = timeline[-1]
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label[-1]
