#!/usr/bin/env python3

import torch
import numpy as np
import os
import networkx as nx

from utils.helpers import *


def load_all_data(data_dir):
    # loop through subdirectories in data_dir
    samples = []

    dirs = os.listdir(data_dir)
    for subdir in dirs:
        gs = []
        for filename in sorted(
            os.listdir(os.path.join(data_dir, subdir)),
            key=lambda x: int(x.split("_")[0]),
        ):
            # read .txt file and loop each line starting from third line
            filename_dir = os.path.join(data_dir, subdir, filename)

            g = nx.read_edgelist(filename_dir, nodetype=int, data=(("weight", float),))

            node_fts = extract_node_features(g)
            edge_fts = extract_edge_features(g)
            graph_fts = extract_graph_features(g)

            edges = torch.tensor([[e[0], e[1]] for e in g.edges()]).t().contiguous()
            g_tensor = [
                edges,
                node_fts,
                edge_fts,
                graph_fts,
            ]
            gs.append(g_tensor)
        samples.append(gs)
    return samples


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir="dataset", transform=None, target_transform=None):
        self.dir = data_dir
        self.data = load_all_data(data_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx][:-1]
        label = self.data[idx][-1]
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label
