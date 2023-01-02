from asyncore import read
from datetime import datetime
import os
import networkx as nx
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

EPS = 1e-8


def group_and_sort_by_unixtime():
    # loop through files in data-1 directory in the ascending order of unix timestamp which is the value after - in the file name
    filenames_sorted = []
    for file in os.listdir("data-1"):
        # get the file path
        file_path = os.path.join("data-1", file)
        # get unix timestamp which is the value after - in filename
        timestamp_split = file.split("-")[1].split(".")[0]
        timestamp = int(timestamp_split)
        # append the file path and timestamp to the list
        filenames_sorted.append((timestamp, file_path))

    # sort the list by unix timestamp
    filenames_sorted.sort(key=lambda x: x[0])
    # partition the list into groups of 9
    # filenames_sorted = [
    #    filenames_sorted[i : i + 9] for i in range(0, len(filenames_sorted), 9)
    # ]

    return filenames_sorted


def humanize_unixtime(unix_time):
    time = datetime.fromtimestamp(int(unix_time)).strftime("%d-%m-%Y %H.%M.%S")
    return time


def read_graph():
    filenames_sorted = group_and_sort_by_unixtime()

    valid_timeline_counter = 0
    # loop through each group
    for i, (timestamp, file_path) in enumerate(filenames_sorted):
        # read .txt file and loop each line starting from third line
        with open(file_path, "r") as f:
            next(f)
            next(f)
            G = nx.Graph()
            edges = collections.defaultdict(lambda: 0)

            for line in f:
                # split line into list
                line = line.split()
                # each line starts with source node_id and pairs of (neighbor_id, time)
                source = line[0]
                # loop each pair of (neighbor_id, time)
                for k in range(1, len(line), 2):
                    neighbor = line[k]
                    quality = float(line[k + 1])
                    # add edge to graph
                    edges[(source, neighbor)] = quality

            data = [v for _, v in edges.items()]
            if len(data) > 0:
                q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
                iqr = q75 - q25
                cut_off = iqr * 1.5
                lower, upper = q25 - cut_off, q75 + cut_off
                non_outliers = [x for x in data if x > lower and x < upper]
                edges_scaled = []

                mean, std = np.mean(non_outliers), np.std(non_outliers)

                for k, v in edges.items():
                    if v > lower and v < upper:
                        edges_scaled.append(
                            (
                                k[0],
                                k[1],
                                inverse_soft(
                                    (
                                        ((v - mean) + EPS)
                                        if v - mean == 0
                                        else (v - mean)
                                    )
                                    / std
                                ),
                            )
                        )

                G.add_weighted_edges_from(edges_scaled)

                if len(G.nodes) != 38:
                    continue

                # check if G is connected
                elif nx.number_connected_components(G) != 2:
                    continue
            else:
                continue

            valid_timeline_counter += 1
            # os.mkdir(os.path.join("dataset_all", str(valid_timeline_counter)))
            g = G
            # convert adjancecy matrix to torch tensor and save it in dataset directory
            # get the first connected component
            g_sub = list(nx.connected_components(g))[0]
            g = nx.subgraph(g, g_sub)

            # map each node to a unique integer
            mapping = {node: i for i, node in enumerate(g.nodes)}
            edges = [
                (mapping[e[0]], mapping[e[1]], e[2]["weight"])
                for e in g.edges(data=True)
            ]

            # write one edge per line
            with open(
                os.path.join(
                    "dataset_all",
                    str(valid_timeline_counter) + "_" + "edges.txt",
                ),
                "w",
            ) as f:
                for edge in edges:
                    edge = str(edge[0]) + " " + str(edge[1]) + " " + str(edge[2]) + "\n"
                    f.write(edge)


def inverse_1(x):
    b = pow((1 / x - 1), 1.3)
    return 1 - (1 / (1 + b))


def inverse_soft(x):
    return 1 - (1 / (1 + np.exp(x)))


# read_graph()
