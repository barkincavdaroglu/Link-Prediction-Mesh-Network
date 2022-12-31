import torch
import networkx as nx
import numpy as np
from sklearn.preprocessing import PowerTransformer

import os
import pickle


# @torch.jit.script
def unsorted_segment_sum(data, segment_ids: torch.Tensor, num_segments: int):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all(
        [i in data.shape for i in segment_ids.shape]
    ), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    # TODO convert *data.shape[1:] to torch.jit compatible code
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long()
        segment_ids = segment_ids.repeat_interleave(s).view(
            segment_ids.shape[0], *data.shape[1:]
        )

    assert (
        data.shape == segment_ids.shape
    ), "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).scatter_add(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor


def extract_node_features(G) -> torch.Tensor:
    """
    Extracts node features from the graph.
    """
    # try calculating closeness_vitality

    try:
        closeness_vit_per_node = nx.closeness_vitality(G)  # dict
    except nx.exception.NetworkXError:
        print("Oops! Error calculating closeness_vitality for graph: ", G)

    barycenter_per_node = set(nx.barycenter(G))  # list

    degree_per_node = dict(G.degree())

    information_centrality_per_node = nx.information_centrality(
        G, weight="weight"
    )  # dict

    betweenness_centrality_per_node = nx.betweenness_centrality(
        G, weight="weight"
    )  # dict

    # No aps in this dataset - disable for now
    # articulation_points = set(nx.articulation_points(G))

    # Compute the average degree of the graph.
    avg_neighbor_degree = nx.average_neighbor_degree(G, weight="weight")

    # for each node, create a tensor of above features
    # return a tensor of shape (num_nodes, num_features)
    node_fts = torch.zeros((len(G), 6))

    for i, node in enumerate(G.nodes()):
        node_fts[i, 0] = closeness_vit_per_node[node]
        node_fts[i, 1] = int(node in barycenter_per_node)
        node_fts[i, 2] = degree_per_node[node]
        node_fts[i, 3] = information_centrality_per_node[node]
        node_fts[i, 4] = betweenness_centrality_per_node[node]
        # node_fts[i, 5] = int(node in articulation_points)
        node_fts[i, 5] = avg_neighbor_degree[node]

    return node_fts


def extract_edge_features(G) -> torch.Tensor:
    """
    Extracts edge features from the graph.
    """
    edge_betweenness_per_edge = nx.edge_betweenness_centrality(
        G, weight="weight"
    )  # dict

    # for each edge, create a tensor of above features plus the edge weight
    # return a tensor of shape (num_edges, num_features)
    edge_fts = torch.zeros((len(G.edges()), 2))

    for i, edge in enumerate(G.edges().data("weight")):
        edge_fts[i, 0] = edge[2]
        edge_fts[i, 1] = edge_betweenness_per_edge[(edge[0], edge[1])]

    return edge_fts


def extract_graph_features(G) -> torch.Tensor:
    """
    Extracts graph features from the graph.
    """
    laplacian = nx.laplacian_matrix(G).toarray()

    # Compute diameter of G
    diameter = nx.diameter(G)

    # Compute the number of spanning trees of a graph using Kirchhoff's theorem
    # delete the last row and column of the laplacian matrix
    laplacian_cofactor = laplacian[:-1, :-1]
    num_span_trees = abs(round(np.linalg.det(laplacian_cofactor)))

    # Compute an approximation for node connectivity of G.
    node_connectivity = nx.node_connectivity(G)

    # Compute degree assortativity of G.
    deg_assort_coef = nx.degree_assortativity_coefficient(G, weight="weight")

    # Compute the number of triangles in G.
    num_triangles = nx.triangles(G)

    # Compute the value of mincut of G using Stoer-Wagner algorithm.
    min_cut_value, _ = nx.stoer_wagner(G, weight="weight")

    # MST will be the one that connects all nodes with worst connections
    # If the median of all STs is closer to weight of MST than the max of all STs,
    # then the graph is more connected
    min_st = nx.minimum_spanning_tree(G, weight="weight")
    min_st_weight = min_st.size(weight="weight")

    max_st = nx.maximum_spanning_tree(G, weight="weight")
    max_st_weight = max_st.size(weight="weight")

    num_bridges = len(list(nx.bridges(G)))

    return torch.tensor(
        [
            diameter,
            num_span_trees,
            node_connectivity,
            deg_assort_coef,
            sum(num_triangles.values()) / 3,
            min_cut_value,
            min_st_weight,
            max_st_weight,
            num_bridges,
        ]
    )


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def load_all_data(data_dir="dataset_all", mode="pickle"):
    samples = []

    all_node_fts = []
    all_edge_fts = []
    all_graph_fts = []

    for i, filename in enumerate(
        sorted(
            os.listdir(data_dir),
            key=lambda x: int(x.split("_")[0]),
        )
    ):
        # read .txt file and loop each line starting from third line
        filename_dir = os.path.join(data_dir, filename)

        g = nx.read_edgelist(filename_dir, nodetype=int, data=(("weight", float),))

        # for node in g.nodes():
        #    g.add_edge(node, node, weight=0.0)

        node_fts = extract_node_features(g)
        all_node_fts.append(node_fts)

        edge_fts = extract_edge_features(g)
        all_edge_fts.append(edge_fts)

        graph_fts = extract_graph_features(g)
        all_graph_fts.append(graph_fts)

        adj_numpy = nx.to_numpy_matrix(g)

        assert check_symmetric(adj_numpy)

        adj = torch.tensor(adj_numpy, dtype=torch.float)

        edges = torch.tensor([[e[0], e[1]] for e in g.edges()]).t().contiguous()

        g_tensor = [
            edges,
            node_fts,
            edge_fts,
            graph_fts,
            adj,
        ]

        # gs.append(g_tensor)
        # For faster loading, uncomment the block below to save the processed data as pickle files
        # with open("dataset_all_processed/" + str(i) + ".pickle", "wb") as ft_tensors:
        #    pickle.dump(g_tensor, ft_tensors)
        samples.append(g_tensor)

    pt_node_fts = PowerTransformer()
    all_node_fts = torch.vstack(all_node_fts)
    pt_node_fts.fit(all_node_fts)

    pt_edge_fts = PowerTransformer()
    all_edge_fts = torch.vstack(all_edge_fts)
    pt_edge_fts.fit(all_edge_fts)

    pt_graph_fts = PowerTransformer()
    all_graph_fts = torch.vstack(all_graph_fts)
    pt_graph_fts.fit(all_graph_fts)

    for i in range(len(samples)):
        samples[i][1] = torch.tensor(pt_node_fts.transform(samples[i][1])).float()
        samples[i][2] = torch.tensor(pt_edge_fts.transform(samples[i][2])).float()
        samples[i][3] = torch.tensor(
            pt_graph_fts.transform(samples[i][3].reshape(1, -1))
        ).float()
        with open("dataset_all_processed/" + str(i) + ".pickle", "wb") as ft_tensors:
            pickle.dump(samples[i], ft_tensors)

    return samples


# load_all_data()
