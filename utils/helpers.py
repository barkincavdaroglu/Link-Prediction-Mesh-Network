import torch
import networkx as nx
import numpy as np

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
    articulation_points = set(nx.articulation_points(G))

    # Compute the average degree of the graph.
    avg_neighbor_degree = nx.average_neighbor_degree(G, weight="weight")

    # for each node, create a tensor of above features
    # return a tensor of shape (num_nodes, num_features)
    node_fts = torch.zeros((len(G), 7))

    for i, node in enumerate(G.nodes()):
        node_fts[i, 0] = closeness_vit_per_node[node]
        node_fts[i, 1] = node in barycenter_per_node
        node_fts[i, 2] = degree_per_node[node]
        node_fts[i, 3] = information_centrality_per_node[node]
        node_fts[i, 4] = betweenness_centrality_per_node[node]
        node_fts[i, 5] = node in articulation_points
        node_fts[i, 6] = avg_neighbor_degree[node]

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


def load_all_data(data_dir="dataset", mode="pickle"):
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

            for node in g.nodes():
                g.add_edge(node, node, weight=0.0)

            node_fts = extract_node_features(g)
            edge_fts = extract_edge_features(g)
            graph_fts = extract_graph_features(g)
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

            gs.append(g_tensor)
            # For faster loading, uncomment the block below to save the processed data as pickle files
            # with open("data_processed/" + subdir + ".pickle", "wb") as ft_tensors:
            #    pickle.dump(gs, ft_tensors)
        samples.append(gs)
    return samples
