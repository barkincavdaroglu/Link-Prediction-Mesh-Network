import torch
import pickle
import os
from typing import List, Tuple


def collate_fn(
    batch: List[Tuple[List[List[torch.Tensor]], torch.Tensor]]
) -> List[torch.Tensor]:
    target_collated = torch.tensor([])
    batch_index_shift = torch.tensor([0])
    temporal_indices_shifts = torch.tensor([])

    batch_edges_collated = torch.tensor([])
    batch_node_fts_collated = torch.tensor([])
    batch_edge_fts_collated = torch.tensor([])
    batch_graph_fts_collated = torch.tensor([])

    for i, (sequence, target) in enumerate(batch):
        b_index_shift = batch_index_shift[-1]

        temporal_index_shift = torch.tensor([0])
        temporal_edges_collated = torch.tensor([])
        temporal_node_fts_collated = torch.tensor([])
        temporal_edge_fts_collated = torch.tensor([])
        temporal_graph_fts_collated = torch.tensor([])

        # each element in the sequence a list of tensors
        # element = [edges, node_fts, edge_fts, graph_fts, adj]
        for element in sequence:
            t_index_shift = temporal_index_shift[-1]
            num_nodes_in_current_graph = element[1].shape[0]
            # shift the node index in the edges
            temporal_edges = element[0] + t_index_shift

            temporal_edges_collated = torch.cat(
                (temporal_edges_collated, temporal_edges), dim=1
            )
            # concatenate the node features
            temporal_node_fts_collated = torch.cat(
                (temporal_node_fts_collated, element[1]), dim=0
            )
            # concatenate the edge features
            temporal_edge_fts_collated = torch.cat(
                (temporal_edge_fts_collated, element[2]), dim=0
            )
            # concatenate the graph features
            temporal_graph_fts_collated = torch.cat(
                (temporal_graph_fts_collated, element[3]), dim=0
            )
            # update the index shift
            temporal_index_shift = torch.cat(
                (
                    temporal_index_shift,
                    torch.tensor([t_index_shift + num_nodes_in_current_graph]),
                )
            )

        target_collated = torch.cat((target_collated, target), dim=0)

        temporal_indices_shifts = torch.cat(
            (temporal_indices_shifts, temporal_index_shift), dim=0
        )

        # shift the node index in the edges
        batch_edges = temporal_edges_collated + b_index_shift
        batch_edges_collated = torch.cat((batch_edges_collated, batch_edges), dim=1)
        # concatenate the node features
        batch_node_fts_collated = torch.cat(
            (batch_node_fts_collated, temporal_node_fts_collated), dim=0
        )
        # concatenate the edge features
        batch_edge_fts_collated = torch.cat(
            (batch_edge_fts_collated, temporal_edge_fts_collated), dim=0
        )
        # concatenate the graph features
        batch_graph_fts_collated = torch.cat(
            (batch_graph_fts_collated, temporal_graph_fts_collated), dim=0
        )
        # update the index shift
        batch_index_shift = torch.cat(
            (
                batch_index_shift,
                torch.tensor([batch_index_shift[-1] + temporal_index_shift[-1]]),
            )
        )

    return [
        batch_index_shift,
        temporal_indices_shifts,
        batch_edges_collated,
        batch_node_fts_collated,
        batch_edge_fts_collated,
        batch_graph_fts_collated,
        target_collated,
    ]


def load_all_pickle(data_dir):
    data = []
    for i in range(len(os.listdir(data_dir))):
        with open(data_dir + "/" + str(i) + ".pickle", "rb") as pickle_file:
            data.append(pickle.load(pickle_file))
    return data
