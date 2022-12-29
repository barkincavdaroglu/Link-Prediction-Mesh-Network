import torch
import torch.nn as nn
from .MultiHeadNodeAttention import MultiHeadNodeAttention
from .MultiHeadGraphAttention import MultiHeadGraphAttention
from .EdgeUpdate import EdgeUpdate
from .DiffCenAGG import DiffCenAGG


class GNBlock(nn.Module):
    def __init__(
        self,
        graph_in_fts,
        graph_out_fts,
        node_in_fts,
        node_out_fts,
        edge_in_fts,
        edge_out_fts,
        num_heads_node,
        num_heads_graph,
        head_agg_mode="mean",
    ):
        super(GNBlock, self).__init__()
        node_in_w_head = 0
        if head_agg_mode == "concat":
            node_in_w_head = (
                num_heads_node * node_out_fts + num_heads_node * edge_out_fts
            )
        else:
            node_in_w_head = node_out_fts + edge_out_fts
        self.multihead_node_attention_hop_1 = MultiHeadNodeAttention(
            node_in_fts=node_in_fts,
            node_out_fts=node_out_fts,
            edge_in_fts=edge_out_fts,
            edge_out_fts=edge_out_fts,
            num_heads=num_heads_node,
            head_agg_mode="mean",
        )
        self.multihead_node_attention_hop_2 = MultiHeadNodeAttention(
            node_in_fts=node_in_w_head,
            node_out_fts=node_out_fts,
            edge_in_fts=edge_out_fts,
            edge_out_fts=edge_out_fts,
            num_heads=num_heads_node,
            head_agg_mode="mean",
        )
        self.layer_norm = torch.nn.LayerNorm(node_in_w_head)

        """self.multihead_graph_attention = MultiHeadGraphAttention(
            graph_in_fts=graph_in_fts,
            graph_out_fts=graph_out_fts,
            node_in_fts=node_in_fts,
            node_out_fts=node_out_fts,
            edge_in_fts=edge_in_fts,
            edge_out_fts=edge_out_fts,
            num_heads=num_heads_graph,
        )"""
        self.edge_update = EdgeUpdate(
            in_fts=node_in_fts,
            edge_in_fts=edge_in_fts,
            out_dim=edge_out_fts,
        )
        self.node_agg = DiffCenAGG(
            in_dim=node_in_w_head,
            out_dim=node_out_fts,
        )

    def forward(self, input):
        node_fts, edge_fts, edges, adj = input

        edge_fts = self.edge_update([node_fts, edge_fts, edges])

        node_fts = self.multihead_node_attention_hop_1([node_fts, edge_fts, edges])
        # node_fts = self.multihead_node_attention_hop_2([node_fts, edge_fts, edges])
        # edge_fts = self.edge_update([node_fts, edge_fts, edges])

        agg_node_fts = self.node_agg([node_fts, adj])
        agg_edge_fts = torch.mean(edge_fts, dim=0)

        return agg_node_fts, agg_edge_fts, node_fts, edge_fts, edges
