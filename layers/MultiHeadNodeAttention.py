import torch
import torch.nn as nn
from .NodeAttention import NodeAttentionHead


class MultiHeadNodeAttention(nn.Module):
    def __init__(
        self,
        specs,
        attention_head,
    ):
        super().__init__()
        self.node_in_fts = specs.node_in_fts
        self.node_out_fts = specs.node_out_fts
        self.edge_in_fts = specs.edge_in_fts
        self.num_heads = specs.num_heads_node
        self.head_agg_mode = specs.head_agg_mode

        self.head_coef = nn.Parameter(
            torch.ones(specs.num_heads_node, 19, specs.node_out_fts)
        )
        self.leaky_relu = nn.LeakyReLU(specs.alpha)

        self.attention_heads = nn.ModuleList(
            [
                attention_head(
                    specs.node_in_fts,
                    specs.node_out_fts,
                    specs.edge_in_fts,
                    specs.node_agg_mode,
                    specs.messagenorm_learn_scale,
                    specs.alpha,
                    specs.kernel_init,
                    specs.kernel_reg,
                )
                for _ in range(specs.num_heads_node)
            ]
        )

    def forward(self, inputs):
        """
        Args:
            inputs: list of [node_fts, edge_fts, edges]
            where
                node_fts: tensor of shape (num_nodes, node_in_fts)
                edge_fts: tensor of shape (num_edges, edge_in_fts)
                edges: tensor of shape (num_edges, 2)
        Returns:
            node_ft_embeds: tensor of shape (num_nodes, num_heads * node_out_fts + num_heads * edge_out_fts)
        """
        node_fts, edge_fts, edges = inputs

        node_ft_embeds, node_attentions_var = (
            torch.tensor([]),
            torch.tensor([]),
        )

        for head in self.attention_heads:
            node_out, node_attention_var = head([node_fts, edge_fts, edges])
            # TODO: Log the attention variance for each head
            node_attention_var = node_attention_var.unsqueeze(dim=0)

            node_ft_embeds = torch.cat((node_ft_embeds, node_out.unsqueeze(0)), dim=0)
            node_attentions_var = torch.cat(
                (node_attentions_var, node_attention_var), dim=0
            )

        node_attentions_var = torch.exp(node_attentions_var)
        node_attentions_var = node_attentions_var / torch.sum(node_attentions_var)

        if self.head_agg_mode == "concat":
            node_ft_embeds = torch.cat(
                [
                    node_ft_embeds[i] * node_attentions_var[i]
                    for i in range(self.num_heads)
                ],
                dim=1,
            )
        elif self.head_agg_mode == "weighted_mean":
            # TODO: Is nonlinear activation necessary here?
            node_ft_embeds = self.leaky_relu(
                torch.mul(
                    node_attentions_var.unsqueeze(1).unsqueeze(1), node_ft_embeds
                ).sum(dim=0)
                / node_attention_var.sum()
            )
        else:
            node_ft_embeds = node_ft_embeds.mean(dim=0)

        return node_ft_embeds
