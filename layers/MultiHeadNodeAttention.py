import torch
import torch.nn as nn
from .NodeAttention import NodeAttentionHead


class MultiHeadNodeAttention(nn.Module):
    def __init__(
        self,
        node_in_fts,
        node_out_fts,
        edge_in_fts,
        edge_out_fts,
        num_heads,
        head_agg_mode="concat",
        alpha=0.2,
        kernel_init=nn.init.xavier_uniform_,
        kernel_reg=None,
    ):
        super().__init__()
        self.node_in_fts = node_in_fts
        self.node_out_fts = node_out_fts
        self.edge_in_fts = edge_in_fts
        self.edge_out_fts = edge_out_fts
        self.num_heads = num_heads
        self.head_agg_mode = head_agg_mode

        self.kernel_init = kernel_init
        self.kernel_reg = kernel_reg

        self.layer_norm = nn.LayerNorm(
            num_heads * node_out_fts + num_heads * edge_out_fts
        )
        self.activation = nn.LeakyReLU(alpha)

        self.attention_heads = nn.ModuleList(
            [
                NodeAttentionHead(
                    node_in_fts,
                    node_out_fts,
                    edge_in_fts,
                    edge_out_fts,
                    alpha,
                    kernel_init,
                    kernel_reg,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, inputs):
        """
        Args:
            inputs: list of [node_fts, edge_fts, edges]
        Returns:
            node_ft_embeds: tensor of shape (num_nodes, num_heads * node_out_fts + num_heads * edge_out_fts)
        """
        node_fts, edge_fts, edges = inputs

        node_ft_embeds, edge_ft_embeds, node_attentions_var, edge_attentions_var = (
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
        )

        for head in self.attention_heads:
            node_out, edge_out, node_attention_var, edge_attention_var = head(
                [node_fts, edge_fts, edges]
            )

            node_attention_var = node_attention_var.unsqueeze(dim=0)
            edge_attention_var = edge_attention_var.unsqueeze(dim=0)

            node_ft_embeds = torch.cat((node_ft_embeds, node_out.unsqueeze(0)), dim=0)
            edge_ft_embeds = torch.cat((edge_ft_embeds, edge_out.unsqueeze(0)), dim=0)
            node_attentions_var = torch.cat(
                (node_attentions_var, node_attention_var), dim=0
            )
            edge_attentions_var = torch.cat(
                (edge_attentions_var, edge_attention_var), dim=0
            )

        # normalize attention scores using softmax
        node_attentions_var = torch.exp(torch.clamp(node_attentions_var, -2, 2))
        node_attentions_var = node_attentions_var / torch.sum(node_attentions_var)

        edge_attentions_var = torch.exp(torch.clamp(edge_attentions_var, -2, 2))
        edge_attentions_var = edge_attentions_var / torch.sum(edge_attentions_var)

        if self.head_agg_mode == "concat":
            # multiply each node_ft_embed with respective node_attentions_var and concat results
            node_ft_embeds = torch.cat(
                [
                    node_ft_embeds[i] * node_attentions_var[i]
                    for i in range(self.num_heads)
                ],
                dim=1,
            )
            edge_ft_embeds = torch.cat(
                [
                    edge_ft_embeds[i] * edge_attentions_var[i]
                    for i in range(self.num_heads)
                ],
                dim=1,
            )
        else:
            # compute weighted mean of node_ft_embeds and edge_ft_embeds
            node_ft_embeds = (
                torch.mul(
                    node_attentions_var.unsqueeze(1).unsqueeze(1), node_ft_embeds
                ).sum(dim=0)
                / node_attention_var.sum()
            )
            edge_ft_embeds = (
                torch.mul(
                    edge_attentions_var.unsqueeze(1).unsqueeze(1), edge_ft_embeds
                ).sum(dim=0)
                / edge_attention_var.sum()
            )

        return torch.cat([node_ft_embeds, edge_ft_embeds], dim=1)
