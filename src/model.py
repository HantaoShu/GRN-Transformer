
import torch
import torch.nn as nn

from .modules import (
    GRNTransformer,
    MLP,
    ESM1bLayerNorm)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_X = nn.Sequential(nn.Linear(1, self.args.embed_dim), nn.ReLU(),
                                     nn.Linear(self.args.embed_dim, self.args.embed_dim // 2))
        self.zeros_embed = nn.Linear(1, self.args.embed_dim // 2)
        self.layers = nn.ModuleList(
            [
                GRNTransformer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    self.args.attention_heads,
                )
                for _ in range(self.args.layers)
            ]
        )
        self.connect_layers = nn.Linear(self.args.embed_dim, 1)
        self.emb_layer_norm_before = ESM1bLayerNorm(self.args.embed_dim)
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.lm_head = MLP(embed_dim=self.args.embed_dim,output_dim=1)

    def forward(self, x, zero, network,train=True) :
        x = torch.cat([self.embed_X(x.unsqueeze(-1)), self.zeros_embed(zero.unsqueeze(-1))], -1)

        x = self.emb_layer_norm_before(x)

        row_attn_weights = []
        x = x.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):

            x = layer(x, network,train=train)
            x, row_attn, l = x
            row_attn_weights.append(row_attn.permute(1, 0, 2, 3))
        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)
        x = self.lm_head(x)
        result = {"logits": x}
        row_attentions = torch.stack(row_attn_weights, 1)
        result["row_attentions"] = row_attentions.view(-1, row_attentions.shape[-1], row_attentions.shape[-1]).permute(2,1,0).contiguous().detach().cpu().numpy()
        return result


    def loss(self, y, pred, d_mask):
        return torch.sum(((y - pred) * d_mask) ** 2) / (torch.sum(d_mask) + 1e-5)
