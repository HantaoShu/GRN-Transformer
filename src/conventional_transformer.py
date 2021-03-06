import math

import torch
import torch.nn as nn

from src.modules import FeedForwardNetwork, ESM1bLayerNorm, NormalizedResidualBlock, MLP
# We followed implement in https://github.com/facebookresearch/esm

class Row_Attention(nn.Module):

    def __init__(
            self,
            embed_dim,
            num_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.attn_shape = "hnij"
        self.embed_dim = embed_dim
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k1_proj = nn.Linear(1, 8)
        self.q1_proj = nn.Linear(embed_dim // num_heads, 8)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.l = torch.nn.Parameter(torch.autograd.Variable(torch.randn(self.num_heads, 1, 1, 1)), requires_grad=True)

    def align_scaling(self, q):
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def attention_network_selection(self, q, k, network, batch_size, num_cols):
        network = network.view(batch_size, num_cols, num_cols, -1)
        q = self.q1_proj(torch.einsum('rinhd,rjnhd->hnijd', q, k)).mean(0)  # hnijd
        network_k = self.k1_proj(network.unsqueeze(-1))  # hnijmd
        network_attention = torch.einsum('nijd,nijmd->nijm', q, network_k)
        return network_attention

    def pred_attention_network_selection(self, q, network, batch_size, num_cols):
        network = network.view(batch_size, num_cols, num_cols, -1)
        network_k = self.k1_proj(network.unsqueeze(-1))  # hnijmd
        network_attention = torch.einsum('nijd,nijmd->nijm', q, network_k)
        return network_attention

    def compute_attention_weights(self, x, network, scaling, num_rows, num_cols, batch_size):

        q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        q *= scaling
        network_attention = self.attention_network_selection(q, k, network, batch_size, num_cols).repeat(self.num_heads, 1, 1, 1)
        attn_weights = torch.einsum(f"rinhd,rjnhd->hnij", q, k)
        return attn_weights, network_attention

    def pred_compute_attention_weights(self, x, scaling, num_rows, num_cols, batch_size):

        q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        q *= scaling
        attn_weights = torch.einsum(f"rinhd,rjnhd->hnij", q, k)
        q2 = (torch.einsum('rinhd,rjnhd->hnijd', q, k))
        return attn_weights, q2

    def compute_attention_update(self, x, attn_probs):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        context = torch.einsum(f"hnij,rjnhd->rinhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output

    def forward(self, x, network, return_attn=False):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if not return_attn:
            scaling = self.align_scaling(x)
            attn_weights, network_attention = self.compute_attention_weights(x, network, scaling, num_rows, num_cols, batch_size)

            network = network.view(batch_size, num_cols, num_cols, -1)
            network_attention = network_attention.softmax(-1)

            network = (network * network_attention).sum(-1).unsqueeze(1)

            attn_weights = attn_weights + self.l * network

            mask = torch.eye(attn_weights.shape[2]).repeat(self.num_heads, 1, 1, 1).to(attn_weights.device).bool()
            attn_weight = attn_weights.masked_fill(mask, -1000000000)
            attn_probs = attn_weight.softmax(-1)
            output = self.compute_attention_update(x, attn_probs)
            return output, attn_probs, self.l

        else:
            return self.pred(x, network)

    def pred(self, x, network):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        m = 1
        n = math.ceil(len(x) / m)
        scaling = self.align_scaling(x)
        attn_weights = 0
        qs = 0
        for i in range(n):
            x_in = x[i * m:min(len(x), (i + 1) * m)]
            attn_weight, q2 = self.pred_compute_attention_weights(x_in, scaling, x_in.shape[0], num_cols, batch_size)
            attn_weights += attn_weight
            qs += q2
        network_attentions = self.pred_attention_network_selection(self.q1_proj(qs).mean(0), network, batch_size,
                                                                   num_cols).repeat(self.num_heads, 1, 1, 1)
        network = network.view(batch_size, num_cols, num_cols, -1)
        network_attentions = network_attentions.softmax(-1)

        network = (network * network_attentions).sum(-1).unsqueeze(1)

        attn_weights = attn_weights + self.l * network

        mask = torch.eye(attn_weights.shape[2]).repeat(self.num_heads, 1, 1, 1).to(attn_weights.device).bool()
        attn_weight = attn_weights.masked_fill(mask, -1000000000)
        attn_probs = attn_weight.softmax(-1)
        output = []
        for i in (range(n)):
            x_in = x[i * m:min(len(x), (i + 1) * m)]
            output.append(self.compute_attention_update(x_in, attn_probs))
        output = torch.cat(output, 0)
        return output, attn_probs, self.l


class ColumnSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def align_scaling(self, q):
        num_cols = q.size(1)
        return self.scaling / math.sqrt(num_cols)

    def compute_attention_weights(self, x, scaling: float):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        q *= scaling

        attn_weights = torch.einsum(f"icnhd,jcnhd->hnij", q, k)

        return attn_weights

    def compute_attention_update(self, x, attn_probs):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        context = torch.einsum(f"hnij,jcnhd->icnhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output

    def forward(self, x, return_attn=True):
        if not return_attn:
            scaling = self.align_scaling(x)
            attn_weights = self.compute_attention_weights(x, scaling)

            attn_weights = attn_weights
            attn_probs = attn_weights.softmax(-1)

            output = self.compute_attention_update(x, attn_probs)
            return output, attn_probs
        else:
            return self.pred(x)

    def pred(self, x):
        m = 1
        n = math.ceil(x.shape[1] / m)
        scaling = self.align_scaling(x)
        attn_weights = 0
        for i in range(n):
            x_in = x[:, i * m:min(x.shape[1], (i + 1) * m)]
            attn_weights += self.compute_attention_weights(x_in, scaling)
        attn_weights = attn_weights
        attn_probs = attn_weights.softmax(-1)
        output = []
        for i in range(n):
            x_in = x[:, i * m:min(x.shape[1], (i + 1) * m)]
            output.append(self.compute_attention_update(x_in, attn_probs))
        output = torch.cat(output, 1)

        return output, attn_probs


class GRNTransformer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 64,
            ffn_embedding_dim: int = 64,
            num_attention_heads: int = 4,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        row_self_attention = Row_Attention(
            embedding_dim,
            num_attention_heads,
        )
        self.row_self_attention = self.build_residual(row_self_attention)

        self.column_self_attention = self.build_residual(ColumnSelfAttention(embedding_dim, num_attention_heads))

        feed_forward_layer = FeedForwardNetwork(
            embedding_dim,
            ffn_embedding_dim,
        )
        self.feed_forward_layer = self.build_residual(feed_forward_layer)

    def build_residual(self, layer: nn.Module):
        return NormalizedResidualBlock(
            layer,
            self.embedding_dim,
        )

    def forward(self, x, network, return_attn=False, ):
        x, row_attn, l = self.row_self_attention(x, network, return_attn=return_attn)
        x, col_attn = self.column_self_attention(x, return_attn)
        x = self.feed_forward_layer(x)
        return x, row_attn, col_attn, l


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
        self.lm_head = MLP(embed_dim=self.args.embed_dim, output_dim=1)

    def forward(self, x, zero, network, return_attn=False):
        x = torch.cat([self.embed_X(x.unsqueeze(-1)), self.zeros_embed(zero.unsqueeze(-1))], -1)

        x = self.emb_layer_norm_before(x)

        row_attn_weights = []
        col_attn_weights = []
        x = x.permute(1, 2, 0, 3)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, network, return_attn=return_attn)
            x, row_attn, col_attn, l = x
            row_attn_weights.append(row_attn.permute(1, 0, 2, 3))
            col_attn_weights.append(col_attn.permute(1, 0, 2, 3))
        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)
        x = self.lm_head(x)
        result = {"logits": x}
        row_attentions = torch.stack(row_attn_weights, 1)
        col_attentions = torch.stack(col_attn_weights, 1)
        result["row_attentions"] = row_attentions.view(-1, row_attentions.shape[-1],
                                                       row_attentions.shape[-1]).permute(2, 1,
                                                                                         0).contiguous().detach().cpu().numpy()
        result["col_attentions"] = col_attentions.view(-1, col_attentions.shape[-1],
                                                       col_attentions.shape[-1]).permute(2, 1,
                                                                                         0).contiguous().detach().cpu().numpy()
        return result

    def loss(self, y, pred, d_mask):
        return torch.sum(((y - pred) * d_mask) ** 2) / (torch.sum(d_mask) + 1e-5)
