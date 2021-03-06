import torch
import torch.nn as nn
from torch import einsum

from src.modules import FeedForwardNetwork, ESM1bLayerNorm, NormalizedResidualBlock, MLP
# We followed implement in https://github.com/lucidrains/linear-attention-transformer


class Gene_Linear_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.attn_shape = "hnij"
        self.embed_dim = embed_dim
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

    def attention(self, q, k, v):
        dim = q.shape[-1]  # ncell,head,ngene,dim
        k = k.softmax(dim=-2)
        q = q * dim ** -0.5
        context = einsum('hnd,hne->hde', k, v)
        attn = einsum('hnd,hde-> hne', q, context)
        return attn.reshape(*v.shape)

    def forward(self, x, return_attn=True):
        num_rows, num_cols, embed_dim = x.size()
        if not return_attn:
            q = self.q_proj(x).view(num_rows, num_cols, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(num_rows, num_cols, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(num_rows, num_cols, self.num_heads, self.head_dim)
            q = q.permute(2, 1, 0, 3).softmax(-1).reshape(self.num_heads, num_cols, -1)
            k = k.permute(2, 1, 0, 3).reshape(self.num_heads, num_cols, -1)
            v = v.permute(2, 1, 0, 3).reshape(self.num_heads, num_cols, -1)
            output = self.attention(q, k, v)
            output = output.view(self.num_heads, num_cols, num_rows, -1).permute(2, 1, 0, 3)
            output = output.reshape(num_rows, num_cols, -1)
            return output, None
        else:
            return self.pred(x)

    def pred(self, x):
        num_rows, num_cols, embed_dim = x.size()
        q = self.q_proj(x).view(num_rows, num_cols, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_rows, num_cols, self.num_heads, self.head_dim)
        q = q.permute(2, 1, 0, 3).softmax(dim=-1).reshape(self.num_heads, num_cols, -1)
        k = k.permute(2, 1, 0, 3).reshape(self.num_heads, num_cols, -1)
        v = v.permute(2, 1, 0, 3).reshape(self.num_heads, num_cols, -1)
        dim = q.shape[-1]
        k = k.softmax(dim=-2)
        q = q * dim ** -0.5
        attn_weights = (q @ k.transpose(2, 1))
        context = einsum('hnd,hne->hde', k, v)
        output = einsum('hnd,hde-> hne', q, context)
        output = output.view(self.num_heads, num_cols, num_rows, -1).permute(2, 1, 0, 3)
        output = output.reshape(num_rows, num_cols, -1)
        return output, attn_weights


class Cell_Linear_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.attn_shape = "hnij"
        self.embed_dim = embed_dim
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

    def attention(self, q, k, v):
        dim = q.shape[-1]  # ncell,head,ngene,dim
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        q = q * dim ** -0.5
        context = einsum('hnd,bhne->bhde', k, v)
        attn = einsum('hnd,bhde-> bhne', q, context)
        return attn.reshape(*v.shape)

    def forward(self, x, return_attn=True):
        num_rows, num_cols, embed_dim = x.size()
        if not return_attn:
            q = self.q_proj(x).view(num_rows, num_cols, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(num_rows, num_cols, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(num_rows, num_cols, self.num_heads, self.head_dim)
            q = q.permute(0, 2, 1, 3).mean(0)
            k = k.permute(0, 2, 1, 3).mean(0)
            v = v.permute(0, 2, 1, 3)
            output = self.attention(q, k, v)
            output = output.permute(0, 2, 1, 3)
            output = output.reshape(num_rows, num_cols, -1)
            return output, None
        else:
            return self.pred(x)

    def pred(self, x):
        num_rows, num_cols, embed_dim = x.size()
        q = self.q_proj(x).view(num_rows, num_cols, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_rows, num_cols, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3).mean(0)
        k = k.permute(0, 2, 1, 3).mean(0)
        v = v.permute(0, 2, 1, 3)
        dim = q.shape[-1]
        q = q.softmax(dim=-1)  # row,head,col,dim
        k = k.softmax(dim=-2)
        q = q * dim ** -0.5
        attn_weights = (q @ k.transpose(2, 1))
        context = einsum('hnd,bhne->bhde', k, v)
        output = einsum('hnd,bhde-> bhne', q, context)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(num_rows, num_cols, -1)
        return output, attn_weights


class GRNTransformer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 64,
            ffn_embedding_dim: int = 64,
            num_attention_heads: int = 4,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        row_self_attention = Gene_Linear_Attention(
            embedding_dim,
            num_attention_heads,
        )

        col_self_attention = Cell_Linear_Attention(
            embedding_dim,
            num_attention_heads,
        )
        feed_forward_layer = FeedForwardNetwork(
            embedding_dim,
            ffn_embedding_dim,
        )

        self.attention = self.build_residual(row_self_attention)
        self.col_attention = self.build_residual(col_self_attention)
        self.feed_forward_layer = self.build_residual(feed_forward_layer)

    def build_residual(self, layer: nn.Module):
        return NormalizedResidualBlock(
            layer,
            self.embedding_dim,
        )

    def forward(self, x, return_attn=True, ):
        x, attn = self.attention(x, return_attn=return_attn)
        x = x.transpose(0, 1)
        x, _ = self.col_attention(x, return_attn=return_attn)
        x = x.transpose(0, 1)
        x = self.feed_forward_layer(x)
        return x, attn


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
        x, zero = x.squeeze(0), zero.squeeze(0)
        n_cell, n_gene = x.shape
        x = torch.cat([self.embed_X(x.unsqueeze(-1)), self.zeros_embed(zero.unsqueeze(-1))], -1)
        x = self.emb_layer_norm_before(x)
        row_attn_weights = []
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(x, return_attn=return_attn)
            row_attn_weights.append(attn)
        x = self.emb_layer_norm_after(x)
        # x = x.permute(2, 0, 1, 3)
        x = self.lm_head(x)
        if return_attn:
            result = {"logits": x.unsqueeze(0),
                      'row_attentions': torch.cat(row_attn_weights).permute(2, 1, 0).contiguous().detach().cpu().numpy()}
        else:
            result = {"logits": x.unsqueeze(0)}

        # row_attentions = torch.stack(row_attn_weights, 1)
        return result

    def loss(self, y, pred, d_mask):
        return torch.sum(((y - pred) * d_mask) ** 2) / (torch.sum(d_mask) + 1e-5)
