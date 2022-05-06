import math
from distutils.version import LooseVersion
from functools import partial

import torch
import torch.nn as nn
from einops import repeat

from src.modules import FeedForwardNetwork, ESM1bLayerNorm, NormalizedResidualBlock, MLP

TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')
# We followed implement in https://github.com/lucidrains/performer-pytorch

def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    if TORCH_GE_1_8_0:
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    else:
        q, r = torch.qr(unstructured_block.cpu(), some=True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data -
                          torch.amax(data_dash, dim=-1, keepdim=True)) + eps)
    else:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True)) + eps)

    return data_dash.type_as(data)


class Gene_Performer_attn(nn.Module):
    def __init__(self, embed_dim, num_heads, ortho_scaling=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.nb_features = embed_dim
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features, nb_columns=self.head_dim,
                                         scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = False
        self.kernel_fn = nn.ReLU()

        self.no_projection = False
        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)

    def linear_attention(self, q, k, v):
        k_cumsum = k.sum(dim=-2)
        D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))  # q batch head gene hidden
        context = torch.einsum('...nd,...ne->...de', k, v)
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        return out

    def forward(self, x, return_attn=True):
        if not return_attn:
            device = x.device
            num_rows, num_cols, embed_dim = x.size()
            q = self.to_q(x).view(num_rows, num_cols, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.to_k(x).view(num_rows, num_cols, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.to_v(x).view(num_rows, num_cols, self.num_heads, self.head_dim).transpose(1, 2)
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)
            q = q.permute(1, 2, 0, 3).reshape(self.num_heads, num_cols, -1)
            k = k.permute(1, 2, 0, 3).reshape(self.num_heads, num_cols, -1)
            v = v.permute(1, 2, 0, 3).reshape(self.num_heads, num_cols, -1)

            attn_fn = self.linear_attention
            out = attn_fn(q, k, v)
            out = out.view(self.num_heads, num_cols, num_rows, self.head_dim).transpose(0, 2).reshape(num_rows, num_cols,
                                                                                                      embed_dim)
            return out, None
        else:
            return self.pred(x)

    def pred(self, x):
        device = x.device
        num_rows, num_cols, embed_dim = x.size()
        q = self.to_q(x).view(num_rows, num_cols, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(num_rows, num_cols, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(num_rows, num_cols, self.num_heads, self.head_dim).transpose(1, 2)
        create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
        q = create_kernel(q, is_query=True)
        k = create_kernel(k, is_query=False)
        q = q.permute(1, 2, 0, 3).reshape(self.num_heads, num_cols, -1)
        k = k.permute(1, 2, 0, 3).reshape(self.num_heads, num_cols, -1)
        v = v.permute(1, 2, 0, 3).reshape(self.num_heads, num_cols, -1)

        k_cumsum = k.sum(dim=-2)
        D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))  # q  head gene hidden
        context = torch.einsum('...nd,...ne->...de', k, v)
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        out = out.view(self.num_heads, num_cols, num_rows, self.head_dim).transpose(0, 2).reshape(num_rows, num_cols, embed_dim)
        attn = torch.einsum('hqf,hkf->hqk', q, k)
        return out, attn


class cell_Performer_attn(nn.Module):
    def __init__(self, embed_dim, num_heads, ortho_scaling=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.nb_features = embed_dim
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features, nb_columns=self.head_dim,
                                         scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = False
        self.kernel_fn = nn.ReLU()

        self.no_projection = False
        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)

    def linear_attention(self, q, k, v):
        k_cumsum = k.sum(dim=-2)
        D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))  # q batch head gene hidden
        context = torch.einsum('...nd,...ne->...de', k, v)
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        return out

    def forward(self, x, return_attn=True):
        if not return_attn:
            device = x.device
            num_rows, num_cols, embed_dim = x.size()
            q = self.to_q(x).view(num_rows, num_cols, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.to_k(x).view(num_rows, num_cols, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.to_v(x).view(num_rows, num_cols, self.num_heads, self.head_dim).transpose(1, 2)
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)
            q = q.mean(0)
            k = k.mean(0)
            v = v.permute(1, 2, 0, 3).reshape(self.num_heads, num_cols, -1)

            attn_fn = self.linear_attention
            out = attn_fn(q, k, v)
            out = out.view(self.num_heads, num_cols, num_rows, self.head_dim).transpose(0, 2).reshape(num_rows, num_cols,
                                                                                                      embed_dim)
            return out, None
        else:
            return self.pred(x)

    def pred(self, x):
        device = x.device
        num_rows, num_cols, embed_dim = x.size()
        q = self.to_q(x).view(num_rows, num_cols, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(num_rows, num_cols, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(num_rows, num_cols, self.num_heads, self.head_dim).transpose(1, 2)
        create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
        q = create_kernel(q, is_query=True)
        k = create_kernel(k, is_query=False)
        q = q.mean(0)
        k = k.mean(0)
        v = v.permute(1, 2, 0, 3).reshape(self.num_heads, num_cols, -1)

        k_cumsum = k.sum(dim=-2)
        D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))  # q  head gene hidden
        context = torch.einsum('...nd,...ne->...de', k, v)
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        out = out.view(self.num_heads, num_cols, num_rows, self.head_dim).transpose(0, 2).reshape(num_rows, num_cols, embed_dim)
        attn = torch.einsum('hqf,hkf->hqk', q, k)
        return out, attn


class GRNTransformer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 64,
            ffn_embedding_dim: int = 64,
            num_attention_heads: int = 4,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        row_self_attention = Gene_Performer_attn(
            embedding_dim,
            num_attention_heads,
        )

        feed_forward_layer = FeedForwardNetwork(
            embedding_dim,
            ffn_embedding_dim,
        )

        self.attention = self.build_residual(row_self_attention)
        self.col_attention = self.build_residual(cell_Performer_attn(embedding_dim, num_attention_heads))

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
        x = self.lm_head(x)[:, :n_gene]
        if return_attn:
            result = {"logits": x.unsqueeze(0),
                      'row_attentions': torch.cat(row_attn_weights).permute(2, 1, 0).contiguous().detach().cpu().numpy()}
        else:
            result = {"logits": x.unsqueeze(0)}

        # row_attentions = torch.stack(row_attn_weights, 1)
        return result

    def loss(self, y, pred, d_mask):
        return torch.sum(((y - pred) * d_mask) ** 2) / (torch.sum(d_mask) + 1e-5)
