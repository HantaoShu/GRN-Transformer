import math

import torch
import torch.nn as nn


class Attention(nn.Module):

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

    def compute_attention_weights(self, x, network, scaling, num_rows, num_cols, batch_size):

        q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        q *= scaling
        network_attention = self.attention_network_selection(q, k, network, batch_size, num_cols).repeat(self.num_heads, 1, 1, 1)
        attn_weights = torch.einsum(f"rinhd,rjnhd->hnij", q, k)
        return attn_weights, network_attention

    def compute_attention_update(self, x, attn_probs):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        context = torch.einsum(f"hnij,rjnhd->rinhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output

    def forward(self, x, network, train=True):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if train:
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
        network_attentions = 0
        for i in (range(n)):
            x_in = x[i * m:min(len(x), (i + 1) * m)]
            attn_weight, network_attention = self.compute_attention_weights(x_in, network, scaling, m, num_cols, batch_size)
            attn_weights += attn_weight
            network_attentions += network_attention
        network_attentions = network_attentions.softmax(-1)
        network = network.view(batch_size, num_cols, num_cols, -1)
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
