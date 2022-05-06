import math
from functools import wraps, partial, reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.modules import FeedForwardNetwork, ESM1bLayerNorm, NormalizedResidualBlock, MLP

TOKEN_SELF_ATTN_VALUE = -5e4  # carefully set for half precision to work

# We followed implement in https://github.com/lucidrains/reformer-pytorch


def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)


def process_inputs_chunk(fn, chunks=1, dim=0):
    def inner_fn(*args, **kwargs):
        keys, values, len_args = kwargs.keys(), kwargs.values(), len(args)
        chunked_args = list(zip(*map(lambda x: x.chunk(chunks, dim=dim), list(args) + list(values))))
        all_args = map(lambda x: (x[:len_args], dict(zip(keys, x[len_args:]))), chunked_args)
        outputs = [fn(*c_args, **c_kwargs) for c_args, c_kwargs in all_args]
        return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))

    return inner_fn


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def exists(val):
    return val is not None


def batched_index_select(values, indices):
    last_dim = values.shape[-1]  # batch gene dim[batch,gene,
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def default(val, default_val):
    return default_val if val is None else val


def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)


def expand_dim(dim, k, t):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(qk, sinu_pos):
    sinu_pos = sinu_pos.type(qk.dtype)
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j=2)
    sin, cos = sinu_pos.unbind(dim=-2)
    sin, cos = map(lambda t: repeat(t, 'n d -> n (d j)', j=2), (sin, cos))
    seq_len = sin.shape[0]
    qk, qk_pass = qk[:, :seq_len], qk[:, seq_len:]
    qk = (qk * cos) + (rotate_every_two(qk) * sin)
    return torch.cat((qk, qk_pass), dim=1)


def cache_method_decorator(cache_attr, cache_namespace, reexecute=False):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache=True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'

            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val

        return wrapper

    return inner_fn


class LSHAttention(nn.Module):
    def __init__(self,
                 dropout=0.,
                 bucket_size=64,
                 n_hashes=4,
                 causal=False,
                 allow_duplicate_attention=True,
                 attend_across_buckets=True,
                 rehash_each_round=True,
                 drop_for_hash_rate=0.0,
                 random_rotations_per_head=False, dim_per_head=16,
                 return_attn=False):
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.bucket_size = bucket_size

        self.n_hashes = n_hashes
        self.dim_per_head = dim_per_head

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

        # cache buckets for reversible network, reported by authors to make Reformer work at depth
        self._cache = {}

    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        device = vecs.device

        assert n_buckets % 2 == 0

        rot_size = n_buckets
        rotations_shape = (
            # batch_size if self._random_rotations_per_head else 1,
            1,
            vecs.shape[-1],
            self.n_hashes,
            rot_size // 2)

        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1,
                                                                                                -1)  # batch,hidden,nhash,bin

        dropped_vecs = self.dropout_for_hash(vecs)  # batch,ngene,hidden
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)  # batch,nhash,ngene,bin

        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
        buckets = torch.argmax(rotated_vecs, dim=-1)  # batch,nhash,ngene,whichbin

        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
        buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        return buckets

    def forward(self, qk, v, query_len=None, return_attn=False, input_mask=None, **kwargs):
        batch_size, seqlen, dim, device = *qk.shape, qk.device
        query_len = default(query_len, seqlen)
        is_reverse = kwargs.pop('_reverse', False)
        depth = kwargs.pop('_depth', None)

        assert seqlen % (
                self.bucket_size * 2) == 0, f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - ' \
            f'{self.bucket_size * 2}'

        n_buckets = seqlen // self.bucket_size
        buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training)  # batch,
        # (nhash,ngene,whichbin)

        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen

        total_hashes = self.n_hashes

        ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        buckets_and_t = buckets_and_t.detach()

        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sticker.sort(dim=-1)
        del ticker

        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()

        st = (sticker % seqlen)
        sqk = batched_index_select(qk, st)  # batch gene dim
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = total_hashes * n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))

        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)
        lens = bk.shape[-1] // self.dim_per_head
        # Dot-product attention.
        dots = 0
        tk = 64
        if return_attn:
            for i in range(lens // tk):
                dots += (torch.einsum('bhie,bhje->bhij', bq[:, :, :, i * tk * self.dim_per_head:(i + 1) * tk * self.dim_per_head],
                                      bk[:, :, :, i * tk * self.dim_per_head:(i + 1) * tk * self.dim_per_head]) * (dim ** -0.5))
            if lens % tk != 0:
                dots += (torch.einsum('bhie,bhje->bhij', bq[:, :, :, -(len(bk) % tk) * self.dim_per_head:],
                                      bk[:, :, :, -(len(bk) % tk) * self.dim_per_head:]) * (dim ** -0.5))
        else:
            dots = (torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5))
        masked_value = max_neg_value(dots)

        if input_mask is not None:
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            # print((mask==1).sum(),batch_size,chunk_size)
            dots.masked_fill_(mask == 0, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, chunk_size, -1))
        bkv_buckets = look_one_back(bkv_buckets)
        bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
        dots.masked_fill_(bucket_mask, masked_value)
        del bucket_mask

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type_as(dots)

        bo = torch.einsum('buij,buje->buie', dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

        # unsort logits
        o = batched_index_select(so, undo_sort)
        logits = slogits.gather(1, undo_sort)

        o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))

        if query_len != seqlen:
            query_slice = (slice(None), slice(None), slice(0, query_len))
            o, logits = o[query_slice], logits[query_slice]

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

        attn = torch.empty(0, device=device)

        # return unsorted attention weights
        if return_attn:
            attn_unsort = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long()
            unsorted_dots = torch.zeros(batch_size * total_hashes, seqlen * seqlen, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, total_hashes, seqlen, seqlen)
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)

        # return output, attention matrix, and bucket distribution
        return out, attn, buckets


class LSHSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, bucket_size=64, n_hashes=4, causal=False, dim_head=None, attn_chunks=1,
                 random_rotations_per_head=False, attend_across_buckets=True, allow_duplicate_attention=True, num_mem_kv=0,
                 use_full_attn=False, full_attn_thres=None, post_attn_dropout=0.,
                 dropout=0., n_local_attn_heads=0, **kwargs):
        super().__init__()
        assert dim_head or (dim % heads) == 0, 'dimensions must be divisible by number of heads'
        assert n_local_attn_heads < heads, 'local attention heads must be less than number of heads'

        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.attn_chunks = default(attn_chunks, 1)
        self.dim_per_head = dim // heads
        v_dim = dim_heads

        self.toqk = nn.Linear(dim, dim_heads, bias=False)
        self.tov = nn.Linear(dim, v_dim, bias=False)

        self.bucket_size = bucket_size
        self.lsh_attn = LSHAttention(bucket_size=bucket_size, n_hashes=n_hashes, causal=causal, dim_per_head=self.dim_per_head,
                                     random_rotations_per_head=random_rotations_per_head,
                                     attend_across_buckets=attend_across_buckets,
                                     allow_duplicate_attention=allow_duplicate_attention,
                                     dropout=dropout, **kwargs)

        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.use_full_attn = use_full_attn
        self.full_attn_thres = default(full_attn_thres, bucket_size)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True)) if num_mem_kv > 0 else None

        self.n_local_attn_heads = n_local_attn_heads

        self.callback = None

    def forward(self, x, return_attn, input_mask=None, **kwargs):
        b, t, e, h, dh, m = *x.shape, self.heads, self.dim_head, self.num_mem_kv
        kv_len = t

        qk = self.toqk(x)  # batch gene dim
        v = self.tov(x)

        def merge_heads(v):
            return v.view(b, kv_len, h, -1).transpose(1, 2)  # batch,head,len,feature

        merge_batch_and_heads = partial(merge_dims, 0, 1)

        qk, v = map(merge_heads, (qk, v))
        qk = qk.permute(1, 2, 0, 3).reshape(h, t, b * e // h)
        v = v.permute(1, 2, 0, 3).reshape(h, t, b * e // h)
        # qk, v = map(merge_batch_and_heads, (qk, v))
        # qk = qk.permute(1,2,3,0).view()
        # v = v.permute(1,2,3,0)
        masks = {}
        if input_mask is not None:

            mask = input_mask.expand(h, t, )
            masks['input_mask'] = mask
        attn_fn = self.lsh_attn
        partial_attn_fn = partial(attn_fn, return_attn=return_attn, query_len=t, **kwargs)
        attn_fn_in_chunks = process_inputs_chunk(partial_attn_fn, chunks=self.attn_chunks)
        out, attn, buckets = attn_fn_in_chunks(qk, v, **masks)
        out = out.reshape(h, t, e // h, b).permute(2, 0, 1, 3)
        if self.callback is not None:
            self.callback(attn.reshape(b, h, t, -1), buckets.reshape(b, h, -1))

        def split_heads(v):
            return v.reshape(b, h, t, -1).transpose(1, 2).contiguous()

        out = split_heads(out).view(b, t, -1)
        return out, attn


class GRNTransformer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 64,
            ffn_embedding_dim: int = 64,
            num_attention_heads: int = 4, bucket_size=64
    ) -> None:
        super().__init__()
        self.bucket_size = bucket_size
        self.embedding_dim = embedding_dim

        row_self_attention = LSHSelfAttention(
            embedding_dim,
            num_attention_heads, bucket_size=bucket_size
        )
        col_self_attention = LSHSelfAttention(
            embedding_dim,
            num_attention_heads, bucket_size=bucket_size
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

    def forward(self, x, return_attn=True, n_cell=0, n_gene=0):
        pad_len = math.ceil(x.shape[1] / (self.bucket_size * 2)) * (self.bucket_size * 2) - x.shape[1]
        x = F.pad(x, [0, 0, 0, pad_len])
        input_mask = torch.ones(x.shape[1]).to(x.device)
        input_mask[-pad_len:] = 0
        x, attn = self.attention(x, return_attn=return_attn, input_mask=input_mask)

        x = x[:, :n_gene]
        x = x.transpose(0, 1)

        pad_len2 = math.ceil(x.shape[1] / (self.bucket_size * 2)) * (self.bucket_size * 2) - x.shape[1]
        x = F.pad(x, [0, 0, 0, pad_len2])
        input_mask2 = torch.ones(x.shape[1]).to(x.device)
        input_mask2[-pad_len2:] = 0
        x, _ = self.col_attention(x, return_attn=return_attn, input_mask=input_mask2)
        x = x.transpose(0, 1)[:n_cell]
        x = self.feed_forward_layer(x)
        return x, attn


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_X = nn.Sequential(nn.Linear(1, self.args.embed_dim), nn.ReLU(),
                                     nn.Linear(self.args.embed_dim, self.args.embed_dim // 2))
        self.zeros_embed = nn.Linear(1, self.args.embed_dim // 2)
        self.bucket_size = 64
        self.layers = nn.ModuleList(
            [
                GRNTransformer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    self.args.attention_heads, bucket_size=self.bucket_size
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
            x, attn = layer(x, return_attn=return_attn, n_cell=n_cell, n_gene=n_gene)
            row_attn_weights.append(attn)
        x = self.emb_layer_norm_after(x)
        x = self.lm_head(x)[:, :n_gene]
        x = x[:n_cell]
        if return_attn:
            result = {"logits": x.unsqueeze(0),
                      'row_attentions': torch.cat(row_attn_weights).permute(2, 1, 0).contiguous().detach().cpu().numpy()[:,
                                        :n_gene, :n_gene]}
        else:
            result = {"logits": x.unsqueeze(0)}

        return result

    def loss(self, y, pred, d_mask):
        return torch.sum(((y - pred) * d_mask) ** 2) / (torch.sum(d_mask) + 1e-5)
