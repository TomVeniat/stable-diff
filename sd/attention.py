import math

import torch
from torch import nn, Tensor


def attention(q, k, v, causal_mask=False):
    """

    :param q: (b, h, seq_len, d)
    :param k: (b, h, seq_len, d)
    :param v: (b, h, seq_len, d)
    :return: (b, seq_len, d_model)
    """

    d_k = k.shape[-1]
    attention_weights = q @ k.transpose(-2, -1)  # (b, h, seq_len, seq_len)

    if causal_mask:
        mask = torch.ones_like(attention_weights, dtype=torch.bool).triu(1)
        attention_weights.masked_fill_(mask, -torch.inf)

    attention_weights /= math.sqrt(d_k)

    attention_weights = torch.softmax(attention_weights, dim=-1)

    out = attention_weights @ v  # (b, h, seq_len, d)
    return out


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_att: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.n_heads = n_heads
        assert d_att % n_heads == 0

        self.in_proj = nn.Linear(d_embed, d_att * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_att, d_embed, bias=out_proj_bias)
        self.d_heads = d_att // n_heads

    def forward(self, x: Tensor, causal_mask=False):
        """

        :param x: (b, seq_len, d_embed)
        :param causal_mask:
        :return:
        """
        in_shape = x.shape
        batch_size, seq_len, d_embed = in_shape

        interm_shape = (batch_size, seq_len, self.n_heads, self.d_heads)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interm_shape).transpose(1, 2)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)

        attention_maps = attention(q, k, v)

        attention_maps = attention_maps.transpose(1, 2)

        attention_maps = attention_maps.reshape(in_shape)

        out = self.out_proj(attention_maps)

        return out  # (b, seq_len, d_embed)


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, d_att: int, in_proj_bias: True, out_proj_bias=True):
        super().__init__()

        self.n_heads = n_heads
        self.d_att = d_att
        assert d_att % n_heads == 0
        self.d_heads = d_att // n_heads

        self.q_proj = nn.Linear(d_embed, d_att, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_att, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_att, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_att, d_embed, bias=out_proj_bias)

    def forward(self, x, y, causal_mask=False):
        """

        :param x: latent var (b, seq_len, d)
        :param y: context var, prompt (b, seq_len, d)
        :return:
        """
        q = x
        k = v = y
        in_shape = q.shape
        b, seq_len, d_in = in_shape

        tmp_shape = (b, -1, self.n_heads, self.d_heads)

        # print(f"q shape : {q.shape}")
        q = self.q_proj(q)
        # print(f"q shape : {q.shape}")
        q = q.view(tmp_shape).transpose(1, 2)
        # print(f"k shape : {k.shape}")
        k = self.k_proj(k)
        # print(f"k shape : {k.shape}")
        k = k.view(tmp_shape).transpose(1, 2)
        # print(f"v shape : {v.shape}")
        v = self.v_proj(v)
        # print(f"v shape : {v.shape}")
        v = v.view(tmp_shape).transpose(1, 2)

        scores = attention(q, k, v, causal_mask=causal_mask)  # (b, h, seq_len, d_head)
        scores = scores.transpose(1, 2).contiguous().view(b, seq_len, self.d_att)

        return self.out_proj(scores)
