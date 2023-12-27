import torch
from torch import nn
from torch.nn import functional as F

from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        x = self.token_embedding(tokens)

        x += self.position_embedding
        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_heads, n_embed):
        super().__init__()

        self.norm_1 = nn.LayerNorm(n_embed)
        self.norm_2 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_heads, n_embed, n_embed)
        self.proj_1 = nn.Linear(n_embed, 4 * n_embed)
        self.proj_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        x = self.norm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        residue = x
        x = self.norm_2(x)
        x = self.proj_1(x)
        x = x * torch.sigmoid(1.702 * x)  # QuickGelu activation
        x = self.proj_2(x)

        return x + residue


class CLIP(nn.Module):
    def __init__(self, n_feats: int = 768):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, n_feats, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, n_feats) for _ in range(12)
        ])

        self.layernorm = nn.LayerNorm(n_feats)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)  # (b, seq_len, n_feats)

        for layer in self.layers:
            state = layer(state)

        out = self.layernorm(state)
        return out
