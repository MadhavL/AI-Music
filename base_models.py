import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hparams import hparams_cross_modal_transformer, hparams_motion_transformer

class Norm(nn.Module):
    """Layer normalization."""

    def __init__(self, fn, normalized_shape):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=1e-5)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class Residual(nn.Module):
    """Residual layer."""

    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class MLP(nn.Module):
    """Feedforward layer."""

    def __init__(self, out_dim, hidden_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """Attention layer."""

    def __init__(self, dim, heads=8):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = torch.softmax(dots, dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """Transformer Encoder."""

    def __init__(self, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072, initializer_range=0.02):
        super(Transformer, self).__init__()
        blocks = []
        for _ in range(num_hidden_layers):
            blocks.extend([
                Residual(Norm(Attention(hidden_size, heads=num_attention_heads), hidden_size)),
                Residual(Norm(MLP(hidden_size, intermediate_size), hidden_size))
            ])
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

class LinearEmbedding(nn.Module):
    """Linear projection."""

    def __init__(self, dim_in, dim_out):
        super(LinearEmbedding, self).__init__()
        self.net = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return self.net(x)

class PositionEmbedding(nn.Module):
    """Position Embedding layer."""

    def __init__(self, seq_length, dim):
        super(PositionEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(seq_length, dim)) #Not truncated normal!

    def forward(self, x):
        return x + self.pos_embedding

class CrossModalLayer(nn.Module):
    """Cross-modal layer."""

    def __init__(self):
        super(CrossModalLayer, self).__init__()
        self.transformer_layer = Transformer(
            hidden_size=hparams_cross_modal_transformer['d_model'],
            num_hidden_layers=hparams_cross_modal_transformer['num_layers'],
            num_attention_heads=hparams_cross_modal_transformer['num_heads'],
            intermediate_size=hparams_cross_modal_transformer['intermediate_size'], # Confirm
            initializer_range=hparams_cross_modal_transformer['initializer_range']) #Confirm

        self.cross_output_layer = nn.Linear(
            in_features=hparams_cross_modal_transformer['d_model'], 
            out_features=hparams_motion_transformer['motion_num_features'])

    def forward(self, modal_a_sequences, modal_b_sequences):
        if modal_a_sequences.size(-1) != modal_b_sequences.size(-1):
            raise ValueError(
                "The modal_a hidden size (%d) should be the same with the modal_b "
                "hidden size (%d)" % (modal_a_sequences.size(-1), modal_b_sequences.size(-1)))
        merged_sequences = torch.cat([modal_a_sequences, modal_b_sequences], dim=1)
        merged_sequences = self.transformer_layer(merged_sequences)
        logits = self.cross_output_layer(merged_sequences)
        return logits
