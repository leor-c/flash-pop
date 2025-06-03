import math
import torch
import torch.nn as nn
from einops import rearrange


"""
Taken from https://github.com/facebookresearch/DiT/blob/main/models.py
Adapted for finer time conditioning e.g., per-token or per-frame
"""


#################################################################################
#                       Embedding Layers for Timesteps                          #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a Tensor of (Batch, [Timesteps], [Tokens per step]) time "indices", 
        one per batch element. These may be fractional.
        :param dim: the embedding dimension of the output (features).
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (*input_shape, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        shape = t.shape
        args = t.flatten()[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.reshape(*shape, dim)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

