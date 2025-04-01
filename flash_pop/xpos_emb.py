from typing import Union, Optional

from functools import lru_cache
import torch
import torch.nn as nn
from torch import Tensor

from einops import rearrange, repeat


@lru_cache(maxsize=1)
def _build_position_thetas(
    head_dim: int,
    scale: float = 10000,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Positional thetas are different for each value along head_dim, following the
    prescribed method in the paper.  These are used to update the positional
    embeddings in both the parallel and recurrent formulations of retention.
    See: https://arxiv.org/pdf/2307.08621v3.pdf, Section 2.1 (Retention)

    NOTE: The actual values for thetas are not specified in the paper, so I
    copied these values from the official implementation.
    See: https://github.com/microsoft/torchscale/blob/7d231743f4f96c460b7cf0aa0cf242bb192b34f8/torchscale/architecture/retnet.py#L27C1-L28C59
    """
    x = torch.linspace(0, 1, steps=head_dim // 2, device=device, dtype=dtype)
    thetas = 1 / (scale**x)
    return repeat(thetas, "d -> (d n)", n=2)


@torch.compile()
def _multiply_by_i(x: Tensor) -> Tensor:
    """Multiply a complex-valued tensor by the imaginary unit 'i'."""
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(start_dim=-2)


@torch.compile()
def _theta_shift(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (_multiply_by_i(x) * sin)


@torch.compile()
def _get_sin_cos(
        seq_len: int,
        start_idx: Union[int, torch.Tensor],
        thetas: Tensor,
) -> tuple[Tensor, Tensor]:
    device, dtype = thetas.device, thetas.dtype
    indices = torch.arange(seq_len, device=device, dtype=dtype)

    if isinstance(start_idx, int):
        assert thetas is not None
        # Combined (cross + intra chunk):
        indices = start_idx + indices
        indices = indices.reshape(1, -1, 1, 1)

    elif isinstance(start_idx, torch.Tensor):
        # designed for the training phase of POP, where we flatten suffixes into the batch dimension
        # here we assume that start_idx has the (final) batch dimension
        # start_idx entries determine the offsets of individual batch entries.
        assert start_idx.dim() == 1
        indices = start_idx.view(-1, 1) + indices.view(1, -1)
        indices = indices.reshape(start_idx.shape[0], indices.shape[1], 1, 1)

    else:
        assert False, f"Unsupported type for start_index. Expected int or LongTensor, got '{type(start_idx)}'."

    thetas = thetas.reshape(1, 1, 1, -1)
    angles = indices * thetas
    sin = torch.sin(angles)
    cos = torch.cos(angles)

    return sin, cos


class XPos(nn.Module):

    def __init__(
            self,
            head_dim_qk: int,
            seq_len_estimate: int = 2 ** 12,
            start_idx: Union[int, torch.Tensor] = 0,
            device=None,
            dtype=None,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cos_cache = None
        self.sin_cache = None
        self.start_idx = start_idx

        # 'thetas' parameter for updating the relative position embeddings.
        self.thetas = _build_position_thetas(
            head_dim=head_dim_qk, device=device, dtype=dtype
        )
        # self.register_buffer("thetas", self.thetas)

        self.sin_cache, self.cos_cache = _get_sin_cos(
            seq_len=seq_len_estimate,
            start_idx=start_idx,
            thetas=self.thetas,
        )
        # self.register_buffer("cos_cache", self.cos_cache)
        # self.register_buffer("sin_cache", self.sin_cache)

        self.seq_len = self.sin_cache.shape[1]

    def get_sin_cos(self, seq_len, start_idx):
        if isinstance(start_idx, int):
            if seq_len > self.seq_len:
                self.seq_len = seq_len
                self.sin_cache, self.cos_cache = _get_sin_cos(seq_len, start_idx, self.thetas)
            end_idx = start_idx + seq_len
            return self.sin_cache[:, start_idx:end_idx], self.cos_cache[:, start_idx:end_idx]
        else:
            assert isinstance(start_idx, torch.Tensor)
            if start_idx.numel() != self.start_idx.numel() or torch.any(start_idx != self.start_idx):
                self.sin_cache, self.cos_cache = _get_sin_cos(seq_len, start_idx, self.thetas)
            return self.sin_cache, self.cos_cache

    def __call__(self, q, k, start_idx: Union[int, torch.Tensor], *args, **kwargs):
        dtype = q.dtype
        sin, cos = self.get_sin_cos(seq_len=q.size(1), start_idx=start_idx)
        q = _theta_shift(q, sin, cos).to(dtype=dtype)
        k = _theta_shift(k, sin, cos).to(dtype=dtype)

        return q, k



