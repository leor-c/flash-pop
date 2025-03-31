from typing import Optional, Union
from dataclasses import dataclass

import torch
from torch import Tensor

from retnet import MultiScaleRetention, _theta_shift


@torch.compile()
def apply_relative_position_pred_tokens(
        q, k, start_idx: Union[int, torch.Tensor], thetas: Tensor, tokens_per_block: int
) -> tuple[Tensor, Tensor]:
    # b num_blocks len h dk
    assert q.dim() == 5 and k.dim() == 5
    indices = torch.arange(q.size(2), device=q.device, dtype=q.dtype).reshape(1, -1)
    block_steps = torch.arange(q.size(1), device=q.device, dtype=q.dtype).reshape(-1, 1) * tokens_per_block
    indices = indices + block_steps
    indices = indices.flatten()
    # b t k h d -> b (t k) h d
    # q = q.flatten(1, 2)
    # k = k.flatten(1, 2)

    if isinstance(start_idx, int):
        assert thetas is not None
        # Combined (cross + intra chunk):
        indices = start_idx + indices
        indices = indices.reshape(1, 1, -1, 1)

    elif isinstance(start_idx, torch.Tensor):
        assert start_idx.dim() == 1
        indices = start_idx.view(-1, 1) + indices.view(1, -1)
        indices = indices.reshape(indices.shape[0], 1, indices.shape[1], 1)

    else:
        assert False, f"Unsupported type for start_index. Expected int or LongTensor, got '{type(start_idx)}'."

    thetas = thetas.reshape(1, 1, 1, -1)
    angles = indices * thetas
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    q = _theta_shift(q, sin, cos)
    k = _theta_shift(k, sin, cos)

    return q, k


class POPMultiScaleRetention(MultiScaleRetention):
    @dataclass(kw_only=True)
    class Config(MultiScaleRetention.Config):
        block_size: int

    def __init__(self, config: MultiScaleRetention.Config):
        super().__init__(config)
        self.config = config

    def pop_forward(self, x: Tensor, suffixes: Tensor, start_index: int, prev_state: Optional[Tensor]) -> Tensor:
        assert x.dim() == 3, f"Got {x.dim()}"  # b t (h d)
        assert suffixes.dim() == 4, f"Got {suffixes.dim()}"  # b n k (h d) where n=num blocks, k is sfx length





