# Based on https://github.com/fkodom/yet-another-retnet/blob/main/yet_another_retnet/retention.py
# Copyright (c) 2022 Frank Odom
# Copyright (c) 2025 Lior Cohen
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, log
from typing import Union, Callable, Optional, List, Sequence, Tuple, Literal

import numpy as np
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from einops import rearrange, einsum, repeat
from torch import Tensor

from fused_retention import fused_chunk_retention

ActivationString = Literal["swish", "gelu", "relu"]


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """Return an activation function given a string"""
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    else:
        raise RuntimeError(
            f"Unsupported activation string '{activation}'. "
            "Supported: 'swish', 'gelu', 'relu'"
        )


DECAY_SCALE_MIN_NUM_BLOCKS = 4
DECAY_SCALE_MAX_NUM_BLOCKS = 512


@lru_cache(maxsize=1)
def _build_decay_gammas(
        num_heads: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
) -> Tensor:
    """Decay values are different for each retention head, following the prescribed
    method in the paper.  Conceptually, I think of each head having a different
    "retention window", which is the effective number of steps back in time that
    the head can attend to.  Retention windows are effectively determined by
    these decay coefficients.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Section 3.1 (Setup)
    """
    if xmin is None:
        xmin = log(1 / 32)
    if xmax is None:
        xmax = log(1 / 512)
    x = torch.linspace(xmin, xmax, steps=num_heads, device=device, dtype=dtype)
    return 1 - x.exp_()


@lru_cache(maxsize=1)
def get_decays(num_heads: int, decay_range: Optional[tuple[float, float]] = None) -> np.ndarray:
    if decay_range is None:
        decay_exp = -5 -np.arange(num_heads)
    else:
        decay_exp = -np.linspace(decay_range[0], decay_range[1], num_heads)
    return 1 - np.exp2(decay_exp)


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
    # TODO: Add docstring
    return (x * cos) + (_multiply_by_i(x) * sin)


@torch.compile()
def apply_relative_position(q, k, start_idx: Union[int, torch.Tensor], thetas: Tensor) -> Tuple[Tensor, Tensor]:
    indices = torch.arange(q.size(2), device=q.device, dtype=q.dtype)

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


def retention_recurrent(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    head_decays: Tensor,
    prev_state: Optional[Tensor],
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    assert head_decays.dim() == 4 and head_decays.size(0) == 1 and head_decays.size(2) == 1 and head_decays.size(3) == 1
    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - d: hidden_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    state = einsum(key, value, "b h dk, b h dv -> b h dk dv")
    if prev_state is not None:
        state = state + prev_state * head_decays
    retention = einsum(query, state, "b h dk, b h dk dv -> b h dv")

    return retention, state


class MultiScaleRetention(nn.Module):
    """Multi-scale retention (MSR) layer.  Intended to be (mostly) a drop-in replacement
        for nn.MultiheadAttention, but with the option to use either the parallel or
        recurrent formulation of retention. (Attention only has the parallel formulation.)

        NOTE: As presented in the paper, Multi-Scale Retention includes an explicit
        position embedding, which is based on xPos.  IMO, this is unnecessary and overly
        specific to language modeling, since other domains (e.g. computer vision,
        heterogeneous graphs) will have different positional semantics.

        I have made the relational position embedding optional, so that this module
        can (in theory) support more modalities. Setting 'relative_position=False' will
        remove the positional embedding, and instead rely on the query and key
        embeddings to encode positional information ahead of time (if needed at all).
        See: https://github.com/microsoft/torchscale/issues/48

        Reference:
            "Retentive Network: A Successor to Transformer for Large Language Models"
            https://arxiv.org/pdf/2307.08621v3.pdf
        """
    @dataclass(kw_only=True)
    class Config:
        num_heads: int
        head_dim_v: int
        head_dim_qk: int = None
        dropout: float = 0.0
        head_decays_range: Optional[Tuple[float, float]] = None
        relative_position: bool = True
        bias: bool = True
        activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish"
        group_norm_eps: float = 1e-6
        device: Optional[Union[torch.device, str]] = torch.device('cuda')
        dtype: Optional[torch.dtype] = torch.bfloat16
    def __init__(
            self,
            config: Config,
    ):
        """"""
        activation = config.activation
        if isinstance(config.activation, str):
            activation = _get_activation_fn(config.activation)

        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim_v = config.head_dim_v
        self.head_dim_qk = config.head_dim_v if config.head_dim_qk is None else config.head_dim_qk
        embed_dim = self.head_dim_v * self.num_heads
        self.embed_dim = embed_dim
        self.dropout = config.dropout
        self.relative_position = config.relative_position
        self.bias = config.bias
        self.activation = activation
        self.head_decays = tuple(get_decays(self.num_heads, config.head_decays_range).tolist())
        self.head_decays_torch = torch.tensor(self.head_decays, dtype=torch.float32, device=config.device)
        self.head_decays_torch = rearrange(self.head_decays_torch, "h -> () h () ()")

        if embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        if not self.head_dim_v % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {self.head_dim_v}) must be divisible by 8"
            )

        device, dtype = config.device, config.dtype
        bias = config.bias

        # The q/k/v projection layers are the same as in vanilla MHA.
        self.q_proj = nn.Linear(
            embed_dim, self.num_heads * self.head_dim_qk, bias=bias, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            embed_dim, self.num_heads * self.head_dim_qk, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.group_norm = nn.GroupNorm(
            num_groups=self.num_heads,
            num_channels=embed_dim,
            affine=False,
            eps=config.group_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )
        # The output project is slightly different, due to the gated "swish" layer.
        self.g_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        # 'thetas' parameter for updating the relative position embeddings.
        thetas: Optional[Tensor] = None
        if self.relative_position:
            thetas = _build_position_thetas(
                head_dim=self.head_dim_qk, device=device, dtype=dtype
            )
        self.thetas: Optional[Tensor]
        self.register_buffer("thetas", thetas)

        self._reset_parameters()

    def _reset_parameters(self):
        # TODO: Double-check that we're following the same initialization as in
        # the paper.  This is a generic initialization for MHA linear layers.
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)
        nn.init.xavier_normal_(self.v_proj.weight)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)
        nn.init.xavier_normal_(self.g_proj.weight)
        if self.g_proj.bias is not None:
            nn.init.constant_(self.g_proj.bias, 0)

    def forward_chunkwise(
            self,
            x: Tensor,
            start_idx: Union[int, torch.LongTensor],
            prev_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # einstein notation:
        # b - batch size
        # n - sequence length
        # h - number of heads
        # d - head dimension
        #
        # Input shape: (b, n, dim_v)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Unfold 'd' dimension into 'h' separate retention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.num_heads)

        if self.relative_position:
            # global (cross-chunk) + intra-chunk relative position embedding
            assert self.thetas is not None
            q, k = apply_relative_position(q, k, start_idx, self.thetas)

        if prev_state is None:
            batch_size, seq_len, num_heads, dim_v = v.shape
            dim_qk = q.shape[-1]
            prev_state = torch.zeros((batch_size, num_heads, dim_qk, dim_v), device=x.device, dtype=x.dtype)

        # Apply retention then group norm.
        retention, state = fused_chunk_retention(
            q, k, v, prev_state, self.head_decays
        )
        # To apply group norm in an equivalent way to the recurrent formulation,
        # we fold the sequence dimension into the batch dimension.  Otherwise,
        # normalization would be applied over the entire input sequence.
        batch_size = retention.size(0)
        retention = rearrange(retention, "b n h d -> (b n) (h d)")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) e -> b n e", b=batch_size)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.
        gate = self.activation(self.g_proj(x))
        retention = self.out_proj(retention * gate)

        return retention, state

    def forward_recurrent(
            self,
            x: Tensor,
            seq_idx: int,
            prev_state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # einstein notation:
        # b - batch size
        # h - number of heads
        # d - embedding dimension
        #
        # input shape: (b, d)
        q: Tensor = self.q_proj(x)
        k: Tensor = self.k_proj(x)
        v: Tensor = self.v_proj(x)

        # Unfold 'd' dimension into 'h' separate retention heads.
        q = rearrange(q, "b (h d) -> b h d", h=self.num_heads)
        k = rearrange(k, "b (h d) -> b h d", h=self.num_heads)
        v = rearrange(v, "b (h d) -> b h d", h=self.num_heads)

        if self.relative_position:
            assert self.thetas is not None
            thetas = rearrange(self.thetas, "d -> () () d")
            angles = seq_idx * thetas
            sin = torch.sin(angles)
            cos = torch.cos(angles)

            q = _theta_shift(q, sin, cos)
            k = _theta_shift(k, sin, cos)

        # Apply retention then group norm.
        retention, state = retention_recurrent(q, k, v, self.head_decays_torch, prev_state=prev_state)
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        # Fold heads back into the embedding dimension.
        retention = rearrange(retention, "b h d -> b (h d)")
        retention = self.group_norm(retention)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.
        gate = self.activation(self.g_proj(x))
        retention = self.out_proj(retention * gate)

        return retention, state


class RetNetDecoderLayer(nn.Module):

    # NOTE: Mostly pulled from 'nn.TransformerDecoderLayer', but with changes:
    #   - use MultiScaleRetention instead of MultiheadAttention
    #   - no cross-attention layer, since retention doesn't play well with that

    @dataclass(kw_only=True)
    class Config:
        num_heads: int
        head_dim_v: int
        head_dim_qk: int = None
        dim_feedforward: int = 2048
        dropout: float = 0.1
        head_decays_range: tuple[float, float] = None
        activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish"
        norm_first: bool = True
        layer_norm_eps: float = 1e-6
        device: Optional[Union[torch.device, str]] = torch.device('cuda')
        dtype: Optional[torch.dtype] = torch.bfloat16

    def __init__(
        self,
        config: Config,
    ) -> None:
        """

        :param num_heads: number of attention heads
        :param head_dim_v: the dimension of each attention head. This defines d_model, i.e., embedding dimension,
        through d_model = num_heads * head_dim_v.
        :param head_dim_qk: the query and key dimension of each attention head. If none, `head_dim_v` is used.
        Lower values (around 0.5-0.75*head_dim_v) were shown to be effective while reducing computational cost.
        :param dim_feedforward: the dimension of feedforward layer (hidden)
        :param dropout:
        :param activation:
        :param norm_first:
        :param layer_norm_eps:
        :param device:
        :param dtype:
        """
        activation = config.activation
        if isinstance(config.activation, str):
            activation = _get_activation_fn(config.activation)

        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.activation = activation
        self.norm_first = config.norm_first
        d_model = config.num_heads * config.head_dim_v
        # retention block
        self.norm1 = nn.LayerNorm(
            d_model,
            eps=config.layer_norm_eps,
            device=config.device,
            dtype=config.dtype
        )
        self.retention = MultiScaleRetention(MultiScaleRetention.Config(
            num_heads=config.num_heads,
            head_dim_v=config.head_dim_v,
            head_dim_qk=config.head_dim_qk,
            dropout=config.dropout,
            head_decays_range=config.head_decays_range,
            activation=activation,
            device=config.device,
            dtype=config.dtype,
        ))
        # feedforward block
        self.norm2 = nn.LayerNorm(
            d_model,
            eps=config.layer_norm_eps,
            device=config.device,
            dtype=config.dtype
        )
        self.linear1 = nn.Linear(d_model, config.dim_feedforward, device=config.device, dtype=config.dtype)
        self.linear2 = nn.Linear(config.dim_feedforward, d_model, device=config.device, dtype=config.dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        # TODO: Check that we're following the same initialization as the paper
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def _feedforward_block(self, x: Tensor) -> Tensor:
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

    def forward_chunkwise(
            self, x: Tensor, start_idx: int, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        # retention block
        if self.norm_first:
            y, state = self.retention.forward_chunkwise(self.norm1(x), start_idx=start_idx, prev_state=prev_state)
            x = x + y
            x = x + self._feedforward_block(self.norm2(x))
        else:
            y, state = self.retention.forward_chunkwise(x, start_idx=start_idx, prev_state=prev_state)
            x = x + self.norm1(y)
            x = x + self.norm2(self._feedforward_block(x))

        return x, state

    def forward_recurrent(
            self, x: Tensor, seq_idx: int, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        def _retention_block(x: Tensor) -> Tuple[Tensor, Tensor]:
            x, state = self.retention.forward_recurrent(
                x, seq_idx=seq_idx, prev_state=prev_state
            )
            return self.dropout(x), state

        # retention block
        if self.norm_first:
            y, state = _retention_block(self.norm1(x))
            x = x + y
            x = x + self._feedforward_block(self.norm2(x))
        else:
            y, state = _retention_block(x)
            x = x + self.norm1(y)
            x = x + self.norm2(self._feedforward_block(x))

        return x, state

    def forward(self, x: Tensor, start_idx: int = 0, prev_state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        return self.forward_chunkwise(x, start_idx, prev_state)


class RetNetDecoder(nn.Module):
    def __init__(self, layer_config: RetNetDecoderLayer.Config, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [RetNetDecoderLayer(layer_config) for _ in range(num_layers)]
        )

    def forward_recurrent(
            self, x: Tensor, seq_idx: int, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, List[Tensor]]:
        if not prev_states:
            prev_states = [None] * self.num_layers
        elif len(prev_states) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} previous states, got {len(prev_states)}"
            )

        states: List[Tensor] = []
        for layer, prev_state in zip(self.layers, prev_states):
            assert isinstance(layer, RetNetDecoderLayer)
            x, state = layer.forward_recurrent(x, seq_idx, prev_state)
            states.append(state)
        return x, states

    def forward_chunkwise(
            self, x: Tensor, start_idx: int = 0, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, List[Tensor]]:
        if not prev_states:
            prev_states = [None] * self.num_layers
        elif len(prev_states) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} previous states, got {len(prev_states)}"
            )

        states: List[Tensor] = []
        for layer, prev_state in zip(self.layers, prev_states):
            assert isinstance(layer, RetNetDecoderLayer)
            x, state = layer.forward_chunkwise(x, start_idx, prev_state)
            states.append(state)
        return x, states

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        return self.forward_chunkwise(x)
