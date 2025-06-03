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

from flash_pop.fused_retention import fused_chunk_retention
from flash_pop.xpos_emb import XPos
from flash_pop.retnet import ActivationString, _get_activation_fn, MultiScaleRetention
from modules.adaln import AdaLayerNorm, AdaLayerNormZero
from modules.timestep import TimestepEmbedder


class DiffusionRetNetDecoderLayer(nn.Module):

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
        use_post_retention_dropout: bool = True
        head_decays_range: tuple[float, float] = None
        activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish"
        layer_norm_eps: float = 1e-6
        ada_ln_lora_dim: Optional[int] = None
        device: Optional[Union[torch.device, str]] = torch.device('cuda')
        dtype: Optional[torch.dtype] = torch.bfloat16

    def __init__(
        self,
        config: Config,
        xpos_embedder: Optional[XPos] = None
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
        self.config = config
        activation = config.activation
        if isinstance(config.activation, str):
            activation = _get_activation_fn(config.activation)

        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.use_post_retention_dropout = config.use_post_retention_dropout
        self.activation = activation
        d_model = config.num_heads * config.head_dim_v
        # retention block
        self.ada_ln1 = AdaLayerNormZero(
            d_model,
            eps=config.layer_norm_eps,
            lora_hidden_dim=config.ada_ln_lora_dim,
            device=config.device,
            dtype=config.dtype,
        )
        self.retention = self._build_multi_scale_retention(xpos_embedder=xpos_embedder)
        # feedforward block
        self.ada_ln2 = AdaLayerNormZero(
            d_model,
            eps=config.layer_norm_eps,
            lora_hidden_dim=config.ada_ln_lora_dim,
            device=config.device,
            dtype=config.dtype,
        )
        self.linear1 = nn.Linear(d_model, config.dim_feedforward, device=config.device, dtype=config.dtype)
        self.linear2 = nn.Linear(config.dim_feedforward, d_model, device=config.device, dtype=config.dtype)

        self._reset_parameters()

    def _build_multi_scale_retention(self, xpos_embedder: Optional[XPos] = None):
        return MultiScaleRetention(
            MultiScaleRetention.Config(
                num_heads=self.config.num_heads,
                head_dim_v=self.config.head_dim_v,
                head_dim_qk=self.config.head_dim_qk,
                dropout=self.config.dropout,
                head_decays_range=self.config.head_decays_range,
                activation=self.activation,
                device=self.config.device,
                dtype=self.config.dtype,
            ),
            xpos_embedder=xpos_embedder,
        )

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
        x, gate = self.ada_ln1(x)
        y, state = self.retention.forward_chunkwise(x, start_idx=start_idx, prev_state=prev_state)
        y = y * gate
        if self.use_post_retention_dropout:
            y = self.dropout(y)
        x = x + y
        x, gate = self.ada_ln2(x)
        x = x + self._feedforward_block(x) * gate

        return x, state

    def forward_recurrent(
            self, x: Tensor, seq_idx: int, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        # retention block
        x, gate = self.ada_ln1(x)
        y, state = self.retention.forward_recurrent(
            x, seq_idx=seq_idx, prev_state=prev_state
        )
        y = y * gate
        if self.config.use_post_retention_dropout:
            y = self.dropout(y)
        x = x + y
        x, gate = self.ada_ln2(x)
        x = x + self._feedforward_block(x) * gate

        return x, state

    def forward(self, x: Tensor, start_idx: int = 0, prev_state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        return self.forward_chunkwise(x, start_idx, prev_state)


class DiffusionRetNetDecoder(nn.Module):
    @dataclass(kw_only=True)
    class Config:
        layer_config: DiffusionRetNetDecoderLayer.Config
        num_layers: int
        out_dim: int

    def __init__(self, config: Config):
        super().__init__()
        layer_config = config.layer_config
        self.layer_config = config.layer_config
        self.num_layers = config.num_layers
        self.xpos_embedder = XPos(
            layer_config.head_dim_qk,
            device=layer_config.device,
            dtype=layer_config.dtype,
        )
        self.layers = nn.ModuleList(self._build_layers(config.num_layers))
        d_model = layer_config.head_dim_v * layer_config.num_heads
        self.final_ada_ln = AdaLayerNorm(
            d_model,
            eps=layer_config.layer_norm_eps,
            lora_hidden_dim=layer_config.ada_ln_lora_dim,
            device=layer_config.device,
            dtype=layer_config.dtype,
        )
        self.final_linear = nn.Linear(d_model, config.out_dim)

    def _build_layers(self, num_layers: int):
        return [DiffusionRetNetDecoderLayer(self.layer_config, self.xpos_embedder) for _ in range(num_layers)]

    def forward_recurrent(
            self, x: Tensor, seq_idx: int, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, Tensor]:
        if prev_states is None or len(prev_states) == 0:
            prev_states = [None] * self.num_layers
        elif len(prev_states) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} previous states, got {len(prev_states)}"
            )

        states: List[Tensor] = []
        for layer, prev_state in zip(self.layers, prev_states):
            assert isinstance(layer, DiffusionRetNetDecoderLayer)
            x, state = layer.forward_recurrent(x, seq_idx, prev_state)
            states.append(state)
        
        # Final layer as in DiT:
        x = self.final_ada_ln(x)
        x = self.final_linear(x)

        return x, torch.stack(states)

    def forward_chunkwise(
            self, x: Tensor, start_idx: int = 0, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, Tensor]:
        if prev_states is None or len(prev_states) == 0:
            prev_states = [None] * self.num_layers
        elif len(prev_states) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} previous states, got {len(prev_states)}"
            )

        states: List[Tensor] = []
        for layer, prev_state in zip(self.layers, prev_states):
            assert isinstance(layer, DiffusionRetNetDecoderLayer)
            x, state = layer.forward_chunkwise(x, start_idx, prev_state)
            states.append(state)
        
        # Final layer as in DiT:
        x = self.final_ada_ln(x)
        x = self.final_linear(x)

        return x, torch.stack(states)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.forward_chunkwise(x)
