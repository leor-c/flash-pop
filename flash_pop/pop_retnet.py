from typing import Optional, Union
from dataclasses import dataclass

import torch
from torch import Tensor
from einops import rearrange, einsum, repeat

from xpos_emb import XPos
from retnet import MultiScaleRetention, RetNetDecoderLayer, RetNetDecoder
from pop_retention import flash_pop_retention


class POPMultiScaleRetention(MultiScaleRetention):
    @dataclass(kw_only=True)
    class Config(MultiScaleRetention.Config):
        block_size: int

    def __init__(self, config: Config, xpos_embedder: Optional[XPos] = None):
        super().__init__(config, xpos_embedder=xpos_embedder)
        self.config = config

    def _pop_retention_kernel(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            prev_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        retention, states = flash_pop_retention(
            q, k, v, prev_state, self.head_decays, self.config.block_size
        )
        return retention, states

    def pop_chunkwise(
            self,
            x: Tensor,
            start_index: int,
            prev_state: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        return self._retention_chunkwise(
            x,
            start_index,
            prev_state,
            self._pop_retention_kernel
        )


class POPDecoderLayer(RetNetDecoderLayer):
    @dataclass(kw_only=True)
    class Config(RetNetDecoderLayer.Config):
        block_size: int

    def __init__(
            self,
            config: Config,
            xpos_embedder: Optional[XPos] = None,
            suffixes_xpos_embedder: Optional[XPos] = None,
    ) -> None:
        super().__init__(config, xpos_embedder)
        self.suffixes_xpos_embedder = suffixes_xpos_embedder
        self.config = config

    def _build_multi_scale_retention(self, xpos_embedder: Optional[XPos] = None):
        return POPMultiScaleRetention(
            POPMultiScaleRetention.Config(
                block_size=self.config.block_size,
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

    def pop_forward(
            self,
            x: Tensor,
            start_index: int,
            prev_state: Optional[Tensor],
            suffixes: Optional[Tensor] = None,
            suffixes_start_indices: Optional[Tensor] = None,
    ):
        """
        Retention chunkwise of `x` with state computations every full 'block'.
        If suffixes are provided, another retention chunkwise is computed in a large batch
        form, starting from the computed states and using the suffixes as inputs.
        :param x: Tensor. shape: (batch_size, seq_len, num_heads * dim_v) where seq_len = N * block_size
        for positive integer N.
        :param start_index:
        :param prev_state:
        :param suffixes: Tensor. shape: (batch_size * N+1, sfx_len, num_heads * dim_v).
        Note that given a sequence `x` of N blocks and a previous state we can predict N+1 blocks!
        :return: If suffixes are provided, their corresponding outputs are also returned. Otherwise,
        the retention outputs of `x` and the states are returned.
        """
        assert x.dim() == 3, f"Got {x.dim()}"  # b t (h d)

        if self.norm_first:
            y, states = self.retention.pop_chunkwise(self.norm1(x), start_index=start_index, prev_state=prev_state)
            x = x + y
            x = x + self._feedforward_block(self.norm2(x))
        else:
            y, states = self.retention.pop_chunkwise(x, start_index=start_index, prev_state=prev_state)
            x = x + self.norm1(y)
            x = x + self.norm2(self._feedforward_block(x))

        if suffixes is not None:
            assert suffixes_start_indices is not None and isinstance(suffixes_start_indices, Tensor)
            assert suffixes.dim() == 3, f"Got {suffixes.dim()}"  # (b n) t (h d) where n=num blocks, t is sfx length
            batch_size = x.size(0)
            num_blocks = suffixes.size(0) // batch_size

            assert states.size(1)+1 == num_blocks, (f"got {states.size(1)+1} states != {num_blocks} num_blocks. "
                                                    f"make sure there is one additional suffix block.")

            if prev_state is None:
                prev_state = torch.zeros_like(states[:, 0:1])
            prev_states = torch.cat((prev_state, states), dim=1).flatten(0, 1)
            suffixes, _ = self.retention._retention_chunkwise(
                suffixes,
                start_idx=suffixes_start_indices,
                prev_state=prev_states,
                xpos_embedder=self.suffixes_xpos_embedder,
            )
            last_state = states[:, -1].clone()

            return x, last_state, suffixes

        else:
            return x, states, None


def _get_suffixes_start_indices(batch_size, num_blocks, start_index: int, block_size: int, device):
    start_idx = start_index + torch.arange(num_blocks, device=device) * block_size
    start_idx = repeat(start_idx, 'n -> (b n)', b=batch_size)

    return start_idx


class POPRetNetDecoder(RetNetDecoder):
    def __init__(self, layer_config: POPDecoderLayer.Config, num_layers: int):
        self.suffixes_xpos_embedder = XPos(
            layer_config.head_dim_qk,
            device=layer_config.device,
            dtype=layer_config.dtype,
        )
        super().__init__(layer_config, num_layers)


    def _build_layers(self, num_layers: int):
        return [
            POPDecoderLayer(
                self.layer_config,
                xpos_embedder=self.xpos_embedder,
                suffixes_xpos_embedder=self.suffixes_xpos_embedder
            ) for _ in range(num_layers)
        ]

    def pop_forward(
            self,
            x: Tensor,
            start_idx: int = 0,
            prev_states: Optional[tuple[Tensor, ...]] = (),
            suffixes: Optional[Tensor] = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """

        :param x: Tensor of shape (batch_size, seq_len, num_heads * dim_v).
        :param start_idx:
        :param prev_states:
        :param suffixes: Tensor of shape (batch_size, num_blocks, sfx_seq_len, num_heads * dim_v).
        :return:
        """
        if not prev_states:
            prev_states = [None] * self.num_layers
        elif len(prev_states) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} previous states, got {len(prev_states)}"
            )

        suffixes_start_indices = None
        if suffixes is not None:
            assert suffixes.dim() == 4, f"Got {suffixes.dim()}"
            suffixes_start_indices = _get_suffixes_start_indices(
                x.size(0), suffixes.size(1), start_idx, self.layer_config.block_size, x.device
            )
            suffixes = suffixes.flatten(0, 1)

        states: list[Tensor] = []
        for layer, prev_state in zip(self.layers, prev_states):
            assert isinstance(layer, POPDecoderLayer)

            x, state, suffixes = layer.pop_forward(
                x,
                start_idx,
                prev_state,
                suffixes,
                suffixes_start_indices=suffixes_start_indices
            )
            states.append(state)
        return x, states, rearrange(suffixes, '(b n) t d -> b n t d', b=x.size(0))


