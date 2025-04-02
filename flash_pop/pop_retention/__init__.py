import torch
import tilelang

from flash_pop.fused_retention import _get_decay_mask
from .pop_fwd import fused_pop_retention_fwd
from .pop_bwd import pop_retention_bwd_dk_dv_ds
from flash_pop.fused_retention.fused_chunk_bwd import fused_retention_bwd_dq
from flash_pop.utils import cached


class FlashPOPRetention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, s, head_decays: tuple[float, ...], block_size):
        batch_size, seq_len, num_heads, dim_qk = q.shape
        dim_v = v.shape[-1]
        dtype = "bfloat16" if q.dtype == torch.bfloat16 else "float"
        assert dtype == 'bfloat16', f'currently, TileLang does not support float32'

        block_K, block_V, block_T = 64, 64, 32

        assert len(head_decays) == num_heads
        chunk_decays = _get_decay_mask(head_decays, block_T)

        # kernel = tilelang.cached(fused_pop_retention_fwd, [5, 6], batch_size, num_heads, seq_len, block_size, dim_qk, dim_v, block_K, block_V, block_T)
        # f = fused_pop_retention_fwd(batch_size, num_heads, seq_len, block_size, dim_qk, dim_v, block_K, block_V, block_T)
        # kernel = tilelang.cached(f, [5, 6], target='cuda')
        kernel = cached(fused_pop_retention_fwd, [5, 6], batch_size, num_heads, seq_len, block_size, dim_qk, dim_v, block_K, block_V, block_T)
        o, block_states = kernel(q, k, v, s, chunk_decays)
        ctx.save_for_backward(q, k, v, s, chunk_decays)
        return o.sum(0), block_states.sum(0)

    @staticmethod
    def backward(ctx, do, d_block_states):
        q, k, v, s, chunk_decays = ctx.saved_tensors

        batch_size, seq_len, num_heads, dim_qk = q.shape
        dim_v = v.shape[-1]
        num_full_blocks = d_block_states.shape[1]
        block_size = seq_len // num_full_blocks
        block_K, block_V, block_T = 64, 64, 32

        kernel1 = cached(pop_retention_bwd_dk_dv_ds, [6, 7, 8], batch_size, num_heads, seq_len, block_size, dim_qk, dim_v, block_K, block_V, block_T)
        # kernel1 = tilelang.compile(pop_retention_bwd_dk_dv_ds(batch_size, num_heads, seq_len, block_size, dim_qk, dim_v, block_K, block_V, block_T), [6, 7, 8], target='cuda')
        dK, dV, dS = kernel1(q, k, v, chunk_decays, do, d_block_states)
        kernel2 = cached(fused_retention_bwd_dq, [5], batch_size, num_heads, seq_len, dim_qk, dim_v, block_K, block_V, block_T)
        # kernel2 = tilelang.compile(fused_retention_bwd_dq(batch_size, num_heads, seq_len, dim_qk, dim_v, block_K, block_V, block_T), [5], target='cuda')
        dQ = kernel2(k, v, s, chunk_decays, do)

        return dQ.sum(0), dK.sum(0), dV.sum(0), dS.sum(0), None, None

flash_pop_retention = FlashPOPRetention.apply
