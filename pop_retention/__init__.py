import torch
import tilelang

from fused_retention import _get_decay_mask
from pop_retention.pop_fwd import fused_pop_retention_fwd


class FusedPOPRetention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, s, head_decays: tuple[float, ...], block_size):
        batch_size, seq_len, num_heads, dim_qk = q.shape
        dim_v = v.shape[-1]

        block_K, block_V, block_T = 64, 64, 32

        assert len(head_decays) == num_heads
        chunk_decays = _get_decay_mask(head_decays, block_T)

        mod = tilelang.cached(fused_pop_retention_fwd, [5, 6], batch_size, num_heads, seq_len, block_size, dim_qk, dim_v, block_K, block_V, block_T)
        o, block_states = mod(q, k, v, s, chunk_decays)
        ctx.save_for_backward(q, k, v, s, chunk_decays)
        return o.sum(0), block_states.sum(0)

    # @staticmethod
    # def backward(ctx, dO, dS_new):
    #     q, k, v, s, chunk_decays = ctx.saved_tensors
    #
    #     batch_size, seq_len, num_heads, dim_qk = q.shape
    #     dim_v = v.shape[-1]
    #     block_K, block_V, block_T = 64, 64, 64
    #
    #     mod = tilelang.cached(fused_retention_bwd_dk_dv_ds, [6, 7, 8], batch_size, num_heads, seq_len, dim_qk, dim_v, block_K, block_V, block_T)
    #     dK, dV, dS = mod(q, k, v, chunk_decays, dO, dS_new)
    #     mod2 = tilelang.cached(fused_retention_bwd_dq, [5], batch_size, num_heads, seq_len, dim_qk, dim_v, block_K, block_V, block_T)
    #     dQ = mod2(k, v, s, chunk_decays, dO)
    #
    #     return dQ.sum(0), dK.sum(0), dV.sum(0), dS.sum(0), None

fused_pop_retention = FusedPOPRetention.apply
