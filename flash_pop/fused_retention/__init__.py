import torch
import tilelang
from einops import rearrange

from .fused_chunk_fwd import fused_chunk_retention_fwd
from .fused_chunk_bwd import fused_retention_bwd_dk_dv_ds, fused_retention_bwd_dq
from utils import cached


class FusedChunkRetention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, s, head_decays: tuple[float, ...]):
        batch_size, seq_len, num_heads, dim_qk = q.shape
        dim_v = v.shape[-1]

        block_K, block_V, block_T = 64, 64, 32  # for GPUs with large memory try BT=64
        threads = 64

        assert len(head_decays) == num_heads
        chunk_decays = _get_decay_mask(head_decays, block_T)

        # key = ('fused_chunk_retention_fwd', batch_size, num_heads, seq_len, dim_qk, dim_v, block_K, block_V, block_T)
        # f = fused_chunk_retention_fwd(batch_size, num_heads, seq_len, dim_qk, dim_v, block_K, block_V, block_T, threads=threads)
        # fn = lambda: tilelang.compile(f, [5, 6], target='cuda')
        # mod = _get_kernel(key, fn)
        # mod = tilelang.cached(f, [5, 6], target='cuda')
        kernel = cached(fused_chunk_retention_fwd, [5, 6], batch_size, num_heads, seq_len, dim_qk, dim_v, block_K, block_V, block_T, threads, target='cuda')
        o, s_out = kernel(q, k, v, s, chunk_decays)
        ctx.save_for_backward(q, k, v, s, chunk_decays)
        return o.sum(0), s_out.sum(0)

    @staticmethod
    def backward(ctx, dO, dS_new):
        q, k, v, s, chunk_decays = ctx.saved_tensors

        batch_size, seq_len, num_heads, dim_qk = q.shape
        dim_v = v.shape[-1]
        block_K, block_V, block_T = 64, 64, 32  # for GPUs with large memory try BT=64

        kernel1 = cached(fused_retention_bwd_dk_dv_ds, [6, 7, 8], batch_size, num_heads, seq_len, dim_qk, dim_v, block_K, block_V, block_T, target='cuda')
        dK, dV, dS = kernel1(q, k, v, chunk_decays, dO, dS_new)
        threads = 64
        kernel2 = cached(fused_retention_bwd_dq, [5], batch_size, num_heads, seq_len, dim_qk, dim_v, block_K, block_V, block_T, threads, target='cuda')
        dQ = kernel2(k, v, s, chunk_decays, dO)

        return dQ.sum(0), dK.sum(0), dV.sum(0), dS.sum(0), None
        #
        # def maybe_contiguous(x):
        #     if x.stride(-1) != 1:
        #         return x.contiguous()
        #     return x
        #
        # do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        # block_M = 128
        # block_N = 128 if D_HEAD <= 64 else 32
        # mod_prep = cached(flashattn_bwd_preprocess, [2], BATCH, H, N_CTX, D_HEAD)
        # mod_post = cached(flashattn_bwd_postprocess, [1], BATCH, H, N_CTX, D_HEAD)
        # delta = mod_prep(o, do)
        # mod = cached(flashattn_bwd, [6, 7, 8], BATCH, H, N_CTX, D_HEAD, ctx.causal, block_M,
        #              block_N)
        # dq, dk, dv = mod(q, k, v, do, lse, delta)
        # dq = mod_post(dq)
        # return dq, dk, dv, None


fused_chunk_retention = FusedChunkRetention.apply


cached_masks = {}
cached_head_decays = {}

def _get_decay_mask(head_decays, seq_len, device="cuda", dtype=torch.float32, return_head_decays: bool = False):
    head_decays = tuple(head_decays)
    key = tuple([seq_len, *head_decays])
    global cached_mask
    if key in cached_masks:
        if return_head_decays:
            return cached_masks[key], cached_head_decays[head_decays]
        return cached_masks[key]

    if head_decays in cached_head_decays:
        head_decays_t = cached_head_decays[head_decays]
    else:
        head_decays_t = torch.tensor(head_decays, device=device, dtype=dtype)
        cached_head_decays[head_decays] = head_decays_t

    query_pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze_(-1)
    key_pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze_(0)
    distance = query_pos - key_pos

    distance = rearrange(distance, "n s -> () n s")
    decay_gammas = rearrange(head_decays_t, "h -> h () ()")
    decay_mask = decay_gammas ** distance
    decay_mask = decay_mask.tril()

    cached_masks[key] = decay_mask

    if return_head_decays:
        return decay_mask, head_decays_t

    return decay_mask
