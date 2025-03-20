from math import ceil

import torch
import tilelang
import tilelang.language as T
from tilelang import cached
from torch import Tensor
from einops import rearrange, einsum


# Heavily modified https://github.com/sustcsonglin/fla-tilelang/blob/main/linear_attn/fused_chunk.py


def fused_retention_fwd(batch, heads, seq_len, dim_qk, dim_v, BK, BV, BT):
    NK = T.ceildiv(dim_qk, BK)
    qk_shape = [batch, seq_len, heads, dim_qk]
    v_shape = [batch, seq_len, heads, dim_v]
    o_shape = [NK, batch, seq_len, heads, dim_v]  # we have to reduce the first dimension
    state_shape = [batch, heads, dim_qk, dim_v]
    out_state_shape = [NK, batch, heads, dim_qk, dim_v]
    dtype = "bfloat16"
    accum_dtype = "float"

    sqrt_dim_qk = dim_qk ** 0.5

    @T.macro
    def compute_retention_chunk_outputs(
            acc_A_local: T.Buffer([BT, BT], accum_dtype),
            acc_A_cast: T.Buffer([BT, BT], dtype),
            mask: T.Buffer([BT, BT], accum_dtype),
            acc_s_shared: T.Buffer([BK, BV], dtype),
            acc_o_shared: T.Buffer([BT, BV], dtype),
            acc_o_local: T.Buffer([BT, BV], accum_dtype),
            Q_shared: T.Buffer([BT, BK], dtype),
            K_shared: T.Buffer([BT, BK], dtype),
            V_shared: T.Buffer([BT, BV], dtype),
            Output: T.Buffer(o_shape, dtype),
            decay: T.float32,
            i_bk: T.int32,
            i_batch: T.int32,
            i_head: T.int32,
            i_bv: T.int32,
            k: T.int32
    ):
        # Compute chunk attention scores (within chunk):
        T.clear(acc_A_local)
        T.gemm(Q_shared, K_shared, acc_A_local, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
        for i, j in T.Parallel(BT, BT):
            acc_A_local[i, j] = (acc_A_local[i, j] / sqrt_dim_qk) * mask[i, j]
            # acc_A_local[i, j] = T.if_then_else(i >= j, acc_A_local[i, j] * T.pow(decay, i - j), 0)
        T.copy(acc_A_local, acc_A_cast)

        # Compute outputs:
        T.clear(acc_o_local)
        T.gemm(Q_shared, acc_s_shared, acc_o_local, policy=T.GemmWarpPolicy.FullCol)
        for i, j in T.Parallel(BT, BV):
            acc_o_local[i, j] = acc_o_local[i, j] * (mask[i, 0] * decay)

        T.gemm(acc_A_cast, V_shared, acc_o_local, policy=T.GemmWarpPolicy.FullCol)

        T.copy(acc_o_local, acc_o_shared)
        T.copy(acc_o_shared, Output[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV])

    @T.macro
    def update_recurrent_state(
            mask: T.Buffer([BT, BT], accum_dtype),
            K_shared: T.Buffer([BT, BK], dtype),
            K_local_trans: T.Buffer([BK, BT], accum_dtype),
            K_local_trans_cast: T.Buffer([BK, BT], dtype),
            V_shared: T.Buffer([BT, BV], dtype),
            acc_s_local: T.Buffer([BK, BV], accum_dtype),
            acc_s_shared: T.Buffer([BK, BV], dtype),
            decay: T.float32,
            effective_chunk_size_correction: T.int32
    ):
        # transpose k first because T.gemm does not have a good support for transposing the first operand according to the authors
        c = effective_chunk_size_correction  # if last chunk is shorter (c>0), decays exponents should be adjusted
        for i, j in T.Parallel(BK, BT):
            # Also apply decay terms:
            K_local_trans[i, j] = (K_shared[j, i] / sqrt_dim_qk) * mask[BT-1, j+c]  # T.pow(decay, BT - j - 1)
        T.copy(K_local_trans, K_local_trans_cast)

        cross_chunk_decay = mask[BT - 1, c] * decay  # T.pow(decay, BT)
        for i, j in T.Parallel(BK, BV):
            acc_s_local[i, j] = acc_s_local[i, j] * cross_chunk_decay
        T.gemm(K_local_trans_cast, V_shared, acc_s_local, policy=T.GemmWarpPolicy.FullCol)
        T.copy(acc_s_local, acc_s_shared)

    @T.prim_func
    def main(
            Q: T.Buffer(qk_shape, dtype),
            K: T.Buffer(qk_shape, dtype),
            V: T.Buffer(v_shape, dtype),
            state: T.Buffer(state_shape, dtype),
            head_decays: T.Buffer([heads], accum_dtype),
            decays_block: T.Buffer([heads, BT, BT], accum_dtype),
            Output: T.Buffer(o_shape, dtype),
            out_state: T.Buffer(out_state_shape, dtype),
    ):
        with T.Kernel(heads, batch, T.ceildiv(dim_v, BV) * NK, threads=128) as (i_head, i_batch, bz):
            i_bk = bz % NK
            i_bv = bz // NK
            Q_shared = T.alloc_shared([BT, BK], dtype)
            K_shared = T.alloc_shared([BT, BK], dtype)
            K_local_trans = T.alloc_fragment([BK, BT], accum_dtype)
            K_local_trans_cast = T.alloc_fragment([BK, BT], dtype)
            V_shared = T.alloc_shared([BT, BV], dtype)

            acc_o_local = T.alloc_fragment((BT, BV), accum_dtype)
            acc_o_shared = T.alloc_shared([BT, BV], dtype)

            acc_s_local = T.alloc_fragment((BK, BV), accum_dtype)
            acc_A_local = T.alloc_fragment((BT, BT), accum_dtype)
            acc_A_cast = T.alloc_shared((BT, BT), dtype)
            mask = T.alloc_shared((BT, BT), accum_dtype)

            acc_s_shared = T.alloc_fragment((BK, BV), dtype, scope="shared")

            decays_shared = T.alloc_shared((heads,), accum_dtype)

            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                acc_o_shared: tilelang.layout.make_swizzled_layout(acc_o_shared),
                acc_s_shared: tilelang.layout.make_swizzled_layout(acc_s_shared),
            })

            T.clear(acc_s_local)
            T.copy(state[i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV], acc_s_shared)
            T.copy(acc_s_shared, acc_s_local)

            # init decay values:
            T.copy(head_decays[:heads], decays_shared)
            decay = decays_shared[i_head]

            T.copy(decays_block[i_head, :, :], mask)
            # for i, j in T.Parallel(BT, BT):
            #     mask_local[i, j] = T.if_then_else(i >= j, T.pow(decay, i-j), 0)

            loop_range = T.ceildiv(seq_len, BT)
            for k in T.Pipelined(loop_range, num_stages=2):
                T.copy(K[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK], K_shared)
                T.copy(Q[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK], Q_shared)
                T.copy(V[i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV], V_shared)
                effective_chunk_size_correction = T.max(0, ((k+1)*BT) - seq_len)

                compute_retention_chunk_outputs(
                    acc_A_local,
                    acc_A_cast,
                    mask,
                    acc_s_shared,
                    acc_o_shared,
                    acc_o_local,
                    Q_shared,
                    K_shared,
                    V_shared,
                    Output,
                    decay,
                    i_bk, i_batch, i_head, i_bv, k
                )

                update_recurrent_state(
                    mask,
                    K_shared,
                    K_local_trans,
                    K_local_trans_cast,
                    V_shared,
                    acc_s_local,
                    acc_s_shared,
                    decay,
                    effective_chunk_size_correction
                )
            T.copy(acc_s_shared, out_state[i_bk, i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV])

    return main


def fused_retention_bwd(batch, heads, seq_len, dim_qk, dim_v, BK, BV, BT):
    NK = T.ceildiv(dim_qk, BK)
    qk_shape = [batch, seq_len, heads, dim_qk]
    v_shape = [batch, seq_len, heads, dim_v]
    dv_shape = [NK, batch, seq_len, heads, dim_v]  # we have to reduce the first dimension
    dqk_shape = [NK, batch, seq_len, heads, dim_qk]  # we have to reduce the first dimension
    state_shape = [batch, heads, dim_qk, dim_v]
    d_state_shape = [NK, batch, heads, dim_qk, dim_v]
    dtype = "bfloat16"
    accum_dtype = "float"

    sqrt_dim_qk = dim_qk ** 0.5

    @T.prim_func
    def main(
            Q: T.Buffer(qk_shape, dtype),
            K: T.Buffer(qk_shape, dtype),
            V: T.Buffer(v_shape, dtype),
            state: T.Buffer(state_shape, dtype),
            head_decays: T.Buffer([heads], accum_dtype),
            decays_block: T.Buffer([heads, BT, BT], accum_dtype),
            dO: T.Buffer(v_shape, dtype),
            dS_new: T.Buffer(state_shape, dtype),
            dQ: T.Buffer(dqk_shape, dtype),
            dK: T.Buffer(dqk_shape, dtype),
            dV: T.Buffer(dv_shape, dtype),
            dS: T.Buffer(d_state_shape, dtype),
    ):
        with T.Kernel(heads, batch, T.ceildiv(dim_v, BV) * NK, threads=128) as (i_head, i_batch, bz):
            i_bk = bz % NK
            i_bv = bz // NK
            Q_shared = T.alloc_shared([BT, BK], dtype)
            K_shared = T.alloc_shared([BT, BK], dtype)
            BK_BT_cast = T.alloc_fragment([BK, BT], dtype)
            V_shared = T.alloc_shared([BT, BV], dtype)
            s_shared = T.alloc_shared([BK, BV], dtype)

            dO_shared = T.alloc_shared([BT, BV], dtype)
            dO_decay_shared = T.alloc_shared([BT, BV], dtype)
            dS_new_shared = T.alloc_shared([BK, BV], dtype)

            dQ_shared = T.alloc_shared([BT, BK], dtype)
            dS_shared = T.alloc_shared([BK, BV], dtype)

            mask = T.alloc_shared((BT, BT), accum_dtype)
            decays_shared = T.alloc_shared((heads,), accum_dtype)

            BT_BT_buffer = T.alloc_fragment((BT, BT), accum_dtype)
            BT_BT_shared = T.alloc_shared((BT, BT), dtype)
            BT_BV_buffer = T.alloc_fragment((BT, BV), accum_dtype)
            BT_BV_buffer_cast = T.alloc_fragment((BT, BV), dtype)
            BT_BK_buffer = T.alloc_fragment((BT, BK), accum_dtype)
            BT_BK_cast = T.alloc_fragment((BT, BK), dtype)
            BK_BV_buffer = T.alloc_fragment((BK, BV), accum_dtype)

            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                dO_shared: tilelang.layout.make_swizzled_layout(dO_shared),
                dS_new_shared: tilelang.layout.make_swizzled_layout(dS_new_shared),
                s_shared: tilelang.layout.make_swizzled_layout(s_shared),
            })

            # prepare:
            T.copy(decays_block[i_head, :, :], mask)
            T.copy(head_decays[:heads], decays_shared)
            decay = decays_shared[i_head]

            T.copy(state[i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV], s_shared)
            T.copy(dS_new[i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV], dS_new_shared)

            loop_range = T.ceildiv(seq_len, BT)
            for k_tag in T.Pipelined(loop_range, num_stages=2):
                k = loop_range - k_tag
                effective_chunk_size_correction = T.max(0, ((k + 1) * BT) - seq_len)

                # Compute dQ:
                # dO_VT_D:
                # T.clear(BT_BV_buffer)
                T.copy(dO[i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV], dO_shared)
                T.copy(dO_shared, BT_BV_buffer_cast)
                T.clear(BT_BT_buffer)
                T.copy(V[i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV], V_shared)
                T.gemm(BT_BV_buffer_cast, V_shared, BT_BT_buffer, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(BT, BT):
                    BT_BT_buffer[i, j] *= mask[i, j]
                    BT_BT_buffer[i, j] /= sqrt_dim_qk
                T.copy(BT_BT_buffer, BT_BT_shared)

                T.clear(BT_BK_buffer)
                T.gemm(BT_BT_shared, K_shared, BT_BK_buffer, policy=T.GemmWarpPolicy.FullCol)

                # dO * inner_decay:
                T.copy(dO_shared, BT_BV_buffer)
                for i, j in T.Parallel(BT, BV):
                    BT_BV_buffer[i, j] *= (mask[i, 0] * decay)
                # T.copy(BT_BV_buffer, BT_BV_buffer_cast)
                T.copy(BT_BV_buffer, dO_decay_shared)

                T.gemm(dO_decay_shared, s_shared, BT_BK_buffer, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(BT_BK_buffer, dQ_shared)
                T.copy(dQ_shared, dQ[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK])

                # Compute dS:
                cross_chunk_decay = mask[BT - 1, effective_chunk_size_correction] * decay
                T.clear(BK_BV_buffer)
                T.copy(dS_new_shared, BK_BV_buffer)
                for i, j in T.Parallel(BK, BV):
                    BK_BV_buffer[i, j] *= cross_chunk_decay
                # No support for transpose of first argument, need to do it mannually:
                T.copy(Q[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK], Q_shared)
                # T.clear(BK_BT_cast)
                for i, j in T.Parallel(BT, BK):
                    BK_BT_cast[j, i] = Q_shared[i, j]

                T.gemm(BK_BT_cast, dO_decay_shared, BK_BV_buffer, policy=T.GemmWarpPolicy.FullCol)
                T.copy(BK_BV_buffer, dS_shared)
                T.copy(dS_shared, dS[i_bk, i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV])


                # Compute dK:
                T.clear(BT_BK_buffer)
                T.gemm(V_shared, dS_new_shared, BT_BK_buffer, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(BT, BK):
                    BT_BK_buffer[i, j] *= mask[BT-1, i+effective_chunk_size_correction] / sqrt_dim_qk

                # reuse dO_VT_D, but transposed:
                for i, j in T.Parallel(BT, BT):
                    BT_BT_shared[i, j] = BT_BT_buffer[j, i]

                T.gemm(BT_BT_shared, Q_shared, BT_BK_buffer, policy=T.GemmWarpPolicy.FullCol)
                # T.copy(BT_BK_buffer, dK_shared)
                T.copy(BT_BK_buffer, dK[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK])

                # Compute dV:
                T.copy(K[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK], K_shared)
                for i, j in T.Parallel(BT, BK):
                    BT_BK_buffer[i, j] = K_shared[i, j] * mask[BT-1, i+effective_chunk_size_correction] / sqrt_dim_qk
                    BT_BK_cast[i, j] = BT_BK_buffer[i, j]
                T.clear(BT_BV_buffer)
                T.gemm(BT_BK_cast, dS_new_shared, BT_BV_buffer, policy=T.GemmWarpPolicy.FullCol)

                T.clear(BT_BT_buffer)
                T.gemm(Q_shared, K_shared, BT_BT_buffer, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(BT, BT):
                    BT_BT_buffer[i, j] *= mask[i, j] / sqrt_dim_qk
                for i, j in T.Parallel(BT, BT):
                    BT_BT_shared[i, j] = BT_BT_buffer[j, i]
                T.gemm(BT_BT_shared, dO_shared, BT_BV_buffer, policy=T.GemmWarpPolicy.FullCol)
                # T.copy(BT_BV_buffer, dV_shared)
                T.copy(BT_BV_buffer, dV[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV])

                # update dS_new_shared:
                T.copy(dS_shared, dS_new_shared)
    return main



class FusedChunkRetention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, s, head_decays: tuple[float, ...]):
        batch_size, seq_len, num_heads, dim_qk = q.shape
        dim_v = v.shape[-1]

        block_K, block_V, block_T = 64, 64, 64

        assert len(head_decays) == num_heads
        chunk_decays, head_decays_ = _get_decay_mask(head_decays, block_T)

        mod = tilelang.cached(fused_retention_fwd, [6, 7], batch_size, num_heads, seq_len, dim_qk, dim_v, block_K, block_V, block_T)
        o, s_new = mod(q, k, v, s, head_decays_, chunk_decays)
        ctx.save_for_backward(q, k, v, s, head_decays_, chunk_decays)
        return o.sum(0), s_new.sum(0)

    @staticmethod
    def backward(ctx, dO, dS_new):
        q, k, v, s, head_decays_, chunk_decays = ctx.saved_tensors

        batch_size, seq_len, num_heads, dim_qk = q.shape
        dim_v = v.shape[-1]
        block_K, block_V, block_T = 64, 64, 64

        mod = tilelang.cached(fused_retention_bwd, [8, 9, 10, 11], batch_size, num_heads, seq_len, dim_qk, dim_v, block_K, block_V, block_T)
        dQ, dK, dV, dS = mod(q, k, v, s, head_decays_, chunk_decays, dO, dS_new)

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

def _get_decay_mask(head_decays, seq_len, device="cuda", dtype=torch.float32):
    key = tuple([seq_len, *head_decays])
    global cached_mask
    if key in cached_masks:
        return cached_masks[key], cached_head_decays[key]

    head_decays = torch.tensor(head_decays, device=device, dtype=dtype)
    cached_head_decays[key] = head_decays

    query_pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze_(-1)
    key_pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze_(0)
    distance = query_pos - key_pos

    distance = rearrange(distance, "n s -> () n s")
    decay_gammas = rearrange(head_decays, "h -> h () ()")
    decay_mask = decay_gammas ** distance
    decay_mask = decay_mask.tril()

    cached_masks[key] = decay_mask

    return decay_mask, head_decays


def ref_program_(Q, K, V, prev_state, head_decays):
    qk = torch.einsum('bqhd,bkhd->bhqk', Q, K).tril()

    device, dtype = Q.device, Q.dtype
    seq_len = Q.size(1)
    decay_mask, head_decays_ = _get_decay_mask(head_decays, seq_len, device, torch.float32)
    # decay_mask = decay_mask / decay_mask.sum(dim=-1, keepdim=True).sqrt()

    qkm = (qk * decay_mask.unsqueeze(0).to(dtype=dtype)).tril()
    # r = qkm.sum(dim=-1, keepdim=True).abs()
    # r = torch.where(r >= 1, r, torch.ones_like(r))
    # qkm = qkm / r

    # qkm = qk * mask
    # r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
    o = torch.einsum('bhqk,bkhd->bqhd', qkm.to(dtype=dtype), V)

    # cross-chunk (derived from recurrent retention)
    decay_gammas = rearrange(head_decays_, "h -> () h () ()")
    device = K.device
    dtype = K.dtype
    inner_pos = rearrange(
        torch.arange(K.size(1), device=device, dtype=dtype) + 1,
        "n -> () () n ()",
    )
    state_decays = decay_gammas ** (K.size(1) - inner_pos)
    discounted_key = einsum(K, state_decays.to(dtype=dtype), "b n h d, _ h n _ -> b h n d")
    state = einsum(discounted_key, V, "b h n d1, b n h d2 -> b h d1 d2")

    if prev_state is not None:
        # update recurrent state using prev_state:
        chunk_decay = decay_gammas ** K.size(1)
        state = state + prev_state * chunk_decay

        # Update the retention Tensor, based on cross-chunk information
        inner_decay = rearrange(decay_gammas ** inner_pos, "b h n d -> b n h d")
        o = o + (
                einsum(Q.to(dtype=dtype), prev_state.to(dtype=dtype), "b n h d1, b h d1 d2 -> b n h d2") * inner_decay.to(dtype=dtype)
        )

    return o.to(dtype=dtype), state.to(dtype=dtype)


def ref_program(Q, K, V, prev_state, head_decays, chunk_size: int = 512, *args):
    seq_len = Q.size(1)
    res = []
    state_t = prev_state
    K = K / (K.size(3) ** 0.5)
    # gn = torch.nn.LayerNorm(normalized_shape=V.size(3), device=Q.device, dtype=Q.dtype)
    for i in range(ceil(seq_len / chunk_size)):
        start, end = i * chunk_size, (i + 1) * chunk_size
        res_t, state_t = ref_program_(
            Q[:, start:end],
            K[:, start:end],
            V[:, start:end],
            state_t, head_decays)
        # res_t = gn(res_t)
        res.append(res_t)

    return torch.cat(res, dim=1), state_t


def reference_grads(Q: Tensor, K: Tensor, V: Tensor, prev_state: Tensor, head_decays, dO: Tensor, dS_new: Tensor, *args):
    d_qk = K.size(3)
    sqrt_d_qk = (d_qk ** 0.5)
    assert d_qk == Q.size(3)
    device, dtype = Q.device, Q.dtype
    seq_len = Q.size(1)

    D, head_decays_ = _get_decay_mask(head_decays, seq_len, device, torch.float32)
    decay_gammas = rearrange(head_decays_, "h -> () h () ()")
    inner_pos = rearrange(torch.arange(K.size(1), device=device, dtype=dtype) + 1, "n -> () () n ()")
    inner_decay = rearrange(decay_gammas ** inner_pos, "b h t d -> b t h d")
    state_decays = decay_gammas ** (K.size(1) - inner_pos)
    chunk_decay = decay_gammas ** K.size(1)

    dO_VT_D = einsum(dO, V, "b t1 h dv, b t2 h dv -> b h t1 t2") / sqrt_d_qk
    dO_VT_D = einsum(dO_VT_D, D, "b h t1 t2, h t1 t2 -> b h t1 t2")
    dQ1 = einsum(dO_VT_D.to(dtype=dtype), K, "b h t1 t2, b t2 h dk -> b t1 h dk")
    dO_decay = dO * inner_decay
    dQ2 = einsum(dO_decay.to(dtype=dtype), prev_state, "b t h dv, b h dk dv -> b t h dk")
    dQ = dQ1 + dQ2
    # dQ = dQ2

    # Compute dK:
    dK = (
        einsum(dO_VT_D.to(dtype=dtype), Q, "b h t1 t2, b t1 h dk -> b t2 h dk") +
        einsum(V, dS_new, state_decays, "b t h dv, b h dk dv, b h t dk -> b t h dk") / sqrt_d_qk
    )

    A = einsum(Q, K, D, "b t1 h dk, b t2 h dk, h t1 t2 -> b h t1 t2") / sqrt_d_qk
    dV = (
        einsum(A.to(dtype=dtype), dO, "b h t1 t2, b t1 h dv -> b t2 h dv") +
        einsum(K / sqrt_d_qk, state_decays.to(dtype=dtype), dS_new, "b t h dk, b h t dk, b h dk dv -> b t h dv")
    )

    dS = (
        einsum(Q, dO_decay.to(dtype=dtype), "b t h dk, b t h dv -> b h dk dv") +
        chunk_decay * dS_new
    )

    return dQ, dK, dV, dS
