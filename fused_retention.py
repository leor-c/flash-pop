from math import ceil

import torch
import tilelang
import tilelang.language as T
from einops import rearrange

# Heavily modified https://github.com/sustcsonglin/fla-tilelang/blob/main/linear_attn/fused_chunk.py


def fused_retention_fwd(batch, heads, seq_len, dim_qk, dim_v, BK, BV, BT):
    NK = T.ceildiv(dim_qk, BK)
    qk_shape = [batch, seq_len, heads, dim_qk]
    v_shape = [batch, seq_len, heads, dim_v]
    o_shape = [NK, batch, seq_len, heads, dim_v]  # we have to reduce the first dimension
    state_shape = [batch, heads, dim_qk, dim_v]
    out_state_shape = [NK, batch, heads, dim_qk, dim_v]
    dtype = "bfloat16"
    # dtype = "float"
    accum_dtype = "float"

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
        #       Compute attn. scores + apply mask + normalization:
        T.clear(acc_A_local)
        T.gemm(Q_shared, K_shared, acc_A_local, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
        # sqrt_dim_qk = T.pow(dim_qk, 0.5)
        for i, j in T.Parallel(BT, BT):
            acc_A_local[i, j] = acc_A_local[i, j] * mask[i, j]
            # acc_A_local[i, j] = T.if_then_else(i >= j, acc_A_local[i, j] * T.pow(decay, i - j), 0)
        T.copy(acc_A_local, acc_A_cast)

        # Compute outputs:
        T.clear(acc_o_local)
        T.gemm(Q_shared, acc_s_shared, acc_o_local, policy=T.GemmWarpPolicy.FullCol)
        for i, j in T.Parallel(BT, BV):
            acc_o_local[i, j] = acc_o_local[i, j] * (mask[i, 0] * decay)  #decay_BT[i]

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
    ):
        # Update state:
        # transpose k first because T.gemm does not have a good support for transposing the first operand according to the authors
        # T.copy(K_shared, K_local)
        # sqrt_dim_qk = T.pow(dim_qk, 0.5)
        for i, j in T.Parallel(BK, BT):
            # Also apply decay terms:
            K_local_trans[i, j] = K_shared[j, i] * mask[BT-1, j]  # T.pow(decay, BT - j - 1)
        T.copy(K_local_trans, K_local_trans_cast)

        cross_chunk_decay = mask[BT - 1, 0] * decay  # T.pow(decay, BT)
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

            # hidden state in register tiles, must be in fp32?
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
            if state is not None:
                T.copy(state[i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV], acc_s_shared)
                T.copy(acc_s_shared, acc_s_local)
            else:
                T.fill(acc_s_local, 0)
                T.copy(acc_s_local, acc_s_shared)

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

                # Scale (commented out due to some weird Tilelang bug):
                # sqrt_dim_qk = T.pow(dim_qk, 0.5)
                # T.copy(K_shared, K_local)
                # for i, j in T.Parallel(BT, BK):
                #     K_local[i, j] /= sqrt_dim_qk
                # T.copy(K_local, K_shared)


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
                )
            T.copy(acc_s_shared, out_state[i_bk, i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV])

    return main


def _get_decay_mask(head_decays, seq_len, device="cuda", dtype=torch.float32):
    query_pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze_(-1)
    key_pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze_(0)
    distance = query_pos - key_pos

    distance = rearrange(distance, "n s -> () n s")
    decay_gammas = rearrange(head_decays, "h -> h () ()")
    decay_mask = decay_gammas ** distance
    return decay_mask.tril()


def ref_program_(Q, K, V, prev_state, head_decays):
    sqrt_dim_qk = torch.sqrt(Q.new_tensor(Q.size(3)))
    # K = K/sqrt_dim_qk
    qk = torch.einsum('bqhd,bkhd->bhqk', Q, K).tril()

    from einops import rearrange, einsum
    device, dtype = Q.device, Q.dtype
    f32_dtype = torch.float32
    seq_len = Q.size(1)
    decay_mask = _get_decay_mask(head_decays, seq_len, device, f32_dtype)
    # decay_mask = decay_mask / decay_mask.sum(dim=-1, keepdim=True).sqrt()

    qkm = (qk * decay_mask.unsqueeze(0).to(dtype=dtype)).tril()
    # r = qkm.sum(dim=-1, keepdim=True).abs()
    # r = torch.where(r >= 1, r, torch.ones_like(r))
    # qkm = qkm / r

    # qkm = qk * mask
    # r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
    o = torch.einsum('bhqk,bkhd->bqhd', qkm.to(dtype=dtype), V)

    # cross-chunk (derived from recurrent retention)
    decay_gammas = rearrange(head_decays, "h -> () h () ()")
    device = K.device
    dtype = K.dtype
    inner_pos = rearrange(
        torch.arange(K.size(1), device=device, dtype=dtype) + 1,
        "n -> () () n ()",
    )
    state_decays = decay_gammas ** (K.size(1) - inner_pos)
    discounted_key = einsum(K, state_decays.to(dtype=dtype), "b n h d, _ h n _ -> b h n d")
    state = einsum(discounted_key, V, "b h n d1, b n h d2 -> b h d1 d2")

    # update recurrent state using prev_state:
    chunk_decay = decay_gammas ** K.size(1)
    state = state + prev_state * chunk_decay

    # Update the retention Tensor, based on cross-chunk information
    inner_decay = rearrange(decay_gammas ** inner_pos, "b h n d -> b n h d")
    o = o + (
            einsum(Q.to(dtype=dtype), prev_state.to(dtype=dtype), "b n h d1, b h d1 d2 -> b n h d2") * inner_decay.to(dtype=dtype)
    )

    return o.to(dtype=dtype), state.to(dtype=dtype)


def ref_program(Q, K, V, prev_state, head_decays, *args):
    seq_len = Q.size(1)
    res = []
    chunk_size = 64
    state_t = prev_state
    # gn = torch.nn.GroupNorm(num_groups=Q.size(2), num_channels=Q.size(3), device=Q.device, dtype=Q.dtype)
    gn = torch.nn.LayerNorm(normalized_shape=V.size(3), device=Q.device, dtype=Q.dtype)
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

