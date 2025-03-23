import torch
import tilelang
from tilelang import cached
from torch import Tensor
from einops import rearrange, einsum


def fused_retention_bwd_dk_dv_ds(batch, heads, seq_len, dim_qk, dim_v, BK, BV, BT):
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
            chunk_decays_mask: T.Buffer([heads, BT, BT], accum_dtype),
            dO: T.Buffer(v_shape, dtype),
            dS_new: T.Buffer(state_shape, dtype),
            dK: T.Buffer(dqk_shape, dtype),
            dV: T.Buffer(dv_shape, dtype),
            dS: T.Buffer(d_state_shape, dtype),
    ):
        with T.Kernel(heads, batch, T.ceildiv(dim_v, BV) * NK, threads=128) as (i_head, i_batch, bz):
            Q_shared = T.alloc_shared([BT, BK], dtype)
            K_shared = T.alloc_shared([BT, BK], dtype)
            BK_BT_cast = T.alloc_fragment([BK, BT], dtype)
            V_shared = T.alloc_shared([BT, BV], dtype)

            dO_shared = T.alloc_shared([BT, BV], dtype)
            BT_BV_shared = T.alloc_shared([BT, BV], dtype)
            dS_new_shared = T.alloc_shared([BK, BV], dtype)

            mask = T.alloc_shared((BT, BT), accum_dtype)

            BT_BT_buffer = T.alloc_fragment((BT, BT), accum_dtype)
            BT_BT_buffer2 = T.alloc_fragment((BT, BT), accum_dtype)
            BT_BT_cast = T.alloc_shared((BT, BT), dtype)
            BT_BT_cast2 = T.alloc_shared((BT, BT), dtype)
            BT_BV_buffer = T.alloc_fragment((BT, BV), accum_dtype)
            BT_BV_buffer2 = T.alloc_fragment((BT, BV), accum_dtype)
            BT_BK_buffer = T.alloc_fragment((BT, BK), accum_dtype)
            BT_BK_buffer2 = T.alloc_fragment((BT, BK), accum_dtype)
            BT_BK_cast = T.alloc_shared((BT, BK), dtype)
            dS_local = T.alloc_fragment((BK, BV), accum_dtype)

            # T.annotate_layout({
            #     Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
            #     dO_shared: tilelang.layout.make_swizzled_layout(dO_shared),
            #     dS_new_shared: tilelang.layout.make_swizzled_layout(dS_new_shared),
            # })

            i_bk = bz % NK
            i_bv = bz // NK

            # prepare:
            T.copy(chunk_decays_mask[i_head, :, :], mask)
            decay = mask[1, 0]

            T.copy(dS_new[i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV], dS_new_shared)

            loop_range = T.ceildiv(seq_len, BT)
            for k_tag in T.Pipelined(loop_range, num_stages=1):
                k = loop_range - 1 - k_tag
                effective_chunk_size_correction = T.max(0, ((k + 1) * BT) - seq_len)

                T.copy(dO[i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV], dO_shared)
                T.copy(V[i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV], V_shared)
                T.copy(K[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK], K_shared)

                # Compute dK:
                T.clear(BT_BK_buffer)
                T.gemm(V_shared, dS_new_shared, BT_BK_buffer, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(BT, BK):
                    BT_BK_buffer[i, j] = BT_BK_buffer[i, j] * mask[BT-1, i+effective_chunk_size_correction] / sqrt_dim_qk

                # dO_VT_D:
                T.clear(BT_BT_buffer)
                T.gemm(dO_shared, V_shared, BT_BT_buffer, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(BT, BT):
                    BT_BT_buffer[i, j] = BT_BT_buffer[i, j] * mask[i, j] / sqrt_dim_qk
                T.copy(BT_BT_buffer, BT_BT_cast)
                # reuse dO_VT_D, but transposed:
                for i, j in T.Parallel(BT, BT):
                    BT_BT_buffer[i, j] = BT_BT_cast[j, i]
                T.copy(BT_BT_buffer, BT_BT_cast)

                T.copy(Q[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK], Q_shared)
                T.gemm(BT_BT_cast, Q_shared, BT_BK_buffer, policy=T.GemmWarpPolicy.FullCol)
                # T.copy(BT_BK_buffer, dK_shared)
                T.copy(BT_BK_buffer, dK[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK])

                # Compute dV:
                # T.clear(BT_BK_buffer)
                for i, j in T.Parallel(BT, BK):
                    BT_BK_buffer2[i, j] = K_shared[i, j] * mask[BT-1, i+effective_chunk_size_correction] / sqrt_dim_qk
                T.copy(BT_BK_buffer2, BT_BK_cast)
                T.clear(BT_BV_buffer)
                T.gemm(BT_BK_cast, dS_new_shared, BT_BV_buffer, policy=T.GemmWarpPolicy.FullCol)

                T.clear(BT_BT_buffer2)
                # Compute A^T @ dO:
                T.gemm(K_shared, Q_shared, BT_BT_buffer2, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(BT, BT):
                    BT_BT_buffer2[i, j] *= mask[j, i] / sqrt_dim_qk
                T.copy(BT_BT_buffer2, BT_BT_cast2)
                T.gemm(BT_BT_cast2, dO_shared, BT_BV_buffer, policy=T.GemmWarpPolicy.FullCol)
                # T.copy(BT_BV_buffer, dV_shared)
                T.copy(BT_BV_buffer, dV[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV])

                # Compute dS:
                cross_chunk_decay = mask[BT - 1, effective_chunk_size_correction] * decay
                T.copy(dS_new_shared, dS_local)
                for i, j in T.Parallel(BK, BV):
                    dS_local[i, j] = dS_local[i, j] * cross_chunk_decay
                # No support for transpose of first argument, need to do it mannually:
                for i, j in T.Parallel(BT, BK):
                    BK_BT_cast[j, i] = Q_shared[i, j]

                # dO * inner_decay:
                T.copy(dO_shared, BT_BV_buffer2)
                for i, j in T.Parallel(BT, BV):
                    BT_BV_buffer2[i, j] = BT_BV_buffer2[i, j] * (mask[i, 0] * decay)
                T.copy(BT_BV_buffer2, BT_BV_shared)

                T.gemm(BK_BT_cast, BT_BV_shared, dS_local, policy=T.GemmWarpPolicy.FullCol)

                # update dS_new_shared:
                T.copy(dS_local, dS_new_shared)
            T.copy(dS_new_shared, dS[i_bk, i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV])

    return main


def fused_retention_bwd_dq(batch, heads, seq_len, dim_qk, dim_v, BK, BV, BT):
    NK = T.ceildiv(dim_qk, BK)
    qk_shape = [batch, seq_len, heads, dim_qk]
    v_shape = [batch, seq_len, heads, dim_v]
    dqk_shape = [NK, batch, seq_len, heads, dim_qk]  # we have to reduce the first dimension
    state_shape = [batch, heads, dim_qk, dim_v]
    dtype = "bfloat16"
    accum_dtype = "float"

    sqrt_dim_qk = dim_qk ** 0.5

    @T.prim_func
    def main(
            K: T.Buffer(qk_shape, dtype),
            V: T.Buffer(v_shape, dtype),
            state: T.Buffer(state_shape, dtype),
            chunk_decays_mask: T.Buffer([heads, BT, BT], accum_dtype),
            dO: T.Buffer(v_shape, dtype),
            dQ: T.Buffer(dqk_shape, dtype),
    ):
        with T.Kernel(heads, batch, T.ceildiv(dim_v, BV) * NK, threads=128) as (i_head, i_batch, bz):
            K_shared = T.alloc_shared([BT, BK], dtype)
            BK_BT_cast = T.alloc_fragment([BK, BT], dtype)
            V_shared = T.alloc_shared([BT, BV], dtype)
            s_shared = T.alloc_shared([BK, BV], dtype)

            dO_shared = T.alloc_shared([BT, BV], dtype)
            BT_BV_shared = T.alloc_shared([BT, BV], dtype)

            mask = T.alloc_shared((BT, BT), accum_dtype)

            BT_BT_buffer = T.alloc_fragment((BT, BT), accum_dtype)
            BT_BT_cast = T.alloc_shared((BT, BT), dtype)
            BT_BV_buffer = T.alloc_fragment((BT, BV), accum_dtype)
            BT_BK_buffer = T.alloc_fragment((BT, BK), accum_dtype)
            dS_local = T.alloc_fragment((BK, BV), accum_dtype)

            # T.annotate_layout({
            #     Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
            #     dO_shared: tilelang.layout.make_swizzled_layout(dO_shared),
            # })

            i_bk = bz % NK
            i_bv = bz // NK

            # prepare:
            T.copy(chunk_decays_mask[i_head, :, :], mask)
            decay = mask[1, 0]

            T.copy(state[i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV], s_shared)

            loop_range = T.ceildiv(seq_len, BT)
            for k in T.Pipelined(loop_range, num_stages=2):
                # Compute dQ:
                # dO_VT_D:
                T.copy(dO[i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV], dO_shared)
                T.copy(V[i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV], V_shared)
                T.copy(K[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK], K_shared)

                T.clear(BT_BT_buffer)
                T.gemm(dO_shared, V_shared, BT_BT_buffer, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(BT, BT):
                    BT_BT_buffer[i, j] = BT_BT_buffer[i, j] * mask[i, j] / sqrt_dim_qk
                T.copy(BT_BT_buffer, BT_BT_cast)
                T.clear(BT_BK_buffer)
                T.gemm(BT_BT_cast, K_shared, BT_BK_buffer, policy=T.GemmWarpPolicy.FullCol)

                T.copy(dO_shared, BT_BV_buffer)
                for i, j in T.Parallel(BT, BV):
                    BT_BV_buffer[i, j] = BT_BV_buffer[i, j] * (mask[i, 0] * decay)
                T.copy(BT_BV_buffer, BT_BV_shared)

                T.gemm(BT_BV_shared, s_shared, BT_BK_buffer, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(BT_BK_buffer, dQ[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK])

                # update the state:
                c = T.max(0, ((k + 1) * BT) - seq_len)
                for i, j in T.Parallel(BK, BT):
                    # Also apply decay terms:
                    BT_BK_buffer[j, i] = (K_shared[j, i] / sqrt_dim_qk) * mask[BT - 1, j + c]  # T.pow(decay, BT - j - 1)
                    BK_BT_cast[i, j] = BT_BK_buffer[j, i]

                cross_chunk_decay = mask[BT - 1, c] * decay  # T.pow(decay, BT)
                T.copy(s_shared, dS_local)
                for i, j in T.Parallel(BK, BV):
                    dS_local[i, j] = dS_local[i, j] * cross_chunk_decay
                T.gemm(BK_BT_cast, V_shared, dS_local, policy=T.GemmWarpPolicy.FullCol)
                T.copy(dS_local, s_shared)

    return main
