import tilelang
import tilelang.language as T

from .fused_chunk_fwd import chunk_state_update_macro


def compute_do_dk_macro(dim_qk, BK, BV, BT):
    dtype, accum_dtype = "bfloat16", "float"
    sqrt_dim_qk = dim_qk ** 0.5

    @T.macro
    def compute_retention_bwd_dk_do(
            Q: T.Buffer((BT, BK), dtype),
            Q_shared: T.Buffer((BT, BK), dtype),
            V_shared: T.Buffer((BT, BV), dtype),
            mask: T.Buffer((BT, BT), accum_dtype),
            dO_shared: T.Buffer((BT, BV), dtype),
            BT_BK_buffer: T.Buffer((BT, BK), accum_dtype),
            BT_BT_buffer: T.Buffer((BT, BT), accum_dtype),
            BT_BT_cast: T.Buffer((BT, BT), dtype),
    ):
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

        T.copy(Q, Q_shared)
        T.gemm(BT_BT_cast, Q_shared, BT_BK_buffer, policy=T.GemmWarpPolicy.FullCol)
    return compute_retention_bwd_dk_do


def accumulate_ds_out_dk_macro(dim_qk, BK, BV, BT):
    dtype, accum_dtype = "bfloat16", "float"
    sqrt_dim_qk = dim_qk ** 0.5

    @T.macro
    def accumulate_ds_out_dk(
            V_shared: T.Buffer((BT, BV), dtype),
            mask: T.Buffer((BT, BT), accum_dtype),
            dS_out_shared: T.Buffer((BK, BV), dtype),
            BT_BK_buffer: T.Buffer((BT, BK), accum_dtype),
            BT_BV_buffer: T.Buffer((BT, BK), accum_dtype),
            BT_BV_shared: T.Buffer((BT, BK), dtype),
            effective_chunk_size_correction: T.int32
    ):
        # compute dS_out / dK:
        T.copy(V_shared, BT_BV_buffer)
        c = effective_chunk_size_correction
        for i, j in T.Parallel(BT, BV):
            mask_clipped = T.if_then_else(i + c <= BT - 1, mask[BT - 1, i + c], 0)
            BT_BV_buffer[i, j] = BT_BV_buffer[i, j] * mask_clipped / sqrt_dim_qk
        T.copy(BT_BV_buffer, BT_BV_shared)
        T.gemm(BT_BV_shared, dS_out_shared, BT_BK_buffer, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
        # T.gemm(V_shared, dS_out_shared, BT_BK_buffer, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
        # c = effective_chunk_size_correction
        # for i, j in T.Parallel(BT, BK):
        #     mask_clipped = T.if_then_else(j + c <= BT - 1, mask[BT - 1, i + c], 0)
        #     BT_BK_buffer[i, j] = BT_BK_buffer[i, j] * mask_clipped / sqrt_dim_qk

    return accumulate_ds_out_dk


def compute_dk_macro(dim_qk, BK, BV, BT):
    dtype, accum_dtype = "bfloat16", "float"

    compute_dO_dK = compute_do_dk_macro(dim_qk, BK, BV, BT)
    accumulate_dS_out_dK = accumulate_ds_out_dk_macro(dim_qk, BK, BV, BT)

    @T.macro
    def compute_dK(
            Q: T.Buffer((BT, BK), dtype),
            Q_shared: T.Buffer((BT, BK), dtype),
            V_shared: T.Buffer((BT, BV), dtype),
            mask: T.Buffer((BT, BT), accum_dtype),
            dO_shared: T.Buffer((BT, BV), dtype),
            dS_out_shared: T.Buffer((BK, BV), dtype),
            BT_BK_buffer: T.Buffer((BT, BK), accum_dtype),
            BT_BV_buffer: T.Buffer((BT, BK), accum_dtype),
            BT_BV_shared: T.Buffer((BT, BK), dtype),
            BT_BT_buffer: T.Buffer((BT, BT), accum_dtype),
            BT_BT_cast: T.Buffer((BT, BT), dtype),
            dK: T.Buffer((BT, BK), dtype),
            effective_chunk_size_correction: T.int32
    ):
        # compute dS_out / dK:
        T.clear(BT_BK_buffer)
        accumulate_dS_out_dK(V_shared, mask, dS_out_shared, BT_BK_buffer, BT_BV_buffer, BT_BV_shared, effective_chunk_size_correction)

        compute_dO_dK(Q, Q_shared, V_shared, mask, dO_shared, BT_BK_buffer, BT_BT_buffer, BT_BT_cast)
        T.copy(BT_BK_buffer, dK)

    return compute_dK


def compute_do_dv_macro(dim_qk, BK, BV, BT):
    dtype, accum_dtype = "bfloat16", "float"
    sqrt_dim_qk = dim_qk ** 0.5

    @T.macro
    def compute_retention_bwd_do_dv(
            K_shared: T.Buffer((BT, BK), dtype),
            Q_shared: T.Buffer((BT, BK), dtype),
            mask: T.Buffer((BT, BT), accum_dtype),
            dO_shared: T.Buffer((BT, BV), dtype),
            BT_BV_buffer: T.Buffer((BT, BV), accum_dtype),
            BT_BT_buffer: T.Buffer((BT, BT), accum_dtype),
            BT_BT_cast: T.Buffer((BT, BT), dtype),
    ):
        T.clear(BT_BT_buffer)
        # Compute A^T @ dO:
        T.gemm(K_shared, Q_shared, BT_BT_buffer, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
        for i, j in T.Parallel(BT, BT):
            BT_BT_buffer[i, j] *= mask[j, i] / sqrt_dim_qk
        T.copy(BT_BT_buffer, BT_BT_cast)
        T.gemm(BT_BT_cast, dO_shared, BT_BV_buffer, policy=T.GemmWarpPolicy.FullCol)

    return compute_retention_bwd_do_dv


def accumulate_ds_out_dv_macro(dim_qk, BK, BV, BT):
    dtype, accum_dtype = "bfloat16", "float"
    sqrt_dim_qk = dim_qk ** 0.5

    @T.macro
    def accumulate_ds_out_dv(
            K_shared: T.Buffer((BT, BK), dtype),
            mask: T.Buffer((BT, BT), accum_dtype),
            dS_out_shared: T.Buffer((BK, BV), dtype),
            BT_BK_buffer: T.Buffer((BT, BK), accum_dtype),
            BT_BK_cast: T.Buffer((BT, BK), dtype),
            BT_BV_buffer: T.Buffer((BT, BV), accum_dtype),
            effective_chunk_size_correction: T.int32
    ):
        c = effective_chunk_size_correction
        for i, j in T.Parallel(BT, BK):
            mask_clipped = T.if_then_else(i + c <= BT - 1, mask[BT - 1, i + c], 0)
            BT_BK_buffer[i, j] = K_shared[i, j] * mask_clipped / sqrt_dim_qk
        T.copy(BT_BK_buffer, BT_BK_cast)
        T.gemm(BT_BK_cast, dS_out_shared, BT_BV_buffer, policy=T.GemmWarpPolicy.FullCol)

    return accumulate_ds_out_dv


def compute_dv_macro(dim_qk, BK, BV, BT):
    dtype, accum_dtype = "bfloat16", "float"

    compute_do_dv = compute_do_dv_macro(dim_qk, BK, BV, BT)
    accumulate_ds_out_dv = accumulate_ds_out_dv_macro(dim_qk, BK, BV, BT)

    @T.macro
    def compute_dv(
            K_shared: T.Buffer((BT, BK), dtype),
            Q_shared: T.Buffer((BT, BK), dtype),
            mask: T.Buffer((BT, BT), accum_dtype),
            dO_shared: T.Buffer((BT, BV), dtype),
            dS_out_shared: T.Buffer((BK, BV), dtype),
            BT_BK_buffer: T.Buffer((BT, BK), accum_dtype),
            BT_BK_cast: T.Buffer((BT, BK), dtype),
            BT_BV_buffer: T.Buffer((BT, BV), accum_dtype),
            BT_BT_buffer: T.Buffer((BT, BT), accum_dtype),
            BT_BT_cast: T.Buffer((BT, BT), dtype),
            dV: T.Buffer((BT, BV), dtype),
            effective_chunk_size_correction: T.int32
    ):
        T.clear(BT_BV_buffer)
        accumulate_ds_out_dv(K_shared, mask, dS_out_shared, BT_BK_buffer, BT_BK_cast, BT_BV_buffer, effective_chunk_size_correction)

        compute_do_dv(K_shared, Q_shared, mask, dO_shared, BT_BV_buffer, BT_BT_buffer, BT_BT_cast)
        T.copy(BT_BV_buffer, dV)

    return compute_dv


def compute_ds_macro(BK, BV, BT):
    dtype, accum_dtype = "bfloat16", "float"

    @T.macro
    def compute_ds(
            Q_shared: T.Buffer((BT, BK), dtype),
            mask: T.Buffer((BT, BT), accum_dtype),
            do_shared: T.Buffer((BT, BV), dtype),
            ds_out_shared: T.Buffer((BK, BV), accum_dtype),
            ds_local: T.Buffer((BK, BV), accum_dtype),
            BK_BT_cast: T.Buffer((BK, BT), dtype),
            BT_BV_buffer: T.Buffer((BT, BV), accum_dtype),
            BT_BV_shared: T.Buffer((BT, BV), dtype),
            effective_chunk_size_correction: T.int32,
            decay: T.float32
    ):
        cross_chunk_decay = mask[BT - 1, effective_chunk_size_correction] * decay
        T.copy(ds_out_shared, ds_local)
        for i, j in T.Parallel(BK, BV):
            ds_local[i, j] = ds_local[i, j] * cross_chunk_decay

        # No support for transpose of first argument, need to do it mannually:
        for i, j in T.Parallel(BT, BK):
            BK_BT_cast[j, i] = Q_shared[i, j]
        # dO * inner_decay:
        T.copy(do_shared, BT_BV_buffer)
        for i, j in T.Parallel(BT, BV):
            BT_BV_buffer[i, j] = BT_BV_buffer[i, j] * (mask[i, 0] * decay)
        T.copy(BT_BV_buffer, BT_BV_shared)

        T.gemm(BK_BT_cast, BT_BV_shared, ds_local, policy=T.GemmWarpPolicy.FullCol)

        # update dS_out_shared:
        T.copy(ds_local, ds_out_shared)

    return compute_ds


def fused_retention_bwd_dk_dv_ds(batch, heads, seq_len, dim_qk, dim_v, BK, BV, BT, threads=64):
    NK = T.ceildiv(dim_qk, BK)
    qk_shape = [batch, seq_len, heads, dim_qk]
    v_shape = [batch, seq_len, heads, dim_v]
    dv_shape = [NK, batch, seq_len, heads, dim_v]  # we have to reduce the first dimension
    dqk_shape = [NK, batch, seq_len, heads, dim_qk]  # we have to reduce the first dimension
    state_shape = [batch, heads, dim_qk, dim_v]
    d_state_shape = [NK, batch, heads, dim_qk, dim_v]
    dtype, accum_dtype = "bfloat16", "float"

    compute_dK = compute_dk_macro(dim_qk, BK, BV, BT)
    compute_dV = compute_dv_macro(dim_qk, BK, BV, BT)
    compute_dS = compute_ds_macro(BK, BV, BT)

    @T.prim_func
    def main(
            Q: T.Buffer(qk_shape, dtype),
            K: T.Buffer(qk_shape, dtype),
            V: T.Buffer(v_shape, dtype),
            chunk_decays_mask: T.Buffer([heads, BT, BT], accum_dtype),
            dO: T.Buffer(v_shape, dtype),
            dS_out: T.Buffer(state_shape, dtype),
            dK: T.Buffer(dqk_shape, dtype),
            dV: T.Buffer(dv_shape, dtype),
            dS: T.Buffer(d_state_shape, dtype),
    ):
        with T.Kernel(heads, batch, T.ceildiv(dim_v, BV) * NK, threads=threads) as (i_head, i_batch, bz):
            Q_shared = T.alloc_shared([BT, BK], dtype)
            K_shared = T.alloc_shared([BT, BK], dtype)
            BK_BT_cast = T.alloc_fragment([BK, BT], dtype)
            V_shared = T.alloc_shared([BT, BV], dtype)

            do_shared = T.alloc_shared([BT, BV], dtype)
            BT_BV_shared = T.alloc_shared([BT, BV], dtype)
            BT_BV_shared2 = T.alloc_shared([BT, BV], dtype)
            ds_out_cast = T.alloc_shared([BK, BV], dtype)
            ds_out_shared = T.alloc_shared((BK, BV), accum_dtype)
            ds_local = T.alloc_fragment((BK, BV), accum_dtype)

            mask = T.alloc_shared((BT, BT), accum_dtype)

            BT_BT_buffer = T.alloc_fragment((BT, BT), accum_dtype)
            BT_BT_cast = T.alloc_shared((BT, BT), dtype)
            BT_BV_buffer = T.alloc_fragment((BT, BV), accum_dtype)
            BT_BV_buffer2 = T.alloc_fragment((BT, BV), accum_dtype)
            dv_local = T.alloc_fragment((BT, BV), accum_dtype)
            dk_local = T.alloc_fragment((BT, BK), accum_dtype)
            BT_BK_buffer2 = T.alloc_fragment((BT, BK), accum_dtype)
            BT_BK_cast = T.alloc_shared((BT, BK), dtype)
            dk_shared = T.alloc_shared((BT, BK), dtype)
            dv_shared = T.alloc_shared((BT, BV), dtype)

            # T.annotate_layout({
            #     Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
            #     do_shared: tilelang.layout.make_swizzled_layout(do_shared),
            #     ds_out_cast: tilelang.layout.make_swizzled_layout(ds_out_cast),
            # })

            i_bk = bz % NK
            i_bv = bz // NK

            # prepare:
            T.copy(chunk_decays_mask[i_head, :, :], mask)
            decay = mask[1, 0]

            T.copy(dS_out[i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV], ds_out_cast)
            T.copy(ds_out_cast, ds_out_shared)

            loop_range = T.ceildiv(seq_len, BT)
            for k_tag in T.Pipelined(loop_range, num_stages=0):
                k = loop_range - 1 - k_tag
                effective_chunk_size_correction = T.max(0, ((k + 1) * BT) - seq_len)

                T.copy(dO[i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV], do_shared)
                T.copy(V[i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV], V_shared)
                T.copy(K[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK], K_shared)

                compute_dK(
                    Q[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK],
                    Q_shared,
                    V_shared,
                    mask,
                    do_shared,
                    ds_out_cast,
                    dk_local,
                    BT_BV_buffer,
                    BT_BV_shared,
                    BT_BT_buffer,
                    BT_BT_cast,
                    dk_shared,
                    effective_chunk_size_correction
                )

                compute_dV(
                    K_shared,
                    Q_shared,
                    mask,
                    do_shared,
                    ds_out_cast,
                    BT_BK_buffer2,
                    BT_BK_cast,
                    dv_local,
                    BT_BT_buffer,
                    BT_BT_cast,
                    dv_shared,
                    effective_chunk_size_correction
                )

                compute_dS(
                    Q_shared,
                    mask,
                    do_shared,
                    ds_out_shared,
                    ds_local,
                    BK_BT_cast,
                    BT_BV_buffer2,
                    BT_BV_shared2,
                    effective_chunk_size_correction,
                    decay
                )
                T.copy(ds_local, ds_out_cast)
                # T.copy(dk_shared, dk_local)
                # T.copy(dv_shared, BT_BV_buffer)
                T.copy(dk_shared, dK[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK])
                T.copy(dv_shared, dV[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV])

            T.copy(ds_out_cast, dS[i_bk, i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV])

    return main


def fused_retention_bwd_dq(batch, heads, seq_len, dim_qk, dim_v, BK, BV, BT, threads=64):
    NK = T.ceildiv(dim_qk, BK)
    qk_shape = [batch, seq_len, heads, dim_qk]
    v_shape = [batch, seq_len, heads, dim_v]
    dqk_shape = [NK, batch, seq_len, heads, dim_qk]  # we have to reduce the first dimension
    state_shape = [batch, heads, dim_qk, dim_v]
    dtype = "bfloat16"
    accum_dtype = "float"

    sqrt_dim_qk = dim_qk ** 0.5
    chunk_state_update = chunk_state_update_macro(dim_qk, BK, BV, BT, False)

    @T.prim_func
    def main(
            K: T.Buffer(qk_shape, dtype),
            V: T.Buffer(v_shape, dtype),
            state: T.Buffer(state_shape, dtype),
            chunk_decays_mask: T.Buffer([heads, BT, BT], accum_dtype),
            dO: T.Buffer(v_shape, dtype),
            dQ: T.Buffer(dqk_shape, dtype),
    ):
        with T.Kernel(heads, batch, T.ceildiv(dim_v, BV) * NK, threads=threads) as (i_head, i_batch, bz):
            K_shared = T.alloc_shared([BT, BK], dtype)
            BK_BT_cast = T.alloc_shared([BK, BT], dtype)
            V_shared = T.alloc_shared([BT, BV], dtype)
            state_cast = T.alloc_shared([BK, BV], dtype)

            dO_shared = T.alloc_shared([BT, BV], dtype)
            BT_BV_shared = T.alloc_shared([BT, BV], dtype)

            mask = T.alloc_shared((BT, BT), accum_dtype)

            BT_BT_buffer = T.alloc_fragment((BT, BT), accum_dtype)
            BT_BT_cast = T.alloc_shared((BT, BT), dtype)
            BT_BV_buffer = T.alloc_fragment((BT, BV), accum_dtype)
            BT_BK_buffer = T.alloc_fragment((BT, BK), accum_dtype)
            dq_shared = T.alloc_shared((BT, BK), accum_dtype)
            BK_BT_buffer = T.alloc_fragment((BK, BT), accum_dtype)
            state_local = T.alloc_fragment((BK, BV), accum_dtype)
            state_shared = T.alloc_shared((BK, BV), accum_dtype)

            # T.annotate_layout({
            #     dQ: tilelang.layout.make_swizzled_layout(dQ),
            #     dO_shared: tilelang.layout.make_swizzled_layout(dO_shared),
            # })
            # T.annotate_layout({dO_shared: tilelang.layout.make_swizzled_layout(dO_shared)})

            i_bk = bz % NK
            i_bv = bz // NK

            # prepare:
            T.copy(chunk_decays_mask[i_head, :, :], mask)
            decay = mask[1, 0]

            T.copy(state[i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV], state_shared)

            loop_range = T.ceildiv(seq_len, BT)
            for k in T.Pipelined(loop_range, num_stages=0):
                # Compute dQ:
                # dO_VT_D:
                T.copy(dO[i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV], dO_shared)
                T.copy(V[i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV], V_shared)

                T.gemm(dO_shared, V_shared, BT_BT_buffer, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                for i, j in T.Parallel(BT, BT):
                    BT_BT_buffer[i, j] = BT_BT_buffer[i, j] * mask[i, j] / sqrt_dim_qk
                T.copy(BT_BT_buffer, BT_BT_cast)
                T.copy(K[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK], K_shared)
                T.gemm(BT_BT_cast, K_shared, BT_BK_buffer, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                T.copy(BT_BK_buffer, dq_shared)

                # Second term (dO * \zeta) @ S^T:
                # T.copy(dO_shared, BT_BV_buffer)
                for i, j in T.Parallel(BT, BV):
                    BT_BV_buffer[i, j] = dO_shared[i, j] * (mask[i, 0] * decay)
                T.copy(BT_BV_buffer, BT_BV_shared)

                T.copy(state_shared, state_cast)
                T.copy(dq_shared, BT_BK_buffer)
                T.gemm(BT_BV_shared, state_cast, BT_BK_buffer, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(BT_BK_buffer, dq_shared)

                # update the state:
                c = T.max(0, ((k + 1) * BT) - seq_len)
                T.copy(state_shared, state_local)
                chunk_state_update(
                    mask,
                    K_shared,
                    BK_BT_buffer,
                    BK_BT_cast,
                    V_shared,
                    state_local,
                    state_shared,
                    decay,
                    c
                )

                T.copy(dq_shared, dQ[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK])


    return main
