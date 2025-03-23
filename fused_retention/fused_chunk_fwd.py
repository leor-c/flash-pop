import tilelang
from tilelang import language as T


def chunk_outputs_macro(batch, heads, seq_len, dim_qk, dim_v, BK, BV, BT):
    NK = T.ceildiv(dim_qk, BK)
    o_shape = [NK, batch, seq_len, heads, dim_v]  # we have to reduce the first dimension
    dtype = "bfloat16"
    accum_dtype = "float"

    sqrt_dim_qk = dim_qk ** 0.5

    @T.macro
    def compute_retention_chunk_outputs(
            attention_scores_local: T.Buffer([BT, BT], accum_dtype),
            attention_scores_cast: T.Buffer([BT, BT], dtype),
            mask: T.Buffer([BT, BT], accum_dtype),
            state_shared: T.Buffer([BK, BV], dtype),
            output_shared: T.Buffer([BT, BV], dtype),
            output_local: T.Buffer([BT, BV], accum_dtype),
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
        T.clear(attention_scores_local)
        T.gemm(Q_shared, K_shared, attention_scores_local, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
        for i, j in T.Parallel(BT, BT):
            attention_scores_local[i, j] = (attention_scores_local[i, j] / sqrt_dim_qk) * mask[i, j]
            # attention_scores_local[i, j] = T.if_then_else(i >= j, attention_scores_local[i, j] * T.pow(decay, i - j), 0)
        T.copy(attention_scores_local, attention_scores_cast)

        # Compute outputs:
        T.clear(output_local)
        T.gemm(Q_shared, state_shared, output_local, policy=T.GemmWarpPolicy.FullCol)
        for i, j in T.Parallel(BT, BV):
            output_local[i, j] = output_local[i, j] * (mask[i, 0] * decay)

        T.gemm(attention_scores_cast, V_shared, output_local, policy=T.GemmWarpPolicy.FullCol)

        T.copy(output_local, output_shared)
        T.copy(output_shared, Output[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV])

    return compute_retention_chunk_outputs


def chunk_state_update_macro(dim_qk, BK, BV, BT):
    dtype = "bfloat16"
    accum_dtype = "float"

    sqrt_dim_qk = dim_qk ** 0.5

    @T.macro
    def update_recurrent_state(
            mask: T.Buffer([BT, BT], accum_dtype),
            K_shared: T.Buffer([BT, BK], dtype),
            K_local_trans: T.Buffer([BK, BT], accum_dtype),
            K_local_trans_cast: T.Buffer([BK, BT], dtype),
            V_shared: T.Buffer([BT, BV], dtype),
            state_local: T.Buffer([BK, BV], accum_dtype),
            state_shared: T.Buffer([BK, BV], dtype),
            decay: T.float32,
            effective_chunk_size_correction: T.int32
    ):
        # transpose k first because T.gemm does not have a good support for transposing the first operand according to the authors
        c = effective_chunk_size_correction  # if last chunk is shorter (c>0), decays exponents should be adjusted
        for i, j in T.Parallel(BK, BT):
            # Also apply decay terms:
            mask_clipped = T.if_then_else(j+c <= BT-1, mask[BT - 1, j+c], 0)
            K_local_trans[i, j] = (K_shared[j, i] / sqrt_dim_qk) * mask_clipped  # T.pow(decay, BT - j - 1)
        T.copy(K_local_trans, K_local_trans_cast)

        # cross_chunk_decay = T.if_then_else(effective_chunk_size_correction < BT, mask[BT - 1, c] * decay, 1)  # T.pow(decay, BT)
        cross_chunk_decay = T.if_then_else(c < BT, mask[BT - 1, c] * decay, mask[0, 0])  # mask[0, 0] = 1
        for i, j in T.Parallel(BK, BV):
            state_local[i, j] = state_local[i, j] * cross_chunk_decay
        T.gemm(K_local_trans_cast, V_shared, state_local, policy=T.GemmWarpPolicy.FullCol)
        T.copy(state_local, state_shared)

    return update_recurrent_state


def fused_chunk_retention_fwd(batch, heads, seq_len, dim_qk, dim_v, BK, BV, BT):
    NK = T.ceildiv(dim_qk, BK)
    qk_shape = [batch, seq_len, heads, dim_qk]
    v_shape = [batch, seq_len, heads, dim_v]
    o_shape = [NK, batch, seq_len, heads, dim_v]  # we have to reduce the first dimension
    state_shape = [batch, heads, dim_qk, dim_v]
    out_state_shape = [NK, batch, heads, dim_qk, dim_v]
    dtype = "bfloat16"
    accum_dtype = "float"

    compute_retention_chunk_outputs = chunk_outputs_macro(batch, heads, seq_len, dim_qk, dim_v, BK, BV, BT)

    update_recurrent_state = chunk_state_update_macro(dim_qk, BK, BV, BT)

    @T.prim_func
    def main(
            Q: T.Buffer(qk_shape, dtype),
            K: T.Buffer(qk_shape, dtype),
            V: T.Buffer(v_shape, dtype),
            state: T.Buffer(state_shape, dtype),
            chunk_decays_mask: T.Buffer([heads, BT, BT], accum_dtype),
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

            output_local = T.alloc_fragment((BT, BV), accum_dtype)
            output_shared = T.alloc_shared([BT, BV], dtype)

            state_local = T.alloc_fragment((BK, BV), accum_dtype)
            attention_scores_local = T.alloc_fragment((BT, BT), accum_dtype)
            attention_scores_cast = T.alloc_shared((BT, BT), dtype)
            mask = T.alloc_shared((BT, BT), accum_dtype)

            state_shared = T.alloc_fragment((BK, BV), dtype, scope="shared")

            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                output_shared: tilelang.layout.make_swizzled_layout(output_shared),
                state_shared: tilelang.layout.make_swizzled_layout(state_shared),
            })

            T.clear(state_local)
            T.copy(state[i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV], state_shared)
            T.copy(state_shared, state_local)

            # init decay values:
            T.copy(chunk_decays_mask[i_head, :, :], mask)
            decay = mask[1, 0]

            loop_range = T.ceildiv(seq_len, BT)
            for k in T.Pipelined(loop_range, num_stages=2):
                T.copy(K[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK], K_shared)
                T.copy(Q[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK], Q_shared)
                T.copy(V[i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV], V_shared)
                effective_chunk_size_correction = T.max(0, ((k+1)*BT) - seq_len)

                compute_retention_chunk_outputs(
                    attention_scores_local,
                    attention_scores_cast,
                    mask,
                    state_shared,
                    output_shared,
                    output_local,
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
                    state_local,
                    state_shared,
                    decay,
                    effective_chunk_size_correction
                )
            T.copy(state_shared, out_state[i_bk, i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV])

    return main
