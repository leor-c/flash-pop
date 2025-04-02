import tilelang
import tilelang.language as T

from flash_pop.fused_retention.fused_chunk_bwd import (
    compute_dk_macro, compute_dv_macro, compute_ds_macro, accumulate_ds_out_dk_macro, accumulate_ds_out_dv_macro,
)


def pop_retention_bwd_dk_dv_ds(batch, heads, seq_len, block_size, dim_qk, dim_v, BK, BV, BT, dtype="bfloat16"):
    NK = T.ceildiv(dim_qk, BK)
    num_full_blocks = seq_len // block_size  # only keep states at the end/beginning of full blocks
    qk_shape = [batch, seq_len, heads, dim_qk]
    v_shape = [batch, seq_len, heads, dim_v]
    dv_shape = [NK, batch, seq_len, heads, dim_v]  # we have to reduce the first dimension
    dqk_shape = [NK, batch, seq_len, heads, dim_qk]  # we have to reduce the first dimension
    d_block_states_shape = [batch, num_full_blocks, heads, dim_qk, dim_v]
    d_state_shape = [NK, batch, heads, dim_qk, dim_v]
    assert dtype == "bfloat16"
    accum_dtype = "float"

    compute_dK = compute_dk_macro(dim_qk, BK, BV, BT)
    compute_dV = compute_dv_macro(dim_qk, BK, BV, BT)
    compute_dS = compute_ds_macro(BK, BV, BT)

    accumulate_ds_out_dk = accumulate_ds_out_dk_macro(dim_qk, BK, BV, BT)
    accumulate_ds_out_dv = accumulate_ds_out_dv_macro(dim_qk, BK, BV, BT)
    accumulate_ds_out_dk = accumulate_ds_out_dk_macro(dim_qk, BK, BV, BT)

    @T.prim_func
    def main(
            Q: T.Buffer(qk_shape, dtype),
            K: T.Buffer(qk_shape, dtype),
            V: T.Buffer(v_shape, dtype),
            chunk_decays_mask: T.Buffer([heads, BT, BT], accum_dtype),
            dO: T.Buffer(v_shape, dtype),
            d_block_states: T.Buffer(d_block_states_shape, dtype),
            dK: T.Buffer(dqk_shape, dtype),
            dV: T.Buffer(dv_shape, dtype),
            dS: T.Buffer(d_state_shape, dtype),
    ):
        with T.Kernel(heads, batch, T.ceildiv(dim_v, BV) * NK, threads=64) as (i_head, i_batch, bz):
            Q_shared = T.alloc_shared([BT, BK], dtype)
            K_shared = T.alloc_shared([BT, BK], dtype)
            BK_BT_cast = T.alloc_fragment([BK, BT], dtype)
            V_shared = T.alloc_shared([BT, BV], dtype)

            do_shared = T.alloc_shared([BT, BV], dtype)
            BT_BV_shared = T.alloc_shared([BT, BV], dtype)
            ds_out_cast = T.alloc_shared([BK, BV], dtype)
            ds_out_cast2 = T.alloc_shared([BK, BV], dtype)
            ds_out_shared = T.alloc_shared((BK, BV), accum_dtype)

            mask = T.alloc_shared((BT, BT), accum_dtype)

            BT_BT_buffer = T.alloc_fragment((BT, BT), accum_dtype)
            BT_BT_cast = T.alloc_shared((BT, BT), dtype)
            BT_BV_buffer = T.alloc_fragment((BT, BV), accum_dtype)
            BT_BV_buffer2 = T.alloc_fragment((BT, BV), accum_dtype)
            BT_BK_buffer = T.alloc_fragment((BT, BK), accum_dtype)
            d_state_dk_buffer = T.alloc_fragment((BT, BK), accum_dtype)
            BT_BK_buffer2 = T.alloc_fragment((BT, BK), accum_dtype)
            BT_BK_cast = T.alloc_shared((BT, BK), dtype)
            ds_local = T.alloc_fragment((BK, BV), accum_dtype)
            dk_shared = T.alloc_shared((BT, BK), accum_dtype)
            dv_shared = T.alloc_shared((BT, BV), accum_dtype)

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

            T.clear(ds_local)
            T.copy(ds_local, ds_out_shared)
            T.copy(ds_out_shared, ds_out_cast)

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
                    d_state_dk_buffer,  # BT_BK_buffer
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
                    BT_BK_buffer,
                    BT_BK_cast,
                    BT_BV_buffer,
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
                    BT_BV_buffer,
                    BT_BV_shared,
                    effective_chunk_size_correction,
                    decay
                )

                # determine how many blocks start within the current chunk:
                first_token_block = T.FloorDiv(k * BT, block_size)
                last_token_block = T.FloorDiv((k + 1) * BT - 1, block_size)
                is_last_token_block_end = T.if_then_else(T.Mod((k + 1) * BT, block_size) == 0, 1, 0)
                num_iterations = last_token_block - first_token_block + is_last_token_block_end
                block_idx = first_token_block
                next_block_start_idx = (block_idx + 1) * block_size - k * BT

                T.copy(dk_shared, d_state_dk_buffer)
                T.copy(dv_shared, BT_BV_buffer)
                for i_block in T.Pipelined(num_iterations, num_stages=0):
                    T.copy(d_block_states[i_batch, block_idx + i_block, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV], ds_out_cast2)
                    c = BT - (next_block_start_idx + i_block * block_size)

                    accumulate_ds_out_dk(V_shared, mask, ds_out_cast2, d_state_dk_buffer, BT_BV_buffer2, BT_BV_shared, c)
                    accumulate_ds_out_dv(K_shared, mask, ds_out_cast2, BT_BK_buffer2, BT_BK_cast, BT_BV_buffer, c)
                    # Accumulate d_state_out / d_state:
                    cross_chunk_decay = mask[BT - 1, c] * decay
                    for i, j in T.Parallel(BK, BV):
                        ds_local[i, j] += ds_out_cast2[i, j] * cross_chunk_decay

                T.copy(ds_local, ds_out_shared)
                T.copy(ds_out_shared, ds_out_cast)
                T.copy(d_state_dk_buffer, dK[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK])
                T.copy(BT_BV_buffer, dV[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV])

            T.copy(ds_out_cast, dS[i_bk, i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV])

    return main
