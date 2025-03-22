import torch
import tilelang
import tilelang.language as T

from fused_retention import chunk_outputs_macro, chunk_state_update_macro, _get_decay_mask


def fused_pop_retention_fwd(batch, heads, seq_len, block_size, dim_qk, dim_v, BK, BV, BT):
    NK = T.ceildiv(dim_qk, BK)
    num_full_blocks = seq_len // block_size  # only keep states at the end/beginning of full blocks
    qk_shape = [batch, seq_len, heads, dim_qk]
    v_shape = [batch, seq_len, heads, dim_v]
    o_shape = [NK, batch, seq_len, heads, dim_v]  # we have to reduce the first dimension
    state_shape = [batch, heads, dim_qk, dim_v]
    block_states_shape = [NK, batch, num_full_blocks+1, heads, dim_qk, dim_v]
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
            block_states: T.Buffer(block_states_shape, dtype),
    ):
        """

        Args:
            Q:
            K:
            V:
            state:
            chunk_decays_mask:
            Output:
            block_states: states at the end of complete blocks, incl. initial state. For example, for sequence
            of length 40 with blocks of 15 tokens, this will include states (index): initial state, 14, and 29.

        Returns:

        """
        with T.Kernel(heads, batch, T.ceildiv(dim_v, BV) * NK, threads=64) as (i_head, i_batch, bz):
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
            segment_state_local = T.alloc_fragment((BK, BV), accum_dtype)

            state_shared = T.alloc_fragment((BK, BV), dtype, scope="shared")
            segment_state_shared = T.alloc_shared((BK, BV), dtype)

            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                output_shared: tilelang.layout.make_swizzled_layout(output_shared),
                state_shared: tilelang.layout.make_swizzled_layout(state_shared),
                segment_state_shared: tilelang.layout.make_swizzled_layout(segment_state_shared),
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

                # determine how many blocks start within the current chunk:
                first_token_block = T.FloorDiv(k * BT, block_size)
                last_token_block = T.FloorDiv((k+1) * BT - 1, block_size)
                is_first_token_block_start = T.if_then_else(T.Mod(k * BT, block_size) > 0, 0, 1)
                num_iterations = last_token_block - first_token_block + is_first_token_block_start
                block_idx = first_token_block + 1 - is_first_token_block_start
                block_start_idx = block_idx * block_size - k * BT

                for i_block in T.Pipelined(num_iterations, num_stages=0):
                    T.copy(state_shared, segment_state_local)
                    c = BT - (block_start_idx + i_block * block_size)
                    update_recurrent_state(
                        mask,
                        K_shared,
                        K_local_trans,
                        K_local_trans_cast,
                        V_shared,
                        segment_state_local,
                        block_states[i_bk, i_batch, block_idx+i_block, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV],
                        decay,
                        c,
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

    return main


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
