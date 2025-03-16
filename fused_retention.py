import argparse
from math import ceil

import torch
import tilelang
import tilelang.language as T
from fla.ops.linear_attn import fused_chunk_linear_attn as chunk_linear_attn
# from fla.ops.linear_attn import chunk_linear_attn
from functools import partial


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
    accum_dtype = "float32"

    @T.macro
    def compute_retention_chunk_outputs(
            acc_A_local: T.Buffer([BT, BT], accum_dtype),
            acc_A_cast: T.Buffer([BT, BT], dtype),
            mask_local: T.Buffer([BT, BT], accum_dtype),
            mask_reduce_buffer: T.Buffer([BT,], accum_dtype),
            acc_s_shared: T.Buffer([BK, BV], dtype),
            acc_o_shared: T.Buffer([BT, BV], dtype),
            acc_o_local: T.Buffer([BT, BV], accum_dtype),
            acc_o_local2: T.Buffer([BT, BV], accum_dtype),
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
        #       Compute decay AR mask:
        for i, j in T.Parallel(BT, BT):
            mask_local[i, j] = T.pow(decay, i - j)
            mask_local[i, j] = T.if_then_else(i >= j, mask_local[i, j], 0)

        T.reduce_sum(mask_local, mask_reduce_buffer, dim=1)
        for i, j in T.Parallel(BT, BT):
            mask_local[i, j] /= T.pow(mask_reduce_buffer[i], 0.5)

        #       Compute attn. scores + apply mask + normalization:
        T.clear(acc_A_local)
        T.gemm(Q_shared, K_shared, acc_A_local, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
        sqrt_d = T.pow(dim_qk, 0.5)
        for i, j in T.Parallel(BT, BT):
            # acc_A_local[i, j] /= sqrt_d
            acc_A_local[i, j] *= (mask_local[i, j] / sqrt_d)
        T.reduce_sum(acc_A_local, mask_reduce_buffer, dim=1)
        for i, j in T.Parallel(BT, BT):
            acc_A_local[i, j] /= T.max(T.abs(mask_reduce_buffer[i]), 1)
        T.copy(acc_A_local, acc_A_cast)

        # Compute outputs:
        T.clear(acc_o_local2)
        T.gemm(Q_shared, acc_s_shared, acc_o_local2, policy=T.GemmWarpPolicy.FullCol)

        T.clear(acc_o_local)
        T.gemm(acc_A_cast, V_shared, acc_o_local, policy=T.GemmWarpPolicy.FullCol)
        for i, j in T.Parallel(BT, BV):
            acc_o_local[i, j] += acc_o_local2[i, j] * T.pow(decay, i + 1)
        T.copy(acc_o_local, acc_o_shared)
        T.copy(acc_o_shared, Output[i_bk, i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV])

    @T.macro
    def update_recurrent_state(
            K_shared: T.Buffer([BT, BK], dtype),
            K_local: T.Buffer([BT, BK], dtype),
            K_local_trans: T.Buffer([BT, BK], accum_dtype),
            K_local_trans_cast: T.Buffer([BT, BK], dtype),
            V_shared: T.Buffer([BT, BV], dtype),
            acc_s_local: T.Buffer([BK, BV], accum_dtype),
            acc_s_local2: T.Buffer([BK, BV], accum_dtype),
            acc_s_shared: T.Buffer([BK, BV], dtype),
            decay: T.float32,
    ):
        # Update state:
        # transpose k first because T.gemm does not have a good support for transposing the first operand according to the authors
        T.copy(K_shared, K_local)
        for i, j in T.Parallel(BK, BT):
            # Also apply decay terms:
            K_local_trans[i, j] = K_local[j, i]
            K_local_trans[i, j] *= T.pow(decay, BT - j - 1)
        T.copy(K_local_trans, K_local_trans_cast)
        T.clear(acc_s_local2)
        T.gemm(K_local_trans_cast, V_shared, acc_s_local2, policy=T.GemmWarpPolicy.FullCol)

        cross_decay = T.pow(decay, BT)
        for i, j in T.Parallel(BK, BV):
            acc_s_local2[i, j] += acc_s_local[i, j] * cross_decay
        T.copy(acc_s_local2, acc_s_local)
        T.copy(acc_s_local, acc_s_shared)

    @T.prim_func
    def main(
            Q: T.Buffer(qk_shape, dtype),
            K: T.Buffer(qk_shape, dtype),
            V: T.Buffer(v_shape, dtype),
            state: T.Buffer(state_shape, dtype),
            head_decays: T.Buffer([heads], accum_dtype),
            Output: T.Buffer(o_shape, dtype),
            out_state: T.Buffer(out_state_shape, dtype),
    ):
        with T.Kernel(heads, batch, T.ceildiv(dim_v, BV) * NK, threads=128) as (i_head, i_batch, bz):
            i_bk = bz % NK
            i_bv = bz // NK
            Q_shared = T.alloc_shared([BT, BK], dtype)
            K_shared = T.alloc_shared([BT, BK], dtype)
            K_local = T.alloc_fragment([BT, BK], dtype)
            K_local_trans = T.alloc_fragment([BK, BT], accum_dtype)
            K_local_trans_cast = T.alloc_fragment([BK, BT], dtype)
            V_shared = T.alloc_shared([BT, BV], dtype)

            acc_o_local = T.alloc_fragment((BT, BV), accum_dtype)
            acc_o_local2 = T.alloc_fragment((BT, BV), accum_dtype)
            acc_o_shared = T.alloc_shared([BT, BV], dtype)

            # hidden state in register tiles, must be in fp32?
            acc_s_local = T.alloc_fragment((BK, BV), accum_dtype)
            acc_s_local2 = T.alloc_fragment((BK, BV), accum_dtype)
            acc_A_local = T.alloc_fragment((BT, BT), accum_dtype)
            acc_A_cast = T.alloc_fragment((BT, BT), dtype)
            mask_local = T.alloc_fragment((BT, BT), accum_dtype)
            mask_reduce_buffer = T.alloc_fragment((BT,), accum_dtype)

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
            T.clear(acc_s_local2)

            # init decay values:
            T.copy(head_decays[:heads], decays_shared)
            decay = decays_shared[i_head]


            loop_range = T.ceildiv(seq_len, BT)
            for k in T.Pipelined(loop_range, num_stages=2):
                T.copy(K[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK], K_shared)
                T.copy(Q[i_batch, k * BT:(k + 1) * BT, i_head, i_bk * BK:(i_bk + 1) * BK], Q_shared)
                T.copy(V[i_batch, k * BT:(k + 1) * BT, i_head, i_bv * BV:(i_bv + 1) * BV], V_shared)

                compute_retention_chunk_outputs(
                    acc_A_local,
                    acc_A_cast,
                    mask_local,
                    mask_reduce_buffer,
                    acc_s_shared,
                    acc_o_shared,
                    acc_o_local,
                    acc_o_local2,
                    Q_shared,
                    K_shared,
                    V_shared,
                    Output,
                    decay,
                    i_bk, i_batch, i_head, i_bv, k
                )

                update_recurrent_state(
                    K_shared,
                    K_local,
                    K_local_trans,
                    K_local_trans_cast,
                    V_shared,
                    acc_s_local,
                    acc_s_local2,
                    acc_s_shared,
                    decay
                )
            T.copy(acc_s_shared, out_state[i_bk, i_batch, i_head, i_bk * BK:(i_bk + 1) * BK, i_bv * BV:(i_bv + 1) * BV])

    return main


def ref_program_(Q, K, V, prev_state, head_decays):
    qk = torch.einsum('bqhd,bkhd->bhqk', Q, K).tril() / torch.sqrt(Q.new_tensor(Q.size(1)))

    from einops import rearrange, einsum
    device, dtype = Q.device, Q.dtype
    query_pos = torch.arange(Q.size(1), device=device, dtype=dtype).unsqueeze_(-1)
    key_pos = torch.arange(K.size(1), device=device, dtype=dtype).unsqueeze_(0)
    distance = torch.abs(query_pos - key_pos)

    distance = rearrange(distance, "n s -> () n s")
    decay_gammas = rearrange(head_decays, "h -> h () ()")
    decay_mask = decay_gammas ** distance
    decay_mask = decay_mask / decay_mask.sum(dim=-1, keepdim=True).sqrt()

    qkm = (qk * decay_mask.tril().unsqueeze(0)).tril()
    r = qkm.sum(dim=-1, keepdim=True).abs()
    r = torch.where(r >= 1, r, torch.ones_like(r))
    qkm = qkm / r

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

    return o.to(dtype=io_dtype), state.to(dtype=io_dtype)


def ref_program(Q, K, V, prev_state, head_decays):
    seq_len = Q.size(1)
    res = []
    chunk_size = 64
    state_t = prev_state
    for i in range(ceil(seq_len / chunk_size)):
        start, end = i * chunk_size, (i + 1) * chunk_size
        res_t, state_t = ref_program_(
            Q[:, start:end],
            K[:, start:end],
            V[:, start:end],
            state_t, head_decays)
        res.append(res_t)

    return torch.cat(res, dim=1), state_t


def get_decays(num_heads: int, decay_range = None, device='cuda') -> torch.Tensor:
    if decay_range is None:
        decay_exp = -5 -torch.tensor(range(num_heads), dtype=torch.float, device=device)
    else:
        decay_exp = -torch.linspace(decay_range[0], decay_range[1], num_heads, dtype=torch.float, device=device)
    return 1 - torch.tensor(2., dtype=torch.float, device=device).pow(decay_exp)


def get_abs_err(x, y):
    return (x - y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--h', type=int, default=4, help='Number of heads')
    parser.add_argument('--n_ctx', type=int, default=2048, help='Context size')
    parser.add_argument('--dim_qk', type=int, default=64, help='Head dimension')
    parser.add_argument('--dim_v', type=int, default=64, help='Head dimension')
    args = parser.parse_args()
    BATCH, H, N_CTX, dim_qk, dim_v = args.batch, args.h, args.n_ctx, args.dim_qk, args.dim_v
    total_flops = 2.0 * BATCH * H * N_CTX * N_CTX * (dim_qk + dim_v)
    BLOCK_K = 64
    BLOCK_V = 64
    BLOCK_T = 64

    # io_dtype = torch.float32
    io_dtype = torch.bfloat16

    # TODO: auto padding
    assert dim_qk % BLOCK_K == 0
    assert dim_v % BLOCK_V == 0

    program = fused_retention_fwd(BATCH, H, N_CTX, dim_qk, dim_v, BLOCK_K, BLOCK_V, BLOCK_T)
    mod, params = tilelang.lower(program)

    mod = tilelang.Profiler(mod, params, [5, 6], tilelang.TensorSupplyType.Normal)

    ins = []
    head_decays = get_decays(num_heads=H, decay_range=(5, 12))
    for i in range(len(mod.params)):
        if i not in mod.result_idx:
            shape = [int(x) for x in mod.params[i].shape]
            print(f"Shape of {i}: {shape}")
            if len(shape) == 1:
                ins.append(head_decays)
            else:
                ins.append(torch.randn(shape, device="cuda", dtype=io_dtype))

    qk_shape = (BATCH, N_CTX, H, dim_qk)
    v_shape = (BATCH, N_CTX, H, dim_v)
    magnitude_qk = dim_qk ** -0.5
    ins = [
        torch.randn(qk_shape, device="cuda", dtype=io_dtype),
        torch.randn(qk_shape, device="cuda", dtype=io_dtype),
        torch.randn(v_shape, device="cuda", dtype=io_dtype),
        torch.randn((BATCH, H, dim_qk, dim_v), device="cuda", dtype=io_dtype),
        head_decays
    ]
    ins32 = [v.clone().float() for v in ins]

    ref_outs, ref_state = ref_program(*ins)
    torch.cuda.synchronize()
    ref32_outs, ref32_state = ref_program(*ins32)
    torch.cuda.synchronize()
    lib_outs, lib_state = mod.func(*ins)
    torch.cuda.synchronize()
    lib_outs = lib_outs.float().sum(0).to(dtype=io_dtype)
    lib_state = lib_state.float().sum(0).to(dtype=io_dtype)

    print(f"Ref32 state: {ref32_state.flatten()[:10]}")
    print(f"Ref state: {ref_state.flatten()[:10]}")
    print(f"Tile state: {lib_state.flatten()[:10]}")

    if isinstance(lib_outs, torch.Tensor):
        lib_outs = [lib_outs]
    if isinstance(ref_outs, torch.Tensor):
        ref_outs = [ref_outs]
    assert len(lib_outs) == len(ref_outs)

    for lhs, rhs in zip(lib_outs, ref_outs):
        print(f"LHS: {lhs.shape}, RHS: {rhs.shape}")
        print(f"Relative error: ", get_err_ratio(lhs, rhs))
        print("If it is < 0.005, it is okayish")
        print(f"Abs error: ", get_abs_err(lhs, rhs))
        print(f"Abs error ref32-ref: ", get_abs_err(ref32_outs, rhs.to(dtype=torch.float32)))
        print(f"Abs error ref32-tile: ", get_abs_err(ref32_outs, lhs.to(dtype=torch.float32)))
        print(f"Ref32: {ref32_outs.flatten()[:10]}")
        print(f"Tile: {lhs.flatten()[:10]}")
        print(f"Ref: {rhs.flatten()[:10]}")

    print("Caveat: TFLOPs might be misleading here, but the larger the faster..")

    latency = mod.do_bench(ref_program, n_warmup=10, n_repeat=10, profiler="torch")
    print("Ref: {:.2f} ms".format(latency))
    print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="torch")
    print("tilelang: {:.2f} ms".format(latency))
    print("tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    chunk_head_first = partial(chunk_linear_attn, head_first=False)
    latency = mod.do_bench(lambda x1,x2,x3,x4,x5: chunk_head_first(q=x1,k=x2,v=x3), n_warmup=10, n_repeat=10, profiler="torch")
    print("FLA: {:.2f} ms".format(latency))
    print("FLA: {:.2f} TFlops".format(total_flops / latency * 1e-9))
