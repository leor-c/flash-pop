from math import ceil
from functools import partial

import torch
from tilelang.profiler import do_bench
from loguru import logger

from fused_retention import ref_program_
from pop_retention import fused_pop_retention
from test_fused_retention import Config as RetNetConfig, generate_inputs, get_err_ratio


def ref_pop(Q, K, V, prev_state, head_decays, block_size: int = 512, *args):
    seq_len = Q.size(1)
    res = []
    states = [prev_state]
    state_t = prev_state
    K = K / (K.size(3) ** 0.5)
    # gn = torch.nn.LayerNorm(normalized_shape=V.size(3), device=Q.device, dtype=Q.dtype)
    for i in range(ceil(seq_len / block_size)):
        start, end = i * block_size, (i + 1) * block_size
        res_t, state_t = ref_program_(
            Q[:, start:end],
            K[:, start:end],
            V[:, start:end],
            state_t, head_decays)
        # res_t = gn(res_t)
        res.append(res_t)
        if end <= seq_len:
            states.append(state_t)

    return torch.cat(res, dim=1), torch.stack(states, dim=1)


def test_correctness():
    cfg = RetNetConfig(
        batch_size=128,
        num_heads=4,
        seq_len=287,
        dim_qk=64,
        dim_v=128,
        block_K=64,
        block_V=64,
        block_T=64,
        decay_range=(5, 12),
    )
    block_size = 144

    Q, K, V, S, head_decays = generate_inputs(cfg, False)

    O_ref, states_ref = ref_pop(Q, K, V, S, head_decays, block_size=block_size)
    O, states = fused_pop_retention(Q, K, V, S, head_decays, block_size)

    o_diff = (O_ref - O).abs()
    err_ratio = get_err_ratio(O, O_ref)
    print(f"Err ratio: {err_ratio}, Mean out diff: {o_diff.mean()}, max out diff: {o_diff.max()}")

    print(f"states shape: {states.shape}, states_ref shape: {states_ref.shape}")
    assert states.shape == states_ref.shape, f"got {states.shape} != {states_ref.shape}"
    s_diff = (states_ref - states).abs()
    err_ratio = get_err_ratio(states, states_ref)
    print(f"Err ratio: {err_ratio}, Mean states diff: {s_diff.mean()}, max states diff: {s_diff.max()}")

    print(f"{states[0, 1, 0]}")
    print(f"{states_ref[0, 1, 0]}")

    from fused_retention import fused_chunk_retention
    latency = do_bench(partial(fused_pop_retention, Q, K, V, S, head_decays, block_size))
    logger.info("Ref: {:.2f} ms".format(latency))
    latency = do_bench(partial(fused_chunk_retention, Q, K, V, S, head_decays))
    logger.info("tilelang: {:.2f} ms".format(latency))


if __name__ == '__main__':
    test_correctness()
