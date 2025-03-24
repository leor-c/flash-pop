from math import ceil
from functools import partial

import torch
from tilelang.profiler import do_bench
from loguru import logger

from fused_retention.reference import ref_program_
from pop_retention import flash_pop_retention
from test_fused_retention import Config as RetNetConfig, generate_inputs, get_err_ratio, log_error_info, detail_level


def ref_pop(Q, K, V, prev_state, head_decays, block_size: int = 512, *args):
    seq_len = Q.size(1)
    res = []
    states = []
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


def run_fwd_test(cfg, block_size):
    Q, K, V, S, head_decays = generate_inputs(cfg, False)

    O_ref, states_ref = ref_pop(Q, K, V, S, head_decays, block_size=block_size)
    O, states = flash_pop_retention(Q, K, V, S, head_decays, block_size)

    o_diff = (O_ref - O).abs()
    err_ratio = get_err_ratio(O, O_ref)
    print(f"Err ratio: {err_ratio}, Mean out diff: {o_diff.mean()}, max out diff: {o_diff.max()}")

    print(f"states shape: {states.shape}, states_ref shape: {states_ref.shape}")
    assert states.shape == states_ref.shape, f"got {states.shape} != {states_ref.shape}"
    s_diff = (states_ref - states).abs()
    err_ratio = get_err_ratio(states, states_ref)
    print(f"Err ratio: {err_ratio}, Mean states diff: {s_diff.mean()}, max states diff: {s_diff.max()}")

    print(f"{states[0, 0, 0]}")
    print(f"{states_ref[0, 0, 0]}")

    from fused_retention import fused_chunk_retention
    latency = do_bench(partial(flash_pop_retention, Q, K, V, S, head_decays, block_size))
    logger.info("POP: {:.2f} ms".format(latency))
    latency = do_bench(partial(fused_chunk_retention, Q, K, V, S, head_decays))
    logger.info("RetNet: {:.2f} ms".format(latency))
    latency = do_bench(partial(ref_pop, Q, K, V, S, head_decays, block_size))
    logger.info("Pytorch Naive POP: {:.2f} ms".format(latency))


def run_bwd_test(cfg, block_size):
    Q, K, V, S, head_decays = generate_inputs(cfg, False)
    Q, K, V, S = Q.requires_grad_(), K.requires_grad_(), V.requires_grad_(), S.requires_grad_()

    Q32, K32, V32, S32 = Q.float(), K.float(),V.float(),S.float()
    O_ref, states_ref = ref_pop(Q32, K32, V32, S32, head_decays, block_size=block_size)
    O, states = flash_pop_retention(Q, K, V, S, head_decays, block_size)

    d_out = torch.randn_like(O)
    d_states = torch.randn_like(states)

    ((O * d_out.clone()).sum() + (states * d_states.clone()).sum()).backward(retain_graph=True)
    dQ, Q.grad = Q.grad.clone(), None
    dK, K.grad = K.grad.clone(), None
    dV, V.grad = V.grad.clone(), None
    dS, S.grad = S.grad.clone(), None

    ((O_ref * d_out.clone().float()).sum() + (states_ref * d_states.clone().float()).sum()).backward(retain_graph=True)
    dQ_ref, Q.grad = Q.grad.clone(), None
    dK_ref, K.grad = K.grad.clone(), None
    dV_ref, V.grad = V.grad.clone(), None
    dS_ref, S.grad = S.grad.clone(), None

    logger.log(detail_level, f"dQ_ref: {dQ_ref.flatten()[:10]}")
    logger.log(detail_level, f"dQ: {dQ.flatten()[:10]}")
    log_error_info(dQ, dQ_ref, 'dQ')

    logger.log(detail_level, f"dK_ref: {dK_ref.flatten()[:10]}")
    logger.log(detail_level, f"dK: {dK.flatten()[:10]}")
    log_error_info(dK, dK_ref, 'dK')

    logger.log(detail_level, f"dV_ref: {dV_ref.flatten()[:10]}")
    logger.log(detail_level, f"dV: {dV.flatten()[:10]}")
    log_error_info(dV, dV_ref, 'dV')

    logger.log(detail_level, f"dS_ref: {dS_ref.flatten()[:10]}")
    logger.log(detail_level, f"dS: {dS.flatten()[:10]}")
    log_error_info(dS, dS_ref, 'dS')


def test_generation_scenario():
    """
    Test the scenario where the input is composed of 1 full (obs-action) block, followed by a suffix of pred tokens.
    We want to output only the state at the end of the real block, and exclude the suffix from the state.
    Returns:

    """
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

    run_fwd_test(cfg, block_size)

    cfg = RetNetConfig(
        batch_size=128,
        num_heads=4,
        seq_len=28,
        dim_qk=64,
        dim_v=128,
        block_K=64,
        block_V=64,
        block_T=64,
        decay_range=(5, 12),
    )
    block_size = 15

    run_fwd_test(cfg, block_size)


def test_training_scenario():
    """
    Test the scenario where the input is composed of 1 full (obs-action) block, followed by a suffix of pred tokens.
    We want to output only the state at the end of the real block, and exclude the suffix from the state.
    Returns:

    """
    block_size = 144
    cfg = RetNetConfig(
        batch_size=8,
        num_heads=4,
        seq_len=block_size*20,
        dim_qk=64,
        dim_v=128,
        block_K=64,
        block_V=64,
        block_T=64,
        decay_range=(5, 12),
    )
    run_fwd_test(cfg, block_size)

    block_size = 32
    cfg = RetNetConfig(
        batch_size=8,
        num_heads=4,
        seq_len=block_size * 20,
        dim_qk=64,
        dim_v=128,
        block_K=64,
        block_V=64,
        block_T=64,
        decay_range=(5, 12),
    )
    run_fwd_test(cfg, block_size)

    block_size = 15
    cfg = RetNetConfig(
        batch_size=8,
        num_heads=4,
        seq_len=block_size*20,
        dim_qk=64,
        dim_v=128,
        block_K=64,
        block_V=64,
        block_T=64,
        decay_range=(5, 12),
    )
    run_fwd_test(cfg, block_size)


def test_bwd():
    block_size = 32
    cfg = RetNetConfig(
        batch_size=8,
        num_heads=4,
        seq_len=2**15,
        dim_qk=64,
        dim_v=64,
        block_K=64,
        block_V=64,
        block_T=64,
        decay_range=(5, 12),
    )
    run_bwd_test(cfg, block_size)


if __name__ == '__main__':
    # test_generation_scenario()
    # test_training_scenario()
    test_bwd()
