import argparse
from functools import partial

from dataclasses import dataclass
from math import ceil

import torch
from tilelang.profiler import do_bench
from einops import rearrange
from loguru import logger
import sys

detail_level = "detail"
logger.level(detail_level, no=15, color="<yellow>")
logger.remove()
logger.add(sys.stderr, level="INFO")

# from fla.ops.linear_attn import fused_chunk_linear_attn as chunk_linear_attn
# from fla.ops.linear_attn import chunk_linear_attn

from fused_retention import fused_chunk_retention, ref_program



def get_decays(num_heads: int, decay_range = None, device='cuda') -> torch.Tensor:
    if decay_range is None:
        decay_exp = -5 -torch.tensor(range(num_heads), dtype=torch.float, device=device)
    else:
        decay_exp = -torch.linspace(decay_range[0], decay_range[1], num_heads, dtype=torch.float, device=device)
    return 1 - torch.tensor(2., dtype=torch.float, device=device).pow(decay_exp)


def get_max_abs_err(x, y):
    return (x - y).flatten().abs().max().item()


def get_mean_abs_err(x, y):
    return (x - y).flatten().abs().mean().item()


def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


@dataclass
class Config:
    batch_size: int
    num_heads: int
    seq_len: int
    dim_qk: int
    dim_v: int
    block_K: int
    block_V: int
    block_T: int
    decay_range: tuple[float, float]
    dtype: torch.dtype = torch.bfloat16


def generate_inputs(cfg: Config, apply_layer_norm: bool):
    qk_shape = (cfg.batch_size, cfg.seq_len, cfg.num_heads, cfg.dim_qk)
    v_shape = (cfg.batch_size, cfg.seq_len, cfg.num_heads, cfg.dim_v)
    ln_qk = torch.nn.LayerNorm(cfg.dim_qk, device="cuda", dtype=cfg.dtype) if apply_layer_norm else lambda x: x
    ln_v = torch.nn.LayerNorm(cfg.dim_v, device="cuda", dtype=cfg.dtype) if apply_layer_norm else lambda x: x
    head_decays = tuple(get_decays(num_heads=cfg.num_heads, decay_range=cfg.decay_range).cpu().numpy().tolist())
    # head_decays = torch.zeros_like(head_decays)
    ins = [
        ln_qk(torch.randn(qk_shape, device="cuda", dtype=cfg.dtype)),
        ln_qk(torch.randn(qk_shape, device="cuda", dtype=cfg.dtype)),
        ln_v(torch.randn(v_shape, device="cuda", dtype=cfg.dtype)),
        torch.zeros((cfg.batch_size, cfg.num_heads, cfg.dim_qk, cfg.dim_v), device="cuda", dtype=cfg.dtype),
        head_decays,
    ]
    return ins


def compute_forward(inputs: list):
    ins32 = [v.clone().float() if isinstance(v, torch.Tensor) else v for v in inputs]

    # Compute reference outputs:
    try:
        ref_outs, ref_state = ref_program(*inputs)
        torch.cuda.synchronize()
        ref32_outs, ref32_state = ref_program(*ins32)
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        logger.error("Reference (Pytorch) program out of memory.")
        return None

    # Compute Tilelang outputs:
    dtype = ins32[0].dtype
    lib_outs, lib_state = fused_chunk_retention(*inputs)
    torch.cuda.synchronize()
    # gn = torch.nn.LayerNorm(normalized_shape=dim_v, device="cuda", dtype=io_dtype)
    # lib_outs = gn(lib_outs)

    return lib_outs, lib_state, ref_outs, ref_state, ref32_outs, ref32_state


def benchmark_run_times(cfg: Config):
    # total_flops = 2.0 * cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.seq_len * (cfg.dim_qk + cfg.dim_v)
    # print("Caveat: TFLOPs might be misleading here, but the larger the faster..")

    inputs = generate_inputs(cfg, apply_layer_norm=False)

    latency = do_bench(partial(ref_program, *inputs))
    logger.info("Ref: {:.2f} ms".format(latency))
    # logger.info("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = do_bench(partial(fused_chunk_retention, *inputs))
    logger.info("tilelang: {:.2f} ms".format(latency))
    # logger.info("tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))


def evaluate_states(kernel_state, ref_state, ref32_state):
    logger.log(detail_level, f"Ref32 state: {ref32_state.flatten()[:10]}")
    logger.log(detail_level, f"Ref state: {ref_state.flatten()[:10]}")
    logger.log(detail_level, f"Tile state: {kernel_state.flatten()[:10]}")
    relative_error = get_err_ratio(kernel_state, ref32_state)
    logger.log(detail_level, f"State relative error: {relative_error}")
    assert relative_error < 0.005, f"Got {relative_error}"


def evaluate_outputs(cfg, kernel_outs, ref_outs, ref32_outs, verbosity: int = 1):
    assert kernel_outs.shape == (cfg.batch_size, cfg.seq_len, cfg.num_heads, cfg.dim_v)
    assert kernel_outs.shape == ref_outs.shape
    relative_error = get_err_ratio(kernel_outs, ref32_outs)
    logger.log(detail_level, f"Output relative error: {relative_error}")
    logger.log(detail_level, "If it is < 0.005, it is okayish")
    logger.log(detail_level, f"Max/Avg Abs error: {get_max_abs_err(kernel_outs, ref_outs)}/{get_mean_abs_err(kernel_outs, ref_outs)}")
    logger.log(detail_level, f"Abs error ref32-ref: {get_max_abs_err(ref32_outs, ref_outs.to(dtype=torch.float32))}/{get_mean_abs_err(ref32_outs, ref_outs.to(dtype=torch.float32))}")
    logger.log(detail_level, f"Abs error ref32-tile: {get_max_abs_err(ref32_outs, kernel_outs.clone().to(dtype=torch.float32))}/{get_mean_abs_err(ref32_outs, kernel_outs.clone().to(dtype=torch.float32))}")
    assert relative_error < 0.005, f"Got {relative_error}"

    argmax = torch.argmax(kernel_outs)
    i4 = argmax % cfg.dim_v
    i3 = (argmax // cfg.dim_v) % cfg.num_heads
    i2 = (argmax // (cfg.dim_v * cfg.num_heads)) % cfg.seq_len
    i1 = (argmax // (cfg.dim_v * cfg.num_heads * cfg.seq_len)) % cfg.batch_size
    logger.log(detail_level, f"Tile argmax: ({i1},{i2},{i3},{i4}), value: {kernel_outs.clone().flatten()[argmax]}")
    logger.log(detail_level, f"Ref32: {ref32_outs.flatten()[:10]}")
    logger.log(detail_level, f"Ref: {ref_outs.flatten()[:10]}")
    logger.log(detail_level, f"Tile: {kernel_outs.flatten()[:10]}")


def run_from_cfg(cfg: Config, inputs):
    results = compute_forward(inputs)
    if results is not None:
        kernel_outs, kernel_state, ref_outs, ref_state, ref32_outs, ref32_state = results

        evaluate_states(kernel_state, ref_state, ref32_state)

        evaluate_outputs(cfg, kernel_outs, ref_outs, ref32_outs)

    benchmark_run_times(cfg)


def test_single_chunk_forward():
    cfg = Config(
        batch_size=128,
        num_heads=4,
        seq_len=64,
        dim_qk=64,
        dim_v=64,
        block_K=64,
        block_V=64,
        block_T=64,
        decay_range=(5, 12),
    )

    inputs = generate_inputs(cfg, False)
    run_from_cfg(cfg, inputs)


def test_multi_chunk_forward():
    cfg = Config(
        batch_size=8,
        num_heads=4,
        seq_len=2048,
        dim_qk=64,
        dim_v=128,
        block_K=64,
        block_V=64,
        block_T=64,
        decay_range=(5, 12),
    )
    for dim_v in [64, 128]:
        logger.info(f"Testing 'dim_v'={dim_v}...")
        cfg.dim_v = dim_v
        inputs = generate_inputs(cfg, True)
        run_from_cfg(cfg, inputs)
    cfg.dim_v = 64

    for seq_len in [256, 400, 2048]:
        logger.info(f"Testing 'seq_len'={seq_len}...")
        cfg.seq_len = seq_len
        inputs = generate_inputs(cfg, True)
        run_from_cfg(cfg, inputs)

    logger.info(f"Testing different 'batch_size' values...")
    for batch_size in [1, 3, 8, 64, 128, 256]:
        logger.info(f"Testing 'batch_size'={batch_size}...")
        cfg.batch_size = batch_size
        inputs = generate_inputs(cfg, True)
        run_from_cfg(cfg, inputs)


def test_ref_chunkwise_correctness():
    cfg = Config(
        batch_size=8,
        num_heads=4,
        seq_len=256,
        dim_qk=64,
        dim_v=128,
        block_K=64,
        block_V=64,
        block_T=64,
        decay_range=(5, 12),
        dtype=torch.float32,
    )
    Q, K, V, S, head_decays = generate_inputs(cfg, False)
    ref_O, ref_S_new = ref_program(Q, K, V, S, head_decays)
    O, S_new = ref_program(Q, K, V, S, head_decays, chunk_size=128)

    diff = torch.abs(O - ref_O)
    logger.log(detail_level, f"max diff {diff.max()}, avg: {diff.mean()}, num places > 0.1: {(diff > 0.1).sum()}")
    assert torch.allclose(S_new, ref_S_new, rtol=1e-2, atol=1e-2)
    assert torch.allclose(O, ref_O, rtol=1e-1, atol=1e-1)


def test_reference_grads():
    cfg = Config(
        batch_size=8,
        num_heads=4,
        seq_len=256*2,
        dim_qk=64,
        dim_v=128,
        block_K=64,
        block_V=64,
        block_T=64,
        decay_range=(5, 12),
        dtype=torch.float32,
    )

    from fused_retention import reference_grads
    Q, K, V, S, head_decays = generate_inputs(cfg, False)
    Q, K, V, S = Q.requires_grad_(), K.requires_grad_(), V.requires_grad_(), S.requires_grad_()
    dO = torch.randn_like(V)
    dS_new = torch.randn_like(S)

    O, S_new = ref_program(Q, K, V, S, head_decays)
    O.backward(dO, retain_graph=True)
    S_new.backward(dS_new, retain_graph=True)

    dQ_ref, Q.grad = Q.grad.clone(), None
    dK_ref, K.grad = K.grad.clone(), None
    dV_ref, V.grad = V.grad.clone(), None
    dS_ref, S.grad = S.grad.clone(), None

    dQ, dK, dV, dS = reference_grads(Q, K, V, S, head_decays, dO, dS_new)

    logger.log(detail_level, f"dQ_ref: {dQ_ref.flatten()[:10]}")
    logger.log(detail_level, f"dQ: {dQ.flatten()[:10]}")

    diff = torch.abs(dQ_ref - dQ)
    logger.log(detail_level, f"max diff: {diff.max()}, avg diff: {diff.mean()}, places with diff > 0.1: {(diff > 0.1).sum()}")

    assert torch.allclose(dQ, dQ_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dK, dK_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dV, dV_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dS, dS_ref, rtol=1e-2, atol=1e-2)


def test_backward_pass():
    cfg = Config(
        batch_size=8,
        num_heads=4,
        seq_len=256 * 2,
        dim_qk=64,
        dim_v=128,
        block_K=64,
        block_V=64,
        block_T=64,
        decay_range=(5, 12),
    )

    Q, K, V, S, head_decays = generate_inputs(cfg, False)
    Q, K, V, S = Q.requires_grad_(), K.requires_grad_(), V.requires_grad_(), S.requires_grad_()
    dO = torch.randn_like(V)
    dS_new = torch.randn_like(S)

    O, S_new = ref_program(Q, K, V, S, head_decays)
    O.backward(dO, retain_graph=True)
    S_new.backward(dS_new, retain_graph=True)

    dQ_ref, Q.grad = Q.grad.clone(), None
    dK_ref, K.grad = K.grad.clone(), None
    dV_ref, V.grad = V.grad.clone(), None
    dS_ref, S.grad = S.grad.clone(), None


    O, S_new = fused_chunk_retention(Q, K, V, S, head_decays)
    O.backward(dO, retain_graph=True)
    S_new.backward(dS_new, retain_graph=True)

    dQ, Q.grad = Q.grad.clone(), None
    dK, K.grad = K.grad.clone(), None
    dV, V.grad = V.grad.clone(), None
    dS, S.grad = S.grad.clone(), None

    logger.log(detail_level, f"dQ_ref: {dQ_ref.flatten()[:10]}")
    logger.log(detail_level, f"dQ: {dQ.flatten()[:10]}")

    diff = torch.abs(dQ_ref - dQ)
    logger.log(detail_level,
               f"max diff: {diff.max()}, avg diff: {diff.mean()}, places with diff > 0.1: {(diff > 0.1).sum()}")

    assert torch.allclose(dQ, dQ_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dK, dK_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dV, dV_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dS, dS_ref, rtol=1e-2, atol=1e-2)


def test_utilization():
    cfg = Config(
        batch_size=8,
        num_heads=4,
        seq_len=2048,
        dim_qk=64,
        dim_v=128,
        block_K=64,
        block_V=64,
        block_T=64,
        decay_range=(5, 12),
    )
    inputs = generate_inputs(cfg, False)
    logger.info("Testing utilization...")
    for _ in range(10*3000):
        fused_chunk_retention(*inputs)


if __name__ == "__main__":
    # test_single_chunk_forward()
    # test_multi_chunk_forward()
    test_ref_chunkwise_correctness()
    test_reference_grads()
    test_backward_pass()
    # test_utilization()


