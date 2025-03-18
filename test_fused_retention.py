import argparse
from functools import partial

from dataclasses import dataclass
from math import ceil

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from einops import rearrange

# from fla.ops.linear_attn import fused_chunk_linear_attn as chunk_linear_attn
# from fla.ops.linear_attn import chunk_linear_attn

from fused_retention import fused_chunk_retention, ref_program, _get_decay_mask



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
    ln = torch.nn.LayerNorm(cfg.dim_v, device="cuda", dtype=cfg.dtype) if apply_layer_norm else lambda x: x
    head_decays = tuple(get_decays(num_heads=cfg.num_heads, decay_range=cfg.decay_range).cpu().numpy().tolist())
    # head_decays = torch.zeros_like(head_decays)
    ins = [
        ln(torch.randn(qk_shape, device="cuda", dtype=cfg.dtype)),
        ln(torch.randn(qk_shape, device="cuda", dtype=cfg.dtype)),
        ln(torch.randn(v_shape, device="cuda", dtype=cfg.dtype)),
        torch.zeros((cfg.batch_size, cfg.num_heads, cfg.dim_qk, cfg.dim_v), device="cuda", dtype=cfg.dtype),
        head_decays,
    ]
    return ins


def compute_forward(inputs: list):
    ins32 = [v.clone().float() if isinstance(v, torch.Tensor) else v for v in inputs]

    # Compute reference outputs:
    ref_outs, ref_state = ref_program(*inputs)
    torch.cuda.synchronize()
    ref32_outs, ref32_state = ref_program(*ins32)
    torch.cuda.synchronize()

    # Compute Tilelang outputs:
    dtype = ins32[0].dtype
    lib_outs, lib_state = fused_chunk_retention(*inputs)
    torch.cuda.synchronize()
    lib_outs = lib_outs.float().sum(0).to(dtype=dtype)
    # gn = torch.nn.LayerNorm(normalized_shape=dim_v, device="cuda", dtype=io_dtype)
    # lib_outs = gn(lib_outs)
    lib_state = lib_state.float().sum(0).to(dtype=dtype)

    return lib_outs, lib_state, ref_outs, ref_state, ref32_outs, ref32_state


def benchmark_run_times(cfg: Config):
    total_flops = 2.0 * cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.seq_len * (cfg.dim_qk + cfg.dim_v)
    print("Caveat: TFLOPs might be misleading here, but the larger the faster..")

    inputs = generate_inputs(cfg, apply_layer_norm=False)

    latency = do_bench(partial(ref_program, *inputs))
    print("Ref: {:.2f} ms".format(latency))
    print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = do_bench(partial(fused_chunk_retention, *inputs))
    print("tilelang: {:.2f} ms".format(latency))
    print("tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))

    # chunk_head_first = partial(chunk_linear_attn, head_first=False)
    # latency = mod.do_bench(lambda x1, x2, x3, x4, x5, x6: chunk_head_first(q=x1, k=x2, v=x3), n_warmup=10, n_repeat=10,
    #                        profiler="torch")
    # print("FLA: {:.2f} ms".format(latency))
    # print("FLA: {:.2f} TFlops".format(total_flops / latency * 1e-9))


def evaluate_states(kernel_state, ref_state, ref32_state):
    print(f"Ref32 state: {ref32_state.flatten()[:10]}")
    print(f"Ref state: {ref_state.flatten()[:10]}")
    print(f"Tile state: {kernel_state.flatten()[:10]}")


def evaluate_outputs(cfg, kernel_outs, ref_outs, ref32_outs):
    assert kernel_outs.shape == (cfg.batch_size, cfg.seq_len, cfg.num_heads, cfg.dim_v)
    assert kernel_outs.shape == ref_outs.shape
    print(f"Relative error: ", get_err_ratio(kernel_outs, ref32_outs))
    print("If it is < 0.005, it is okayish")
    print(f"Max/Avg Abs error: {get_max_abs_err(kernel_outs, ref_outs)}/{get_mean_abs_err(kernel_outs, ref_outs)}")
    print(f"Abs error ref32-ref: {get_max_abs_err(ref32_outs, ref_outs.to(dtype=torch.float32))}/{get_mean_abs_err(ref32_outs, ref_outs.to(dtype=torch.float32))}")
    print(f"Abs error ref32-tile: {get_max_abs_err(ref32_outs, kernel_outs.clone().to(dtype=torch.float32))}/{get_mean_abs_err(ref32_outs, kernel_outs.clone().to(dtype=torch.float32))}")
    argmax = torch.argmax(kernel_outs)
    i4 = argmax % cfg.dim_v
    i3 = (argmax // cfg.dim_v) % cfg.num_heads
    i2 = (argmax // (cfg.dim_v * cfg.num_heads)) % cfg.seq_len
    i1 = (argmax // (cfg.dim_v * cfg.num_heads * cfg.seq_len)) % cfg.batch_size
    print(f"Tile argmax: ({i1},{i2},{i3},{i4}), value: {kernel_outs.clone().flatten()[argmax]}")
    print(f"Ref32: {ref32_outs.flatten()[:10]}")
    print(f"Ref: {ref_outs.flatten()[:10]}")
    print(f"Tile: {kernel_outs.flatten()[:10]}")


def run_from_cfg(cfg: Config, inputs):

    kernel_outs, kernel_state, ref_outs, ref_state, ref32_outs, ref32_state = compute_forward(inputs)

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


def test_multi_chunk_small_batch_forward():
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
    run_from_cfg(cfg, inputs)


def test_multi_chunk_large_batch_forward():
    cfg = Config(
        batch_size=128,
        num_heads=4,
        seq_len=2048,
        dim_qk=64,
        dim_v=64,
        block_K=64,
        block_V=64,
        block_T=64,
        decay_range=(5, 12),
    )

    inputs = generate_inputs(cfg, False)
    run_from_cfg(cfg, inputs)



if __name__ == "__main__":
    # test_single_chunk_forward()
    test_multi_chunk_small_batch_forward()
    # test_multi_chunk_large_batch_forward()


