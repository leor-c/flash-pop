import time

from dataclasses import dataclass
import torch
from tilelang.profiler import do_bench
from functools import partial

from flash_pop import DiffusionRetNetDecoder, DiffusionRetNetDecoderLayer


@dataclass()
class TestConfig:
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim_qk: int
    head_dim_v: int
    decay_range: tuple[float, float] = None
    dtype: torch.dtype = torch.bfloat16



def run_test_layer(cfg: TestConfig):
    device = torch.device("cuda")
    layer = DiffusionRetNetDecoderLayer(DiffusionRetNetDecoderLayer.Config(
        num_heads=cfg.num_heads,
        head_dim_v=cfg.head_dim_v,
        head_dim_qk=cfg.head_dim_qk,
        dim_feedforward=cfg.head_dim_v*2,
    ))

    d_model = cfg.head_dim_v * cfg.num_heads
    x = torch.randn(cfg.batch_size, cfg.seq_len, d_model, device=device, dtype=torch.bfloat16)
    t = torch.randn(cfg.batch_size, cfg.seq_len // 4, device=device, dtype=torch.bfloat16)

    def run():
        layer.forward_chunkwise(x, t, 0, prev_state=None)

    latency = do_bench(run, warmup=100)
    print(f"RetNetDecoderLayer latency (ms): {latency}")


def test_retnet(cfg: TestConfig):
    device = torch.device("cuda")
    retnet_cfg = DiffusionRetNetDecoderLayer.Config(
        num_heads=cfg.num_heads,
        head_dim_v=cfg.head_dim_v,
        head_dim_qk=cfg.head_dim_qk,
        dim_feedforward=cfg.head_dim_v * 2,
        ada_ln_lora_dim=64
    )
    num_layers = 20
    retnet = DiffusionRetNetDecoder(DiffusionRetNetDecoder.Config(
        layer_config=retnet_cfg, 
        num_layers=num_layers, 
        out_dim=64,
    ))

    d_model = cfg.head_dim_v * cfg.num_heads
    x = torch.randn(cfg.batch_size, cfg.seq_len, d_model, device=device, dtype=torch.bfloat16)
    t = torch.randn(cfg.batch_size, cfg.seq_len // 4, device=device, dtype=torch.bfloat16)

    run = partial(retnet, x=x, diffusion_times=t)

    # test case: t per frame
    with torch.no_grad():
        latency = do_bench(run, warmup=100, rep=100)
    print(f"RetNet latency (ms): {latency}")

    # test case: t shape = x shape
    t = torch.randn(cfg.batch_size, cfg.seq_len, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        latency = do_bench(run, warmup=100, rep=100)
    print(f"RetNet latency (ms): {latency}")

    # test case: single t per batch sample
    t = torch.randn(cfg.batch_size, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        latency = do_bench(run, warmup=100, rep=100)
    print(f"RetNet latency (ms): {latency}")


def test_t():
    device = 'cuda'
    decoder_layer = torch.nn.TransformerDecoderLayer(d_model=512, nhead=4, device=device)
    transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=5)
    memory = torch.rand(10, 2**11, 512, device=device)
    tgt = torch.rand(20, 2**11, 512, device=device)

    with torch.no_grad():
        t0 = time.time()
        for _ in range(100):
            out = transformer_decoder(tgt, memory)
        t1 = time.time()
        print(f"TransformerDecoder total time (s): {t1-t0}")



def sanity_check():
    cfg = TestConfig(
        batch_size=8,
        num_heads=4,
        seq_len=2**14,
        head_dim_qk=64,
        head_dim_v=128,
    )
    # run_test(cfg)
    # run_test_layer(cfg)
    test_retnet(cfg)
    # test_t()


if __name__ == '__main__':
    sanity_check()

