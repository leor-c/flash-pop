import time

from dataclasses import dataclass
import torch

from retention_layer import MultiScaleRetention, RetNetDecoderLayer


@dataclass()
class TestConfig:
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim_qk: int
    head_dim_v: int
    decay_range: tuple[float, float] = None
    dtype: torch.dtype = torch.bfloat16


def run_test(cfg: TestConfig):
    device = torch.device("cuda")
    msr = MultiScaleRetention(MultiScaleRetention.Config(
        cfg.num_heads,
        cfg.head_dim_v,
        cfg.head_dim_qk,
        device=device
    ))

    d_model = cfg.head_dim_v * cfg.num_heads
    x = torch.randn(cfg.batch_size, cfg.seq_len, d_model, device=device, dtype=torch.bfloat16)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(2000):
            # t1 = time.time()
            msr.forward_chunkwise(x, 0, prev_state=None)
            # t2 = time.time()
            # print(f"forward: {t2 - t1}")
        t = time.time() - t0
        print(f"MultiScaleRetention total time (s): {t}")


def run_test_layer(cfg: TestConfig):
    device = torch.device("cuda")
    layer = RetNetDecoderLayer(RetNetDecoderLayer.Config(
        cfg.num_heads,
        cfg.head_dim_v,
        cfg.head_dim_qk,
        dim_feedforward=cfg.head_dim_v*2,
    ))

    d_model = cfg.head_dim_v * cfg.num_heads
    x = torch.randn(cfg.batch_size, cfg.seq_len, d_model, device=device, dtype=torch.bfloat16)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(2000):
            # t1 = time.time()
            layer.forward_chunkwise(x, 0, prev_state=None)
            # t2 = time.time()
            # print(f"forward: {t2 - t1}")
        t = time.time() - t0
        print(f"RetNetDecoderLayer total time (s): {t}")


def sanity_check():
    cfg = TestConfig(
        batch_size=64,
        num_heads=4,
        seq_len=2**13,
        head_dim_qk=64,
        head_dim_v=64,
    )
    run_test(cfg)
    run_test_layer(cfg)


if __name__ == '__main__':
    sanity_check()

