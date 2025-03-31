import time

from dataclasses import dataclass
import torch
from tilelang.profiler import do_bench

from retnet import MultiScaleRetention, RetNetDecoderLayer, RetNetDecoder


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
        num_heads=cfg.num_heads,
        head_dim_v=cfg.head_dim_v,
        head_dim_qk=cfg.head_dim_qk,
        device=device
    ))

    d_model = cfg.head_dim_v * cfg.num_heads
    x = torch.randn(cfg.batch_size, cfg.seq_len, d_model, device=device, dtype=torch.bfloat16)

    def run():
        msr.forward_chunkwise(x, 0, prev_state=None)

    latency = do_bench(run, warmup=100)
    print(f"MultiScaleRetention latency (ms): {latency}")

    # warmup:
    # msr.forward_chunkwise(x, 0, prev_state=None)

    # t0 = time.time()
    # with torch.no_grad():
    #     for _ in range(2000):
    #         # t1 = time.time()
    #         msr.forward_chunkwise(x, 0, prev_state=None)
    #         # t2 = time.time()
    #         # print(f"forward: {t2 - t1}")
    #     t = time.time() - t0
    #     print(f"MultiScaleRetention total time (s): {t}")


def run_test_layer(cfg: TestConfig):
    device = torch.device("cuda")
    layer = RetNetDecoderLayer(RetNetDecoderLayer.Config(
        num_heads=cfg.num_heads,
        head_dim_v=cfg.head_dim_v,
        head_dim_qk=cfg.head_dim_qk,
        dim_feedforward=cfg.head_dim_v*2,
    ))

    d_model = cfg.head_dim_v * cfg.num_heads
    x = torch.randn(cfg.batch_size, cfg.seq_len, d_model, device=device, dtype=torch.bfloat16)

    def run():
        layer.forward_chunkwise(x, 0, prev_state=None)

    latency = do_bench(run, warmup=100)
    print(f"RetNetDecoderLayer latency (ms): {latency}")

    # warmup:
    # layer.forward_chunkwise(x, 0, prev_state=None)
    #
    # t0 = time.time()
    # with torch.no_grad():
    #     for _ in range(2000):
    #         # t1 = time.time()
    #         layer.forward_chunkwise(x, 0, prev_state=None)
    #         # t2 = time.time()
    #         # print(f"forward: {t2 - t1}")
    #     t = time.time() - t0
    #     print(f"RetNetDecoderLayer total time (s): {t}")


def test_retnet(cfg: TestConfig):
    device = torch.device("cuda")
    retnet_cfg = RetNetDecoderLayer.Config(
        num_heads=cfg.num_heads,
        head_dim_v=cfg.head_dim_v,
        head_dim_qk=cfg.head_dim_qk,
        dim_feedforward=cfg.head_dim_v * 2,
    )
    num_layers = 20
    # layers = [RetNetDecoderLayer(retnet_cfg) for _ in range(num_layers)]
    retnet = RetNetDecoder(retnet_cfg, num_layers)

    d_model = cfg.head_dim_v * cfg.num_heads
    x = torch.randn(cfg.batch_size, cfg.seq_len, d_model, device=device, dtype=torch.bfloat16)

    def run():
        retnet(x)
        # y = x
        # for i in range(num_layers):
        #     y = layers[i](y)[0]
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

