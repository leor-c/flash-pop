from dataclasses import dataclass
import torch
from tilelang.profiler import do_bench


@dataclass()
class TestConfig:
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim_qk: int
    head_dim_v: int
    decay_range: tuple[float, float] = None
    dtype: torch.dtype = torch.bfloat16


def test_pop_retnet(cfg: TestConfig):
    device = torch.device("cuda")
    from flash_pop.pop_retnet import POPRetNetDecoder, POPDecoderLayer
    retnet_cfg = POPDecoderLayer.Config(
        block_size=32,
        num_heads=cfg.num_heads,
        head_dim_v=cfg.head_dim_v,
        head_dim_qk=cfg.head_dim_qk,
        dim_feedforward=cfg.head_dim_v * 2,
        device='cuda'
    )
    num_layers = 20
    # layers = [RetNetDecoderLayer(retnet_cfg) for _ in range(num_layers)]
    retnet = POPRetNetDecoder(retnet_cfg, num_layers)

    d_model = cfg.head_dim_v * cfg.num_heads
    x = torch.randn(cfg.batch_size, cfg.seq_len, d_model, device=device, dtype=torch.bfloat16)
    num_blocks = cfg.seq_len // retnet_cfg.block_size
    suffixes = torch.randn(cfg.batch_size, num_blocks+1, retnet_cfg.block_size-1, d_model, device=device, dtype=torch.bfloat16)

    def run():
        retnet.pop_forward(x, suffixes=suffixes)
        # y = x
        # for i in range(num_layers):
        #     y = layers[i](y)[0]
    with torch.no_grad():
        latency = do_bench(run, warmup=100, rep=100)
    print(f"POP RetNet latency (ms): {latency}")

def sanity_check():
    cfg = TestConfig(
        batch_size=8,
        num_heads=4,
        seq_len=2**14,
        head_dim_qk=64,
        head_dim_v=128,
    )
    test_pop_retnet(cfg)


if __name__ == '__main__':
    sanity_check()
