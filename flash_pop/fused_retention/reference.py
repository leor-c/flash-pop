from math import ceil

import torch
from torch import Tensor
from einops import rearrange, einsum

from flash_pop.fused_retention import _get_decay_mask


def ref_program_(Q, K, V, prev_state, head_decays):
    qk = torch.einsum('bqhd,bkhd->bhqk', Q, K).tril()

    device, dtype = Q.device, Q.dtype
    seq_len = Q.size(1)
    decay_mask, head_decays_ = _get_decay_mask(head_decays, seq_len, device, torch.float32, return_head_decays=True)
    # decay_mask = decay_mask / decay_mask.sum(dim=-1, keepdim=True).sqrt()

    qkm = (qk * decay_mask.unsqueeze(0).to(dtype=dtype)).tril()
    # r = qkm.sum(dim=-1, keepdim=True).abs()
    # r = torch.where(r >= 1, r, torch.ones_like(r))
    # qkm = qkm / r

    # qkm = qk * mask
    # r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
    o = torch.einsum('bhqk,bkhd->bqhd', qkm.to(dtype=dtype), V)

    # cross-chunk (derived from recurrent retention)
    decay_gammas = rearrange(head_decays_, "h -> () h () ()")
    device = K.device
    dtype = K.dtype
    inner_pos = rearrange(
        torch.arange(K.size(1), device=device, dtype=dtype) + 1,
        "n -> () () n ()",
    )
    state_decays = decay_gammas ** (K.size(1) - inner_pos)
    discounted_key = einsum(K, state_decays.to(dtype=dtype), "b n h d, _ h n _ -> b h n d")
    state = einsum(discounted_key, V, "b h n d1, b n h d2 -> b h d1 d2")

    if prev_state is not None:
        # update recurrent state using prev_state:
        chunk_decay = decay_gammas ** K.size(1)
        state = state + prev_state * chunk_decay

        # Update the retention Tensor, based on cross-chunk information
        inner_decay = rearrange(decay_gammas ** inner_pos, "b h n d -> b n h d")
        o = o + (
                einsum(Q.to(dtype=dtype), prev_state.to(dtype=dtype), "b n h d1, b h d1 d2 -> b n h d2") * inner_decay.to(dtype=dtype)
        )

    return o.to(dtype=dtype), state.to(dtype=dtype)


def ref_program(Q, K, V, prev_state, head_decays, chunk_size: int = 512, *args):
    seq_len = Q.size(1)
    res = []
    state_t = prev_state
    K = K / (K.size(3) ** 0.5)
    # gn = torch.nn.LayerNorm(normalized_shape=V.size(3), device=Q.device, dtype=Q.dtype)
    for i in range(ceil(seq_len / chunk_size)):
        start, end = i * chunk_size, (i + 1) * chunk_size
        res_t, state_t = ref_program_(
            Q[:, start:end],
            K[:, start:end],
            V[:, start:end],
            state_t, head_decays)
        # res_t = gn(res_t)
        res.append(res_t)

    return torch.cat(res, dim=1), state_t


def reference_grads(Q: Tensor, K: Tensor, V: Tensor, prev_state: Tensor, head_decays, dO: Tensor, dS_new: Tensor, *args):
    d_qk = K.size(3)
    sqrt_d_qk = (d_qk ** 0.5)
    assert d_qk == Q.size(3)
    device, dtype = Q.device, Q.dtype
    seq_len = Q.size(1)

    D, head_decays_ = _get_decay_mask(head_decays, seq_len, device, torch.float32, return_head_decays=True)
    decay_gammas = rearrange(head_decays_, "h -> () h () ()")
    inner_pos = rearrange(torch.arange(K.size(1), device=device, dtype=dtype) + 1, "n -> () () n ()")
    inner_decay = rearrange(decay_gammas ** inner_pos, "b h t d -> b t h d")
    state_decays = decay_gammas ** (K.size(1) - inner_pos)
    chunk_decay = decay_gammas ** K.size(1)

    dO_VT_D = einsum(dO, V, "b t1 h dv, b t2 h dv -> b h t1 t2") / sqrt_d_qk
    dO_VT_D = einsum(dO_VT_D, D, "b h t1 t2, h t1 t2 -> b h t1 t2")
    dQ1 = einsum(dO_VT_D.to(dtype=dtype), K, "b h t1 t2, b t2 h dk -> b t1 h dk")
    dO_decay = dO * inner_decay
    dQ2 = einsum(dO_decay.to(dtype=dtype), prev_state, "b t h dv, b h dk dv -> b t h dk")
    dQ = dQ1 + dQ2
    # dQ = dQ2

    # Compute dK:
    dK = (
        einsum(dO_VT_D.to(dtype=dtype), Q, "b h t1 t2, b t1 h dk -> b t2 h dk") +
        einsum(V, dS_new, state_decays, "b t h dv, b h dk dv, b h t dk -> b t h dk") / sqrt_d_qk
    )

    A = einsum(Q, K, D, "b t1 h dk, b t2 h dk, h t1 t2 -> b h t1 t2") / sqrt_d_qk
    dV = (
        einsum(A.to(dtype=dtype), dO, "b h t1 t2, b t1 h dv -> b t2 h dv") +
        einsum(K / sqrt_d_qk, state_decays.to(dtype=dtype), dS_new, "b t h dk, b h t dk, b h dk dv -> b t h dv")
    )

    dS = (
        einsum(Q, dO_decay.to(dtype=dtype), "b t h dk, b t h dv -> b h dk dv") +
        chunk_decay * dS_new
    )

    return dQ, dK, dV, dS
