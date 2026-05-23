"""
MoE torch golden: expert 计算部分（不含 router softmax/topk）。

`torch_experts` 原样来自 vLLM（未改实现）：
  https://github.com/vllm-project/vllm/blob/main/tests/kernels/utils.py
  本地路径: /opt/vllm/tests/kernels/utils.py  (约 L826–L964)

运行需能 import vllm（PYTHONPATH 含 vllm 仓库根目录）。
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/opt/vllm")

import torch

from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.model_executor.custom_op import op_registry
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

# --- default problem size (test_moe layout) ---
NUM_TOKENS = 1024
HIDDEN_SIZE = 4096
INTER_SIZE_TP = 128
NUM_EXPERTS = 512
TOPK = 10
DTYPE = torch.bfloat16


# SPDX-Snippet-Start
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copied verbatim from vllm/tests/kernels/utils.py


def torch_experts(
    a: torch.Tensor, # 这个是hidden states
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    b_bias1: torch.Tensor | None = None,
    b_bias2: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    quant_dtype: torch.dtype | None = None,
    per_act_token_quant=False,
    block_shape: list[int] | None = None,
    apply_router_weights_on_input: bool = False,
    activation: MoEActivation = MoEActivation.SILU,
) -> torch.Tensor:
    assert (
        global_num_experts == -1
        or (global_num_experts == w1.shape[0] and expert_map is None)
        or (expert_map is not None and global_num_experts == expert_map.shape[0])
    )

    if quant_dtype in [torch.float16, torch.bfloat16]:
        quant_dtype = None
    quant_input_only = quant_dtype is not None and w1_scale is None and w2_scale is None
    if quant_input_only:
        assert a1_scale is None and a2_scale is None
        assert per_act_token_quant

    M, K = a.shape
    topk = topk_ids.shape[1]

    if apply_router_weights_on_input:
        assert topk == 1
        a = a * topk_weight.to(a.dtype)

    a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)

    out = torch.zeros(M * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    if a1_scale:
        assert not per_act_token_quant and block_shape is None
    a, a_scale = moe_kernel_quantize_input(
        a, a1_scale, quant_dtype, per_act_token_quant, block_shape
    )

    if quant_input_only:
        a = (a.float() * a_scale.view(-1, 1)).to(w1.dtype)

    num_experts = w1.shape[0]

    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]

    f32 = torch.float32

    act = op_registry[activation.custom_op_name]

    for i in range(num_experts):
        mask = topk_ids == i
        if mask.sum():
            if quant_dtype is None:
                tmp1 = a[mask] @ w1[i].transpose(0, 1)
                if b_bias1 is not None:
                    tmp1 = tmp1 + b_bias1[i].view(1, -1).to(tmp1.dtype)
                tmp2 = act()(tmp1)
                out[mask] = tmp2 @ w2[i].transpose(0, 1)
                if b_bias2 is not None:
                    out[mask] = out[mask] + b_bias2[i].view(1, -1).to(tmp1.dtype)
            elif quant_input_only:
                tmp1 = a[mask] @ w1[i].transpose(0, 1)
                tmp2 = SiluAndMul()(tmp1)
                tmp2, tmp2_scale = moe_kernel_quantize_input(
                    tmp2, None, quant_dtype, per_act_token_quant
                )
                tmp2 = (tmp2.float() * tmp2_scale.view(-1, 1)).to(w2.dtype)
                out[mask] = tmp2 @ w2[i].transpose(0, 1)
            elif block_shape is not None:
                # block quantized
                assert (
                    a_scale is not None
                    and w1_scale is not None
                    and w2_scale is not None
                )
                tmp1 = native_w8a8_block_matmul(
                    a[mask], w1[i], a_scale[mask], w1_scale[i], block_shape, out.dtype
                )
                if b_bias1 is not None:
                    tmp1 = tmp1 + b_bias1[i].view(1, -1).to(tmp1.dtype)
                tmp2 = SiluAndMul()(tmp1)
                tmp2, b_scale = moe_kernel_quantize_input(
                    tmp2, a2_scale, quant_dtype, per_act_token_quant, block_shape
                )

                out[mask] = native_w8a8_block_matmul(
                    tmp2, w2[i], b_scale, w2_scale[i], block_shape, out.dtype
                )
                if b_bias2 is not None:
                    out[mask] = out[mask] + b_bias2[i].view(1, -1).to(tmp1.dtype)
            else:
                assert (
                    a_scale is not None
                    and w1_scale is not None
                    and w2_scale is not None
                )
                scales = a_scale if a_scale.numel() == 1 else a_scale[mask]

                tmp1 = a[mask].to(f32) * scales
                w1_dq = (w1[i].to(f32) * w1_scale[i]).transpose(0, 1)
                tmp1 = (tmp1 @ w1_dq).to(out.dtype)
                if b_bias1 is not None:
                    tmp1 = tmp1 + b_bias1[i].view(1, -1).to(out.dtype)

                tmp2 = SiluAndMul()(tmp1).to(out.dtype)

                tmp2, b_scale = moe_kernel_quantize_input(
                    tmp2, a2_scale, quant_dtype, per_act_token_quant, block_shape
                )
                assert b_scale is not None

                tmp2 = tmp2.to(f32) * b_scale
                w2_dq = (w2[i].to(f32) * w2_scale[i]).transpose(0, 1)
                out[mask] = (tmp2 @ w2_dq).to(out.dtype)
                if b_bias2 is not None:
                    out[mask] = out[mask] + b_bias2[i].view(1, -1).to(out.dtype)

    if apply_router_weights_on_input:
        return out
    else:
        return (
            (out.view(M, -1, w2.shape[1]).to(f32) * topk_weight.view(M, -1, 1))
            .sum(dim=1)
            .to(out.dtype)
        )


# SPDX-Snippet-End


def make_moe_tensors(
    m: int = NUM_TOKENS,
    k: int = HIDDEN_SIZE,
    n: int = INTER_SIZE_TP,
    e: int = NUM_EXPERTS,
    topk: int = TOPK,
    dtype: torch.dtype = DTYPE,
    device: str = "cuda",
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造 expert 输入；topk 由调用方给定，不经 router。"""
    torch.manual_seed(seed)
    hidden = torch.randn(m, k, device=device, dtype=dtype) / 10
    w1 = torch.randn(e, 2 * n, k, device=device, dtype=dtype) / 10
    w2 = torch.randn(e, k, n, device=device, dtype=dtype) / 10
    topk_ids = torch.randint(0, e, (m, topk), device=device, dtype=torch.int64)
    topk_weight = torch.rand(m, topk, device=device, dtype=torch.float32)
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    topk_weight = topk_weight.to(dtype)
    return hidden, w1, w2, topk_weight, topk_ids


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config

    hidden, w1, w2, topk_weight, topk_ids = make_moe_tensors()
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = {}
    with set_current_vllm_config(vllm_config):
        out = torch_experts(hidden, w1, w2, topk_weight, topk_ids)
    print(
        f"M={NUM_TOKENS} K={HIDDEN_SIZE} N={INTER_SIZE_TP} E={NUM_EXPERTS} topk={TOPK}"
    )
    print(f"hidden {tuple(hidden.shape)} w1 {tuple(w1.shape)} w2 {tuple(w2.shape)}")
    print(f"topk_weight {tuple(topk_weight.shape)} topk_ids {tuple(topk_ids.shape)}")
    print(f"out {tuple(out.shape)}")
