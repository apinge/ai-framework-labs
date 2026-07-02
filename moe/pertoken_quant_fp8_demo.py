#!/usr/bin/env python3
"""
Minimal demo: bf16 -> fp8 via aiter.pertoken_quant.

Covers:
  1. Plain per-row quant (each row one scale)
  2. 128x128 block weight quant (same as test_moe_2stage.py -q 5)

Run (ROCm GPU + aiter):
  cd /opt/ai-framework-labs/moe && python pertoken_quant_fp8_demo.py
"""

from __future__ import annotations

import sys

import torch

import aiter
from aiter import dtypes


def is_arch_type(arch):
    props = torch.cuda.get_device_properties()
    return arch in props.gcnArchName


def get_fp8type():
    return torch.float8_e4m3fn if is_arch_type("950") else torch.float8_e4m3fnuz


def dequant_pertoken(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of ``aiter.pertoken_quant`` / FlyDSL ``pertoken_quant``.

    Quant:  ``y = x_fp32 / scale``  (aiter/aiter/ops/quant.py L69–70)
    Dequant: ``x_fp32 ≈ y_fp32 * scale``  (scale ``[..., 1]``, broadcast 到末维)

    出处（同一公式，无单独 library API）:
      - aiter ``op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py``
        ``dequantize_fp8`` (L248–261): ``x_quantized.float() * x_scales``
      - FlyDSL ``tests/kernels/test_ref.py`` ``_dequant`` scalar 分支 (L87–88):
        ``x_f32 * scale``
      - aiter ``aiter/fused_moe.py`` torch ref (L1547–1564): inline ``* a1_scale`` / ``* w1_scale``
    """
    # Broadcasting (demo 1 断点典型 shape):
    #   x_q   [4, 128]  float8  → .to(f32) → [4, 128]
    #   scale [4,   1]  f32     → .to(f32) → [4,   1]
    # 按 PyTorch 广播规则逐维对齐（从最后一维往前）:
    #   dim -1: 128 vs 1  → 1 扩成 128，同一行的 128 个元素共用 scale[i, 0]
    #   dim -2:   4 vs 4  → 相等，逐行配对
    # 等价于: scale.expand(4, 128) 或 scale.repeat(1, 128)，但广播不拷贝内存
    return x_q.to(torch.float32) * scale.to(torch.float32)


def weight_per_128x128_quant(weight: torch.Tensor, quant_dtype):
    """Block-quantize expert weight [E, dim1, dim2] with 128x128 tiles.

    Same layout as aiter/op_tests/test_moe_2stage.py (per_128x128).
    """
    E, dim1, dim2 = weight.shape
    assert dim1 % 128 == 0 and dim2 % 128 == 0

    weight_blocks = weight.view(E, dim1 // 128, 128, dim2 // 128, 128)
    weight_blocks = weight_blocks.permute(0, 1, 3, 2, 4).contiguous()
    weight_blocks = weight_blocks.view(E, -1, 128 * 128)

    weight_qt, weight_scale = aiter.pertoken_quant(
        weight_blocks, quant_dtype=quant_dtype
    )

    weight_qt = weight_qt.view(E, dim1 // 128, dim2 // 128, 128, 128)
    weight_qt = weight_qt.permute(0, 1, 3, 2, 4).contiguous().view(E, dim1, dim2)
    weight_scale = weight_scale.view(E, dim1 // 128, dim2 // 128)
    return weight_qt, weight_scale


def expand_blockscale(weight_q: torch.Tensor, scale: torch.Tensor, blk_n: int, blk_k: int):
    """Expand [E, nblk_n, nblk_k] block scales to element-wise [E, N, K] for dequant check.

    Adapted from FlyDSL ``_expand_blockscale`` in
    ``/opt/FlyDSL/tests/kernels/test_moe_blockscale.py`` (L65–73).
    Same view/repeat/permute pattern as aiter ``torch_moe_blockscale`` in
    ``/opt/aiter/op_tests/test_moe_blockscale.py`` (L70–83, einops rearrange).
    """
    E, nblk_n, nblk_k = scale.shape
    return (
        scale.view(E, -1, 1)
        .repeat(1, 1, blk_n * blk_k)
        .view(E, nblk_n, nblk_k, blk_n, blk_k)
        .permute(0, 1, 3, 2, 4)
        .reshape(E, nblk_n * blk_n, nblk_k * blk_k)
    )


def demo_per_row_quant(device: torch.device, fp8_dtype: torch.dtype):
    print("=" * 60)
    print("1) Per-row pertoken_quant: bf16 [M, K] -> fp8")
    print("=" * 60)

    M, K = 4, 128
    torch.manual_seed(0)
    x_bf16 = (torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.5)
    
    x_fp8, scale = aiter.pertoken_quant(x_bf16, quant_dtype=fp8_dtype)
    x_ref = dequant_pertoken(x_fp8, scale)

    err = (x_ref - x_bf16.float()).abs().max().item()
    print(f"  input : {x_bf16.shape} bf16")
    print(f"  output: {x_fp8.shape} {x_fp8.dtype}, scale {scale.shape} {scale.dtype}")
    print(f"  max |dequant - bf16|: {err:.4e}")
    print(f"  scale[0,0] = {scale[0, 0].item():.6f}")
    assert err < 0.5, f"per-row dequant error too large: {err}"
    print("  OK\n")


def demo_block_weight_quant(device: torch.device, fp8_dtype: torch.dtype):
    print("=" * 60)
    print("2) 128x128 block weight quant (test_moe_2stage -q 5 pattern)")
    print("=" * 60)

    E, dim1, dim2 = 2, 256, 4096  # one expert W1-like: [2N, K]
    torch.manual_seed(1)
    w_bf16 = (torch.randn(E, dim1, dim2, device=device, dtype=torch.bfloat16) * 0.02)

    w_fp8, w_scale = weight_per_128x128_quant(w_bf16, quant_dtype=fp8_dtype)
    # dequant ref: FlyDSL test_moe_blockscale._expand_blockscale + w * scale_expanded
    scale_expanded = expand_blockscale(w_fp8, w_scale, blk_n=128, blk_k=128)
    w_ref = w_fp8.float() * scale_expanded

    err = (w_ref - w_bf16.float()).abs().max().item()
    nblk_n, nblk_k = dim1 // 128, dim2 // 128
    print(f"  weight bf16 : {w_bf16.shape}")
    print(f"  weight fp8  : {w_fp8.shape} {w_fp8.dtype}")
    print(f"  block scale : {w_scale.shape}  ({nblk_n} x {nblk_k} blocks per expert)")
    print(f"  max |dequant - bf16|: {err:.4e}")
    assert err < 0.5, f"block weight dequant error too large: {err}"
    print("  OK\n")


def main():
    if not torch.cuda.is_available():
        print("CUDA/ROCm not available, skip.")
        sys.exit(0)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    fp8_dtype = get_fp8type()
    aiter_fp8 = dtypes.fp8

    print(f"GPU arch : {props.gcnArchName}")
    print(f"fp8 type : {fp8_dtype} (get_fp8type)")
    print(f"aiter fp8: {aiter_fp8} (dtypes.fp8)")
    if fp8_dtype != aiter_fp8:
        print("  note: get_fp8type() and aiter.dtypes.fp8 differ; demo uses aiter.dtypes.fp8 for quant")
    print()

    demo_per_row_quant(device, aiter_fp8)
    demo_block_weight_quant(device, aiter_fp8)
    print("All demos passed.")


if __name__ == "__main__":
    main()
