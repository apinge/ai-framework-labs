import functools
import torch

import flydsl.expr as fx
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import const_expr, range_constexpr
from flydsl._mlir.dialects import llvm as _llvm


# Simple dtypes namespace used by pertoken_quant
class dtypes:
    fp32 = torch.float32
    fp16 = torch.float16
    bf16 = torch.bfloat16
    i8 = torch.int8
    i32 = torch.int32


@functools.lru_cache()
def get_dtype_max(dtype):
    """Get max value for a given dtype."""
    try:
        return torch.finfo(dtype).max
    except Exception:
        return torch.iinfo(dtype).max


def pertoken_quant(
    x,
    scale=None,
    x_scale=None,  # smooth_scale
    scale_dtype=dtypes.fp32,
    quant_dtype=dtypes.i8,
    dtypeMax=None,
):
    """FlyDSL/tests/utils.py"""
    x = x.to(dtypes.fp32)
    if x_scale is None:
        hidden_states = x
    else:
        # smooth quant
        hidden_states = x * x_scale

    if dtypeMax is None:
        dtypeMax = get_dtype_max(quant_dtype)

    # Be robust to rare non-finite values (can appear from FP8 pipelines at extreme shapes):
    # - Avoid producing inf scales (which would later lead to 0*inf -> NaN in dequant).
    # - Avoid propagating NaN/Inf into the quantized tensor.
    hidden_states = torch.nan_to_num(
        hidden_states,
        nan=0.0,
        posinf=float(dtypeMax),
        neginf=-float(dtypeMax),
    )

    per_token_scale = scale
    if scale is None:
        # [m, 1]
        # Avoid materializing a full-size abs() temporary (can be huge for MoE weights).
        # max(abs(x)) = max(max(x), -min(x))
        # 每行单独算一个 scale
        per_token_max = torch.amax(hidden_states, dim=-1, keepdim=True)
        per_token_min = torch.amin(hidden_states, dim=-1, keepdim=True)
        per_token_amax = torch.maximum(per_token_max, -per_token_min)
        per_token_scale = per_token_amax / dtypeMax
        # 如果某一行全是 0，就强行改成 1。
        per_token_scale[per_token_scale == 0] = 1

    per_token_scale = torch.nan_to_num(per_token_scale, nan=1.0, posinf=1.0, neginf=1.0)

    # quant hidden_states
    y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
    y_scale = per_token_scale.to(scale_dtype)
    return y, y_scale


def _run_torch(a, b, scale_a, scale_b, dtype=torch.float32):
    a_f32 = a.to(torch.float32) * scale_a.view(-1, 1)
    b_f32 = b.to(torch.float32) * scale_b.view(-1, 1)
    return torch.mm(a_f32, b_f32.T).to(dtype)


def preshuffle_b(b_t):
    """Permute row-major ``B_T`` ``(N, K)`` for ``b_preshuffled=True``."""
    """in FlyDSL/kernels/fp8_gemm_utils.py
    - 原始 `[N,K]`
    - reshape后:  `[N/16, 16_rows, K/64, 4_groups, 16_cols]`
    - permute后:  `[N/16, K/64,  4_groups, 16_rows, 16_cols]`  ← 行和列的位置互换
    """
    n, k = b_t.shape[-2:]
    assert n % 16 == 0 and k % 64 == 0, f"need N%16==0 and K%64==0, got N={n} K={k}"
    return b_t.reshape(n // 16, 16, k // 64, 4, 16).permute(0, 2, 3, 1, 4).contiguous()


def _as_i8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def divmod(a: int, b: int) -> tuple[int, int]:
    return (a // b, a % b)


# 一个类型系统上的 reinterpret_cast，从 MLIR IR 层面把 i8* 换成 f8E4M3FN*，底层 GPU 内存地址和数据完全不变。
def make_fp8_buffer_tensor(arg_i8, fp8_ir_t):
    # max_size=False with no num_records_bytes: cosize(layout) becomes a
    # runtime expression because TensorAdaptor defaults to layout-dynamic
    # memref (post #554), so the descriptor adapts to the actual tensor
    # extent and no longer bakes the first-call's shape into IR.

    # 步骤1：把 Python 侧的 int8 tensor 包装成硬件 buffer descriptor
    #  max_size=False → buffer 大小用运行时实际 extent，不硬编码
    t_i8 = fx.rocdl.make_buffer_tensor(arg_i8, max_size=False)

    # 步骤2：从 buffer descriptor 拿到迭代器（带类型信息的指针抽象）
    iter_i8 = fx.get_iter(t_i8)  # 类型是 int8 buffer ptr

    # 步骤2：构造一个"fp8 类型的 buffer 指针类型"
    # 地址空间保持 BufferDesc（VGPR buffer 寄存器）
    # alignment 直接从 int8 指针继承
    f8_buf_ptr_ty = fx.PointerType.get(
        elem_ty=fp8_ir_t,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=fx.PointerType(iter_i8.type).alignment,
    )

    # 步骤4：把 int8 迭代器"reinterpret cast"成 fp8 迭代器（零开销，只换类型标签）
    iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)

    # 步骤5：用原 layout（stride/shape）重新包装成 fp8 Tensor 返回
    return fx.Tensor(fx.make_view(iter_f8, fx.get_layout(t_i8)))


def swizzle_128(row, col):
    offset = row * 128 + col
    swizzle = ((offset % (16 * 128)) >> 8) << 4
    swizzled_offset = offset ^ swizzle
    return swizzled_offset // 128, swizzled_offset % 128


def compute_global_swizzle(lane_id, wave_id, K, n_rounds, preshuffled):
    offsets = []
    n_waves = fx.block_dim.x // 64
    for round in range_constexpr(n_rounds):
        if const_expr(preshuffled):
            row = lane_id % 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id // 8) * 16
            offsets.append(
                (row // 16) * (K * 16)
                + (row % 16) * 16
                + (col // 64) * 1024
                + ((col % 64) // 16) * 256
                + (col % 16)
            )
        else:
            row = lane_id // 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id % 8) * 16
            r, c = swizzle_128(row, col)
            offsets.append(r * K + c)
    return offsets


def pack_i32x4_i32x8(lo, hi):
    # Pack two i32x4 as one i32x8
    return lo.shuffle(hi, list(range(8)))


def wait_barrier(count):
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
        constraints="",
        has_side_effects=True,
    )
