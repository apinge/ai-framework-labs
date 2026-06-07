import functools
import torch



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