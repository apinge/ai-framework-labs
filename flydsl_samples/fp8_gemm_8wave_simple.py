# Code from
# FlyDSL/tests/kernels/test_fp8_gemm_rowscale.py
# FlyDSL/kernels/fp8_gemm_8wave.py
# device cdna4

import os
import sys

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch  # noqa: E402

from fp8_gemm_utils import pertoken_quant, _run_torch, preshuffle_b, _as_i8, ceildiv

# from kernels.fp8_gemm_utils import (
#     G2SLoader,
#     Mfma16x16x128,
#     S2RLoader,
#     StoreC,
#     ceildiv,
#     compute_global_swizzle,
#     divmod,
#     make_fp8_buffer_tensor,
#     wait_barrier,
# )



OUT_DTYPE = torch.bfloat16
ARCH = str(get_rocm_arch())

# begin kernel
def compile_fp8_gemm_8w(*, K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, b_preshuffled: bool = False):

    """
    这段在定义一些 kernel用到的宏 
    """
    BLOCK_K = 128  # [DDD] fixed K-tile = 128

    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K  # [DDD] 8192 // 128 = 64

    N_TILES_A = BLOCK_M // 64   # [DDD] 256 // 64 = 4  (A sub-tiles per wave-row)
    N_TILES_B = BLOCK_N // 128  # [DDD] 256 // 128 = 2  (B sub-tiles per wave-col)
    N_ACCUMS = N_TILES_A * N_TILES_B  # [DDD] 4 * 2 = 8 accumulator fragments per wave
    assert N_ACCUMS > 0

    LDS_BLOCK_M = BLOCK_M // 2  # [DDD] 256 // 2 = 128  (half of BLOCK_M, split into A0/A1)
    LDS_BLOCK_N = BLOCK_N // 2  # [DDD] 256 // 2 = 128  (half of BLOCK_N, split into B0/B1)

    N_LDS_STEPS_A = LDS_BLOCK_M // 64  # [DDD] 128 // 64 = 2  (G2S rounds to fill one A half-tile)
    N_LDS_STEPS_B = LDS_BLOCK_N // 64  # [DDD] 128 // 64 = 2  (G2S rounds to fill one B half-tile)
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)  # [DDD] max(2, 2) = 2

    # half size
    a_lds_size = LDS_BLOCK_M * BLOCK_K  # [DDD] 128 * 128 = 16384 fp8 elements per ping/pong A buffer
    b_lds_size = LDS_BLOCK_N * BLOCK_K  # [DDD] 128 * 128 = 16384 fp8 elements per ping/pong B buffer

    """
    16 是16 align
    凑dwordx4
    """
    @fx.struct
    class SharedStorage:
        # [DDD] 8 ping-pong buffers total: 4×A + 4×B, each 16384 fp8 bytes → 8 * 16384 = 131072 B = 128 KB LDS
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]   # [DDD] (16384,) cur  A0 half-tile [BLOCK_M/2 × BLOCK_K]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]   # [DDD] (16384,) cur  A1 half-tile
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]  # [DDD] (16384,) next A0 half-tile (prefetch)
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]  # [DDD] (16384,) next A1 half-tile (prefetch)
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]   # [DDD] (16384,) cur  B0 half-tile [BLOCK_N/2 × BLOCK_K]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]   # [DDD] (16384,) cur  B1 half-tile
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]  # [DDD] (16384,) next B0 half-tile (prefetch)
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]  # [DDD] (16384,) next B1 half-tile (prefetch)


    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        #F8_IR_t = fx.Float8E4M3FN.ir_type
        #n_blocks = ceildiv(c_n, BLOCK_N) # 8192//256 = 32


        pass
    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)  # [DDD] ceil(8192/256)*ceil(8192/256) = 32*32 = 1024 blocks
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm
# end kernel


# begin 测试代码
def _bench_fp8_gemm(
    M: int,
    N: int,
    K: int,
    *,
    use_8w: bool,
    tile_m: int,
    tile_n: int,
    # disable_xcd_remap: bool = False,
    num_warmups: int = 2,
    num_iters: int = 10,
    vs_torch: bool = False,
    b_preshuffled: bool = False,
    static_weight_scale: bool = True,
):
    device = torch.device("cuda")
    a_fp32 = torch.rand(
        M, K, device=device, dtype=torch.float32
    )  # [DDD] (8192, 8192) f32
    b_fp32_t = torch.rand(
        N, K, device=device, dtype=torch.float32
    )  # [DDD] (8192, 8192) f32  (B transposed: N rows, K cols)
    c_out_raw = torch.zeros(
        (M, N), dtype=OUT_DTYPE, device=device
    )  # [DDD] (8192, 8192) bf16

    a_q, scale_a = pertoken_quant(
        a_fp32, quant_dtype=torch.float8_e4m3fn
    )  # [DDD] a_q: (8192, 8192) fp8, scale_a: (8192, 1) → squeeze → (8192,)
    b_q, scale_b = pertoken_quant(
        b_fp32_t, quant_dtype=torch.float8_e4m3fn
    )  # [DDD] b_q: (8192, 8192) fp8, scale_b: (8192, 1) → squeeze → (8192,)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    scale_a = scale_a.squeeze().contiguous() # 压掉(8192,1) -> (8192,)
    scale_b = scale_b.squeeze().contiguous()
 
    c_ref = _run_torch(a_q, b_q, scale_a, scale_b) # torch reference

    # [DDD] b_preshuffled=True: (8192,8192) → reshape(512,16,128,4,16) → permute(0,2,3,1,4) → (512,128,4,16,16) contiguous
    b_kernel = preshuffle_b(b_q) if b_preshuffled else b_q

    launch_fn = compile_fp8_gemm_8w(
            K=K,                          # [DDD] K=8192
            BLOCK_M=tile_m,               # [DDD] BLOCK_M=256
            BLOCK_N=tile_n,               # [DDD] BLOCK_N=256
            b_preshuffled=b_preshuffled,  # [DDD] b_preshuffled=True
        )
    print(
        f"\n[fp8_gemm_8wave] M={M} N={N} K={K} BLOCK_M={tile_m} BLOCK_N={tile_n} "
        f"preshuffle_b={b_preshuffled} static_weight_scale={static_weight_scale}"
    )

    def _args(c, a, b, sa, sb):
        b_flat = _as_i8(b).contiguous().view(-1)  # [DDD] b: (512,128,4,16,16) fp8→int8→view(-1)=(8192*8192,)=(67108864,)
        sa_flat = sa.contiguous().view(-1)          # [DDD] (8192,)
        sb_flat = sb.contiguous().view(-1)          # [DDD] (8192,)
        if static_weight_scale:
            b_flat = flyc.from_dlpack(b_flat)
            sa_flat = flyc.from_dlpack(sa_flat)
            sb_flat = flyc.from_dlpack(sb_flat)
        return (
            _as_i8(a).contiguous().view(-1),  # [DDD] a: (8192,8192) fp8→int8→(67108864,)
            b_flat,
            c.contiguous().view(-1),          # [DDD] c: (8192,8192) bf16→(67108864,)
            sa_flat,
            sb_flat,
            M,
            N,
            torch.cuda.current_stream(),
        )
    #breakpoint()
    
    

    compiled = flyc.compile(launch_fn, *_args(c_out_raw, a_q, b_kernel, scale_a, scale_b))


    
    compiled(*_args(c_out_raw, a_q, b_kernel, scale_a, scale_b))
    torch.cuda.synchronize()

    c_out_f32 = c_out_raw.to(torch.float32)

    torch.cuda.synchronize()

    c_out_f32 = c_out_raw.to(torch.float32)
    #assert torch.allclose(c_out_f32, c_ref, rtol=0.1, atol=0.1)
    pass


if __name__ == "__main__":
    torch.set_default_device("cuda")
    M, N, K = 8192, 8192, 8192
    tile_m, tile_n = 256, 256

    static_weight_scale = True
    num_warmups = 10
    num_iters = 100

    _bench_fp8_gemm(
        M=M,
        N=N,
        K=K,
        use_8w=True,
        tile_m=tile_m,
        tile_n=tile_n,
        # disable_xcd_remap=args.disable_xcd_remap,
        num_warmups=num_warmups,
        num_iters=num_iters,
        vs_torch=True,
        b_preshuffled=True,
        static_weight_scale=True,
    )
