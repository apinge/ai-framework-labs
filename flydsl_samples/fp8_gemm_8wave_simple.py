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
from flydsl.runtime.device import get_rocm_arch 

from flydsl.expr.typing import Vector as Vec
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl.expr import arith, const_expr, range_constexpr

from fp8_gemm_utils import (
    pertoken_quant,
    _run_torch,
    preshuffle_b,
    _as_i8,
    ceildiv,
    divmod,
    make_fp8_buffer_tensor,
    compute_global_swizzle,
    swizzle_128,
    pack_i32x4_i32x8,
)

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

DDD = False  # debug

OUT_DTYPE = torch.bfloat16
ARCH = str(get_rocm_arch())

OUTPUT_TYPST = "layout.typ"

# helper class
class Mfma16x16x128:
    def __init__(self, n_tiles_a, n_tiles_b):
        self.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
        # ↑ 对应硬件指令 v_mfma_scale_f32_16x16x128_f8f6f4

        self.accum_type = Vec.make_type(4, fx.Float32)  # 每 lane 4个 f32
        self.zero_value = Vec.filled(4, 0.0, fx.Float32) # 初始累加器 [0,0,0,0]
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j
        # n_tiles_a=4, n_tiles_b=2:
        # (0,0)→0  (0,1)→1
        # (1,0)→2  (1,1)→3
        # (2,0)→4  (2,1)→5
        # (3,0)→6  (3,1)→7
        # 共 8 个 accumulator slots

    def _do_mma(self, a, b, c):
        return fly_dialect.mma_atom_call_ssa([self.accum_type], self.atom, a, b, c)

    def call(self, a, b, c):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b
        # a: list[8] (n_tiles_a=4 个 i32x8 frag)
        # b: list[4] (n_tiles_b=2 个 i32x8 frag)
        # c: list[8] (4×2=8 个 Vec<4,f32>) 
        """
        block_size = 512 threads = 8 waves

        wave_m = wave_id // 4  → 0 或 1  (M方向 2行)
        wave_n = wave_id % 4   → 0,1,2,3 (N方向 4列)

        每个 wave 用 Mfma16x16x128(N_TILES_A=4, N_TILES_B=2) 负责：
        M方向: N_TILES_A × 16 = 4×16 = 64 行N方向: N_TILES_B × 16 = 2×16 = 32 列

        8 个 wave 合起来恰好覆盖整个 128×128 的半块
        M方向: N_TILES_A × 16 = 4×16 = 64 行
        N方向: N_TILES_B × 16 = 2×16 = 32 列

        M: 2 × 64 = 128 = LDS_BLOCK_M ✓
        N: 4 × 32 = 128 = LDS_BLOCK_N ✓

                           B0 (cols 0~127)    B1 (cols 128~255)
        A0 (rows 0~127)    c00_frag           c01_frag
        A1 (rows 128~255)  c10_frag           c11_frag

        c00_frag[8]: wave 计算 A0(上半) × B0(左半) 的结果
        c01_frag[8]: wave 计算 A0(上半) × B1(右半) 的结果
        c10_frag[8]: wave 计算 A1(下半) × B0(左半) 的结果
        c11_frag[8]: wave 计算 A1(下半) × B1(右半) 的结果
                    ↑ 对应 LDS 里 4 对 cur/next ping-pong buffer

        mfma.call(a0_frag, b0_frag, c00_frag)  → 内部 4×2 = 8次 MFMA
        mfma.call(a0_frag, b1_frag, c01_frag)  → 8次 MFMA
        mfma.call(a1_frag, b0_frag, c10_frag)  → 8次 MFMA
        mfma.call(a1_frag, b1_frag, c11_frag)  → 8次 MFMA
        ─────────────────────────────────────
        每个wave每轮K-iter = 32次 MFMA

        8个wave各自负责不同输出位置

        wave_m=0, wave_n=0..3 的 store 目标：

                B0(cols 0~127)    B1(cols 128~255)
        A0(0~63)    c00 ↗             c01 ↗
        A1(128~191) c10 ↗             c11 ↗

        wave_m=1, wave_n=0..3 的 store 目标：

                  B0(cols 0~127)    B1(cols 128~255)
        A0(64~127)  c00 ↗             c01 ↗
        A1(192~255) c10 ↗             c11 ↗

        """
        for i in range_constexpr(self.n_tiles_a): #  for i in 0..4:      # A 的 tile 行
            for j in range_constexpr(self.n_tiles_b): # for j in 0..2:   # B 的 tile 列
                c[self.idx(i, j)] = self._do_mma(a[i], b[j], c[self.idx(i, j)])
                #  每个 wave 计算 8 次 MFMA，覆盖 64×32 的输出 tile
                # 可以这样想 16x16x128结果C是16X16 那么 覆盖的大小就是16*16*8
        return c

    def call_one(self, a, b, c, i, j):
        assert i < self.n_tiles_a and j < self.n_tiles_b

        return self._do_mma(a[i], b[j], c[self.idx(i, j)])


class G2SLoader: #Global Memory → LDS 搬运
    def __init__(self, gl_src, gl_offsets, n_load_steps, lds_dtype, wave_id):
        self.g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        # ↑ AMD buffer_load_dwordx4_lds 指令：每线程一次搬 128bit = 16 fp8 elements
        # 64 threads × 16 elements = 1024 fp8/wave/次

        self.LdsPtr_t = fx.PointerType.get(lds_dtype, 2, 512)
        # ↑ address_space=2 = LDS(shared memory), alignment=512bit

        self.gl_src = gl_src
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id
        self.n_waves = fx.block_dim.x // 64
    
    """
    LDS 布局（一个半块 = 16384 bytes）
    [0    ..1023 ] wave0 step0  (rows  0~7  , cols 0~127)
    [1024 ..2047 ] wave1 step0  (rows  8~15 , cols 0~127)
    ...
    [7168 ..8191 ] wave7 step0  (rows 56~63 , cols 0~127)
    [8192 ..9215 ] wave0 step1  (rows 64~71 , cols 0~127)
    ...
    [15360..16383] wave7 step1  (rows120~127, cols 0~127)

    """

    def _lds_dst_at(self, lds_dst, step):
        step_off = self.wave_id * 1024 + step * (self.n_waves * 1024)
        #                ↑ 每wave占1024B   ↑ 每step跨越所有wave
        base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr))
        sum_i32 = base_i32 + fx.Int32(step_off)
        lds_ptr = fx.inttoptr(self.LdsPtr_t, sum_i32)
        return fx.make_view(lds_ptr, fx.make_layout(1, 1))

    def load(self, lds_dst, k_offset):
        for step in range_constexpr(self.n_load_steps):
            src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
            #      ↑ 每线程自己的 swizzle 全局偏移（compute_global_swizzle 算出来的）
            dst = self._lds_dst_at(lds_dst, step)
            #     ↑ 这个wave这个step在LDS里的位置（所有线程共享同一base）
            fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))
            #                              ↑ K 方向的推进量（每K-iter加 BLOCK_K 个元素）

    def load_one(self, lds_dst, k_offset, step): 
        # # 只做1个step，用于 ping-pong 流水线的 prologue/epilogue
        src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
        dst = self._lds_dst_at(lds_dst, step)
        fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

class S2RLoader: # LDS → (VGPR）
    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16xf8(self, lds_src, offset):
        # 每次从 LDS 读 16 个 fp8
        off_tup = fx.make_int_tuple(offset)
        ptr_off = fx.add_offset(lds_src.ptr, off_tup)
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        view = fx.make_view(i8_iter, fx.make_layout(16, 1))
        return view.load() # → Vec<16, uint8>

    def load(self, lds_src, preshuffled=False):
        """
        代入 A 的情况（wave_idx=wave_m, n_tiles=4）：
        for i in range(4):  # 4个 tile，每tile对应16行
            row = wave_m * 64 + i * 16 + lane_id % 16
            #     ↑ wave偏移    ↑tile偏移  ↑ lane在tile内的行

            # step=0: col = (lane_id//16)*16 + 0    → 0,16,32,48
            # step=1: col = (lane_id//16)*16 + 64   → 64,80,96,112
            for step in range(2):
                col = (lane_id // 16) * 16 + step * 64
                offset = swizzle_128(row, col)  # 反swizzle找到正确位置
                halves[step] = load_16xf8(offset).bitcast(i32)  # i32x4

            # 两个 i32x4 拼成 i32x8 = 32 fp8 = 一个 MFMA A-frag
            frag[i] = pack_i32x4_i32x8(halves[0], halves[1])
        lane  0~15 (group0): col=0  (step0) + col=64  (step1) → K[0~15,   64~79]
        lane 16~31 (group1): col=16 (step0) + col=80  (step1) → K[16~31,  80~95]
        lane 32~47 (group2): col=32 (step0) + col=96  (step1) → K[32~47,  96~111]
        lane 48~63 (group3): col=48 (step0) + col=112 (step1) → K[48~63, 112~127]
        合计: 覆盖完整 128 列 ✓

        offset 公式
        A: （non-preshuffled : swizzle_128(row, col)
        B: (row//8)*1024 + (row%8)*16 + (col//16)*128

        最终输出
        frag = [i32x8, i32x8, ..., i32x8]
        ↑ n_tiles 个
        A: 4个 i32x8 → 传给 mfma.call(a=frag, ...)
        B: 2个 i32x8 → 传给 mfma.call(b=frag, ...)

        每个 i32x8 = 32 fp8 = MFMA 16×16×128 所需的单 lane A/B fragmen
        """
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            for step in range_constexpr(2):
                col = (self.lane_id // 16) * 16 + step * 64
                if const_expr(preshuffled):
                    offset = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
                else:
                    row_swz, col_swz = swizzle_128(row, col)
                    offset = row_swz * 128 + col_swz
                v = self._vec_load_16xf8(lds_src, offset)
                halves.append(v.bitcast(fx.Int32))
            frag.append(pack_i32x4_i32x8(halves[0], halves[1]))
        return frag

    def load_one(self, lds_src, lds_offset):
        v = self._vec_load_16xf8(lds_src, lds_offset)
        return v.bitcast(fx.Int32)

class StoreC:
    def __init__(self, A_scale, B_scale, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        # Exact byte counts from compile-time shape (BF16 C output, FP32 scales).
        # ``num_records_bytes`` is required when ``max_size=False`` -- see
        # ``make_buffer_tensor`` docstring for the silent-OOB rationale.
        c_nbytes = c_rows * c_cols * 2  # BFloat16 = 2 bytes
        sa_nbytes = c_rows * 4  # Float32 row-wise scale
        sb_nbytes = c_cols * 4  # Float32 col-wise scale
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False, num_records_bytes=sa_nbytes)
        gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=sb_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))

        self.scale_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        self.reg_f32_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        self.reg_bf16_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.BFloat16)

    def _load_scale_vec4(self, row):
        fx.copy(self.scale_atom_4, fx.slice(self.sa_div, (None, fx.Int32(row))), self.reg_f32_4)
        return Vec(fx.memref_load_vec(self.reg_f32_4))

    def _load_scale_scalar(self, col):
        fx.copy(self.scale_atom_1, fx.slice(self.sb_div, (None, fx.Int32(col))), self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def _store_bf16(self, value_bf16, c_index):
        fx.memref_store_vec(Vec.filled(1, value_bf16, fx.BFloat16), self.reg_bf16_1)
        fx.copy(self.out_atom_1, self.reg_bf16_1, fx.slice(self.c_div, (None, fx.Int32(c_index))))

    def store(self, c_frag, base_row, base_col):
        a_scales = [
            self._load_scale_vec4(base_row + i * 16 + (self.lane_id // 16) * 4) for i in range_constexpr(self.n_tiles_a)
        ]
        b_scales = [
            self._load_scale_scalar(base_col + i * 16 + self.lane_id % 16) for i in range_constexpr(self.n_tiles_b)
        ]
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + (self.lane_id // 16) * 4
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                col_valid = col < self.c_cols
                oob = fx.Int32(self.c_rows * self.c_cols)
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    scaled = (vec_f32[i] * (a_scales[ti][i] * b_scales[tj])).to(fx.BFloat16)
                    c_index = (row + i) * self.c_cols + col
                    self._store_bf16(scaled, arith.select(col_valid, c_index, oob))

    
# begin kernel
def compile_fp8_gemm_8w(
    *, K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, b_preshuffled: bool = False
):
    """
    这段在定义一些 kernel用到的宏
    """
    BLOCK_K = 128  # [DDD] fixed K-tile = 128

    assert (
        BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    )
    assert K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K  # [DDD] 8192 // 128 = 64

    N_TILES_A = BLOCK_M // 64  # [DDD] 256 // 64 = 4  (A sub-tiles per wave-row)
    N_TILES_B = BLOCK_N // 128  # [DDD] 256 // 128 = 2  (B sub-tiles per wave-col)
    N_ACCUMS = N_TILES_A * N_TILES_B  # [DDD] 4 * 2 = 8 accumulator fragments per wave
    assert N_ACCUMS > 0

    LDS_BLOCK_M = (
        BLOCK_M // 2
    )  # [DDD] 256 // 2 = 128  (half of BLOCK_M, split into A0/A1)
    LDS_BLOCK_N = (
        BLOCK_N // 2
    )  # [DDD] 256 // 2 = 128  (half of BLOCK_N, split into B0/B1)

    N_LDS_STEPS_A = (
        LDS_BLOCK_M // 64
    )  # [DDD] 128 // 64 = 2  (G2S rounds to fill one A half-tile)
    N_LDS_STEPS_B = (
        LDS_BLOCK_N // 64
    )  # [DDD] 128 // 64 = 2  (G2S rounds to fill one B half-tile)
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)  # [DDD] max(2, 2) = 2

    """
    N_LDS_ROUNDS：G2S 需要几轮才能填满一个需要的LDS_BLOCK_M × BLOCK_K
    这里是这样算的一块 LDS LDS_BLOCK_M × BLOCK_K = 128 × 128 = 16384 fp8 elements
    每轮能搬多少？
    8 waves × 64 threads × 16 bytes = 8192 fp8 elements / round

    填满一个半块需要：
    16384 / 8192 = 2 rounds = N_LDS_STEPS_A ✓

    8个wave覆盖完整的 256×256 输出块，并行执行，不叠加：
    M方向: wave_m=0 → 行{0~63, 128~191}
       wave_m=1 → 行{64~127, 192~255}
       合计: 256行 ✓

    N方向: wave_n=0..3 各负责 64列(32+32)
        合计: 4×64 = 256列 ✓
    """

    # half size
    a_lds_size = (
        LDS_BLOCK_M * BLOCK_K
    )  # [DDD] 128 * 128 = 16384 fp8 elements per ping/pong A buffer
    b_lds_size = (
        LDS_BLOCK_N * BLOCK_K
    )  # [DDD] 128 * 128 = 16384 fp8 elements per ping/pong B buffer

    """
    16 是16 align
    凑dwordx4
    """

    @fx.struct
    class SharedStorage:
        # [DDD] 8 ping-pong buffers total: 4×A + 4×B, each 16384 fp8 bytes → 8 * 16384 = 131072 B = 128 KB LDS
        A_lds_cur_0: fx.Array[
            fx.Float8E4M3FN, a_lds_size, 16
        ]  # [DDD] (16384,) cur  A0 half-tile [BLOCK_M/2 × BLOCK_K]
        A_lds_cur_1: fx.Array[
            fx.Float8E4M3FN, a_lds_size, 16
        ]  # [DDD] (16384,) cur  A1 half-tile
        A_lds_next_0: fx.Array[
            fx.Float8E4M3FN, a_lds_size, 16
        ]  # [DDD] (16384,) next A0 half-tile (prefetch)
        A_lds_next_1: fx.Array[
            fx.Float8E4M3FN, a_lds_size, 16
        ]  # [DDD] (16384,) next A1 half-tile (prefetch)
        B_lds_cur_0: fx.Array[
            fx.Float8E4M3FN, b_lds_size, 16
        ]  # [DDD] (16384,) cur  B0 half-tile [BLOCK_N/2 × BLOCK_K]
        B_lds_cur_1: fx.Array[
            fx.Float8E4M3FN, b_lds_size, 16
        ]  # [DDD] (16384,) cur  B1 half-tile
        B_lds_next_0: fx.Array[
            fx.Float8E4M3FN, b_lds_size, 16
        ]  # [DDD] (16384,) next B0 half-tile (prefetch)
        B_lds_next_1: fx.Array[
            fx.Float8E4M3FN, b_lds_size, 16
        ]  # [DDD] (16384,) next B1 half-tile (prefetch)

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
        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = ceildiv(c_n, BLOCK_N)  # 8192//256 = 32

        # peek()
        #   Reads (loads) all the fields from the allocated LDS storage and returns an actual SharedStorage instance whose attributes are live
        #   DSL pointers into LDS. This is what lets you do lds.A_lds_cur_0 — each attribute is a pointer (SmemPtr / Array) pointing
        #   to that field's region of LDS memory.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        # 8wave的分配方法 
        wave_m = wave_id // 4 
        wave_n = wave_id % 4  

        # fx.block_idx.x//32 fx.block_idx.x%32
        # fx.block_idx.x 对应grid=(1024, 1, 1)
        block_m, block_n = divmod(fx.block_idx.x, n_blocks)

        if DDD:
            if lane_id == 0 and wave_id == 0 and fx.block_idx.x == 1023:
                fx.printf(
                    "fx.block_idx.x={} n_blocks={} block_m={} block_n={} \n",
                    fx.block_idx.x,
                    block_m,
                    block_n,
                    n_blocks,
                )
                # for example fx.block_idx.x=1023 n_blocks=31 block_m=31 block_n=32
                # block_n \in {0,1,2,..31}
                # block_m \in {0,1,2...31}

        A0_gl_offset = (block_m * BLOCK_M) * K  # (block_m*256)*8192
        A1_gl_offset = (
            block_m * BLOCK_M + LDS_BLOCK_M
        ) * K  # (block_m*256+256//2)*8192
        B_K_STEP = (
            (2 * 1024) if b_preshuffled else BLOCK_K
        )  # b_preshuffled=True 2*1024 [???]
        B0_gl_offset = (block_n * BLOCK_N) * K  # (block_n*256)*8192
        B1_gl_offset = (
            block_n * BLOCK_N + LDS_BLOCK_N
        ) * K  # (block_n*256+256//2)*8192

        # 类型系统上的 reinterpret_cast，从 MLIR IR 层面把 i8* 换成 f8E4M3FN*，底层 GPU 内存地址和数据完全不变。
        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)


        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        # fx.make_layout(1, 1) → Layout<Shape<1>, Stride<1>>，即大小为1、步长为1的平凡tile
        # logical_divide 效果：把 (M*K,) 按 tile=(1,stride=1) 分割
        # 结果 a_div shape: ((1, M*K),)  ← 添加了一个平凡的内层维度 [???]

        # [???]
        # 这里主要就是为了减少bank conflict那种swizzle 我们先忽略实现细节
        gl_off_a = compute_global_swizzle(
            lane_id,       # 0~63，本线程在 wavefront 内的编号
            wave_id,       # 0~7，本 wavefront 在 block 内的编号
            K,             # 8192
            N_LDS_ROUNDS,   # G2S 需要几轮才能填满一个半 tile 2
            preshuffled=False,   # A 永远不做 preshuffle
        )

        gl_off_b = compute_global_swizzle(
            lane_id,
            wave_id,
            K,
            N_LDS_ROUNDS,
            preshuffled=True    # B 已经 preshuffle 过
        )

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        #                 ↑ 需要 2D hierarchical tensor
        #                   才能用 (tile_offset, outer_offset) 两级索引计算地址

        a_s2r = S2RLoader(wave_m, N_TILES_A) # N_TILES_A:4 
        b_s2r = S2RLoader(wave_n, N_TILES_B) # N_TILES_B:2

        store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

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
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(
            c_n, BLOCK_N
        )  # [DDD] ceil(8192/256)*ceil(8192/256) = 32*32 = 1024 blocks
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": "512,512",
            },
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
    scale_a = scale_a.squeeze().contiguous()  # 压掉(8192,1) -> (8192,)
    scale_b = scale_b.squeeze().contiguous()

    c_ref = _run_torch(a_q, b_q, scale_a, scale_b)  # torch reference

    # [DDD] b_preshuffled=True: (8192,8192) → reshape(512,16,128,4,16) → permute(0,2,3,1,4) → (512,128,4,16,16) contiguous
    b_kernel = preshuffle_b(b_q) if b_preshuffled else b_q

    launch_fn = compile_fp8_gemm_8w(
        K=K,  # [DDD] K=8192
        BLOCK_M=tile_m,  # [DDD] BLOCK_M=256
        BLOCK_N=tile_n,  # [DDD] BLOCK_N=256
        b_preshuffled=b_preshuffled,  # [DDD] b_preshuffled=True
    )
    print(
        f"\n[fp8_gemm_8wave] M={M} N={N} K={K} BLOCK_M={tile_m} BLOCK_N={tile_n} "
        f"preshuffle_b={b_preshuffled} static_weight_scale={static_weight_scale}"
    )

    def _args(c, a, b, sa, sb):
        b_flat = (
            _as_i8(b).contiguous().view(-1)
        )  # [DDD] b: (512,128,4,16,16) fp8→int8→view(-1)=(8192*8192,)=(67108864,)
        sa_flat = sa.contiguous().view(-1)  # [DDD] (8192,)
        sb_flat = sb.contiguous().view(-1)  # [DDD] (8192,)
        if static_weight_scale:
            b_flat = flyc.from_dlpack(b_flat)
            sa_flat = flyc.from_dlpack(sa_flat)
            sb_flat = flyc.from_dlpack(sb_flat)
        return (
            _as_i8(a)
            .contiguous()
            .view(-1),  # [DDD] a: (8192,8192) fp8→int8→(67108864,)
            b_flat,
            c.contiguous().view(-1),  # [DDD] c: (8192,8192) bf16→(67108864,)
            sa_flat,
            sb_flat,
            M,
            N,
            torch.cuda.current_stream(),
        )

    # breakpoint()

    compiled = flyc.compile(
        launch_fn, *_args(c_out_raw, a_q, b_kernel, scale_a, scale_b)
    )

    compiled(*_args(c_out_raw, a_q, b_kernel, scale_a, scale_b))
    torch.cuda.synchronize()

    c_out_f32 = c_out_raw.to(torch.float32)

    torch.cuda.synchronize()

    c_out_f32 = c_out_raw.to(torch.float32)
    # assert torch.allclose(c_out_f32, c_ref, rtol=0.1, atol=0.1)
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
