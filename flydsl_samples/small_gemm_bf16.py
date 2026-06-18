import os

os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

import torch
try:
    from IPython.display import SVG, Code, display
except ImportError:
    pass

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import typst  # `pip install typst` — bundles the compiler
except ImportError:
    typst = None

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  BF16 GEMM: C[M,N] = A[M,K] @ B[N,K]^T   (B is N×K, not K×N)            │
# │                                                                             │
# │  block tile = 64×64×32, MFMA(16,16,32,BF16), 256 threads (4 waves)        │
# │  no LDS, no preshuffle, no K-loop — block_k == K, one-shot load+compute   │
# │                                                                             │
# │  IR dump:  FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 <this>  │
# │  IR path:  /root/.flydsl/debug/gemm_kernel_0/                              │
# │  Key files:                                                                 │
# │    00_origin.mlir           — fly dialect, see all shapes/layouts           │
# │    08_convert_fly_to_rocdl.mlir — lowered: buffer.load/store, mfma instr   │
# │                                                                             │
# │  Lowered instruction counts:                                                │
# │    4 × buffer.load  i128   (2 for A + 2 for B, each dwordx4 = 128b)       │
# │    4 × mfma                (M_rep=2 × N_rep=2 × K_rep=1)                  │
# │   16 × buffer.store i32    (val=4 × M_rep=2 × N_rep=2, each dword = 32b) │
# └─────────────────────────────────────────────────────────────────────────────┘
block_m, block_n, block_k = 64, 64, 32

@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    # ── 1. Tile definitions ──────────────────────────────────────────────────
    # make_tile 定义逻辑 tile 形状，用于后续 zipped_divide 切分全局张量
    tileA = fx.make_tile(block_m, block_k)   # (64, 32)
    tileB = fx.make_tile(block_n, block_k)   # (64, 32)
    tileC = fx.make_tile(block_m, block_n)   # (64, 64)

    # ── 2. Buffer tensor 包装 ────────────────────────────────────────────────
    # 把 torch.Tensor 转成 AMDGPU buffer descriptor 引用 (ROCDL buffer_desc)
    # IR: !fly.memref<bf16, #fly_rocdl.buffer_desc, (?,?):(?{i64},1)>
    #     shape (?,?) 是运行时传入的 (M,K) / (N,K) / (M,N)，stride 中 1 表示行内连续
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    # ── 3. Block tile 选取 ───────────────────────────────────────────────────
    # zipped_divide: 按 tile 形状切分 → ((tile_inner), (tile_outer))
    #   A: (?,?) / (64,32) → ((64,32),(num_blocks_m, num_blocks_k))
    # slice(..., (None, bid)): 选第 bid 个 block 的 tile，None 保留 tile 内维度
    #
    # 结果 IR:
    #   bA: !fly.memref<bf16, buffer_desc, (64,32):(?{i64},1)>  — 64行×32列 bf16
    #   bB: !fly.memref<bf16, buffer_desc, (64,32):(?{i64},1)>  — 64行×32列 bf16
    #   bC: !fly.memref<f32,  buffer_desc, (64,64):(?{i64},1)>  — 64行×64列 f32
    bA = fx.slice(fx.zipped_divide(A, tileA), (None, bid))
    bB = fx.slice(fx.zipped_divide(B, tileB), (None, bid))
    bC = fx.slice(fx.zipped_divide(C, tileC), (None, bid))

    # ── 4. MMA atom + tiled_mma ──────────────────────────────────────────────
    # MFMA(16, 16, 32, BF16): 单条硬件指令
    #   - 输入 A/B: 每线程 8 个 bf16 (val=K/GroupK=32/4=8)
    #   - 输出 C:   每线程 4 个 f32  (val=4, 16x16 矩阵中 4 行×1 列)
    #   - 64 threads 协作完成一个 16×16×32 的矩阵乘
    # IR: !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x32, (bf16, bf16) -> f32>>
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))

    # atom_layout (2,2,1):(1,2,0) — 在 M/N/K 三维上分布 MMA atom 到 wave-groups
    #   shape (2,2,1):  M 方向 2 个 atom, N 方向 2 个 atom, K 方向 1 个
    #   stride (1,2,0): atom 编号 = m_idx*1 + n_idx*2 + k_idx*0
    #   2×2×1 = 4 个 atom → 对应 4 个 wave (256 threads / 64 threads_per_wave)
    #
    # 每个 wave 覆盖 1 个 atom (16×16), 需要在 M/N 方向各重复 2 次覆盖整个 64×64:
    #   M_rep = block_m / (atom_M × mma_M) = 64 / (2×16) = 2
    #   N_rep = block_n / (atom_N × mma_N) = 64 / (2×16) = 2
    #   K_rep = block_k / (atom_K × mma_K) = 32 / (1×32) = 1
    #
    # IR: !fly.tiled_mma<..., !fly.layout<(2,2,1):(1,2,0)>>
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((2, 2, 1), (1, 2, 0))
    )
    thr_mma = tiled_mma.thr_slice(tid)

    # ── 5. Copy atom + tiled_copy ────────────────────────────────────────────
    # BufferCopy128b + bf16: 每次 load 128 bits = 8 个 bf16 (dwordx4 指令)
    # BufferCopy32b  + f32:  每次 store 32 bits = 1 个 f32  (dword 指令)
    #
    # 为什么 C 用 32b 而不是 128b?
    #   MFMA C output 的 4 个 val 分布在矩阵的 4 行 (stride=N, 不连续),
    #   128b 要求连续地址，所以 C 必须逐元素 32b 写出
    #
    # IR copy atoms:
    #   ab: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<128>, 16>
    #    c: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>,  32>
    copy_atom_ab = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    copy_atom_c  = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

    # make_tiled_copy_A/B/C: 把 copy_atom 和 tiled_mma 的 TV layout 组合
    # 生成的 tiled_copy 包含 TV layout (Thread×Value 到数据坐标的映射):
    #
    # tiled_copy_A TV layout:
    #   ((16,4,2,2),(8,(1,1))):((1,256,16,0),(32,(0,0)))
    #    ↑ Thread 分解        ↑ Value 分解
    #   Thread: 16×4×2×2 = 256 threads
    #     - 16: wave 内 tid (对应 M 方向 16 行)
    #     -  4: wave 内 K 分组 (GroupK=64/16=4)
    #     -  2: atom_layout M 方向的 2 个 wave-group
    #     -  2: M_rep (每 wave 重复 2 次)  ← stride=0: 同一数据不同 wave
    #   Value: 8 个 bf16 per thread (K 方向连续)
    #
    # tiled_copy_B TV layout:
    #   ((16,4,2,2),(8,(1,1))):((1,256,0,16),(32,(0,0)))
    #   同 A，但 stride 中 M↔N 翻转 (stride 第3项=0→N方向, 第4项=16→N_rep)
    #
    # tiled_copy_C TV layout:
    #   ((16,8,2),((4,1),(1,1))):((32,4,512),((1,0),(0,0)))
    #   Thread: 16×8×2 = 256 threads
    #     - 16: C 矩阵行内分组 (stride=32 → 每组跨 32 列)
    #     -  8: C 矩阵 M 方向 8 组 (stride=4 → 跨 4 行)
    #     -  2: M_rep 两次重复 (stride=512 → 跨 32×16=512)
    #   Value: (4,1) = 4 个 f32 per thread (4 行, stride=1 → 连续行)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom_ab, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom_ab, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c,  tiled_mma)

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    # ── 6. 数据分区 (partition) ──────────────────────────────────────────────
    # partition_S: 按 tiled_copy 的 TV layout 把全局 tile 切分到当前线程
    # 结果是当前线程负责的数据视图: (val_per_copy, spatial_rep_0, spatial_rep_1)
    #
    # copy_src_A: bf16, buffer_desc, ((8,1), 2, 1)
    #   (8,1): 每次 copy 搬 8 个 bf16 (1 条 dwordx4); 1 是 padding
    #   2:     M_rep=2, M 方向重复 2 次
    #   1:     K_rep=1, K 方向只搬 1 次
    #   → 共 2 条 buffer.load i128 for A
    #
    # copy_src_B: bf16, buffer_desc, ((8,1), 2, 1) — 同 A
    #   → 共 2 条 buffer.load i128 for B
    #
    # copy_dst_C: f32, buffer_desc, ((1,4), 2, 2)
    #   (1,4): 每次 copy 写 1 个 f32 (1 条 dword), 但 4 个 val 要写 4 次
    #   2:     M_rep=2
    #   2:     N_rep=2
    #   → 共 4×2×2 = 16 条 buffer.store i32 for C
    copy_src_A = thr_copy_A.partition_S(bA)
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_S(bC)

    # ── 7. 寄存器 fragment 分配 ──────────────────────────────────────────────
    # make_fragment: 按 tiled_mma 的分配策略创建寄存器空间
    # fragment shape: (val, rep_0, rep_1) — 各维含义取决于 A/B/C
    #
    # frag_A: bf16, register, (8, 2, 1):(1,8,0)
    #   val=8:   每个 MFMA atom 每线程 8 个 bf16 input (K/GroupK=32/4)
    #   M_rep=2: M 方向重复 2 次 (stride=8, 步过一个 val 块)
    #   K_rep=1: K 方向 1 次 (block_k/mma_K=32/32=1, 无 K 循环)
    #   → 每线程 8×2×1 = 16 个 bf16 = 8 VGPR (bf16 两个一组占 1 VGPR)
    #
    # frag_B: bf16, register, (8, 2, 1):(1,8,0) — 同 frag_A
    #   val=8, N_rep=2, K_rep=1
    #   → 每线程 8 VGPR
    #
    # frag_C: f32, register, ((4,1), 2, 2):((1,0),8,4)
    #   val=(4,1): 每 atom 每线程 4 个 f32 accumulator (4行×1列)
    #   M_rep=2:   M 方向 2 次 (stride=8)
    #   N_rep=2:   N 方向 2 次 (stride=4)
    #   → 每线程 4×2×2 = 16 个 f32 = 16 VGPR
    #   → K 不出现 (归约维度, 多次 MFMA 累加到同一组 C)
    frag_A = thr_mma.make_fragment_A(bA)
    frag_B = thr_mma.make_fragment_B(bB)
    frag_C = thr_mma.make_fragment_C(bC)

    # ── 8. Retile: fragment → copy 兼容形状 ─────────────────────────────────
    # retile 重新排列 fragment 的 val 维度，使其匹配 copy 的 val 分组方式
    #
    # copy_frag_A: bf16, register, ((8,1), 2, 1) — val 从 8 拆成 (8,1)
    #   匹配 copy_src_A 的 ((8,1), 2, 1)，这样 fx.copy 可以 1:1 搬运
    # copy_frag_B: bf16, register, ((8,1), 2, 1) — 同上
    # copy_frag_C: f32, register, ((1,4), 2, 2) — val 从 (4,1) 变成 (1,4)
    #   匹配 copy_dst_C 的 ((1,4), 2, 2)
    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)

    # ── 9. 执行: Load → GEMM → Store ────────────────────────────────────────
    # 本 kernel 是 load-all → compute-all → store-all 的一次性模式:
    #   frag_A/B 在 gemm 之前已被 fx.copy 完全填满，
    #   4 条 MFMA 全部从寄存器取数据，无需额外 load。
    # lowered IR 中三段严格分离，无交叉:
    #   L140-202: 4 × buffer.load i128
    #   L212-223: 4 × mfma
    #   L245-572: 16 × buffer.store i32
    # 若需分批 (手动 K-loop)，fx.copy 和 fx.gemm 会交替出现在循环体内。

    # copy A: copy_src_A((8,1),2,1) → copy_frag_A((8,1),2,1)
    #   2 条 buffer.load i128 (每条 dwordx4, 搬 8 个 bf16)
    fx.copy(copy_atom_ab, copy_src_A, copy_frag_A, pred=None)
    # copy B: 同上, 2 条 buffer.load i128
    fx.copy(copy_atom_ab, copy_src_B, copy_frag_B, pred=None)

    # C accumulator 清零
    frag_C.fill(0)

    # gemm: 展开 M_rep×N_rep×K_rep = 2×2×1 = 4 条 v_mfma_f32_16x16x32_bf16
    # 每条: A[8 bf16] × B[8 bf16] → C[4 f32] (累加)
    # frag_A (8,2,1) 一次性提供了 4 条 MFMA 所需的全部 A 数据:
    #   MFMA #0 (m=0,n=0): A[:, 0, 0]  ×  B[:, 0, 0]  → C[:, 0, 0]
    #   MFMA #1 (m=0,n=1): A[:, 0, 0]  ×  B[:, 1, 0]  → C[:, 0, 1]  ← A 复用
    #   MFMA #2 (m=1,n=0): A[:, 1, 0]  ×  B[:, 0, 0]  → C[:, 1, 0]  ← B 复用
    #   MFMA #3 (m=1,n=1): A[:, 1, 0]  ×  B[:, 1, 0]  → C[:, 1, 1]
    # A 沿 N 方向复用 (同一 M_rep slice 喂两个 N 位置),
    # B 沿 M 方向复用 (同一 N_rep slice 喂两个 M 位置)。
    # IR 可验证: %165/%174 是 A 的 2 个 M_rep, %166/%170 是 B 的 2 个 N_rep,
    #   mfma #0=(%165,%166), #1=(%165,%170), #2=(%174,%166), #3=(%174,%170)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    # store C: copy_frag_C((1,4),2,2) → copy_dst_C((1,4),2,2)
    #   16 条 buffer.store i32 (每条 dword, 写 1 个 f32)
    fx.copy(copy_atom_c, copy_frag_C, copy_dst_C, pred=None)


@flyc.jit
def launch_fn(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    BLOCK_DIM = 256   # 256 threads = 4 waves × 64 threads/wave
    gemm_kernel(A, B, C).launch(
        grid=(1, 1, 1), block=(BLOCK_DIM, 1, 1), stream=stream
    )


# === Test ===
M = block_m   # 64
N = block_n   # 64
K = block_k   # 32 — must equal block_k, no K-loop in kernel

A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
launch_fn(A, B, C, stream=torch.cuda.Stream())
torch.cuda.synchronize()

ref = A.float() @ B.float().T
assert torch.allclose(ref, C, atol=1e-2, rtol=1e-2), f"max error {torch.max(torch.abs(ref - C))}"
print("PASSED")
