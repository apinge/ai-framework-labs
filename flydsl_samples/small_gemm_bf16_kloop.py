import os

os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  BF16 GEMM with K-loop: C[M,N] = A[M,K] @ B[N,K]^T                       │
# │                                                                             │
# │  和 small_gemm_bf16.py 的唯一区别: K 可以是 block_k 的任意倍数,            │
# │  通过 range_constexpr 展开 K 方向的循环。                                  │
# │                                                                             │
# │  block tile = 64×64×32, MFMA(16,16,32,BF16), 256 threads (4 waves)        │
# │  no LDS, no preshuffle — 每次迭代直接 buffer load → register → MFMA       │
# │                                                                             │
# │  vs small_gemm_bf16.py 的结构差异:                                          │
# │    1. A/B 用 flat_divide 代替 zipped_divide, 保留 K 外维                   │
# │    2. partition_S 结果多一个 K 外维: ((8,1), 2, 1, num_k)                  │
# │    3. 执行阶段: fill(0) → K-loop{ load A[k], load B[k], gemm } → store C  │
# │                                                                             │
# │  IR dump:  FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 <this>  │
# │                                                                             │
# │  Lowered instruction counts (num_k=4):                                      │
# │    16 × buffer.load  i128   (4 iters × (2 A + 2 B))                       │
# │    16 × mfma                (4 iters × 4 MFMAs)                            │
# │    16 × buffer.store i32    (val=4 × M_rep=2 × N_rep=2, 不随 num_k 变)    │
# └─────────────────────────────────────────────────────────────────────────────┘
block_m, block_n, block_k = 64, 64, 32
num_k = 4  # K 方向迭代次数, K_total = num_k × block_k = 128


@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    # ── 1. Tile definitions ──────────────────────────────────────────────────
    tileA = fx.make_tile(block_m, block_k)   # (64, 32)
    tileB = fx.make_tile(block_n, block_k)   # (64, 32)
    tileC = fx.make_tile(block_m, block_n)   # (64, 64)

    # ── 2. Buffer tensor 包装 ────────────────────────────────────────────────
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    # ── 3. Block tile 选取 (K-loop 版) ───────────────────────────────────────
    # 和 small_gemm_bf16.py 的区别:
    #   原版用 zipped_divide + slice → (64, 32), K 维度被消费掉
    #   K-loop 版用 flat_divide + 索引 → (64, 32, num_k), 保留 K 外维
    #
    # flat_divide(A, (64, 32)) 把 A(M, K) 拆成 (64, 32, M/64, K/32)
    # [None, None, bid, None]:
    #   None, None = 保留前两维 (tile 内的 M 和 K)
    #   bid        = 选第 bid 个 M block (这里 bid=0)
    #   None       = 保留 K 外维供循环迭代
    # 结果: (64, 32, num_k) — 每个 K 切片是一个 (64, 32) 的 A tile
    bA_k = fx.flat_divide(A, tileA)[None, None, bid, None]  # (64, 32, num_k)
    bB_k = fx.flat_divide(B, tileB)[None, None, bid, None]  # (64, 32, num_k)
    bC   = fx.flat_divide(C, tileC)[None, None, bid, bid]   # (64, 64)

    # ── 4. MMA atom + tiled_mma ──────────────────────────────────────────────
    # 完全不变
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((2, 2, 1), (1, 2, 0))
    )
    thr_mma = tiled_mma.thr_slice(tid)

    # ── 5. Copy atom + tiled_copy ────────────────────────────────────────────
    # 完全不变
    copy_atom_ab = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    copy_atom_c  = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

    tiled_copy_A = fx.make_tiled_copy_A(copy_atom_ab, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom_ab, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c,  tiled_mma)

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    # ── 6. 数据分区 (partition) ──────────────────────────────────────────────
    # 区别: copy_src_A/B 多了 K 外维
    #
    # copy_src_A: ((8,1), 2, 1, num_k)
    #   前三维和原版一样: (val, M_rep, K_rep_inner)
    #   第四维 num_k: K 外维, 循环迭代索引
    #
    # copy_dst_C: ((1,4), 2, 2) — 不变 (C 不涉及 K 迭代)
    copy_src_A = thr_copy_A.partition_S(bA_k)
    copy_src_B = thr_copy_B.partition_S(bB_k)
    copy_dst_C = thr_copy_C.partition_S(bC)

    # ── 7. 寄存器 fragment 分配 ──────────────────────────────────────────────
    # make_fragment_A/B 需要 2D tile 参考, 所以取 bA_k 的第 0 个 K 切片
    # bA_k[None, None, 0] → (64, 32) — 和原版 bA 的 shape 完全一样
    # fragment shape 不变: frag_A (8,2,1), frag_B (8,2,1), frag_C ((4,1),2,2)
    frag_A = thr_mma.make_fragment_A(bA_k[None, None, 0])
    frag_B = thr_mma.make_fragment_B(bB_k[None, None, 0])
    frag_C = thr_mma.make_fragment_C(bC)

    # ── 8. Retile: 同一块寄存器的不同 shape 视图 ────────────────────────────
    # retile 不分配新内存, 只重组 val 维度的 shape, 类似 numpy.view/reshape:
    #   frag_A      shape (8, 2, 1)     ← MMA 视图: val=8 是一条 MFMA 的 input
    #   copy_frag_A shape ((8,1), 2, 1) ← Copy 视图: val=(8,1) 匹配 128b load 粒度
    #   底层 stride 一样: (1, 8, 0) vs ((1,0), 8, 0) — 访问同一组 VGPR
    #
    # IR 证据 (00_origin.mlir):
    #   %70 = fly.mma.make_fragment(a, ...)           ← 分配 VGPR
    #         → !fly.memref<bf16, register, (8,2,1):(1,8,0)>
    #   %73 = fly.tiled_copy.retile(%43, %70)         ← 输入就是 %70, 不新分配
    #         → !fly.memref<bf16, register, ((8,1),2,1):((1,0),8,0)>
    #   retile 的输入是 %70 (frag_A), 输出 %73 (copy_frag_A)
    #   两者都是 register memref, 指向同一组 VGPR
    #
    # lowered IR (08_convert_fly_to_rocdl.mlir):
    #   retile 完全消失——没有生成任何指令 (零开销)
    #   copy lowering 直接用 copy 视图的 stride 索引 VGPR
    #   mma lowering 直接用 mma 视图的 stride 索引同一组 VGPR
    #
    # 这就是为什么下面 fx.copy 写 copy_frag_A, fx.gemm 读 frag_A 能关联:
    #   fx.copy(src, copy_frag_A) → 往 VGPR 写数据 (用 copy 的 val 分组)
    #   fx.gemm(frag_A, ...)      → 从同一组 VGPR 读数据 (用 MMA 的 val 分组)
    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)

    # ── 9. 执行: K-loop ─────────────────────────────────────────────────────
    # 这是和原版的核心区别:
    #   原版: load A → load B → fill(0) → gemm → store C  (一次性)
    #   K-loop: fill(0) → for k in 0..num_k-1: load A[k] → load B[k] → gemm → store C
    #
    # range_constexpr: 编译期展开, 每个 k 是 Python int
    #   k=0 时: copy_src_A[None, None, None, 0] 选第 0 个 K 切片
    #   k=1 时: copy_src_A[None, None, None, 1] 选第 1 个 K 切片
    #   ...
    #   每次迭代生成: 2 A loads + 2 B loads + 4 MFMAs
    #   frag_C 在循环间累加 (MFMA 本身是 accumulate 模式)
    frag_C.fill(0)

    for k in fx.range_constexpr(num_k):
        # load A[:, k*32:(k+1)*32]
        fx.copy(copy_atom_ab, copy_src_A[None, None, None, k], copy_frag_A, pred=None)
        # load B[:, k*32:(k+1)*32]
        fx.copy(copy_atom_ab, copy_src_B[None, None, None, k], copy_frag_B, pred=None)
        # 4 × MFMA, 累加到 frag_C
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    # store C (循环结束后一次性写出)
    fx.copy(copy_atom_c, copy_frag_C, copy_dst_C, pred=None)


@flyc.jit
def launch_fn(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    BLOCK_DIM = 256
    gemm_kernel(A, B, C).launch(
        grid=(1, 1, 1), block=(BLOCK_DIM, 1, 1), stream=stream
    )


# === Test ===
M = block_m            # 64
N = block_n            # 64
K = num_k * block_k    # 128 (= 4 × 32)

A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
launch_fn(A, B, C, stream=torch.cuda.Stream())
torch.cuda.synchronize()

ref = A.float() @ B.float().T
assert torch.allclose(ref, C, atol=1e-2, rtol=1e-2), f"max error {torch.max(torch.abs(ref - C))}"
print("PASSED")
