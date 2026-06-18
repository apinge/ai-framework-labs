import os
import tempfile

# print_typst writes its .typ files as a trace-time side effect of @flyc.jit, and the
# layout cells print at trace time too. A warm JIT disk cache would skip the re-trace
# (and those side effects), so disable it — re-runs then always reproduce. Set before
# importing flydsl.
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
"""
  FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/sgl-workspace/ai-framework-labs/flydsl_samples/ir_gemm_bf32  \
    python /sgl-workspace/ai-framework-labs/flydsl_samples/small_gemm_f32.py
"""
import torch
from IPython.display import SVG, Code, display

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import typst  # `pip install typst` — bundles the compiler
except ImportError:
    typst = None  # without it, diagrams fall back to raw source

from flydsl._mlir import ir

block_m, block_n, block_k = 64, 64, 8
DDD = 1

@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    # Define tiles
    tileA = fx.make_tile(block_m, block_k)
    tileB = fx.make_tile(block_n, block_k)
    tileC = fx.make_tile(block_m, block_n)

    # Wrap as buffer tensors
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    # Divide and select block's tile
    bA = fx.slice(fx.zipped_divide(A, tileA), (None, bid))
    bB = fx.slice(fx.zipped_divide(B, tileB), (None, bid))
    bC = fx.slice(fx.zipped_divide(C, tileC), (None, bid))

    # === MMA setup ===
    # MFMA(M, N, K, AccType) -- hardware instruction shape
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))

    # Tile the MMA atom across threads: 2x2 = 4 MMA atoms per warp
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((2, 2, 1), (1, 2, 0))  # (M_rep, N_rep, K_rep)
    )
    thr_mma = tiled_mma.thr_slice(tid)

    # === Copy setup (matched to MMA layout) ===
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma)

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    # === Partition data ===
    # Copy partitions (for data movement)
    copy_src_A = thr_copy_A.partition_S(bA)
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_S(bC)

    # MMA partitions (for compute)
    part_A = thr_mma.partition_A(bA)
    part_B = thr_mma.partition_B(bB)
    part_C = thr_mma.partition_C(bC)

    # === Allocate fragments (registers) ===
    frag_A = thr_mma.make_fragment_A(bA)
    frag_B = thr_mma.make_fragment_B(bB)
    frag_C = thr_mma.make_fragment_C(bC)

    # Retile fragments for copy compatibility
    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)

    # === Execute: Load A,B -> GEMM -> Store C ===
    fx.copy(copy_atom, copy_src_A, copy_frag_A, pred=None)
    fx.copy(copy_atom, copy_src_B, copy_frag_B, pred=None)
    frag_C.fill(0)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
    fx.copy(copy_atom, copy_frag_C, copy_dst_C, pred=None)


@flyc.jit
def launch_fn(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    #tile_size = BLOCK_DIM * VEC_WIDTH
    BLOCK_DIM = 256
    gemm_kernel(A, B, C).launch(
        grid=(1, 1, 1), block=(BLOCK_DIM, 1, 1), stream=stream
    )



# === Test ===
M = block_m   # 64
N = block_n   # 64
K = block_k   # 8  (no K-loop in kernel, must equal block_k)

A = torch.randn(M, K, dtype=torch.float32, device="cuda")
B = torch.randn(N, K, dtype=torch.float32, device="cuda")
C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
launch_fn(A, B, C, stream=torch.cuda.Stream())
torch.cuda.synchronize()

ref = A @ B.T
assert torch.allclose(ref, C, atol=1e-5, rtol=1e-5), f"max error {torch.max(torch.abs(ref - C))}"
print("PASSED")
