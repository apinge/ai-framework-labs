import os
import tempfile

# print_typst writes its .typ files as a trace-time side effect of @flyc.jit, and the
# layout cells print at trace time too. A warm JIT disk cache would skip the re-trace
# (and those side effects), so disable it — re-runs then always reproduce. Set before
# importing flydsl.
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

import torch
from IPython.display import SVG, Code, display

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import typst  # `pip install typst` — bundles the compiler
except ImportError:
    typst = None  # without it, diagrams fall back to raw source

DDD = 0
@flyc.kernel
def tiled_copy_kernel(A: fx.Tensor, B: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    A = fx.Tensor(fx.make_view(
            fx.get_iter(A),
            fx.make_layout((8, 24), (24, 1))
        ))
    B = fx.Tensor(fx.make_view(
            fx.get_iter(B),
            fx.make_layout((8, 24), (24, 1))
        ))
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    # zipped_divide(A, (8, 24))：在逻辑上把 A 按 8×24 的小矩形 切成很多 tile（形状由 (8,24) 决定）。
    bA = fx.slice(fx.zipped_divide(A, (8, 24)), (None, bid))
    bB = fx.slice(fx.zipped_divide(B, (8, 24)), (None, bid))
    # trace 时打印整块 tile 的 memref 布局与元素个数（与 04_layout 里对 layout 的 size/cosize 一致）
    lyA, lyB = fx.get_layout(bA), fx.get_layout(bB)
    if DDD:
        print("  bA layout:", lyA, "| size:", fx.size(lyA), "| cosize:", fx.cosize(lyA))
        print("  bB layout:", lyB, "| size:", fx.size(lyB), "| cosize:", fx.cosize(lyB))

    thr = fx.make_layout((4, 1), (1, 1)) #  block=(4,1,1) 其实也就4个线程
    val = fx.make_layout((1, 8), (1, 1)) # 每个线程8个元素
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    tile_mn, tv = fx.make_layout_tv(thr, val)
    tiled = fx.make_tiled_copy(atom, tv, tile_mn)
    
    # TV layout  : Layout<(4,8):(1,4)>
    if DDD:
        fx.utils.print_typst(tiled, file="tiled_in_copy.typ")
    # Python 侧类为 flydsl.expr.typing.TiledCopy（见 typing.py）；repr 里是带 copy_atom / TV layout / tile 的 MLIR 类型串
    print("  tiled:", tiled)
   # tiled：静态结构信息 + 怎么分区，不是「已经执行完的一次计算」
    # fx.copy(...)：按这个计划 实际发生的内存操作（在 trace 里会进 MLIR，最后变成指令）
   # 若用类比：tiled 像调度表，copy 才是干活的那一步。
    thr_copy = tiled.get_slice(tid)
    psrc = thr_copy.partition_S(bA)
    pdst = thr_copy.partition_D(bB)
    if DDD:
        print("  per-thread source view:", psrc)  # trace-time: prints once, during compile


    if True:
        z = fx.Int32(0)
        if (bid == z) & (tid == z):
            psrc_layout = fx.get_layout(psrc)
            fx.printf("psrc_layout{}",psrc_layout)
            fx.printf("{}",psrc[(None,None),None,None].load())
    if DDD:
        z = fx.Int32(0)
        psrc_layout = fx.get_layout(psrc)
        if (bid == z) & (tid == z):
            #psrc_layout = fx.get_layout(psrc)
            print(psrc_layout)
            fx.printf("[psrc dbg] bid=0 tid=0: k -> v (layout flat order; values may repeat vs global row scan)\n")
            fx.printf("{}",psrc[(0,None),None,1].load())
            for k in fx.range(96*2):
                #crd = fx.get_flat_coord(k, psrc_layout)
                #vk = fx.memref_load(psrc, crd)
                #fx.printf("[psrc dbg] k={} v={} crd={}\n", fx.Int32(k), vk,crd)
                crd = fx.idx2crd(fx.Int32(k),psrc_layout)
                #crd_flat = fx.get_flat_coord(fx.Int32(k), psrc_layout)
                vk = fx.memref_load(psrc, crd)
                fx.printf("{}: [crd] crd = {},  vk={}",k,crd, vk) 
            # for z in fx.range_constexpr(4):
            #     for i in fx.range_constexpr(2):
            #         for j in fx.range_constexpr(2):
            #             for k in fx.range_constexpr(3):
            #                 offset = fx.idx2crd(fx.make_coord((z,i),j,k),layout=psrc_layout)
            #                 #fx.printf("[crd] i{},j{},k{},offset = {}",i,j,k,offset) 
            #                 print("[crd] i{},j{},k{},offset = {}",z,i,j,k,offset) 
                        

    frag = fx.make_fragment_like(psrc)
    fx.copy(atom, psrc, frag)
    fx.copy(atom, frag, pdst)
    #fx.copy(atom,  psrc, pdst)


@flyc.jit
def run_copy(A: fx.Tensor, B: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    tiled_copy_kernel(A, B).launch(grid=(1, 1, 1), block=(4, 1, 1), stream=stream)
    psrc_layout = fx.make_layout(((4,2),2,3),((1,4),96,8))
    # for z in fx.range_constexpr(4):
    #     for i in fx.range_constexpr(2):
    #         for j in fx.range_constexpr(2):
    #             for k in fx.range_constexpr(3):
    #                 offset = fx.idx2crd(fx.make_coord((z,i),j,k),layout=psrc_layout)
    #                         #fx.printf("[crd] i{},j{},k{},offset = {}",i,j,k,offset) 
    #                 print("[crd] i{},j{},k{},offset = {}",z,i,j,k,offset) 


#M, N = 8 * 3, 24 * 5
M, N = 8 * 1, 24 * 1
A = torch.arange(M * N, dtype=torch.float32).reshape(M, N).cuda()
B = torch.zeros(M, N, dtype=torch.float32).cuda()
run_copy(A, B, stream=torch.cuda.Stream())
torch.cuda.synchronize()
print("copy correct:", bool(torch.allclose(A, B)))