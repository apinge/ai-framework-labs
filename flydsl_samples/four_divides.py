"""
FlyDSL 的四种 divide 操作演示

对一个 layout 按 tile 尺寸切分，四种方式只是结果的 **分组/排列** 不同，
覆盖的元素和索引映射完全相同。

原始 layout shape: (M, N, ...)
Tiler shape:       <TileM, TileN>
RestM = M/TileM,  RestN = N/TileN

四种 divide 结果的 shape:
  logical_divide : ((TileM, RestM), (TileN, RestN), ...)    每个 mode 独立分成 tile+rest
  zipped_divide  : ((TileM, TileN), (RestM, RestN, ...))    tile 的 zip 到一起, rest 的 zip 到一起
  tiled_divide   : ((TileM, TileN), RestM, RestN, ...)      tile 的 zip, rest 保持独立
  flat_divide    : (TileM, TileN, RestM, RestN, ...)         全部展平

可以这样理解它们的关系:
  logical_divide 是基础
  zipped_divide  = logical_divide → 把所有 tile 子 mode zip 在一起, 所有 rest 子 mode zip 在一起
  tiled_divide   = logical_divide → 只把 tile 子 mode zip 在一起, rest 保持独立 mode
  flat_divide    = logical_divide → 全部展平成独立 mode

用法:
  FLYDSL_RUNTIME_ENABLE_CACHE=0 python four_divides.py
"""

import os

os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

import flydsl.compiler as flyc
import flydsl.expr as fx


# ===========================================================================
# 例 1: 2D layout, 2D tiler
# ===========================================================================
@flyc.jit
def demo_2d():
    print("=" * 60)
    print("例 1: 2D layout (6, 8) 按 tile (2, 4) 切分")
    print("=" * 60)

    layout = fx.make_layout((6, 8), (8, 1))  # 6 行 8 列, row-major
    tile = fx.make_tile(2, 4)

    print(f"原始 layout:       {layout}")
    # shape (6, 8), RestM=6/2=3, RestN=8/4=2

    ld = fx.logical_divide(layout, tile)
    print(f"logical_divide:    {ld}")
    # ((TileM, RestM), (TileN, RestN)) = ((2,3), (4,2))
    # 每个 mode 独立拆分

    zd = fx.zipped_divide(layout, tile)
    print(f"zipped_divide:     {zd}")
    # ((TileM, TileN), (RestM, RestN)) = ((2,4), (3,2))
    # tile 维 zip 在一起, rest 维 zip 在一起
    # mode 0 = 一个 tile 内部的坐标
    # mode 1 = tile 的编号 (哪个 block)

    td = fx.tiled_divide(layout, tile)
    print(f"tiled_divide:      {td}")
    # ((TileM, TileN), RestM, RestN) = ((2,4), 3, 2)
    # tile 维 zip, 但 rest 各自独立

    fd = fx.flat_divide(layout, tile)
    print(f"flat_divide:       {fd}")
    # (TileM, TileN, RestM, RestN) = (2, 4, 3, 2)
    # 全部展平

    print()

    # --- 验证: 所有 divide 映射到同样的元素 ---
    # logical_divide: coord ((tm, rm), (tn, rn)) → 线性偏移
    # 等价于: 原始 coord (tm + rm*TileM, tn + rn*TileN) → 同一个线性偏移
    # 验证: 所有 divide 映射到同样的元素
    # logical_divide((0,1), (2,0)): tm=0, rm=1, tn=2, rn=0
    #   → 原始坐标 (0+1*2, 2+0*4) = (2, 2) → 2*8 + 2 = 18
    off_ld = ld((0, 1), (2, 0))
    off_orig = layout(2, 2)
    print(f"  logical_divide((0,1),(2,0)) = {off_ld.get_static_leaf_int}")
    print(f"  original layout(2,2)        = {off_orig.get_static_leaf_int}")

    print()

    # --- zipped_divide 的典型用法: 取某个 tile ---
    # Layout 用 __call__ + None 做 slice（不是 __getitem__）
    # zipped_divide 的典型用法: 取某个 tile
    # zd shape = ((2,4), (3,2)), mode 0 = tile 内, mode 1 = tile 编号
    # 在 Tensor 上用 zipped_divide 时，tensor[None, (1,0)] 会带上 base offset
    # 但在纯 Layout 上，slice 只返回 tile 内部的相对 layout (shape+stride)
    # 想拿绝对偏移可以用 zd((tile_coord), (rest_coord)) 直接算
    print("zipped_divide 取 tile (1,0) 的绝对偏移:")
    print(f"  zd((0,0), (1,0)) = {zd((0, 0), (1, 0)).get_static_leaf_int}")  # 原始(2,0) = 16
    print(f"  zd((1,3), (1,0)) = {zd((1, 3), (1, 0)).get_static_leaf_int}")  # 原始(3,3) = 27

    print()


# ===========================================================================
# 例 2: 1D layout, 1D tiler
# ===========================================================================
@flyc.jit
def demo_1d():
    print("=" * 60)
    print("例 2: 1D layout (12,) 按 tile (4,) 切分")
    print("=" * 60)

    layout = fx.make_layout(12, 1)  # 12 个元素, stride=1
    tile = fx.make_tile(4)

    print(f"原始 layout:       {layout}")

    ld = fx.logical_divide(layout, tile)
    print(f"logical_divide:    {ld}")
    # ((4, 3)) — 一个 mode 里: tile_size=4, rest=3

    zd = fx.zipped_divide(layout, tile)
    print(f"zipped_divide:     {zd}")
    # (4, 3) — tile 内 4 个元素, 3 个 tile

    td = fx.tiled_divide(layout, tile)
    print(f"tiled_divide:      {td}")
    # (4, 3) — 同上 (1D 时和 zipped 一样)

    fd = fx.flat_divide(layout, tile)
    print(f"flat_divide:       {fd}")
    # (4, 3) — 同上 (1D 时全部一样)

    print()

    # 取第 2 个 tile (index=1)
    print("zipped_divide 取 tile 1:")
    tile_1 = zd(None, 1)
    print(f"  tile 1 layout = {tile_1}")
    # tile 1 = 元素 [4,5,6,7], 偏移 4,5,6,7

    print()


# ===========================================================================
# 例 3: 3D layout — 看 zipped vs tiled vs flat 的差异
# ===========================================================================
@flyc.jit
def demo_3d():
    print("=" * 60)
    print("例 3: 3D layout (4, 6, 2) 按 tile (2, 3) 切分前两个 mode")
    print("=" * 60)

    layout = fx.make_layout((4, 6, 2), (12, 2, 1))
    tile = fx.make_tile(2, 3)

    print(f"原始 layout:       {layout}")

    ld = fx.logical_divide(layout, tile)
    print(f"logical_divide:    {ld}")
    # ((2,2), (3,2), 2) — 每个 mode 独立分, 第 3 个 mode 不受影响

    zd = fx.zipped_divide(layout, tile)
    print(f"zipped_divide:     {zd}")
    # ((2,3), (2,2,2)) — tile zip 在一起, rest+剩余 mode 全 zip 在一起

    td = fx.tiled_divide(layout, tile)
    print(f"tiled_divide:      {td}")
    # ((2,3), 2, 2, 2) — tile zip, rest 和第 3 个 mode 各自独立

    fd = fx.flat_divide(layout, tile)
    print(f"flat_divide:       {fd}")
    # (2, 3, 2, 2, 2) — 全展平

    print()
    print("差异对比:")
    print(f"  logical  rank={ld.rank}  — 每个 mode 内部 (tile, rest)")
    print(f"  zipped   rank={zd.rank}  — mode 0=tile内, mode 1=tile编号+其他")
    print(f"  tiled    rank={td.rank}  — mode 0=tile内, 后面各自独立")
    print(f"  flat     rank={fd.rank}  — 全部平铺")
    print()


# ===========================================================================
# 运行
# ===========================================================================
demo_2d()
demo_1d()
demo_3d()
print("ALL DONE")
