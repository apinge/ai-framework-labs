# permutation K layout `(4, 4*4, 2):(1, 8, 4)` 解析

来源: `splitk_gemm/test_gemm_cdna4.py` 第 79-85 行

```python
tiled_mma = fx.make_tiled_mma(
    fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),       # ① atom
    fx.make_layout((1, 1, 4), (0, 0, 1)),                            # ② atom_layout
    fx.make_tile(None, None, fx.make_layout((4, 4 * 4, 2), (1, 8, 4))) # ③ permutation
)
```

## 1. `make_tiled_mma` 三个参数的分工

| 参数 | 回答什么问题 | 谁决定 |
|---|---|---|
| ① mma_atom | 一条 MFMA 指令的 M×N×K、线程数、ThrVal 映射 | 硬件（AMD ISA） |
| ② atom_layout | 多少个 atom 沿 M/N/K 排列 | kernel 作者（根据 wave 数、split-K 策略） |
| ③ permutation | 每个方向覆盖多大范围，K 方向的层次分解 | kernel 作者（根据 MFMA 消耗模式） |

## 2. 各参数具体值

### ① `MFMA(16, 16, 16, BFloat16)` — 一条硬件指令

`mfma_f32_16x16x16bf16_1k`，消耗 A[16×16] × B[16×16] → C[16×16]。

atom 内部的 A/B ThrVal layout（硬件写死）:

```
来源: lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp:17-28

MN = 16
GroupK = 64 / MN = 4       # 64 线程分成 4 组，每组 16 线程
KPerThread = K / GroupK = 4  # 每个线程持有 4 个 K 元素

ThrVal layout = ((16, 4), 4) : ((1, 64), 16)
                  ↑thread↑    ↑value↑

64 线程 × 4 values = 256 元素 = 16 × 16 = M × K ✓
```

### ② `(1, 1, 4):(0, 0, 1)` — atom 排列

shape = (M_atoms=1, N_atoms=1, K_atoms=**4**)

Split-K: 4 个 wave 全部沿 K 方向排列，不在 M/N 方向复制 atom。

总线程 = 每 atom 64 线程 × 4 atoms = **256 = 4 wavefronts**。

### ③ `Tile(None, None, (4, 16, 2):(1, 8, 4))` — permutation

- M = `None` → 默认覆盖 = atomM × atomLayoutM = 16 × 1 = **16**
- N = `None` → 默认覆盖 = 16 × 1 = **16**
- K = layout `(4, 16, 2):(1, 8, 4)` → 自定义层次分解，size = **128**

## 3. `(4, 16, 2):(1, 8, 4)` 到底是什么

### 3.1 它不是数据重排列

展开这个 layout 的所有坐标:

```python
# coord (i, j, k) → linear index = i*1 + j*8 + k*4
# i ∈ [0,4), j ∈ [0,16), k ∈ [0,2)

(0,0,0)→0  (1,0,0)→1  (2,0,0)→2  (3,0,0)→3
(0,0,1)→4  (1,0,1)→5  (2,0,1)→6  (3,0,1)→7
(0,1,0)→8  (1,1,0)→9  (2,1,0)→10 (3,1,0)→11
(0,1,1)→12 ...
```

**结果就是 0, 1, 2, ..., 127** — 和 `(128):(1)` 在覆盖范围上完全等价。

### 3.2 验证：遍历结果是连续的

```python
vals = []
for j in range(16):
    for k in range(2):
        for i in range(4):
            vals.append(i*1 + j*8 + k*4)
assert vals == list(range(128))  # ✓
```

### 3.3 那为什么不直接写 `None` 或 `128`？

因为 **层次结构在 `logical_divide` 时会被保留**。

`make_tiled_mma` 内部用 permutation 做 `logical_divide(trgLayout, permutation2D)`
（见 `include/flydsl/Dialect/Fly/Utils/TiledOpUtils.h:80`），
三级结构会影响后续 `zipped_divide` 和 `composition` 的结果，
最终决定每个线程在每次 MFMA 调用时读哪些 K 位置。

### 3.4 三级结构和 MFMA 消耗模式的对应

```
Level 1:  shape=4,  stride=1   → 一个线程连续读 4 个 K 元素
                                  MFMA K=16, GroupK=4, KPerThread=16/4=4
                                  每个线程组内 4 个连续 K 值

Level 2:  shape=16, stride=8   → 16 个线程组交错排列
                                  4 atoms × 每 atom 4 groups = 16 组
                                  stride=8: 每组占 8 个 K 位置（4 元素 × 2 半）

Level 3:  shape=2,  stride=4   → 重复 2 次 MFMA 调用
                                  每 atom 调 2 次 MFMA 消耗 K=32
                                  stride=4: 插在 Level 1 的 4 元素后面
```

总覆盖: 4 × 16 × 2 = **128** K 元素/步。

## 4. permutation 对 K 覆盖范围的影响

| permutation K | K 覆盖 | 内层循环次数 | 代码对应 |
|---|---|---|---|
| `None` | atomK(16) × atomLayoutK(4) = 64 | 256/64 = 4 | — |
| `(4,16,2):(1,8,4)` | size = 128 | 256/128 = **2** | `for sub_k in range_constexpr(TILE_K // 32)` → 64//32=2 |

> flat_divide 用 `TILE_K * 4 = 256`（第 72 行），每步的 K 覆盖从 permutation 的 size 决定。

## 5. A 和 B 的 K 维度如何各自处理

### B（pre-shuffled）— 物理排布由 layout 描述

```python
# 第 62 行
arg_b layout = ((16, N//16), (8, K//8)) : ((8, 16*K), (1, 128))
```

`shuffle_weight` 把 B 重排成 `(N//16, K//32, 4, 16, 8)` 的物理布局
（见 `/opt/venv/lib/python3.12/site-packages/aiter/ops/shuffle.py:7-27`），
`make_view` 的自定义 layout 精确描述了这个排布。

### A（row-major）— 没有 shuffle

```python
# 第 51 行
arg_a layout = (M, K) : (K, 1)
```

K 方向连续，permutation 的层次切分作用在连续数据上，读出来的 pattern 自然对齐。

### 共享同一个 permutation

A 和 B 共享同一个 `tiled_mma`，所以 permutation 对两者的 **K 维度逻辑切分方式完全一样**。
区别只在于 trgLayout 不同：
- A 的 K stride=1（连续），logical_divide 后 thread 读连续地址
- B 的 K stride=128（shuffled），logical_divide 后 thread 读 shuffled 地址

**permutation 不是为了适配 shuffle；它是为了匹配 MFMA 的 K 消耗层次。
shuffle 是独立地把 B 排成物理连续（对 buffer_load 合并友好），两者目标一致但各管各的。**

## 6. 注释里的原文对照

```python
# K维度((4*2)*4)*4，线程分布依次为：
#   连续的4点为一个lane，            → Level 1: shape=4, stride=1
#   跳过8点为一组并重复4组*4wave，    → Level 2: shape=4*4=16, stride=8
#   最后为重复第二次调用              → Level 3: shape=2, stride=4
```

## 7. 相关代码路径

| 文件 | 行 | 内容 |
|---|---|---|
| `lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp` | 17-28 | MFMA atom A/B ThrVal layout 定义 |
| `lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp` | 63-73 | MFMA atom C ThrVal layout 定义 |
| `include/flydsl/Dialect/Fly/Utils/TiledOpUtils.h` | 71-137 | `layoutTiledMmaThrValView`: permutation → logical_divide → zipped_divide → compose |
| `include/flydsl/Dialect/Fly/Utils/TiledOpUtils.h` | 265-318 | `layoutTiledMmaThrValOperandView`: None → default size, layout → custom tiler |
| `python/flydsl/expr/primitive.py` | 933-936 | `make_tiled_mma` Python 入口 |
| `python/flydsl/expr/primitive.py` | 1214-1231 | `make_tile` 实现 |
| `docs/cute_layout_algebra_guide.md` | 339 | `make_tiled_mma` 用法示例 |
| `examples/03-tiledMma.py` | 35-36 | 简单 tiled_mma 示例 (2×2 atom, 无 permutation) |
