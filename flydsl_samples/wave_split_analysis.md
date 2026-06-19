# Wave 是在哪里切的？—— 从 tid 到数据地址的完整映射

## 问题

`small_gemm_bf16.py`：block_m=64, block_n=64, block_k=32，256 个线程，4 个 wave。

代码里没有任何显式的 "wave 分割" 逻辑，但不同 wave 确实在处理不同的数据。
切分发生在哪里？

**答案：切分藏在 `tiled_copy` 和 `tiled_mma` 的 TV layout 里。**

`tid` 被 TV layout 的 stride 拆解成多层索引，每层对应数据矩阵的一个维度。
不同 wave 的线程因为 `tid` 范围不同，自然映射到不同的数据区域。

---

## TV layout 是什么

TV = Thread × Value。它是一个 layout，描述 "线程 t 的第 v 个值" 对应数据矩阵的
哪个位置。

以 `tiled_copy_A` 为例，IR 里它的 TV layout 是：

```
((16, 4, 2, 2), (8, (1, 1))) : ((1, 256, 16, 0), (32, (0, 0)))
└─ Thread 维度 ─┘  └ Value ┘   └── Thread stride ──┘  └ Val stride ┘
```

Thread 维度 shape `(16, 4, 2, 2)` 的乘积 = 256 = 线程总数。
Value 维度 shape `(8, (1, 1))` 的乘积 = 8 = 每线程加载 8 个 bf16（= 128 bits）。

### 这个 TV layout 是怎么生成的

在 `00_origin.mlir` 里搜 `make_tiled_copy`，能直接看到这个 layout：

```bash
grep "make_tiled_copy" /root/.flydsl/debug/gemm_kernel_0/00_origin.mlir
```

```
fly.make_tiled_copy(%32, %34, %42) :
  (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<128>, 16>,
   !fly.layout<((16,4,2,2),(8,(1,1))):((1,256,16,0),(32,(0,0)))>,    ← 这就是 TV layout
   !fly.tile<[32:1|32:1]>)
```

它对应 Python 代码：

```python
tiled_copy_A = fx.make_tiled_copy_A(copy_atom_ab, tiled_mma)   # derived.py:139
```

#### 生成链条（3 层）

**第 1 层：MMA 硬件定义 atom 级别的 thread-value layout**

`MFMA(16, 16, 32, BF16)` 的 A operand layout 定义在
`lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp:17`：

```cpp
// getThrValLayoutAB()
int MN = 16;                          // M == N for MFMA
int GroupK = 64 / MN;                 // = 4 (64 threads / 16 rows)
int KPerThread = K / GroupK;          // = 32 / 4 = 8

Shape:  (Thr(16, 4), Val(8))          // 64 threads, 8 values per thread
Stride: (Thr(1, 16×8), Val(16))       // = (Thr(1, 128), Val(16))
```

这描述了**单个 MMA atom** 内 64 个线程怎么映射到 A[16, 32] 数据：
- 线程行位置：`tid_in_atom % 16`（stride 1）
- K group：`tid_in_atom / 16`（stride 128 = 16×8）
- 每线程 8 个连续 K 值：stride 16

**第 2 层：`atom_layout (2,2,1):(1,2,0)` 通过 tiled product 展开到 256 线程**

`lib/Bindings/Python/TiledOpTraits.cpp:122-124`：

```cpp
LayoutAttr atomThrLayout = cast<LayoutAttr>(mmaAtom.getThrLayout());  // (64):(1)
LayoutAttr thrLayoutVMNK = layoutTiledProduct(
    attrBuilder, atomThrLayout, materializeConstantLayout(atomLayoutMNK));
```

`layoutTiledProduct` 把 atom 的 64 线程和 atom_layout `(2,2,1):(1,2,0)` 做
"tiled product"——每个 atom 分配 64 线程，4 个 atom 得到 256 线程。
结果是一个 (V, M, N, K) 四维 layout，shape 为 `(64, 2, 2, 1)`，
告诉框架每个线程属于哪个 atom（M 方向第几个、N 方向第几个）。

**第 3 层：投影到 A operand 的 2D 坐标 + 和 copy atom 的 value 粒度对齐**

对于 A operand（`MmaOperand::A`），只涉及 M 和 K 两个维度（`idx0=0, idx1=2`）。
`TiledOpTraits.cpp:126-143` 把 (V, M, N, K) 中的 M 和 N 投影成 2D 偏移，
其中 A 的投影规则是：M stride=1, N stride=0（A 和 N 无关）。

然后通过 `layoutComposition`、`layoutRightInverse`、`layoutComplement` 等 layout
代数运算（`TiledOpTraits.cpp:149-165`），把 atom 内的 thread-value layout
和 atom 间的分布 layout 合并成最终的 TV layout。

#### 最终 TV layout 的 4 个 Thread 维度分别来自哪

```
Thread shape:  (16,     4,     2,       2      )
Thread stride: (1,      256,   16,      0      )
               │        │      │        │
               │        │      │        └─ 来自 atom_layout N 方向
               │        │      │           stride=2 → 映射到 N → A 不用 N → stride=0
               │        │      │
               │        │      └─ 来自 atom_layout M 方向
               │        │         stride=1 → 映射到 M → A 用 M → stride=16 (= mma_M)
               │        │
               │        └─ 来自 MMA atom 内的 K group
               │           atom 内 layout Thr(16, 4):Thr(1, 128)
               │           128 在 per-group tile 坐标系变成 256
               │
               └─ 来自 MMA atom 内的行位置
                  atom 内 layout Thr(16, ...):Thr(1, ...)
                  stride=1
```

最终被 `make_tiled_copy` 和 `BufferCopy128b` 的 value 粒度再做一次 retile，
产生 Value 维度 `(8, (1, 1)):(32, (0, 0))`——8 个连续 bf16，步长 32 对应
`copy_atom` 的 128b value layout 在 tile 坐标系中的偏移。

---

## tid 是怎么被拆成多层索引的

TV layout 的 Thread 维度 `(16, 4, 2, 2):(1, 256, 16, 0)` 是一个嵌套 layout。
给定 `tid`，框架按 shape 从内到外逐层取模、整除：

```
t0 = tid % 16          →  stride 1    →  用于 MMA atom 内的行/列位置
t1 = (tid / 16) % 4    →  stride 256  →  用于 K 方向的 group
t2 = (tid / 64) % 2    →  stride 16   →  用于 M 方向的 atom 分布
t3 = (tid / 128) % 2   →  stride 0    →  不贡献偏移（N 方向重复）
```

### 对应的 lowered IR

在 `08_convert_fly_to_rocdl.mlir` 里可以直接看到这些运算（`%6` = tid）：

```mlir
%54 = arith.remsi %6, %c16_i32       // t0 = tid % 16
%55 = arith.divsi %6, %c16_i32       // tid / 16
%58 = arith.remsi %55, %c4_i32       // t1 = (tid/16) % 4
%59 = arith.divsi %55, %c4_i32       // tid / 64
%63 = arith.remsi %59, %c2_i32       // t2 = (tid/64) % 2
```

这就是切 wave 的地方——不是一个专门的 "切分函数"，而是 **TV layout 的 stride
在 lowering 时自动生成的取模/整除运算链**。

---

## 每层索引控制什么

### A 矩阵的 TV layout：`((16,4,2,2),(8,(1,1))):((1,256,16,0),(32,(0,0)))`

`bA` 的数据 shape 是 `(64, 32)`（M × K），row-major，stride = `(stride_A, 1)`。

TV layout 的 stride 是**在 per-group tile 坐标系**里的偏移。per-group tile = `(32, 32)`。

| 索引 | 值域 | stride | 含义 |
|---|---|---|---|
| t0 = `tid % 16` | 0..15 | 1 | A 的行位置（在 atom 内的 M 位置） |
| t1 = `(tid/16) % 4` | 0..3 | 256 | K 方向的 group（但 stride 256 超出 32×32 tile，被 layout lowering 展开为具体偏移） |
| t2 = `(tid/64) % 2` | 0..1 | 16 | **M 方向的 atom 偏移**——这就是切 wave 的关键 |
| t3 = `(tid/128) % 2` | 0..1 | 0 | 不贡献偏移——两个 N-wave 读同样的 A 数据 |

实际 lowered IR 里 A 的 load 地址：

```
A_base_offset = t0 × stride_A + t1 × 8 + t2 × stride_A × 16
              = (tid%16) × stride_A + ((tid/16)%4) × 8 + ((tid/64)%2) × stride_A × 16
                └── atom 内行 ───┘   └── K group ──┘   └── wave 间的 M 偏移 ──┘
```

### 具体的 wave 分区

| Wave (tid 范围) | t2 = (tid/64)%2 | t3 = (tid/128)%2 | A 负责的 M 区域 |
|---|---|---|---|
| Wave 0: tid 0-63 | 0 | 0 | M rows 0..15 |
| Wave 1: tid 64-127 | 1 | 0 | M rows 16..31 |
| Wave 2: tid 128-191 | 0 | 1 | M rows 0..15（和 Wave 0 相同） |
| Wave 3: tid 192-255 | 1 | 1 | M rows 16..31（和 Wave 1 相同） |

**Wave 0 和 Wave 2 读的是同一块 A 数据**（t3 stride=0 不贡献偏移），但它们在 B
和 C 上负责不同的 N 区域。

### B 矩阵的 TV layout：`((16,4,2,2),(8,(1,1))):((1,256,0,16),(32,(0,0)))`

注意和 A 的区别：**t2 的 stride 是 0，t3 的 stride 是 16**。正好反过来。

| 索引 | stride（A） | stride（B） | 含义 |
|---|---|---|---|
| t2 = (tid/64)%2 | **16** | 0 | A: 区分 M-wave；B: 不区分 |
| t3 = (tid/128)%2 | 0 | **16** | A: 不区分；B: 区分 N-wave |

所以：

| Wave | t2 | t3 | A 的 M 区域 | B 的 N 区域 |
|---|---|---|---|---|
| Wave 0: 0-63 | 0 | 0 | M[0:16] | N[0:16] |
| Wave 1: 64-127 | 1 | 0 | M[16:32] | N[0:16] |
| Wave 2: 128-191 | 0 | 1 | M[0:16] | N[16:32] |
| Wave 3: 192-255 | 1 | 1 | M[16:32] | N[16:32] |

**这就是 `atom_layout (2,2,1):(1,2,0)` 的实际效果**——把 4 个 wave 按 2×2 分布到
M×N 方向。每个 wave 在 copy tile (32×32) 内负责 16×16 的 atom 位置。

---

## 图解：256 个线程怎么覆盖 64×64 的 C 矩阵

### copy tile 和 rep 的关系

首先要理解两个层次：

1. **copy tile (32×32)**：TV layout 覆盖的基本单元。256 个线程 × 4 values = 1024
   = 32×32。**所有 4 个 wave 协作覆盖同一个 32×32 copy tile**，每个 wave 负责其中
   一个 16×16 象限。

2. **M_rep=2, N_rep=2**：把 32×32 的 copy tile 重复 2×2=4 次，覆盖完整的 64×64。
   每次重复，ALL 4 waves 一起参与，各自写自己的 16×16 块。

### 正确的分布图

```
C 矩阵 (64×64) — 每个 wave 的 4 条 MFMA 输出位置 (棋盘格分布)

              col 0-15   col 16-31   col 32-47   col 48-63
            ┌──────────┬──────────┬──────────┬──────────┐
row  0-15   │  Wave 0  │  Wave 2  │  Wave 0  │  Wave 2  │
            ├──────────┼──────────┼──────────┼──────────┤
row 16-31   │  Wave 1  │  Wave 3  │  Wave 1  │  Wave 3  │
            ├──────────┼──────────┼──────────┼──────────┤
row 32-47   │  Wave 0  │  Wave 2  │  Wave 0  │  Wave 2  │
            ├──────────┼──────────┼──────────┼──────────┤
row 48-63   │  Wave 1  │  Wave 3  │  Wave 1  │  Wave 3  │
            └──────────┴──────────┴──────────┴──────────┘

Wave 0 的 4 个 16×16 块:  (0:16,0:16), (0:16,32:48), (32:48,0:16), (32:48,32:48)
Wave 1 的 4 个 16×16 块: (16:32,0:16),(16:32,32:48),(48:64,0:16),(48:64,32:48)
Wave 2 的 4 个 16×16 块:  (0:16,16:32),(0:16,48:64),(32:48,16:32),(32:48,48:64)
Wave 3 的 4 个 16×16 块: (16:32,16:32),(16:32,48:64),(48:64,16:32),(48:64,48:64)
```

**每个 wave 的 4 条 MFMA 输出是棋盘格散布在 64×64 里的 4 个 16×16 块，
不是一个连续的 32×32 象限。**

### 为什么是棋盘格而不是连续象限

关键在于 M_rep/N_rep 的偏移量 = **整个 atom_layout 块的大小** (32)，不是单个 atom
的大小 (16)。以 wave 0 (atom M=0, N=0) 为例：

```
wave 0 atom 位置 = 16×16 块的 (行0-15, 列0-15)

M_rep/N_rep 偏移 = 32 (= 2 atoms × 16)

(M_rep=0, N_rep=0): 行  0 + 0  = 行 0-15,  列  0 + 0  = 列 0-15   ✓ 左上
(M_rep=0, N_rep=1): 行  0 + 0  = 行 0-15,  列  0 + 32 = 列 32-47  ← 跳到第3列块!
(M_rep=1, N_rep=0): 行  0 + 32 = 行 32-47, 列  0 + 0  = 列 0-15   ← 跳到第3行块!
(M_rep=1, N_rep=1): 行  0 + 32 = 行 32-47, 列  0 + 32 = 列 32-47
```

如果 rep 偏移是 16（单个 atom 大小），结果会是连续 32×32 象限。
但实际偏移是 32（atom_layout 块大小），所以跳过了相邻 wave 的 atom，形成棋盘格。

---

## tiled_copy_C 的 TV layout 详细分析

### TV layout

```
((16,8,2),((4,1),(1,1))):((32,4,512),((1,0),(0,0)))
```

### 分析方法：从 lowered IR 追踪 store 地址

**第 1 步：找 tiled_copy_C 的 tid 分解**

在 `08_convert_fly_to_rocdl.mlir` 中搜 `tid` 相关的取模/整除运算。
tiled_copy_C 的 thread shape 是 (16, 8, 2)，所以分解是：

```
t0_C = tid % 16        → 16 个位置
t1_C = (tid / 16) % 8  → 8 个位置
t2_C = tid / 128       → 2 个位置
```

注意 t0_C 和 tiled_copy_A 的 t0 相同 (都是 tid%16)，但 t1_C 和 t2_C 不同。

对应 IR (`08_convert_fly_to_rocdl.mlir`):

```mlir
// t0_C = tid % 16 (复用之前算好的 %54/%56)
%54 = arith.remsi %6, %c16_i32       // %6 = tid
%56 = arith.extsi %54 : i32 to i64

// t1_C = (tid/16) % 8
%77 = arith.remsi %55, %c8_i32       // %55 = tid/16

// t2_C = (tid/16) / 8 = tid/128
%78 = arith.divsi %55, %c8_i32
```

**第 2 步：追踪地址公式**

IR 中 C 的 base offset 计算 (loc #loc4):

```mlir
%76 = arith.muli %1, %c4_i64         // ldc × 4
%80 = arith.muli %79, %76            // t1_C × ldc × 4    → 行偏移
%81 = arith.addi %56, %80            // t0_C + t1_C × ldc × 4
%82 = arith.muli %78, %c16_i32       // t2_C × 16         → 列偏移
%84 = arith.addi %81, %83            // t0_C + t1_C × ldc × 4 + t2_C × 16
```

得到地址公式：

```
C_base = t0_C + t1_C × ldc × 4 + t2_C × 16
```

在 C 矩阵 (64×64), stride (ldc, 1) = (64, 1) 下：
- t0_C: 列偏移 (0..15)
- t1_C × 4: 行偏移 (t1_C=0..7 → row=0,4,8,...,28)
- t2_C × 16: 列偏移 (0 或 16)

所以每线程的 base 位置：
```
row = t1_C × 4           (范围 0, 4, 8, ..., 28)
col = t0_C + t2_C × 16   (范围 0..31)
```

**第 3 步：追踪 value 和 rep 的地址偏移**

逐条 store 的地址 (16 条总共)：

```
store #   frag_C index   offset                  位置
──────    ────────────   ──────────────────      ──────────────────
 1         [0]           base                    (row,     col)
 2         [1]           base + ldc              (row+1,   col)
 3         [2]           base + 2×ldc            (row+2,   col)
 4         [3]           base + 3×ldc            (row+3,   col)
                                                 ↑ 4 个 val, 4 连续行

 5         [8]           base + 32×ldc           (row+32,  col)
 6         [9]           base + 33×ldc           (row+33,  col)
 7        [10]           base + 34×ldc           (row+34,  col)
 8        [11]           base + 35×ldc           (row+35,  col)
                                                 ↑ M_rep=1, 行 +32

 9         [4]           base + 32               (row,     col+32)
10         [5]           base + ldc + 32         (row+1,   col+32)
11         [6]           base + 2×ldc + 32       (row+2,   col+32)
12         [7]           base + 3×ldc + 32       (row+3,   col+32)
                                                 ↑ N_rep=1, 列 +32

13        [12]           base + 32×ldc + 32      (row+32,  col+32)
14        [13]           base + 33×ldc + 32      (row+33,  col+32)
15        [14]           base + 34×ldc + 32      (row+34,  col+32)
16        [15]           base + 35×ldc + 32      (row+35,  col+32)
                                                 ↑ M_rep=1 + N_rep=1
```

关键发现：
- **M_rep 偏移 = 32 行** (= atom_layout_M × mma_M = 2×16)
- **N_rep 偏移 = 32 列** (= atom_layout_N × mma_N = 2×16)
- 这两个偏移都是 **整个 atom_layout 块的大小**，不是单个 atom 的大小

**第 4 步：代入具体 wave 验证**

| Wave | tid | t0_C | t1_C | t2_C | copy tile 内 row | copy tile 内 col |
|---|---|---|---|---|---|---|
| 0 | 0-63 | 0..15 | 0..3 | 0 | 0..15 | 0..15 |
| 1 | 64-127 | 0..15 | 4..7 | 0 | 16..31 | 0..15 |
| 2 | 128-191 | 0..15 | 0..3 | 1 | 0..15 | 16..31 |
| 3 | 192-255 | 0..15 | 4..7 | 1 | 16..31 | 16..31 |

每个 wave 在 copy tile (32×32) 内占 16×16。然后 M_rep/N_rep 各 +32，
所以每个 wave 在 64×64 中写 4 个分散的 16×16 块（棋盘格）。

以 **Wave 0** (row=0..15, col=0..15 in copy tile) 为例：
- (M_rep=0, N_rep=0): C[ 0:16,  0:16]
- (M_rep=1, N_rep=0): C[32:48,  0:16]   ← 行 +32, 跳过 wave 1 的行 16-31
- (M_rep=0, N_rep=1): C[ 0:16, 32:48]   ← 列 +32, 跳过 wave 2 的列 16-31
- (M_rep=1, N_rep=1): C[32:48, 32:48]

---

## atom_layout stride 和 wave 分布的关系

`atom_layout = (2, 2, 1):(1, 2, 0)` 怎么变成上面的 wave 分布？

atom_layout 的 stride 告诉框架：把哪些 atom 分给不同的 wave-group。

```
atom_layout shape:  (M_atoms=2, N_atoms=2, K_atoms=1)
atom_layout stride: (1,         2,         0)
```

group_id = m_atom × 1 + n_atom × 2 + k_atom × 0

| m_atom | n_atom | group_id | 对应 Wave |
|---|---|---|---|
| 0 | 0 | 0×1 + 0×2 = 0 | Wave 0 (tid 0-63) |
| 1 | 0 | 1×1 + 0×2 = 1 | Wave 1 (tid 64-127) |
| 0 | 1 | 0×1 + 1×2 = 2 | Wave 2 (tid 128-191) |
| 1 | 1 | 1×1 + 1×2 = 3 | Wave 3 (tid 192-255) |

框架生成 TV layout 时：
- `m_atom` 索引 (t2) → M 方向偏移 16 (= mma_M)
- `n_atom` 索引 (t3) → N 方向偏移 16 (= mma_N)

这决定了每个 wave 在 copy tile 内的 16×16 位置。然后 rep 以 32 (= 2×16) 为步长
重复，形成棋盘格。

---

## 验证方法

### 方法 1：从 lowered IR 追踪 store/load 地址（最可靠）

```bash
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 your_kernel.py
```

步骤：
1. 在 `08_convert_fly_to_rocdl.mlir` 中找 `remsi`/`divsi` 链 → tid 分解
2. 找 `arith.muli` / `arith.addi` 链 → 地址公式 (如 `t0 + t1*ldc*4 + t2*16`)
3. 找 `buffer.store` 前的 offset → 逐条追踪 rep 偏移
4. 代入具体 wave 的 tid 范围 → 得到每个 wave 实际写到哪些 (row, col)

本文的 C 矩阵棋盘格分布就是这样推导出来的。

### 方法 2：看 `00_origin.mlir` 里 `make_tiled_copy` 的 TV layout

```bash
grep "make_tiled_copy" /root/.flydsl/debug/gemm_kernel_0/00_origin.mlir
```

TV layout 的 thread 维度 shape 就是取模链的 shape。stride 非零的维度
会在数据上产生偏移，stride 为 0 的维度意味着多个 wave 读同样的数据。

### 方法 3：对照 atom_layout 推算每个 wave 的覆盖区域

```
1. 算 copy tile 大小:
   M_tile = M_atoms × mma_M    (= 2 × 16 = 32)
   N_tile = N_atoms × mma_N    (= 2 × 16 = 32)

2. 算 rep:
   M_rep = block_m / M_tile    (= 64 / 32 = 2)
   N_rep = block_n / N_tile    (= 64 / 32 = 2)

3. 每个 wave 在 copy tile 内的 atom 位置:
   M_start = m_atom × mma_M    (wave 0: 0,  wave 1: 16)
   N_start = n_atom × mma_N    (wave 0: 0,  wave 2: 16)
   大小: mma_M × mma_N = 16 × 16

4. rep 的偏移步长 = copy tile 大小 (不是 atom 大小!):
   M_rep_stride = M_tile = 32
   N_rep_stride = N_tile = 32

5. Wave i 在 64×64 中的 16×16 块:
   for m_rep in 0..M_rep-1:
     for n_rep in 0..N_rep-1:
       行: M_start + m_rep × M_rep_stride  到  + mma_M
       列: N_start + n_rep × N_rep_stride  到  + mma_N
```

关键区分：
- rep 步长 = **M_tile** (32) 而不是 mma_M (16) → 棋盘格
- 如果 atom_layout 只有 (1,1,1)（单 wave），M_tile = mma_M，rep 步长 = 16 → 连续

---

## tiled_copy_A 的 TV layout 详细分析

### TV layout

```
((16,4,2,2),(8,(1,1))):((1,256,16,0),(32,(0,0)))
```

### 第 1 步：找 tiled_copy_A 的 tid 分解

Thread shape (16,4,2,2) 的分解和 tiled_copy_C 共用同一组取模/整除运算：

```
t0 = tid % 16          → 用于 MMA atom 内的行位置
t1 = (tid / 16) % 4    → 用于 K 方向的 group
t2 = (tid / 64) % 2    → 用于 M 方向的 atom (= 切 wave 的关键)
t3 = (tid / 128) % 2   → stride=0, 不贡献偏移 (A 与 N 无关)
```

对应 IR (`08_convert_fly_to_rocdl.mlir` L86-98, loc #loc24):

```mlir
%54 = arith.remsi %6, %c16_i32       // t0 = tid % 16
%55 = arith.divsi %6, %c16_i32       // tid / 16
%56 = arith.extsi %54                // t0 (i64)
%57 = arith.muli %56, %5             // t0 × lda    → 行偏移
%58 = arith.remsi %55, %c4_i32       // t1 = (tid/16) % 4
%60 = arith.muli %58, %c8_i32        // t1 × 8      → K group 偏移
%62 = arith.addi %57, %61            // t0×lda + t1×8
%63 = arith.remsi %59, %c2_i32       // t2 = (tid/64) % 2
%65 = arith.muli %64, %53            // t2 × (lda×16) = t2×16×lda  → M atom 偏移
%66 = arith.addi %62, %65            // 最终: t0×lda + t1×8 + t2×16×lda
```

### 第 2 步：地址公式

```
A_base = t0 × lda + t1 × 8 + t2 × 16 × lda
```

A 矩阵是 row-major, stride = (lda, 1)，所以分解成 (row, col):

```
row = t0 + t2 × 16    (来自 t0×lda 和 t2×16×lda 的系数)
col = t1 × 8 + v      (v = 0..7, 来自 value stride 32 在 colex (32,32) 中 = K 方向)
```

- t0 = 0..15: atom 内 16 行
- t1 = 0..3: 4 个 K group, 每组 8 列 → 覆盖 K 的全部 32 列
- t2 = 0 或 1: M 方向偏移 16 行
- v = 0..7: 每线程 8 个连续 K 值 (128 bits = 1 次 buffer load)
- **t3 不出现**: stride=0, Wave 在 N 方向重复读同一块 A

### 第 3 步：逐条 load 追踪

A 有 2 条 `buffer.load i128`：M_rep=0 和 M_rep=1。

```
load #   M_rep   offset                        位置
──────   ─────   ─────────────────────────      ──────────────────
 1        0      block_m_offset + A_base        (row,      col=t1×8..t1×8+7)
 2        1      block_m_offset + A_base        (row+32,   col=t1×8..t1×8+7)
                 + lda×32
```

对应 IR:
```
L140: rocdl.raw.ptr.buffer.load → i128 (8 bf16)
      offset = %28 + %66                     ← M_rep=0
      insert_strided_slice offsets=[0]       ← frag_A[0:8]

L161: rocdl.raw.ptr.buffer.load → i128 (8 bf16)
      offset = %28 + %66 + %52              ← M_rep=1 (加 lda×32)
      insert_strided_slice offsets=[8]       ← frag_A[8:16]
```

%52 = `%5 × 32` = `lda × 32` (L84)。所以 M_rep 步长 = 32 行，和 C 一样。

每线程 frag_A = 16 bf16 (8 per M_rep)。

### 第 4 步：代入具体 wave 验证

| Wave | tid | t0 | t1 | t2 | t3 | copy tile 内 row | copy tile 内 col |
|---|---|---|---|---|---|---|---|
| 0 | 0-63 | 0..15 | 0..3 | 0 | 0 | 0..15 | 0..31 |
| 1 | 64-127 | 0..15 | 0..3 | 1 | 0 | 16..31 | 0..31 |
| 2 | 128-191 | 0..15 | 0..3 | 0 | 1 | **0..15 = Wave 0** | 0..31 |
| 3 | 192-255 | 0..15 | 0..3 | 1 | 1 | **16..31 = Wave 1** | 0..31 |

**Wave 0 和 Wave 2 读完全相同的 A 数据** (t3 stride=0)。
**Wave 1 和 Wave 3 读完全相同的 A 数据** (t3 stride=0)。

加上 M_rep=2 (步长 32 行)，完整 A 矩阵覆盖:

```
A 矩阵 (64M × 32K) — 每个 wave 的 load 区域

                     K=0..7   K=8..15  K=16..23 K=24..31
                    (t1=0)   (t1=1)   (t1=2)   (t1=3)
                  ┌────────┬────────┬────────┬────────┐
M_rep=0  M= 0..15 │       Wave 0 = Wave 2              │  t2=0
                  ├────────┴────────┴────────┴────────┤
         M=16..31 │       Wave 1 = Wave 3              │  t2=1
                  ├────────┬────────┬────────┬────────┤
M_rep=1  M=32..47 │       Wave 0 = Wave 2              │  t2=0, row+32
                  ├────────┴────────┴────────┴────────┤
         M=48..63 │       Wave 1 = Wave 3              │  t2=1, row+32
                  └───────────────────────────────────┘
```

**每个 wave 读 2 × 16 × 32 = 1024 bf16 (两个 M_rep, 16 行 × 32 K)**。
但实际只有 **2 组独立数据** (wave 0/2 一组, wave 1/3 一组)。

### 为什么 A 有冗余读取

A 矩阵只参与 M 和 K 的计算，不关心 N。而 atom_layout `(2,2,1):(1,2,0)` 在 N 方向
分了 2 个 wave-group (n_atom=0 和 n_atom=1)。这两组 wave 在 N 方向做不同的 MFMA，
但需要**相同的 A 数据**：

```
C[M, N] = A[M, K] × B[K, N]

Wave 0 (m=0, n=0): C[0:16, 0:16] = A[0:16, :] × B[0:16, :]
Wave 2 (m=0, n=1): C[0:16, 16:32] = A[0:16, :] × B[16:32, :]
                                      ^^^^^^^^^^
                                      同一份 A!
```

TV layout 通过 **t3 stride=0** 编码了这个重复——不同 N-atom 的 wave 映射到同一个
A 地址。

---

## tiled_copy_B 的 TV layout 详细分析

### TV layout

```
((16,4,2,2),(8,(1,1))):((1,256,0,16),(32,(0,0)))
```

和 A 的唯一区别：**t2 stride 和 t3 stride 互换了**（A 是 `16,0`; B 是 `0,16`）。

### 第 1 步：tid 分解

同 A 一样的 4 层分解。但 B 用 t3 而不是 t2:

```
t0 = tid % 16          → atom 内行位置 (stride=1)
t1 = (tid / 16) % 4    → K group (stride=256)
t2 = (tid / 64) % 2    → stride=0, 不贡献偏移 (B 与 M 无关)
t3 = (tid / 128) % 2   → N 方向 atom 偏移 (stride=16)
```

对应 IR (`08_convert_fly_to_rocdl.mlir` L99-106, loc #loc25):

```mlir
%69 = arith.muli %56, %3             // t0 × ldb    → 行偏移
%70 = arith.addi %69, %61            // t0×ldb + t1×8     (%61 = t1×8, 复用 A 的计算)
%71 = arith.divsi %59, %c2_i32       // t3 = (tid/64)/2 = tid/128
%72 = arith.extsi %71                // t3 (i64)
%73 = arith.muli %72, %68            // t3 × (ldb×16)     (%68 = ldb×16)
%74 = arith.addi %70, %73            // 最终: t0×ldb + t1×8 + t3×16×ldb
```

关键区别: A 用 `t2 × 16 × lda`, B 用 `t3 × 16 × ldb`。t2 不出现在 B 的地址里。

### 第 2 步：地址公式

```
B_base = t0 × ldb + t1 × 8 + t3 × 16 × ldb
```

B 矩阵是 row-major (N×K), stride = (ldb, 1):

```
row = t0 + t3 × 16    (N 方向位置)
col = t1 × 8 + v      (K 方向位置, v = 0..7)
```

- t0 = 0..15: atom 内 16 行
- t1 = 0..3: 4 个 K group → 覆盖全部 32 K 列
- t3 = 0 或 1: N 方向偏移 16 行
- **t2 不出现**: stride=0, Wave 在 M 方向重复读同一块 B

### 第 3 步：逐条 load 追踪

B 也是 2 条 `buffer.load i128`：N_rep=0 和 N_rep=1。

```
load #   N_rep   offset                        位置
──────   ─────   ─────────────────────────      ──────────────────
 1        0      block_n_offset + B_base        (row,      col=t1×8..t1×8+7)
 2        1      block_n_offset + B_base        (row+32,   col=t1×8..t1×8+7)
                 + ldb×32
```

对应 IR:
```
L181: rocdl.raw.ptr.buffer.load → i128 (8 bf16)
      offset = %37 + %74                     ← N_rep=0
      insert_strided_slice offsets=[0]       ← frag_B[0:8]

L202: rocdl.raw.ptr.buffer.load → i128 (8 bf16)
      offset = %37 + %74 + %67              ← N_rep=1 (加 ldb×32)
      insert_strided_slice offsets=[8]       ← frag_B[8:16]
```

%67 = `%3 × 32` = `ldb × 32` (L99)。N_rep 步长 = 32 行。

### 第 4 步：代入具体 wave 验证

| Wave | tid | t0 | t1 | t2 | t3 | copy tile 内 row | copy tile 内 col |
|---|---|---|---|---|---|---|---|
| 0 | 0-63 | 0..15 | 0..3 | 0 | 0 | 0..15 | 0..31 |
| 1 | 64-127 | 0..15 | 0..3 | 1 | 0 | **0..15 = Wave 0** | 0..31 |
| 2 | 128-191 | 0..15 | 0..3 | 0 | 1 | 16..31 | 0..31 |
| 3 | 192-255 | 0..15 | 0..3 | 1 | 1 | **16..31 = Wave 2** | 0..31 |

**Wave 0 和 Wave 1 读完全相同的 B 数据** (t2 stride=0)。
**Wave 2 和 Wave 3 读完全相同的 B 数据** (t2 stride=0)。

注意和 A 对比：A 里冗余的 wave 对是 (0,2)+(1,3)，B 里是 (0,1)+(2,3)。

加上 N_rep=2 (步长 32 行)，完整 B 矩阵覆盖:

```
B 矩阵 (64N × 32K) — 每个 wave 的 load 区域

                     K=0..7   K=8..15  K=16..23 K=24..31
                    (t1=0)   (t1=1)   (t1=2)   (t1=3)
                  ┌────────┬────────┬────────┬────────┐
N_rep=0  N= 0..15 │       Wave 0 = Wave 1              │  t3=0
                  ├────────┴────────┴────────┴────────┤
         N=16..31 │       Wave 2 = Wave 3              │  t3=1
                  ├────────┬────────┬────────┬────────┤
N_rep=1  N=32..47 │       Wave 0 = Wave 1              │  t3=0, row+32
                  ├────────┴────────┴────────┴────────┤
         N=48..63 │       Wave 2 = Wave 3              │  t3=1, row+32
                  └───────────────────────────────────┘
```

### 为什么 B 的冗余 wave 对和 A 不同

因为 atom_layout `(2,2,1):(1,2,0)` 的 M 和 N stride 不同:

```
atom_layout: (M=2, N=2, K=1) : (stride_M=1, stride_N=2, stride_K=0)

Wave 0: m_atom=0, n_atom=0    Wave 1: m_atom=1, n_atom=0
Wave 2: m_atom=0, n_atom=1    Wave 3: m_atom=1, n_atom=1
```

- A 只依赖 M → 同 m_atom 的 wave 共享 A → **Wave 0/2 (m=0)** 和 **Wave 1/3 (m=1)**
- B 只依赖 N → 同 n_atom 的 wave 共享 B → **Wave 0/1 (n=0)** 和 **Wave 2/3 (n=1)**

TV layout 通过哪个维度的 stride=0 编码了这个关系:
- A: t3 (N atom) stride=0 → 不同 N-atom 读同样的 A
- B: t2 (M atom) stride=0 → 不同 M-atom 读同样的 B

---

## A, B, C 三矩阵 wave 分布全景

```
atom_layout (2,2,1):(1,2,0)
Wave 0: m=0, n=0    Wave 1: m=1, n=0    Wave 2: m=0, n=1    Wave 3: m=1, n=1

── A 矩阵 (64M × 32K): 冗余沿 N ──    ── B 矩阵 (64N × 32K): 冗余沿 M ──

    全部 32 K 列                             全部 32 K 列
  ┌──────────────────┐                   ┌──────────────────┐
  │ Wave 0 = Wave 2  │  M[0:16]          │ Wave 0 = Wave 1  │  N[0:16]
  ├──────────────────┤                   ├──────────────────┤
  │ Wave 1 = Wave 3  │  M[16:32]         │ Wave 2 = Wave 3  │  N[16:32]
  ├──────────────────┤                   ├──────────────────┤
  │ Wave 0 = Wave 2  │  M[32:48]         │ Wave 0 = Wave 1  │  N[32:48]
  ├──────────────────┤                   ├──────────────────┤
  │ Wave 1 = Wave 3  │  M[48:64]         │ Wave 2 = Wave 3  │  N[48:64]
  └──────────────────┘                   └──────────────────┘

── C 矩阵 (64M × 64N): 每 wave 唯一, 棋盘格 ──

             N[0:16]    N[16:32]   N[32:48]   N[48:64]
           ┌──────────┬──────────┬──────────┬──────────┐
M[ 0:16]   │  Wave 0  │  Wave 2  │  Wave 0  │  Wave 2  │
           ├──────────┼──────────┼──────────┼──────────┤
M[16:32]   │  Wave 1  │  Wave 3  │  Wave 1  │  Wave 3  │
           ├──────────┼──────────┼──────────┼──────────┤
M[32:48]   │  Wave 0  │  Wave 2  │  Wave 0  │  Wave 2  │
           ├──────────┼──────────┼──────────┼──────────┤
M[48:64]   │  Wave 1  │  Wave 3  │  Wave 1  │  Wave 3  │
           └──────────┴──────────┴──────────┴──────────┘
```

**核心规律**:

```
C[M, N] = A[M, K] × B[K, N]

→ A 只和 M 有关, 不同 N-wave 读同一份 A (冗余)
→ B 只和 N 有关, 不同 M-wave 读同一份 B (冗余)
→ C 同时和 M, N 有关, 每 wave 写唯一区域 (无冗余)
```

每个 wave 每次 MFMA iteration 的实际数据量:
- A: 16 bf16 (8 per M_rep, ×2 M_reps) → 但因冗余, 独立数据只有 2 组
- B: 16 bf16 (8 per N_rep, ×2 N_reps) → 但因冗余, 独立数据只有 2 组
- C: 16 f32  (4 per MFMA, ×4 MFMAs) → 4 组全部独立

---

## 总结

| 问题 | 答案 |
|---|---|
| 在哪里切 wave？ | TV layout 的 stride。lowering 时变成 `tid % N`, `tid / N` 的运算链 |
| 谁生成 TV layout？ | `make_tiled_copy_A/B/C` 从 `tiled_mma` 的 atom_layout + MMA 硬件 layout 推导 |
| atom_layout 的作用？ | 决定哪些 atom 分给不同 wave-group，从而决定 TV layout 里哪些 stride 非零 |
| 代码里需要手动切吗？ | 不需要。写 `atom_layout` + `tiled_copy.get_slice(tid)` + `partition_S/D`，框架自动做 |
| A 的冗余读取？ | 有。stride=0 的 wave 维度意味着多个 wave 读同样的数据（如 Wave 0/2 读同样的 A rows） |
| copy tile 是什么？ | TV layout 覆盖的基本单元 (32×32)。所有 256 线程协作覆盖一个 copy tile |
| 每 wave 是连续 32×32 吗？ | **不是**。每 wave 是 4 个分散的 16×16 块 (棋盘格)，因为 rep 步长 = copy tile 大小 (32)，跳过了相邻 wave 的 atom |
| 怎么验证？ | 从 lowered IR 追踪 buffer.store 地址，代入具体 tid 算 (row, col) |
