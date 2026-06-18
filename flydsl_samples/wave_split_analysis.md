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
M×N 方向。每个 wave 负责 32×32 的 per-group tile 中 16×16 的 atom 位置，
然后通过 M_rep=2, N_rep=2 重复覆盖整个 32×32。

---

## 图解：256 个线程怎么覆盖 64×64 的 C 矩阵

```
C 矩阵 (64×64)
┌───────────────────┬───────────────────┐
│                   │                   │
│  Wave 0 负责      │  Wave 2 负责      │
│  C[0:32, 0:32]    │  C[0:32, 32:64]   │
│                   │                   │
│  (tid 0-63)       │  (tid 128-191)    │
│  M_rep=2, N_rep=2 │  M_rep=2, N_rep=2 │
│                   │                   │
├───────────────────┼───────────────────┤
│                   │                   │
│  Wave 1 负责      │  Wave 3 负责      │
│  C[32:64, 0:32]   │  C[32:64, 32:64]  │
│                   │                   │
│  (tid 64-127)     │  (tid 192-255)    │
│  M_rep=2, N_rep=2 │  M_rep=2, N_rep=2 │
│                   │                   │
└───────────────────┴───────────────────┘
```

每个 wave（64 线程）负责 32×32 的输出区域。
在 32×32 内部，每个线程的 MFMA atom 覆盖 16×16，再通过 M_rep=2, N_rep=2
重复到 32×32。

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

然后框架生成 TV layout 时，把 `m_atom` 的索引（t2）映射到 M 方向偏移 16，
`n_atom` 的索引（t3）映射到 N 方向偏移 16。这就产生了上面的 wave 分布。

---

## 验证方法

### 方法 1：直接看 lowered IR 里的取模链

```bash
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 your_kernel.py
# 在 08_convert_fly_to_rocdl.mlir 里搜 remsi（取模）和 divsi（整除）
grep "remsi\|divsi" /root/.flydsl/debug/gemm_kernel_0/08_convert_fly_to_rocdl.mlir
```

会看到 `tid % 16`, `tid / 16`, `(tid/16) % 4`, `(tid/64) % 2` 这样的链条。
每一级取模/整除对应 TV layout shape 的一个维度。

### 方法 2：看 `00_origin.mlir` 里 `make_tiled_copy` 的 TV layout

```bash
grep "make_tiled_copy" /root/.flydsl/debug/gemm_kernel_0/00_origin.mlir
```

TV layout 的 thread 维度 shape 就是取模链的 shape。stride 非零的维度
会在数据上产生偏移，stride 为 0 的维度意味着多个 wave 读同样的数据。

### 方法 3：对照 atom_layout 推算

```
wave_groups = BLOCK_DIM / threads_per_MFMA
atom_layout shape = (M_atoms, N_atoms, K_atoms)
M_atoms × N_atoms × K_atoms = wave_groups

Wave i 负责的 C 区域:
  M 起始 = (i 在 M 方向的 atom 索引) × mma_M
  N 起始 = (i 在 N 方向的 atom 索引) × mma_N

  per-group tile:
    M_tile = M_atoms × mma_M    (= 2 × 16 = 32)
    N_tile = N_atoms × mma_N    (= 2 × 16 = 32)

  rep:
    M_rep = block_m / M_tile    (= 64 / 32 = 2)
    N_rep = block_n / N_tile    (= 64 / 32 = 2)

Wave i 最终覆盖的 C 区域 = per-group tile × rep = 32 × 32
```

---

## 总结

| 问题 | 答案 |
|---|---|
| 在哪里切 wave？ | TV layout 的 stride。lowering 时变成 `tid % N`, `tid / N` 的运算链 |
| 谁生成 TV layout？ | `make_tiled_copy_A/B/C` 从 `tiled_mma` 的 atom_layout + MMA 硬件 layout 推导 |
| atom_layout 的作用？ | 决定哪些 atom 分给不同 wave-group，从而决定 TV layout 里哪些 stride 非零 |
| 代码里需要手动切吗？ | 不需要。写 `atom_layout` + `tiled_copy.get_slice(tid)` + `partition_S/D`，框架自动做 |
| A 的冗余读取？ | 有。stride=0 的 wave 维度意味着多个 wave 读同样的数据（如 Wave 0/2 读同样的 A rows） |
