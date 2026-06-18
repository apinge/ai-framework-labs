# K_rep 分析：从 IR 的 fragment shape 读懂 MFMA 展开

## 核心问题

`fx.gemm` 一行代码到底会生成几条 MFMA 指令？K 维度是怎么被处理的？

答案全在 IR 里的 **fragment shape** 中——这是框架告诉你"每个线程要算多少"的蓝图。

---

## fragment shape 怎么读

`make_fragment` 在 `00_origin.mlir` 里长这样：

```
fly.mma.make_fragment(a, ...) -> !fly.memref<bf16, register, (8, 2, 4):(1, 32, 8)>
```

shape 部分 `(8, 2, 4)` 是关键，三个维度含义固定：

```
frag_A: (val, M_rep, K_rep)
frag_B: (val, N_rep, K_rep)
frag_C: (val, M_rep, N_rep)      ← 注意：C 没有 K_rep
```

| 维度 | 含义 | 由什么决定 |
|---|---|---|
| val | 每个 MMA atom 里每个线程持有的元素数 | 硬件指令定义，不可更改 |
| M_rep / N_rep | 每个 wave-group 在 M/N 方向重复几个 atom | `block_dim / (atom_layout_dim × mma_dim)` |
| K_rep | K 方向需要几次 MMA atom | `block_k / (atom_layout_K × mma_K)` |

**K_rep 只出现在 frag_A 和 frag_B 里，不出现在 frag_C 里**——因为 K 是归约维度，
多次 MFMA 的结果累加到同一个 C fragment 上。

---

## 三个实际 case 的对比

### Case 1：`small_gemm_bf16.py` — block_k = 32

```
MFMA(16, 16, 32, BF16)      mma_K = 32
atom_layout = (2, 2, 1)     atom_K = 1
block_k = 32
```

IR 输出：

```
frag_A: (8, 2, 1)    →  val=8, M_rep=2, K_rep=1
frag_B: (8, 2, 1)    →  val=8, N_rep=2, K_rep=1
frag_C: ((4,1), 2, 2) → val=4, M_rep=2, N_rep=2
```

计算过程：

```
K_rep = block_k / (atom_K × mma_K) = 32 / (1 × 32) = 1
M_rep = block_m / (atom_M × mma_M) = 64 / (2 × 16) = 2
N_rep = block_n / (atom_N × mma_N) = 64 / (2 × 16) = 2
```

MFMA 指令数 = M_rep × N_rep × K_rep = 2 × 2 × 1 = **4**

验证：`grep -c "mfma" 08_convert_fly_to_rocdl.mlir` → **4** ✓

### Case 2：`small_gemm_bf16_no_k_rep.py` — block_k = 256

```
MFMA(16, 16, 32, BF16)      mma_K = 32
atom_layout = (2, 2, 1)     atom_K = 1
block_k = 256
```

IR 输出：

```
frag_A: (8, 2, 8)    →  val=8, M_rep=2, K_rep=8
frag_B: (8, 2, 8)    →  val=8, N_rep=2, K_rep=8
frag_C: ((4,1), 2, 2) → val=4, M_rep=2, N_rep=2   ← C 不变！
```

计算过程：

```
K_rep = block_k / (atom_K × mma_K) = 256 / (1 × 32) = 8
M_rep = 64 / (2 × 16) = 2     （不变）
N_rep = 64 / (2 × 16) = 2     （不变）
```

MFMA 指令数 = 2 × 2 × 8 = **32**

验证：`grep -c "mfma" 08_convert_fly_to_rocdl.mlir` → **32** ✓

**关键观察：block_k 从 32 变到 256，K_rep 从 1 变到 8，MFMA 从 4 变到 32。
但 frag_C 形状完全不变** ——K 维度的增加只让 A/B 的寄存器变多，C 始终是同一组
accumulator 被反复累加。

### Case 3：`small_gemm_f32.py` — block_k = 8，MFMA(16,16,4,F32)

```
MFMA(16, 16, 4, F32)        mma_K = 4
atom_layout = (2, 2, 1)     atom_K = 1
block_k = 8
```

IR 输出：

```
frag_A: (1, 2, 2)       →  val=1, M_rep=2, K_rep=2
frag_B: (1, 2, 2)       →  val=1, N_rep=2, K_rep=2
frag_C: ((4,1), 2, 2)   →  val=4, M_rep=2, N_rep=2
```

计算过程：

```
K_rep = block_k / (atom_K × mma_K) = 8 / (1 × 4) = 2
M_rep = 64 / (2 × 16) = 2
N_rep = 64 / (2 × 16) = 2
```

MFMA 指令数 = 2 × 2 × 2 = **8**

**注意 val 的差异：** F32 MFMA(16,16,4) 每线程每 atom 只有 1 个 A/B 值（1 个 f32），
而 BF16 MFMA(16,16,32) 有 8 个值（8 个 bf16）。这由硬件决定，和 K_rep 无关。

---

## K_rep 计算的一般公式

```
K_rep = block_k / (atom_layout_K × mma_K)
```

其中：
- `block_k`：你在 Python 里定义的 K 方向 tile 大小
- `atom_layout_K`：`make_tiled_mma` 第二个参数 `(M, N, K)` 中的 K
- `mma_K`：MFMA 指令的 K 维度（MFMA(16,16,**K**,...) 的第三个参数）

### atom_layout 的 K 维度何时 > 1？

当多个 wave-group **分工处理** K 维度时（split-K）。例如 `test_gemm_cdna4.py`：

```python
tiled_mma = fx.make_tiled_mma(
    fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
    fx.make_layout((1, 1, 4), (0, 0, 1)),   # ← K 维度分布 4 个 wave-group
    ...
)
```

这里 `atom_layout = (1, 1, 4)`，4 个 wave-group 各算 K 的一部分，最后 reduce。
这种情况下 `K_rep = block_k / (4 × 16)` 而不是 `block_k / 16`。

**本 case 的三个例子都是 `atom_layout_K = 1`（K 不跨 group 分布），所以
K_rep 就是 `block_k / mma_K`。**

---

## 从 IR 验证 K_rep 的快速方法

### 方法 1：看 fragment shape

```bash
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 your_kernel.py
grep "make_fragment" /root/.flydsl/debug/gemm_kernel_0/00_origin.mlir
```

frag_A 的第 3 维就是 K_rep。frag_C 没有 K 维度。

### 方法 2：数 MFMA 指令

```bash
grep -c "mfma" /root/.flydsl/debug/gemm_kernel_0/08_convert_fly_to_rocdl.mlir
```

应该等于 `M_rep × N_rep × K_rep`。如果不等，说明 atom_layout 或 block 尺寸有问题。

### 方法 3：数 buffer_load 指令

```bash
grep -c "buffer.load" /root/.flydsl/debug/gemm_kernel_0/08_convert_fly_to_rocdl.mlir
```

A 的 load 数 = M_rep × K_rep，B 的 load 数 = N_rep × K_rep。
总 load 数 = (M_rep + N_rep) × K_rep。

| Case | M_rep | N_rep | K_rep | MFMA 数 | load 数 |
|---|---|---|---|---|---|
| bf16, block_k=32 | 2 | 2 | 1 | 4 | 2+2 = 4 |
| bf16, block_k=256 | 2 | 2 | 8 | 32 | 16+16 = 32 |
| f32, block_k=8 | 2 | 2 | 2 | 8 | 4+4 = 8 |

---

## val 维度（第 1 维）怎么来的

val 是 MMA 硬件指令决定的，和 block 大小、atom_layout 都无关。

对于 CDNA MFMA，A/B 的 val 公式在 `lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp`：

```cpp
// getThrValLayoutAB()
int GroupK = 64 / MN;               // MN = M = N（MFMA 要求 M == N）
int KPerThread = K / GroupK;         // 每线程持有的 K 元素数 = val

// Shape:  (Thr(MN, GroupK), Val(KPerThread))
// 64 threads,  KPerThread values per thread
```

| MFMA 指令 | MN | GroupK | val = K / GroupK |
|---|---|---|---|
| MFMA(16, 16, 4, F32) | 16 | 64/16 = 4 | 4/4 = **1** |
| MFMA(16, 16, 16, F16) | 16 | 64/16 = 4 | 16/4 = **4** |
| MFMA(16, 16, 32, BF16) | 16 | 64/16 = 4 | 32/4 = **8** |
| MFMA(32, 32, 8, F16) | 32 | 64/32 = 2 | 8/2 = **4** |

C 的 val 公式也在同一文件：

```cpp
// getThrValLayoutC()
int GroupM = 64 / N;    // = 4 (for 16x16)
int ValM0 = 4;          // 固定
int ValM1 = M / 4 / GroupM;  // = 16/4/4 = 1 (for 16x16)
// val = ValM0 × ValM1 = 4
```

所以 MFMA(16,16,*) 的 C 永远是 val = **4**（4 个 f32 per thread per atom）。

---

## 寄存器用量估算

知道 fragment shape 就能算出每个线程用了多少寄存器：

```
A 寄存器 = val × M_rep × K_rep × sizeof(elem) / 4     (单位：VGPR，32-bit)
B 寄存器 = val × N_rep × K_rep × sizeof(elem) / 4
C 寄存器 = val_C × M_rep × N_rep                       (f32 直接是 VGPR 数)
```

| Case | A VGPR | B VGPR | C VGPR | 合计 |
|---|---|---|---|---|
| bf16, K_rep=1 | 8×2×1×2/4 = 8 | 8 | 4×2×2 = 16 | 32 |
| bf16, K_rep=8 | 8×2×8×2/4 = 64 | 64 | 16 | 144 |
| f32, K_rep=2 | 1×2×2×4/4 = 4 | 4 | 16 | 24 |

**K_rep 越大，A/B 寄存器占用线性增长**。这就是为什么大 K 场景需要手动循环——
不可能把整个 K 维度的数据全放进寄存器。

---

## 什么时候让框架自动展开 K_rep，什么时候手动循环

### 自动展开（一行 `fx.gemm`）

适用条件：

- block_k 较小，A/B 数据一次性全部 load 进寄存器
- 寄存器用量可接受（通常 K_rep ≤ 4~8）
- 不需要 prefetch、不需要 LDS double-buffer

```python
# 一次性 load 全部 A/B
fx.copy(copy_atom, copy_src_A, copy_frag_A)
fx.copy(copy_atom, copy_src_B, copy_frag_B)
frag_C.fill(0)
# 框架自动展开 M_rep × N_rep × K_rep 条 MFMA
fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
```

### 手动循环

适用条件：

- 整个 K 维度太大，必须分块（K-tiling），每块 load 一次
- 需要 prefetch overlap（一边 load 下一块 A/B，一边算当前块的 MFMA）
- 需要 LDS double-buffer（A 通过 LDS，B 直接从 global）

参考 `examples/04-preshuffle_gemm.py`：

```python
# A/B 按 K 方向 flat_divide，得到 k 维度索引
gA_k = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x, None]  # (BM, BK, k)

for k_iter in range(0, K // BLOCK_K):
    # 每轮 load 一块 A/B
    fx.copy(buffer_copy, thr_gA_k[..., k_iter], copy_frag_A, ...)
    fx.copy(buffer_copy, thr_gB_k[..., k_iter], copy_frag_B, ...)
    # 可能还有 sub-K 循环（BLOCK_K > mma_K 时）
    for sub_k in range_constexpr(BLOCK_K // mma_K):
        fx.gemm(tiled_mma, frag_C,
                frag_A[None, None, (None, sub_k)],
                frag_B[None, None, (None, sub_k)],
                frag_C)
```

参考 `splitk_gemm/test_gemm_cdna4.py`（split-K + 手动 sub_k）：

```python
for k in range(loop_start, loop_end, loop_step, init=[acc_init]):
    fx.copy(cp_atom_r, a_tensor_thr[None, None, None, k], a_frag_retile)
    fx.copy(cp_atom_r, b_tensor_thr[None, None, None, k], b_frag_retile)
    for sub_k in range_constexpr(TILE_K // 32):
        fx.gemm(tiled_mma, c_frag,
                b_frag[None, None, (None, sub_k)],
                a_frag[None, None, (None, sub_k)],
                c_frag)
```

### 判断标准

| 标准 | 自动展开 | 手动循环 |
|---|---|---|
| K 数据能否一次装入寄存器 | 能（K_rep ≤ ~8） | 不能 |
| 需要 prefetch | 否 | 是 |
| 需要 LDS | 否 | 通常是 |
| 代码复杂度 | 一行 | 多层循环 |
| 本 repo 的例子 | `small_gemm_*.py`, `03-tiledMma.py` | `04-preshuffle_gemm.py`, `test_gemm_cdna4.py` |
# mfma是wave级别的指令

MFMA 是 wave 级别的指令，不是 workgroup 级别。
   
  一条 rocdl.mfma.f32.16x16x32.bf16 由 1 个 wavefront（64 线程） 协作执行。同一个 workgroup 里的 4 个 wave 各自独立执行自己的 MFMA，互不通信。

  从 IR 层面看：
```
                      workgroup (256 threads)
                      ┌──────────────────────┐
  LLVM IR / ISA 层    │  同一份代码，所有线程执行同样的指令流
  只有一份代码        │  v_mfma_f32_16x16x32_bf16 ...
                      │
                      │  但每个 wave 看到不同的 tid
                      │  → 算出不同的 buffer_load 地址
                      │  → 加载不同的 A/B 数据
                      │  → MFMA 算出不同的 C 结果
                      │  → store 到不同的 C 位置
                      └──────────────────────┘

                      Wave 0 (tid 0-63)     ← 执行 4 条 MFMA，算 C[0:32, 0:32]
                      │  但每个 wave 看到不同的 tid
                      │  → 算出不同的 buffer_load 地址
                      │  → 加载不同的 A/B 数据
                      │  → MFMA 算出不同的 C 结果
                      │  → store 到不同的 C 位置
                      └──────────────────────┘

                      Wave 0 (tid 0-63)     ← 执行 4 条 MFMA，算 C[0:32, 0:32]
                      Wave 1 (tid 64-127)   ← 执行同样 4 条 MFMA，算 C[32:64, 0:32]
                      Wave 2 (tid 128-191)  ← 执行同样 4 条 MFMA，算 C[0:32, 32:64]
                      Wave 3 (tid 192-255)  ← 执行同样 4 条 MFMA，算 C[32:64, 32:64]

  ISA 里没有 "这条 MFMA 给 wave 0 执行、那条给 wave 1" 的区分。所有 wave 跑同一份指令流，区别完全来自 tid 不同 → 地址不同 → 数据不同。这和普通的 v_add 没有本质区别——只是 MFMA 需要 64 个线程协作提供操作数（每线程
  8 个 bf16），硬件在 wave 内部做矩阵乘累加。
```