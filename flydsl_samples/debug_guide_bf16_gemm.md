# FlyDSL BF16 GEMM 调试实录

本文记录 `small_gemm_bf16.py` 从结果错误到修复的完整调试过程。
涉及两个 bug：`atom_layout` 配置错误和 C 矩阵 store 宽度错误。

---

## 背景

原始 kernel 参数：

- 块大小：`block_m=64, block_n=64, block_k=32`
- MMA 指令：`MFMA(16, 16, 32, BF16)` — 64 线程/atom，每线程 8 个 bf16 输入、4 个 f32 输出
- 线程数：256（4 个 wave-group）
- 无 LDS，无 preshuffle，A/B 直接从 global memory 用 `BufferCopy128b` 加载

运行结果：`max error 24.7`，完全不对。

---

## 第一步：从正确的 example 学习模式

在动手改代码之前，先读懂框架里已经能跑通的 GEMM 例子。

### 参考文件

| 文件 | 用途 |
|---|---|
| `examples/03-tiledMma.py` | 最简 global-only GEMM，F32，MFMA(16,16,4) |
| `examples/04-preshuffle_gemm.py` | 带 LDS 的 F16 GEMM，MFMA(16,16,16) |
| `splitk_gemm/test_gemm_cdna4.py` | Split-K BF16，K_rep>1 的写法 |
| `docs/kernel_authoring_guide.md` | Tiled MMA / Tiled Copy 的 API 说明 |

### 查看命令

```bash
# 搜索 make_tiled_mma 的用法模式
grep -n "make_tiled_mma" examples/03-tiledMma.py examples/04-preshuffle_gemm.py splitk_gemm/test_gemm_cdna4.py

# 搜索 C store 用的 copy atom 类型
grep -n "tiled_copy_C\|copy_atom.*Float32\|BufferCopy.*Float32" examples/03-tiledMma.py
```

### 关键发现

Example 03（能跑通的）：

```python
# 64×64 block, MFMA(16,16,4,F32), 256 线程 → 4 wave-groups
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))

# C store 用的是 BufferCopy32b，不是 128b
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
```

原始 `small_gemm_bf16.py`：

```python
# atom_layout 只有 (1,1,1) — 只 tile 了 1 个 atom
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 1)))

# C store 用了 BufferCopy128b — 和 example 03 不同
copy_atom_c = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
```

两处差异，都可能是 bug。但哪个是根本原因？需要逐步验证。

---

## 第二步：修 atom_layout — 第一次尝试 (4,4,1)

### 推理

MFMA(16,16,32) 的 atom 尺寸是 16×16。块大小 64×64 需要 4×4 = 16 个 atom。
直觉上 `M_rep = 64/16 = 4, N_rep = 64/16 = 4`，所以试 `(4, 4, 1)`。

256 线程 / 64 线程每 atom = 4 个 wave-group。stride `(1, 4, 0)` 让 M 方向占
group 0-3，N 方向 stride 4 超出 group 数量，等价于 N 全部在 group 0 内重复。

```python
tiled_mma = fx.make_tiled_mma(
    mma_atom,
    fx.make_layout((4, 4, 1), (1, 4, 0))
)
```

### 运行

```bash
python3 splitk_gemm/small_gemm_bf16.py
# AssertionError: max error 20.87
```

还是错。需要看 IR 确认 fragment shape。

---

## 第三步：用 FLYDSL_DUMP_IR 检查 fragment shape

### 命令

```bash
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 splitk_gemm/small_gemm_bf16.py
```

IR 输出到 `/root/.flydsl/debug/gemm_kernel_0/` 目录。最重要的是 `00_origin.mlir`（trace 后的原始 Fly IR）。

### 查看 fragment shape

```bash
grep "make_fragment\|fly.gemm" /root/.flydsl/debug/gemm_kernel_0/00_origin.mlir
```

输出：

```
fly.mma.make_fragment(a, ...) -> !fly.memref<bf16, register, (8,1,1):(1,0,0)>
fly.mma.make_fragment(b, ...) -> !fly.memref<bf16, register, (8,1,1):(1,0,0)>
fly.mma.make_fragment(c, ...) -> !fly.memref<f32, register, ((4,1),1,1):((1,0),0,0)>
```

### 分析

Fragment shape 解读：`(val, M_rep, K_rep)` 对于 A，`(val, N_rep, K_rep)` 对于 B。

- frag_A = `(8, **1**, 1)` — 只有 1 个 M_rep！预期 4。
- frag_C = `((4,1), **1**, **1**)` — 只有 1×1 rep！预期 4×4。

**根因：** `atom_layout (4,4,1)` 告诉框架"4×4 = 16 个 atom 被分布到不同 group"。
框架计算 per-group tile = `atom_count × mma_dim`：

```
per_group_M = 4 × 16 = 64 = block_m   → 剩余 M_rep = 64 / 64 = 1
per_group_N = 4 × 16 = 64 = block_n   → 剩余 N_rep = 64 / 64 = 1
```

所有 tile 维度都被 atom_layout "消耗"了，每个 group 只分到 1 个 atom。
但实际只有 4 个 group，16 个 atom 不可能均匀分给 4 个 group。

**规则：`M_atoms × N_atoms × K_atoms` 必须等于 wave-group 数量。**

---

## 第四步：改为 (2,2,1) — fragment shape 对了但结果仍错

### 正确的 atom_layout

```python
tiled_mma = fx.make_tiled_mma(
    mma_atom,
    fx.make_layout((2, 2, 1), (1, 2, 0))
)
```

分布：2×2 = 4 个 atom → 4 个 wave-group，一一对应。
Per-group tile = 2×16 = 32，剩余 rep = 64/32 = 2。每个 group 重复 2×2 = 4 个 atom。

### 用 IR 确认

```bash
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 splitk_gemm/small_gemm_bf16.py
grep "make_fragment" /root/.flydsl/debug/gemm_kernel_0/00_origin.mlir
```

输出：

```
fly.mma.make_fragment(a, ...) -> !fly.memref<bf16, register, (8,2,1):(1,8,0)>
fly.mma.make_fragment(b, ...) -> !fly.memref<bf16, register, (8,2,1):(1,8,0)>
fly.mma.make_fragment(c, ...) -> !fly.memref<f32, register, ((4,1),2,2):((1,0),8,4)>
```

Shape 正确了：frag_A = `(8, 2, 1)` = 16 bf16/线程，frag_C = `((4,1), 2, 2)` = 16 f32/线程。

### 但是还是错

```bash
python3 splitk_gemm/small_gemm_bf16.py
# AssertionError: max error 22.57
```

Bug 1 修好了，但存在第二个 bug。

---

## 第五步：全 1 输入测试，发现空间 pattern

随机输入的误差不容易看出规律。改用全 1 输入，`ref = A @ B^T` 的每个元素都应该是 K = 32。

### 操作

复制文件，把 `torch.randn` 改成 `torch.ones`，在 assert 前加打印：

```python
A = torch.ones(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.ones(M, K, dtype=torch.bfloat16, device="cuda")
# ...
print('C nonzero rows:', torch.where(C.sum(dim=1) != 0)[0].tolist())
print('C zero rows:',    torch.where(C.sum(dim=1) == 0)[0].tolist())
```

### 结果

```
C nonzero rows: [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29,
                 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61]
C zero rows:    [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31,
                 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63]
C nonzero cols: [0..63]  ← 全部列都有值
```

**Pattern：每 4 行里只有前 2 行被写入，后 2 行是 0。所有列正常。**

32/64 = 正好一半的行丢失。这是 C store 的地址计算问题。

---

## 第六步：缩到最小 16×16 排除 rep 逻辑

为了区分"是 tiled_mma rep 的 bug"还是"是单个 MMA atom C store 的 bug"，创建一个最小测试：

```python
block_m, block_n, block_k = 16, 16, 32   # 刚好 1 个 atom
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
# 64 线程，1 个 atom，0 个 rep
```

### 结果

```
C nonzero rows: [0, 1, 4, 5, 8, 9, 12, 13]
C zero rows:    [2, 3, 6, 7, 10, 11, 14, 15]
```

**同样的 pattern！** 单个 atom、零 rep 就复现了。所以 bug 在 atom 级别的 C store，
和 tiled_mma 配置无关。

---

## 第七步：分析 MFMA C-operand 的硬件 thread-value layout

### 参考代码

```bash
# 找到 MFMA C layout 的定义
grep -n "getThrValLayoutC" lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp
```

文件 `lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp`，第 63-73 行：

```cpp
Attribute MmaOpCDNA3_MFMAType::getThrValLayoutC() const {
  int M = getM();  // 16
  int N = getN();  // 16
  int GroupM = 64 / N;          // 64 / 16 = 4
  int ValM0 = 4;                // 每线程 4 个连续 M-row
  int ValM1 = M / 4 / GroupM;   // 16 / 4 / 4 = 1

  return FxLayout(
    FxShape(FxThr(N, GroupM), FxVal(ValM0, ValM1)),
    FxStride(FxThr(M, ValM0), FxVal(1, ValM0 * GroupM))
  );
}
```

对于 MFMA(16,16,32,BF16)，C layout 是：

```
Shape:  (Thr(16, 4), Val(4, 1))   → 64 threads × 4 values
Stride: (Thr(16, 4), Val(1, 16))
```

**含义：** 线程 `tid` 拥有 C 矩阵中 4 个 f32 值，位于：

- N-column = `tid % 16`（16 列）
- M-rows = `(tid / 16) × 4 + {0, 1, 2, 3}`（每 group 4 行，4 个 group）

### 内存布局推算

C 是 row-major `(M, N):(N, 1)`，stride = N。线程 0 的 4 个值在：

```
值 0: row=0, col=0 → 偏移 0×N + 0 = 0
值 1: row=1, col=0 → 偏移 1×N + 0 = N
值 2: row=2, col=0 → 偏移 2×N + 0 = 2N
值 3: row=3, col=0 → 偏移 3×N + 0 = 3N
```

4 个值之间的距离是 N 个元素 = N×4 字节（f32）。**不连续。**

---

## 第八步：读 lowered IR 确认 store 地址

### 命令

```bash
# 查看 ROCDL 层的 buffer_store 指令
grep "buffer_store\|buffer_load\|mfma" /root/.flydsl/debug/gemm_kernel_0/08_convert_fly_to_rocdl.mlir
```

输出：

```
rocdl.raw.ptr.buffer.load  ... : i128    ← A load, 128 bits
rocdl.raw.ptr.buffer.load  ... : i128    ← A load (2nd M-rep)
rocdl.raw.ptr.buffer.load  ... : i128    ← B load
rocdl.raw.ptr.buffer.load  ... : i128    ← B load (2nd N-rep)
rocdl.mfma.f32.16x16x32.bf16 ...         ← MFMA #1
rocdl.mfma.f32.16x16x32.bf16 ...         ← MFMA #2
rocdl.mfma.f32.16x16x32.bf16 ...         ← MFMA #3
rocdl.mfma.f32.16x16x32.bf16 ...         ← MFMA #4
rocdl.raw.ptr.buffer.store ... : i128    ← C store (128 bits = 4×f32 连续)
rocdl.raw.ptr.buffer.store ... : i128
rocdl.raw.ptr.buffer.store ... : i128
rocdl.raw.ptr.buffer.store ... : i128
```

### 问题确认

`buffer_store ... : i128` 写 128 位 = 4 个连续 f32。但 MFMA C 的 4 个值在内存中间距
是 N 个元素（不连续）。

`BufferCopy128b` 生成 `buffer_store_dwordx4`，一条指令写 16 字节连续地址。
硬件会把 4 个 f32 写到 `{base, base+4, base+8, base+12}` 字节处——
这是 **4 个相邻列**，而不是 4 个相邻行。

线程 0 应该写 rows {0,1,2,3} × col 0，但实际写了 row 0 × cols {0,1,2,3}。
rows 2, 3 根本没有线程写入 → 全 0。

---

## 第九步：修复 — BufferCopy32b

```python
# 之前（错误）
copy_atom_c = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

# 修复后
copy_atom_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
```

`BufferCopy32b` 生成 4 条独立的 `buffer_store_dword`，每条写 1 个 f32。
`tiled_copy_C` 的 partition 会为每个值计算正确的 strided 地址。

```bash
python3 splitk_gemm/small_gemm_bf16.py
# PASSED
```

---

## 第十步：与 working examples 交叉验证

```bash
# 确认 example 03 用的也是 BufferCopy32b
grep "BufferCopy\|copy_atom" examples/03-tiledMma.py
```

输出：

```python
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
```

一致。`examples/03-tiledMma.py` 用 `BufferCopy32b` 做 F32 C store，是正确的参考。

再跑一遍确认 example 03 本身是通的：

```bash
python3 examples/03-tiledMma.py
# Result correct: True
```

---

## 调试工具箱

### 1. IR Dump：检查 trace 后的 layout 和 fragment shape

```bash
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 your_kernel.py
```

输出目录：`/root/.flydsl/debug/<kernel_name>/`

| 文件 | 内容 | 什么时候看 |
|---|---|---|
| `00_origin.mlir` | Trace 后的 Fly IR，包含所有 layout、fragment shape | 第一个看的文件 |
| `03_fly_layout_lowering.mlir` | Layout 展开后的 IR | 怀疑 layout 计算有误时 |
| `08_convert_fly_to_rocdl.mlir` | ROCDL 层 IR，可看到 buffer_load/store/mfma 指令 | 验证实际发射的指令 |
| `21_final_isa.s` | 最终 ISA 汇编 | 排查指令调度或寄存器问题 |

### 怎么读 `00_origin.mlir`

搜索关键词：

```bash
# fragment shape（最重要）
grep "make_fragment" 00_origin.mlir

# partition shape
grep "partition" 00_origin.mlir

# GEMM 调用
grep "fly.gemm" 00_origin.mlir

# Copy 调用
grep "fly.copy" 00_origin.mlir

# tiled_mma 和 tiled_copy 创建
grep "make_tiled_mma\|make_tiled_copy" 00_origin.mlir
```

Fragment shape 语法：`!fly.memref<dtype, addrspace, shape:stride>`

```
(8, 2, 1):(1, 8, 0)
 │  │  │   │  │  └─ K_rep stride（0 = 单层）
 │  │  │   │  └──── M_rep stride（8 = 每 rep 跨 8 个值）
 │  │  │   └─────── val stride（1 = 连续）
 │  │  └─────────── K_rep = 1
 │  └────────────── M_rep = 2（block_m/32 = 2）
 └───────────────── val = 8（MFMA 16×16×32 每线程 8 bf16）
```

### 2. 全 1 输入测试：发现空间 pattern

```python
A = torch.ones(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.ones(N, K, dtype=torch.bfloat16, device="cuda")
C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

# 运行 kernel ...

# 打印哪些行被写入了
print('nonzero rows:', torch.where(C.sum(dim=1) != 0)[0].tolist())
print('zero rows:',    torch.where(C.sum(dim=1) == 0)[0].tolist())
print('nonzero cols:', torch.where(C.sum(dim=0) != 0)[0].tolist())
print('zero cols:',    torch.where(C.sum(dim=0) == 0)[0].tolist())
```

**解读 pattern：**

| 现象 | 可能的原因 |
|---|---|
| 一半的行是 0 | C store 地址计算错误（本文的 Bug 2） |
| 所有值是 0 | MFMA 没执行 / A/B 没加载成功 |
| 对角线正确、其余错误 | A 和 B 的 partition 用了同一个 N 偏移 |
| 值正确但位置偏移 | block 索引计算错误 / slice 参数错误 |
| 结果是 ref 的整数倍 | 多个 wave-group 写到同一位置（atom_layout stride 冲突） |

### 3. 最小化复现：缩到单 atom

把 block 大小缩到刚好 1 个 MMA atom，线程数设为 `threads_per_MFMA`：

```python
block_m, block_n = 16, 16          # MFMA(16,16,K) 的 M/N 维度
atom_layout = (1, 1, 1)            # 1 个 atom，0 个 rep
BLOCK_DIM = 64                     # MFMA 用 64 线程
```

如果最小 case 就能复现，说明 bug 在 atom 级别（copy atom 或 MMA 本身），
不在 tiled_mma 的 rep 逻辑里。

如果最小 case 正确、放大后才错，说明 bug 在 rep 遍历或 atom_layout 配置。

### 4. 读 C++ MMA atom 源码：理解硬件 layout

```bash
# 找 CDNA3 MFMA 的 A/B/C thread-value layout
cat lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp
```

关键函数：

| 函数 | 返回值 |
|---|---|
| `getThrValLayoutA()` / `getThrValLayoutB()` | A/B operand 的 (Thread, Value) layout |
| `getThrValLayoutC()` | C output 的 (Thread, Value) layout |
| `getThrLayout()` | 单 atom 的线程数（CDNA MFMA 固定 64） |
| `getShapeMNK()` | atom 的 M, N, K 尺寸 |

C layout 公式（CDNA3/4 所有 MxM MFMA 通用）：

```
GroupM = 64 / N
ValM0  = 4
ValM1  = M / 4 / GroupM

Shape:  (Thr(N, GroupM), Val(ValM0, ValM1))
Stride: (Thr(M, ValM0), Val(1, ValM0 × GroupM))
```

对于 MFMA(16,16,32)：GroupM=4, ValM0=4, ValM1=1。
每线程 4 个 f32，位于同一列的 4 个连续行。

### 5. 比较 lowered IR 中的指令数

```bash
grep -c "mfma" 08_convert_fly_to_rocdl.mlir          # MFMA 指令数
grep -c "buffer_load\|buffer.load" 08_convert_fly_to_rocdl.mlir   # load 数
grep -c "buffer_store\|buffer.store" 08_convert_fly_to_rocdl.mlir # store 数
```

预期数量：

| 指令 | 公式 | 本 kernel 预期 |
|---|---|---|
| MFMA | M_rep × N_rep × K_rep | 2 × 2 × 1 = 4 |
| A load | M_rep × K_rep | 2 × 1 = 2 |
| B load | N_rep × K_rep | 2 × 1 = 2 |
| C store | M_rep × N_rep × (vals_per_atom / vals_per_store) | 2 × 2 × (4/1) = 16 (32b) 或 2 × 2 × 1 = 4 (128b) |

如果 MFMA 数 < 预期，说明 fragment rep 不对（atom_layout 问题）。
如果 load 数 < 预期，说明 copy partition 漏了某些 rep。

### 6. 参考文件速查

| 需要了解的内容 | 看什么文件 |
|---|---|
| `make_tiled_mma` 的 atom_layout 参数含义 | `python/flydsl/expr/primitive.py` 搜 `make_tiled_mma` |
| `make_tiled_copy_A/B/C` 如何从 MMA layout 推导 copy layout | `python/flydsl/expr/derived.py` 搜 `make_tiled_copy_A` |
| `retile` 的作用（copy 粒度 vs MMA 粒度的桥接） | `splitk_gemm/demo_tiled_copy_from_mma.py` |
| MFMA 硬件 thread-value layout | `lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp` |
| `BufferCopy` 各宽度对应的 ISA 指令 | `lib/Dialect/FlyROCDL/CDNA3/CopyAtom.cpp` |
| `zipped_divide` / `flat_divide` 的区别 | `docs/layout_system_guide.md` |
| `fx.gemm` 如何遍历 fragment reps | `python/flydsl/expr/primitive.py` 搜 `def gemm` |

---

## 常见陷阱总结

### 陷阱 1：atom_layout shape 与 wave-group 数不匹配

```
wave_groups = BLOCK_DIM / threads_per_MFMA
必须满足：M_atoms × N_atoms × K_atoms = wave_groups
```

如果 shape 乘积 > wave_groups，框架不会报错，但会让 per-group fragment rep
变成 1（甚至 0），实际只有部分线程工作。

### 陷阱 2：宽 copy atom 写非连续的 C output

`BufferCopy128b` / `BufferCopy64b` 要求 value 维度上的元素在内存中连续。

| 场景 | 是否连续 | 推荐 copy atom |
|---|---|---|
| A/B load，K 方向 stride=1 | 连续 | `BufferCopy128b` |
| C store，row-major (M,N):(N,1) | **不连续**（stride = N） | `BufferCopy32b` |
| C store，经过 LDS transpose 后 | 连续 | `BufferCopy128b` |
| C store，col-major (M,N):(1,M) | 连续 | `BufferCopy128b` |

### 陷阱 3：`fx.gemm` 的第一个参数 mma_atom vs tiled_mma

两者都能用，行为不同：

- `mma_atom`：框架根据 fragment shape 自动遍历所有 rep 维度
- `tiled_mma`：同上，但如果有 K permutation 或 traversal_order 需要用 tiled_mma

K_rep > 1 时需要手动用 `(None, sub_k)` 索引切片：

```python
for sub_k in range_constexpr(block_k // mma_k):
    fx.gemm(tiled_mma, frag_C,
            frag_A[None, None, (None, sub_k)],
            frag_B[None, None, (None, sub_k)],
            frag_C)
```

### 陷阱 4：JIT cache 缓存了旧 kernel

修改 kernel 代码后结果不变？可能是 JIT disk cache 返回了旧编译结果。

```bash
# 禁用 cache（在 import flydsl 之前设置）
export FLYDSL_RUNTIME_ENABLE_CACHE=0

# 或者清除 cache
rm -rf ~/.flydsl/cache/
```

在脚本开头加 `os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"` 是最保险的做法。
