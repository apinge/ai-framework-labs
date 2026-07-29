# Preshuffle 最小示例详解

## 0. 注释核对

`shuffle_weight` 中 cursor 写的注释是**正确的**。对 `layout=(16,16)`, `float16` 的场景：

- `IN=16, IK=16` → `BN=16, BK=32, K=8`（K 这里是 `16 // element_size(f16=2字节) = 8`）
- view: `(-1, N//16, 16, K//32, 4, 8)` ✓
- permute `(0,1,3,4,2,5)`: 把 dim2(BN=16) 挪到 dim4 ✓
- contiguous(): 按 permute 后的维度顺序重新写内存 ✓

## 1. MFMA 16×16×16 的 B 矩阵 lane 分配

从 `mfma-layout.py` 输出：

```
mfma_f32_16x16x16_f16: 每个 lane 持有 4 个 f16（= 2 个 VGPR）

B 矩阵 (N=行, K=列), 一次 MFMA 消耗 16×16 的 B tile:
        K[0:3]    K[4:7]    K[8:11]   K[12:15]
N= 0 : lane[ 0]  lane[16]  lane[32]  lane[48]
N= 1 : lane[ 1]  lane[17]  lane[33]  lane[49]
...
N=15 : lane[15]  lane[31]  lane[47]  lane[63]
```

## 2. 原始 B 矩阵 (16×32)

```
行0:  [ 0  1  2  3  4  5  6  7 | 8  9 10 11 12 13 14 15 |16 17 18 19 20 21 22 23 |24 25 26 27 28 29 30 31]
行1:  [32 33 34 35 36 37 38 39 |40 41 42 43 44 45 46 47 |48 ...                                          ]
...
行15: [480 ...                                                                                        511]
```

## 3. 你的问题：thread 0 要的两组数据不连续？

如果把 K=32 简单地切成 **前 16 + 后 16**：

```
MFMA #0: B[:, 0:16]  → thread 0 拿 k[0:3]
MFMA #1: B[:, 16:32] → thread 0 拿 k[16:19]
```

那 thread 0 两次拿的是 `k[0:3]` 和 `k[16:19]`——**确实不连续**，中间隔了 12 个元素。preshuffle 后它们也不挨着，8 个 f16 的 load 塞不进去。

**但这个 kernel 不是这样切 K 的。**

## 4. 关键：K-tile layout 交错排列

tiled_mma 的 K-tile 定义：

```python
fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4)))
```

`(4,4,2):(1,8,4)` 把 K=32 映射到 `(mfma_k=4, k_group=4, k_repeat=2)` 三个维度：
- dim0 (4, stride=1): MFMA atom 内 4 个连续 k → 每 lane 拿的 4 个 f16
- dim1 (4, stride=8): 4 个 k_group → lane 0/16/32/48
- dim2 (2, stride=4): **2 次 MFMA 调用之间的交错**

展开后，两次 MFMA 吃的 K 列是**交错的**，不是连续的：

```
MFMA #0 的 K 列: [0,1,2,3,  8,9,10,11,  16,17,18,19,  24,25,26,27]
MFMA #1 的 K 列: [4,5,6,7,  12,13,14,15, 20,21,22,23,  28,29,30,31]
                  └──────┘  └──────────┘  └───────────┘  └──────────┘
                  k_group0   k_group1      k_group2       k_group3
```

**这跟 "前 16 列 + 后 16 列" 完全不同！**

数学上等价——矩阵乘法是 `C = Σ_k A[:,k]·B[:,k]ᵀ`，加法顺序不影响结果（浮点精度差异可忽略）。

## 5. Thread 0 两次 MFMA 拿的数据

```
MFMA #0: thread 0 (lane 0, k_group=0) 拿 k[0, 1, 2, 3]
MFMA #1: thread 0 (lane 0, k_group=0) 拿 k[4, 5, 6, 7]

合计: k[0, 1, 2, 3, 4, 5, 6, 7] = 连续 8 个 f16!
```

Thread 16 (lane 16, k_group=1)：
```
MFMA #0: k[8, 9, 10, 11]
MFMA #1: k[12, 13, 14, 15]
合计: k[8:16] = 连续 8 个 f16
```

**K-tile 的 `stride=4`（dim2）让两次 MFMA 的 k_group=0 恰好覆盖 k[0:4] 和 k[4:8]，拼起来就是连续的 8 个 f16。**

这不是巧合——K-tile layout 就是为了让每个 thread 需要的 B 数据连续，从而匹配 128-bit load。

## 6. Preshuffle 让这连续的 8 个 f16 在内存里也连续

### 6.1 Preshuffle 后内存

```
sub_k=0 (原始 k[0:8], 每行 8 个 f16 = 16 字节):
  地址   0~ 15: 行0  的 [0,1,2,3,4,5,6,7]        ← thread 0 load
  地址  16~ 31: 行1  的 [32,33,...,39]              ← thread 1 load
  ...
  地址 240~255: 行15 的 [480,...,487]               ← thread 15 load

sub_k=1 (原始 k[8:16]):
  地址 256~271: 行0 的 [8,...,15]                   ← thread 16 load
  ...
```

### 6.2 Thread 0 的 load 和 MFMA 使用

```
buffer_load_dwordx4 → 读地址 [0:16]，得到 8 个 f16:

  v[88:89] = [0, 1, 2, 3]     → MFMA #0 的 src_b (lane 0, k_group=0)
  v[90:91] = [4, 5, 6, 7]     → MFMA #1 的 src_b (lane 0, k_group=0)
```

ISA 验证（`22_final_isa.s`）：

```asm
buffer_load_dwordx4 v[88:91], v76, s[16:19], 0 offen   ; load 8 f16

v_mfma_f32_16x16x16_f16 v[12:15], v[96:97], v[88:89], v[12:15]   ; MFMA #0: B=v[88:89]
v_mfma_f32_16x16x16_f16 v[12:15], v[98:99], v[90:91], v[12:15]   ; MFMA #1: B=v[90:91]
```

### 6.3 行主序下同样 16 个线程 load 的对比

```
Preshuffle (sub_k=0 段):            行主序:
线程 0  → 地址   0~ 15              线程 0  → 地址    0~ 15  (行0, k[0:7])
线程 1  → 地址  16~ 31              线程 1  → 地址   64~ 79  (行1, k[0:7])
线程 2  → 地址  32~ 47              线程 2  → 地址  128~143
...                                 ...
线程 15 → 地址 240~255              线程 15 → 地址  960~975

间距: 16 字节 (无间隙)              间距: 64 字节 (中间 48 字节是 k[8:31])
总跨度: 256 字节连续                总跨度: 976 字节散布
cacheline: 2 个, 100% 利用          cacheline: ~8 个, 12.5% 利用
```

## 7. 完整因果链

```
① K-tile layout (4,4,2):(1,8,4)
   让两次 MFMA 交错而非连续切分 K
   → 每个 thread 两次 MFMA 合计需要连续 8 个 f16

② Preshuffle: view → permute → contiguous
   把 B 的行主序 (行宽=K) 重排成 sub_k 段 (每段行宽=8)
   → 每个 thread 需要的 8 个 f16 在内存中连续

③ buffer_load_dwordx4: 16 字节 = 8 f16
   恰好一次 load 读完 thread 需要的全部 B 数据
   → load 后 retile: 前 4 个给 MFMA #0, 后 4 个给 MFMA #1

④ 16 个 thread 同时 load: 地址连续、无间隙
   → 完美 memory coalescing, 2 个 cacheline 搞定
```

**三者缺一不可：K-tile 交错保证逻辑连续，preshuffle 保证物理连续，128-bit load 保证一次读完。**

## 8. 不同 MFMA 需要换 preshuffle pattern 吗？

**是的。** preshuffle 的分段大小（sub_k = 8 个 f16）是为 K-tile 和 MFMA 共同决定的：

```
sub_k 段宽 = 8 个 f16
           = buffer_load_dwordx4 一次读的量
           = K-tile stride=4 → 每 thread 两次 MFMA 合计拿的 f16 数
```

`shuffle_weight` 的 `layout=(IN, IK)` 参数控制这个：
```python
BK = IK * 2   # sub_k 段宽 = 2 × MMA_K 个元素 = 1 次 128-bit load
BN = IN       # 行数 = MMA 的 N
```

换 MFMA（如 32×32×8）需要不同的 `layout` 参数、不同的 K-tile layout、不同的 preshuffle pattern。

## 9. 验证代码

```python
import torch

N, K = 16, 32
B = torch.arange(N * K, dtype=torch.float16).view(N, K)

# preshuffle
x = B.view(1, 1, 16, 1, 4, 8).permute(0, 1, 3, 4, 2, 5).contiguous().view(N, K)
flat = x.flatten()

# thread 0 load 地址 [0:8] = 原始 B[n=0, k=0:7]
print("Thread 0 load 8 个 f16:", flat[0:8].tolist())
print("  MFMA #0 用前 4 个:", flat[0:4].tolist(), "= B[0, 0:4]")
print("  MFMA #1 用后 4 个:", flat[4:8].tolist(), "= B[0, 4:8]")

# 验证交错 MFMA 等价性
A = torch.randn(16, 32)
k0 = [0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27]
k1 = [4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31]
C_ref = A @ B.float().T
C_inter = A[:, k0] @ B[:, k0].float().T + A[:, k1] @ B[:, k1].float().T
print(f"\n交错 MFMA 等价: {torch.allclose(C_ref, C_inter, atol=1e-3)}")
```


## preshuffle layout 怎么和 `make_layout` 对上

这里以 `tests/flydsl/test_gemm_debug.py:103-106` 的 bf16 splitk 为例：

```python
arg_b = fx.Tensor(fx.make_view(
    fx.get_iter(arg_b_),
    fx.make_layout(
        ((16, N // 16), (8, 4, K // 32)),
        ((8, 16 * K), (1, 128, 512)),
    )
))
```

这个 view 的含义不是普通 row-major `[N, K]`。它是在告诉 FlyDSL：`arg_b_` 指向的是 host 侧 `shuffle_weight(w_)` 后的物理 buffer；如果逻辑上访问 `B[n, k]`，应该按 preshuffle 后的物理 offset 去读。

host 侧 `shuffle_weight(w_, layout=(16, 16))` 的核心实现是：

```python
BK = 16 * 2        # 32 for bf16
K_lane = 16 // 2   # 8 bf16 elements per 16B load
BN = 16

x.view(-1, N // 16, 16, K // 32, 4, 8)
 .permute(0, 1, 3, 4, 2, 5)
 .contiguous()
 .view(N, K)
```
注意premute以后 =>(-1,  N // 16, K // 32, 4,16, 8)
把 batch/outer 维先忽略，只看原始 `B[N, K]`。把逻辑坐标拆成：

```text
n = n_blk * 16 + n_in
k = k_blk * 32 + k_sub * 8 + k_lane

n_in  : 0..15
n_blk : 0..N/16-1
k_lane: 0..7
k_sub : 0..3
k_blk : 0..K/32-1
```

`shuffle_weight` 的 `view -> permute -> contiguous` 后，物理内存顺序变成：

```text
[n_blk, k_blk, k_sub, n_in, k_lane]
```

所以 preshuffled buffer 中 `B[n, k]` 的物理 offset 是：

```text
offset =
    n_blk  * (16 * K) 
  + k_blk  * 512
  + k_sub  * 128
  + n_in   * 8
  + k_lane * 1
```

16*K 来自8*16*4//32

而 `fx.make_layout(((16, N // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 128, 512)))` 正好表达同一个 offset 公式：

```text
shape : ((16, N//16), (8, 4, K//32))
coord : ((n_in, n_blk), (k_lane, k_sub, k_blk))
stride: ((8, 16*K),    (1,      128,   512))

offset =
    n_in   * 8
  + n_blk  * 16K
  + k_lane * 1
  + k_sub  * 128
  + k_blk  * 512
```

这就是 preshuffle 和 layout 对上的关键：`shuffle_weight` 改变了物理存储顺序，`make_layout` 用对应的 stride 把这个物理顺序重新解释成逻辑 `[N, K]`。

几个数字的来源：

- `16` 是 MFMA 的 N 方向行数，也就是 16 rows 一组。
- `8` 是 bf16 下一个 128-bit load 能装的元素数：`16 bytes / 2 bytes = 8`。
- `4` 是一个 `BK=32` 分块里有 4 个 8-element 子块。
- `128 = 16 * 8`，表示跳到下一个 `k_sub` 时要跨过 16 行、每行 8 个元素。
- `512 = 4 * 16 * 8`，表示跳到下一个 `k_blk` 时要跨过完整的 `32 x 16` preshuffle block。
- `16 * K` 是跳到下一个 `n_blk` 时跨过这一组 16 行的全部 K 数据。

为什么要这样排？因为 splitk kernel 的 B 侧 load 和 MFMA K layout 是配套的。`tiled_mma` 里 K 维使用：

```python
fx.make_layout((4, 4 * 4, 2), (1, 8, 4))
```

它不是简单地把 K 切成前 16 列、后 16 列，而是交错地把两次 MFMA 需要的 K 片段排在一起。对一个 lane 来说，两次 MFMA 合起来正好需要同一行上的连续 8 个 bf16。preshuffle 把这 8 个 bf16 在内存里也放连续，于是一个 `BufferCopy128b` / `buffer_load_dwordx4` 就能一次读完。

可以把因果链记成：

```text
tiled_mma 的 K layout
  -> 让每个 lane 逻辑上需要连续 8 个 bf16

shuffle_weight
  -> 把这连续 8 个 bf16 物理上也排到一起

make_layout 的 stride
  -> 告诉 kernel 如何在 preshuffled buffer 上按逻辑 B[n,k] 建 view

make_tiled_copy_B + BufferCopy128b
  -> 每个 lane 一次 16B load 读到正好给 MFMA 用的 B fragment
```

MoE 版 `_make_down_weight_view()` 里的写法是同一个思想的泛化：

```python
element_num = 16 // (p_weight.dtype.width // 8)
fx.make_layout(
    ((16, N // 16), (element_num, K // element_num)),
    ((element_num, 16 * K), (1, 16 * element_num)),
)
```

对 bf16，`element_num=8`，它等价于把 K 维写成 `(8, K//8)`；`test_gemm_debug.py` 里进一步把 `K//8` 拆成 `(4, K//32)`，所以看到的是 `(8, 4, K//32)`。对 fp8，`element_num=16`，一个 128-bit load 能装 16 个 fp8，所以 layout 也随元素大小自然变化。