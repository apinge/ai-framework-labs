# data flow
回到问题的最初 看下gemm要解决什么问题，只看每个workgroup 

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64

每个workgroup要解决 C_tile = A_tile @ B_tile.T
即 A_tile(128, 64) @ B_tile(128, 64).T = C_tile(128, 128)。
其中 A_tile 是 A 沿 M 方向的子块，B_tile 是 B 沿 N 方向的子块（代码 L87: `gB_k = fx.flat_divide(B, (BLOCK_N, BLOCK_K))`）。

还记得我们的B是preshuffle过的吗

把原始(N,K)的B经过 view → permute → contiguous：
- view:    (-1, N//16, 16, K//32, 4, 8)
- permute(0,1,3,4,2,5) 后: (-1, N//16, K//32, 4, 16, 8)
- contiguous() 按新顺序重写内存

每个workgroup的(BLOCK_N, BLOCK_K) = (128, 64) 子块，在 preshuffle 后的维度分解是
(BLOCK_N//16, BLOCK_K//32, 4, 16, 8) = (8, 2, 4, 16, 8)

## L112-117: partition_S / partition_D — 把 tile 按 tiled_copy 的 thread 映射切分

### 概念：partition 做什么

一个 tiled_copy 包含两部分信息：
1. **thread layout**：256 个 thread 怎样瓜分 tile 的元素（谁读/写哪些位置）
2. **value layout**：每个 thread 一次搬运多少个元素

**partition 的结果是视图（view），不分配新存储**——IR 中保持原始地址空间（global 仍是 `#fly_rocdl.buffer_desc`，LDS 仍是 `shared`），只是把 shape/stride 改成了本 thread 的子集。数据搬运发生在 `fx.copy` 执行时，不是 partition 调用时。

与之对比，`make_fragment_A/B/C` 和 `make_fragment_like` **真实分配 VGPR**——IR 中地址空间是 `register`。这些是物理寄存器，MFMA 指令直接从中读写。

`partition_S(tensor)` = 按 thread layout 把 **source** tensor 切分，返回 "本 thread 负责读的那些元素"。
`partition_D(tensor)` = 按 thread layout 把 **destination** tensor 切分，返回 "本 thread 负责写的那些位置"。

结果 shape 的第一维总是 **V**（value，每次搬运的元素数），后面是重复维度。

### 怎么读 partition 的 IR

以 `00_origin.mlir:104` 为例，原始 IR 是一行：

```mlir
%85 = fly.tiled_copy.partition_src(%arg4, %29, %39)
    : (!fly.tiled_copy<!fly.copy_atom<!fly.universal_copy<128>, 16>,
                       !fly.layout<((8,32),(1,8)):((256,1),(1,32))>,
                       !fly.tile<[32|64]>>,
       !fly.memref<f16, #fly_rocdl.buffer_desc, (128,64,?):(?{i64 div=16},1,64)>,
       !fly.int_tuple<?>)
   -> !fly.memref<f16, #fly_rocdl.buffer_desc, ((8,1),4,1,?):((1,0),?{i64 div=512},0,64)>
```

结构是 `partition_src(操作数1, 操作数2, 操作数3) : (类型1, 类型2, 类型3) -> 结果类型`：

```
操作数1 = %arg4  : tiled_copy   ← "怎么切"
操作数2 = %29    : memref       ← "切谁"（被切分的 tensor）
操作数3 = %39    : int_tuple    ← "给哪个 thread"（tid）
结果     = %85   : memref       ← "切完的视图"
```

#### 操作数1：tiled_copy 类型拆解

```
!fly.tiled_copy<
  !fly.copy_atom<!fly.universal_copy<128>, 16>,          ← 128-bit load, 元素 16-bit(f16)
  !fly.layout<((8,32),(1,8)):((256,1),(1,32))>,          ← thread layout（下面展开）
  !fly.tile<[32|64]>                                     ← tile 大小 32行×64列
>
```

**thread layout** `((8,32),(1,8)):((256,1),(1,32))` 是一个 **TV layout**（Thread-Value layout），
左半是 T 维（thread 怎么排），右半是 V 维（每个 thread 搬哪些元素）。

Python 构造（L249-256）：

```python
val_per_thr = 8       # 每 thread 搬 8 个 f16 = 128-bit
thrs_col = 64 // 8    # = 8 个 thread 沿 K 方向
thrs_row = 256 // 8   # = 32 个 thread 沿 M 方向

fx.make_layout(
    shape  = ((thrs_col, thrs_row), (1, val_per_thr)),   # ((8,32), (1,8))
    stride = ((thrs_row*val_per_thr, 1), (1, thrs_row))  # ((256,1), (1,32))
)
```

拆解：

```
         T (Thread)          V (Value)
shape:   (8,      32)       (1,    8)
stride:  (256,     1)       (1,   32)
         ↑t_col  ↑t_row            ↑
                                stride=32 的来源
```

- **T 维 `(8,32):(256,1)`**：thread t 分解为 `t_col = t // 32`（K方向），`t_row = t % 32`（M方向）
- **V 维 `(1,8):(1,32)`**：每 thread 搬 8 个 f16，第 v 个 value 在 tile 中偏移 `v × 32`

TV layout 把 (thread, value) 映射到 tile 中的线性位置：

```
position(t, v) = t_col × 256 + t_row + v × 32
```

**stride=32 的来源**：tile 是 32 行 × 64 列（行主序），行方向有 thrs_row=32 个 thread 交错排列。
每个 thread 的 8 个 value 必须跳过其他 31 个 thread 的元素，步长就是 thrs_row=32。

展开看 thread 0（t_col=0, t_row=0）的 8 个 value：

```
v=0: pos = 0   → tile[row=0, col=0]
v=1: pos = 32  → tile[row=0, col=1]     (stride=32 跳过了 31 个其他 thread 的元素)
...
v=7: pos = 224 → tile[row=0, col=7]
→ thread 0 读第 0 行、K 方向连续 8 列 = 128-bit 合并访问
```

256 个 thread 覆盖整个 32×64 tile：

```
         K=0..7        K=8..15       ...  K=56..63
         t_col=0       t_col=1            t_col=7
row 0:  thr 0 (v0~7)  thr 32(v0~7)      thr 224(v0~7)
row 1:  thr 1 (v0~7)  thr 33(v0~7)      thr 225(v0~7)
...
row 31: thr 31(v0~7)  thr 63(v0~7)      thr 255(v0~7)
```

#### 操作数2：被切分的 tensor

```
!fly.memref<f16, #fly_rocdl.buffer_desc, (128,64,?):(?{i64 div=16},1,64)>
              ↑            ↑                ↑              ↑
           元素类型      地址空间          shape          stride
```

就是 gA_k：128 M × 64 K × ? k_tiles。

#### 结果类型

```
!fly.memref<f16, #fly_rocdl.buffer_desc, ((8,1),4,1,?):((1,0),?{div=512},0,64)>
                                          ↑              ↑
                                        shape          stride
```

**结果 shape 的规律**——第一维总是 V (value per copy)，后面是重复维度，最后继承外层维度：

```
dim0 = (8,1) ← V: 每次搬 8 个 f16 (1 是 padding)
dim1 = 4     ← M 方向重复（128 / 32 = 4 块）
dim2 = 1     ← K 方向重复（64 / 64 = 1 块）
dim3 = ?     ← 原 tensor 的 k_tiles，原样继承
```

**结果 stride** 对应：
```
dim0 = (1,0)       ← 8 个 f16 连续排列，padding stride=0
dim1 = ?{div=512}  ← 每个 M-block 偏移 512 倍数 (= 32行 × ?{div=16} stride)
dim2 = 0           ← K 只有 1 块，无意义
dim3 = 64          ← 继承原 tensor 的 k-tile stride
```

**核心**：不看 thread_layout 内部也能读——**结果 shape 直接告诉你每个 thread 搬多少数据**（V × 重复次数）。thread_layout 决定的是哪些元素分给哪个 thread，但结果 shape 已经是 "本 thread 的份" 了。

---

### L112: thr_gA_k = thr_copy_g2s_A.partition_S(gA_k)

g2s tiled_copy 的 thread layout 是 `((8,32),(1,8)):((256,1),(1,32))`（来自 L255 的 `make_layout`）。
- 256 个 thread 沿 (col=8, row=32) 分布
- 每个 thread 一次搬 8 个 f16（128-bit load）

对 gA_k `(128, 64, k_tiles)` 做 partition_S，每个 thread 得到自己负责搬运的 global A 数据：

**IR 证据**（`00_origin.mlir:104`）：
```mlir
%85 = fly.tiled_copy.partition_src(%arg4, %29, %39)
  tiled_copy: universal_copy<128>, thread_layout ((8,32),(1,8)):((256,1),(1,32)), tile [32|64]
  src tensor: (128, 64, ?) : (?{div=16}, 1, 64)
  → ((8,1), 4, 1, ?) : ((1,0), ?{div=512}, 0, 64)
       ↑      ↑  ↑  ↑
       V=8   VM VK  k_tiles
```

- **dim0 = (8,1)**：V = 8 个 f16 = 一次 128-bit load 的数据量（1 是 padding 维）
- **dim1 = 4**：沿 M 方向重复 4 次（128 行 / 32 thread_row = 4 块）
- **dim2 = 1**：沿 K 方向重复 1 次（64 列 / (8 thread_col × 8 val) = 1）
- **dim3 = ?**：k_tiles（沿 K 方向有多少个 BLOCK_K=64 的 tile）

**含义**：每个 thread 从 global A 搬 4 × 1 × 8 = 32 个 f16 / 每个 k_tile。

### L113: thr_sA = thr_copy_g2s_A.partition_D(sA)

同一个 g2s tiled_copy，对 sA（LDS 中的 shared memory）做 partition_D：

**IR 证据**（`00_origin.mlir:105`）：
```mlir
%86 = fly.tiled_copy.partition_dst(%arg4, %84, %39)
  src: sA, S<3,3,3> o 0 o (128,64,2):(64,1,8192)
  → S<3,3,3> o ?{div=8} o ((8,1), 4, 1, 2) : ((1,0), 2048, 0, 8192)
                             ↑      ↑  ↑  ↑
                             V=8   VM VK  stages
```

shape 与 thr_gA_k 的前三维 **完全一致** `((8,1), 4, 1)`——这是 partition_S 和 partition_D 的对称性保证：
**source 第 i 个元素 → destination 第 i 个元素**，一一对应。

区别在 stride：
- thr_gA_k 的 stride 反映 global 内存布局（row-major）
- thr_sA 的 stride 反映 swizzled LDS 布局（stride=2048 = 32行 × 64列）
- 最后一维从 k_tiles 变成 stages=2（LDS 双缓冲）

**offset `?{div=8}`**：每个 thread 在 LDS 中的起始偏移不同（由 tid 决定），swizzle 后除数为 8。

### L115: thr_sA_s2r = thr_copy_s2r_A.partition_S(sA)

这是**另一个** tiled_copy——s2r（shared→register），从 LDS 读数据到 MFMA 寄存器。它的 thread layout 按 MFMA 的 lane 分配生成（`make_tiled_copy_A(buffer_copy_128b, tiled_mma)`），不同于 g2s 的 thread layout。

**IR 证据**（`00_origin.mlir:106`）：
```mlir
%87 = fly.tiled_copy.partition_src(%52, %84, %53)
  tiled_copy: buffer_copy<128>, thread_layout ((16,4,4),(4,(1,2))):((1,128,0),(16,(0,64))), tile [16:1|32:1]
  src: sA (LDS)
  → S<3,3,3> o ?{div=8} o ((8,1), 8, 2, 2) : ((1,0), 1024, 32, 8192)
                             ↑      ↑  ↑  ↑
                             V=8   VM VK  stages
```

- **dim0 = (8,1)**：V = 8 f16 = 128-bit read
- **dim1 = 8**：沿 M 方向 8 块（128 / 16 = 8，因为 MFMA 每次 16 行）
- **dim2 = 2**：沿 K 方向 2 次（BLOCK_K=64 / 32 = 2 个 block_k_iter）
- **dim3 = 2**：stages

**stride 推导**——从 sA 的 base layout 出发：

sA 的 base layout `(128,64,2):(64,1,8192)` 来自 `make_ordered_layout((128,64,2), (1,0,2))`，
order=(1,0,2) 让 dim1（K）分配最紧密的 stride=1 → **K-major**（K 方向连续存储）。

```
sA base: (M=128, K=64, stages=2) : (stride_M=64, stride_K=1, stride_stage=8192)
                                           ↑            ↑
                                    一行宽=BLOCK_K=64   K 连续(stride=1)

s2r tile: [16:1 | 32:1] = 16 M行 × 32 K列

partition 后各维 stride 的来源：
  V  stride = (1, 0)   ← 8 个 f16 沿 K 连续读（stride_K=1）
  VM stride = 1024      = 16 × stride_M = 16 × 64
                          ↑      ↑
                    MFMA M-tile  sA 每行宽 64（因为 K-major）
  VK stride = 32        = 32 × stride_K = 32 × 1
                          ↑      ↑
                   K-tile 宽度   K 连续所以 ×1
  stage stride = 8192    ← 直接继承 sA 的 stage stride
```

**VK stride=32 正是因为 K stride=1**：跳一个 32 列的 K-tile，线性偏移就是 32×1=32。
如果 sA 不是 K-major（比如 M-major, K stride=128），VK stride 就会是 32×128=4096。

**g2s 是 4 块 M × 1 块 K，s2r 是 8 块 M × 2 块 K**——因为 g2s 每 thread 搬更多（32 f16），s2r 每 thread 读更少（8 f16/iter）但更多次。

### L116: thr_gB_k = thr_copy_g2r_B.partition_S(gB_k)

B 不经 LDS，直接 global→register。用 `make_tiled_copy_B(buffer_copy_128b, tiled_mma)` 生成的 tiled_copy。

**IR 证据**（`00_origin.mlir:107`）：
```mlir
%88 = fly.tiled_copy.partition_src(%63, %33, %64)
  tiled_copy: buffer_copy<128>, thread_layout ((16,4,4),(4,(1,2))):((1,512,16),(64,(0,256))), tile [64:1|32:1]
  src: gB_k, ((16,8),(8,8),64) : ((8,65536),(1,128),1024)
  → ((8,1), 2, 2, 64) : ((1,0), 262144, 512, 1024)
      ↑      ↑  ↑   ↑
      V=8   VN VK  k_tiles
```

- **dim0 = (8,1)**：V = 8 f16 = 128-bit load（= buffer_load_dwordx4）
- **dim1 = 2**：沿 N 方向重复 2 次（每 wave 负责 1 个 16×16 N-tile，但 tile [64:1] = 64 行中这个 thread 要覆盖 2 个 16-行段）
- **dim2 = 2**：沿 K 方向 2 次（block_k_iter）
- **dim3 = 64**：k_tiles

### L117: thr_gC = thr_copy_r2g_C.partition_S(gC)

C 的 register→global 写回。

**IR 证据**（`00_origin.mlir:108`）：
```mlir
%89 = fly.tiled_copy.partition_src(%74, %37, %75)
  tiled_copy: buffer_copy<16>, thread_layout ((16,4,4),((4,1),(1,1))):((16,4,256),((1,0),(0,0))), tile [16:1|64:1]
  src: gC, (128,128) : (?{div=16}, 1)
  → ((1,4), 8, 2) : ((0,?{div=16}), ?{div=256}, 64)
      ↑      ↑  ↑
      V     VM VN
```

V = (1,4): 每次写 1 个 f16（buffer_store_short，16-bit copy），沿 M 重复 4 次。
8 个 M-tile，2 个 N-tile。

### partition vs make_fragment 的关系

**partition_S / partition_D** 和 **make_fragment_A / B / C** 是两个**独立的切分体系**，服务于不同的目的：

| | partition_S / partition_D | make_fragment_A / B / C |
|---|---|---|
| **属于** | tiled_copy（搬运指令） | tiled_mma（计算指令） |
| **决定** | 每个 thread 搬运哪些元素 | 每个 thread 持有哪些 MFMA 操作数 |
| **thread 映射** | copy 的 thread layout | MFMA 的 lane 映射 |
| **典型 shape** | `((8,1), M重复, K重复)` | `(4, M重复, K重复)` |

两者的**桥梁**是 `retile`：
```python
mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)
```
retile 把 make_fragment 产生的 MMA layout 重新解释为 copy 能理解的 layout。
这样 `fx.copy` 写入 retile 视图，`fx.gemm` 读取原始 fragment——**同一块 VGPR，两种视角**。

```
partition_S(gA_k) ──copy──→ copy_frag_A ──copy──→ partition_D(sA)     [g2s: global→LDS]
partition_S(sA)   ──copy──→ retile(mma_frag_A)  = mma_frag_A          [s2r: LDS→reg]
                                                    ↓
                                              fx.gemm 直接使用        [MMA: reg→reg]

partition_S(gB_k) ──copy──→ retile(mma_frag_B) = mma_frag_B           [g2r: global→reg]
```

## L119-167: Fragment 分配与 Pipeline 数据流

### L119: copy_frag_A — Global→LDS 搬运的中转寄存器

```python
copy_frag_A = fx.make_fragment_like(thr_sA[None, None, None, 0])  # (VA, VM, VN)
```

`thr_sA` 是 g2s tiled_copy 对 sA 做 `partition_D` 的结果——每个 thread 负责写入 LDS 的哪些位置。
取 `[None, None, None, 0]` 去掉 stage 维度后，`make_fragment_like` 创建一个**同 shape 的寄存器 fragment**。

**IR 证据**（`00_origin.mlir:111-112`）：
```mlir
%91 = fly.slice(%86, (*,*,*,0)) → shape ((8,1),4,1):((1,0),2048,0)   ; 去掉 stage 维
%92 = fly.make_fragment_like(%91) → register, ((8,1),4,1):((1,0),8,0) ; 同 shape, stride 变紧凑
```

shape `((8,1),4,1)` = 8 个 f16 × 4 块 × 1 = 32 个 f16。即每个 thread 每次从 global 搬 32 个 f16（= 4 × buffer_load_dwordx4）到这个中转寄存器，再写入 LDS。

### L121-123: mma_frag_A / B / C — MFMA 需要的寄存器 fragment

```python
mma_frag_A = thr_mma.make_fragment_A(sA[None, None, 0])   # (4, 8, (2,2))
mma_frag_B = thr_mma.make_fragment_B(gB_k, stages=2)      # (4, 2, (2,2), 2)
mma_frag_C = thr_mma.make_fragment_C(gC)                   # ((4,1), 8, 2)
```

这三个 fragment 按 tiled_mma 的 thread 分配，分好了每个 thread 在 MMA 计算中需要持有的寄存器数据。

**IR 证据**（`00_origin.mlir:114-116`）：
```mlir
%95 = fly.mma.make_fragment(a, ...) → register, (4,8,(2,2)):(1,16,(4,8))     ; 128 f16
%96 = fly.mma.make_fragment(b, ...) → register, (4,2,(2,2),2):(1,16,(4,8),32) ; 64 f16
%97 = fly.mma.make_fragment(c, ...) → register, ((4,1),8,2):((1,0),8,4)       ; 64 f32
```

**维度含义**——以 frag_A `(4,8,(2,2))` 为例：
- dim0=4: MFMA atom 内每 lane 持有 4 个 f16（src_a 是 2 个 VGPR）
- dim1=8: 沿 M 方向重复 8 次（128 / 16 = 8 个 M-tile）
- dim2=(2,2): 沿 K 方向 — 2 个 block_k_iter × 每 iter 2 次 MFMA（K-tile 的 k_repeat=2）

frag_B `(4,2,(2,2),2)` 多了最后一维 `2`：**双缓冲 stages**。B 直接 global→register，不经 LDS，所以用寄存器双缓冲。

frag_C `((4,1),8,2)` = 64 个 f32 = 每个 MFMA 输出 4 个 f32 × 8 个 M-tile × 2 个 N-tile（该 thread 所属 wave 负责 1 个 N-tile，但累加了 k_repeat=2 的结果映射到 2 列）。总共 64 f32 = 16 个 VGPR。

### L125-126: retile — 把 MMA fragment 重组为 copy 形状

```python
mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)
mma_frag_B_retile = thr_copy_g2r_B.retile(mma_frag_B)
```

MMA fragment 的 layout 按 MFMA 指令的寄存器分配排列，但 copy 指令（buffer_load / LDS read）需要不同的排列。
`retile` 在**不移动数据**的情况下重解释 layout，让同一块 VGPR 既能被 copy 填充、又能被 MFMA 消费。

**IR 证据**（`00_origin.mlir:117-118`）：
```mlir
%98 = fly.tiled_copy.retile(s2r_copy, %95)
    : (4,8,(2,2)):(1,16,(4,8))                    ; mma_frag_A 原始 layout
   → ((8,1),8,2):((1,0),16,8)                     ; retile 后的 layout

%99 = fly.tiled_copy.retile(g2r_copy, %96)
    : (4,2,(2,2),2):(1,16,(4,8),32)               ; mma_frag_B 原始 layout
   → ((8,1),2,2,2):((1,0),16,8,32)                ; retile 后的 layout
```

retile 后 dim0 变成 `(8,1)` = 8 个 f16 = 1 次 128-bit load 的宽度。

### L160-161: gA_k_stride / gB_k_stride — K 维度步长

```python
gA_k_stride = fx.get_scalar(gA_k.stride[2])   # 打印结果: 64
gB_k_stride = fx.get_scalar(gB_k.stride[2])   # 打印结果: 1024
```

`gA_k.stride` 和 `gB_k.stride` 是 `flat_divide` 后 tensor 的完整 stride tuple，
`stride[2]` 取第 3 维（k-tile 索引维度）的步长——从一个 BLOCK_K 跳到下一个 BLOCK_K 在线性地址上偏移多少个 f16。

#### gA_k: stride = (?{div=16}, 1, 64)

A 是 row-major `(M, K):(K, 1)`，`flat_divide(A, (128, 64))` 的拆分规则是：
**每个维度就地拆为 (tile_size, repeat_count)，tile_size 维在前，repeat_count 维在后**。

```python
A:            (M,         K)
flat_divide:  (128, 64,   M//128, K//64)
               ↑    ↑      ↑       ↑
              dim0  dim1   dim2    dim3
              tile部分      repeat部分
```

**IR 证据**（`00_origin.mlir:46`）：
```mlir
flat_divide(A, [128|64])
  (?, ?) : (?{div=16}, 1)
→ (128, 64, ?, ?) : (?{div=16}, 1, ?{div=2048}, 64)
   ↑    ↑   ↑  ↑
  BM   BK  M块 K块
```

然后 `[None, None, bid_x, None]` 在 dim2（M块）上选 bid_x，dim3（K块）保留：

```mlir
slice((128,64,?,?), (*,*,bid_x,*))
→ (128, 64, ?) : (?{div=16}, 1, 64)
   ↑    ↑   ↑
  BM   BK  K_tiles    ← 这就是 gA_k
```

```
gA_k.stride 打印: (4096d, 1, 64)
                    ↑      ↑   ↑
                 stride_M  K连续 stride[2] = BLOCK_K = 64
```

stride[2] = 64 的含义：k_tile 从 0 跳到 1，地址偏移 64 个 f16 = 128 字节。
因为 A 是 K-连续(stride=1)，跳过 BLOCK_K=64 列就是偏移 64×1=64。

（`4096d` 中的 `d` 表示动态值，实际等于 K=4096 = A 的 leading dimension。`?{div=2048}` = ?{div=16}×128，被 bid_x 选掉后不再出现。）

#### gB_k: stride = ((8, 65536), (1, 128), 1024)

B 经过 preshuffle，layout 是层次化的。下面从 IR 一步步推导 `flat_divide` 为什么得到这个 shape 和 stride。

**起点**：preshuffle_B 的 layout（`00_origin.mlir:24`）：

```
shape:  ((16, 256), (8, 4, 128))
stride: ((8, 65536), (1, 128, 512))
         ↑ N 维度     ↑ K 维度
```

N 总大小 = 16×256 = 4096，K 总大小 = 8×4×128 = 4096。

**`flat_divide(B, [128|64])`**：沿 N 维切 128 一组，沿 K 维切 64 一组。

**N 维切分**（(16, 256) → tile=128）：

从最内层开始吃：
1. 内层 16（stride=8）全部进 tile，已消耗 16
2. 还需 128/16 = 8，从外层 256 中取 8（stride=65536），剩 256/8 = 32 作为 repeat

结果：tile = (16, 8):(8, 65536)，repeat = 32，repeat_stride = 8 × 65536 = **524288**

**K 维切分**（(8, 4, 128) → tile=64）：

从最内层开始吃：
1. 内层 8（stride=1）全部进 tile，已消耗 8
2. 中层 4（stride=128）全部进 tile，已消耗 8×4 = 32
3. 还需 64/32 = 2，从外层 128（stride=512）中取 2，剩 128/2 = 64 作为 repeat

此时 tile 内层结构是 (8, 4, 2):(1, 128, 512)。
但 (4, 2):(128, 512) 的 stride 是连续的（512 = 128×4），编译器合并为 **8:128**。

结果：tile = (8, 8):(1, 128)，repeat = 64，repeat_stride = 2 × 512 = **1024**

**`flat_divide` 输出**（tile dims 在前，repeat dims 在后）（`00_origin.mlir:50`）：

```mlir
fly.flat_divide(%18, [128|64])
  : ((16,256),(8,4,128)):((8,65536),(1,128,512))
  → ((16,8),(8,8),32,64):((8,65536),(1,128),524288,1024)
      ↑N-tile  ↑K-tile  ↑N-rpt ↑K-rpt
```

**`slice(*,*,bid_y,*)`**：选走 dim2（N-repeat=32），得 gB_k（`00_origin.mlir:52`）：

```
gB_k shape:  ((16, 8), (8, 8), 64)
gB_k stride: ((8, 65536), (1, 128), 1024)
              ↑ BN=128     ↑ BK=64   ↑ K_tiles=64
```

**stride[2] = 1024** 的含义：k_tile 从 0 跳到 1，地址偏移 1024 个 f16 = 2048 字节。
来源：从 128（stride=512）中取了 2 个作为 tile，repeat 每步跨 2×512 = 1024。

#### 用法：soffset 偏移

```python
fx.copy(buffer_copy_128b, thr_gA_k[..., 0], copy_frag_A,
        soffset=next_k * gA_k_stride)    # next_k × 64
fx.copy(buffer_copy_128b, thr_gB_k[..., 0], mma_frag_B_retile[..., write_stage],
        soffset=next_k * gB_k_stride)    # next_k × 1024
```

partition 结果始终指向 k=0 的位置，通过 `soffset = next_k × stride` 在 buffer descriptor 上偏移到任意 k-tile，
避免每次循环重新计算 partition。

### L134-168: run_pipeline_stage — 双缓冲流水线的核心

```python
def run_pipeline_stage(read_stage, next_k, read_next=True):
    write_stage = read_stage ^ 1
```

双缓冲：当前从 `read_stage`（0 或 1）读数据做 MMA，同时预取下一个 k-tile 到 `write_stage`。

#### L137-150: 预取下一个 k-tile（global → register）

```python
fx.copy(buffer_copy_128b, thr_gA_k[..., 0], copy_frag_A, soffset=next_k * gA_k_stride)
fx.copy(buffer_copy_128b, thr_gB_k[..., 0], mma_frag_B_retile[..., write_stage], soffset=next_k * gB_k_stride)
```

- A: global → `copy_frag_A`（中转寄存器），稍后 L167 写入 LDS write_stage
- B: global → `mma_frag_B_retile[..., write_stage]`（直接到 MMA 寄存器的 write_stage 槽）

`soffset` 机制：`thr_gA_k[..., 0]` 始终指向 k=0 的 partition，通过 `soffset = next_k * stride` 在 buffer descriptor 上偏移到正确的 k-tile。避免每次重新计算 partition。

#### L152-165: 当前 k-tile 的 MMA 计算

```python
for block_k_iter in fx.range_constexpr(BLOCK_K // 32):   # 64 // 32 = 2 次
    fx.copy(uni_copy_128b, thr_sA_s2r[..., block_k_iter, read_stage], mma_frag_A_retile[..., block_k_iter])
    fx.gemm(tiled_mma, mma_frag_C, mma_frag_A[..., (*, block_k_iter)], mma_frag_B[..., (*, block_k_iter), read_stage], mma_frag_C)
```

每个 block_k_iter 处理 K=32 列：
1. **LDS → register**：从 sA 的 read_stage 读 A 数据到 `mma_frag_A_retile`（= mma_frag_A 的 retile 视图）
2. **GEMM**：用 `mma_frag_A` 的 `(*, block_k_iter)` 切片 + `mma_frag_B` 的 `(*, block_k_iter)` 切片执行 MFMA，累加到 `mma_frag_C`

**注意 retile 的妙处**：`fx.copy` 写入 `mma_frag_A_retile`，`fx.gemm` 读取 `mma_frag_A` —— **它们是同一块 VGPR 的两种视图**。copy 按 `((8,1),8)` 排列写入，MFMA 按 `(4,8,2)` 排列读出，无需数据搬运。

**IR 证据**（`00_origin.mlir:174-179`，block_k_iter=0 展开）：
```mlir
; LDS → register (s2r copy)
fly.copy(uni_copy, sA_s2r[*,*,0,read_stage], mma_frag_A_retile[*,*,0])
    src: S<3,3,3> o ?{div=8} o ((8,1),8):((1,0),1024)    ; LDS 中 8 f16 × 8 行
    dst: register, ((8,1),8):((1,0),16)                    ; VGPR

; GEMM
fly.gemm(tiled_mma, frag_C,
    frag_A[*,*,(*,0)] → (4,8,2):(1,16,4),                 ; 64 f16
    frag_B[*,*,(*,0),read_stage] → (4,2,2):(1,16,4),      ; 16 f16
    frag_C)
```

每次 `fx.gemm` 展开为 **16 次 MFMA**（8 M-tile × 2 k_repeat，见 preshuffle_gemm.md）。
两次 block_k_iter = **32 次 MFMA** / pipeline stage。

#### L167-168: 写入 LDS 并同步

```python
fx.copy(uni_copy_128b, copy_frag_A, thr_sA[..., write_stage])
fx.gpu.barrier()
```

把之前 L139-144 预取到 `copy_frag_A` 的 A 数据写入 sA 的 write_stage。
然后 barrier 确保所有 thread 写完才能进入下一个 pipeline stage（下一轮从 write_stage 读）。

### 完整 pipeline 时序图

```
                        k=0           k=1           k=2           ...
                   ┌──────────┐  ┌──────────┐  ┌──────────┐
copy_frag_A(reg):  │ prefetch │→│ prefetch │→│ prefetch │→ ...
                   │ A[k=0]   │  │ A[k=1]   │  │ A[k=2]   │
                   └────┬─────┘  └────┬─────┘  └────┬─────┘
                        ↓             ↓             ↓
sA (LDS):         ┌─stage0──┐  ┌─stage1──┐  ┌─stage0──┐
                   │ A[k=0]  │  │ A[k=1]  │  │ A[k=2]  │
                   └────┬────┘  └────┬────┘  └────┬────┘
                        ↓             ↓             ↓
mma_frag_A (reg):  ┌── s2r ──┐  ┌── s2r ──┐
                   │ read k=0│  │ read k=1│  ...
                   └────┬────┘  └────┬────┘
                        ↓             ↓
                   ┌── MMA ──┐  ┌── MMA ──┐
mma_frag_C (reg):  │ C += A₀B₀│ │ C += A₁B₁│  ...
                   └─────────┘  └─────────┘

mma_frag_B (reg):  ┌─stage0──┐  ┌─stage1──┐  ┌─stage0──┐
                   │ B[k=0]  │  │ B[k=1]  │  │ B[k=2]  │  (直接 global→reg, 双缓冲)
                   └─────────┘  └─────────┘  └─────────┘
```

关键：**A 走 global→reg→LDS→reg→MMA（三跳），B 走 global→reg→MMA（两跳）**。
A 需要 LDS 是因为 row-major A 沿 M 方向分发给不同 lane 需要转置（通过 LDS swizzle 消除 bank conflict）。
B 经过 preshuffle 后内存布局已经 MFMA 友好，直接 buffer_load 到寄存器即可。