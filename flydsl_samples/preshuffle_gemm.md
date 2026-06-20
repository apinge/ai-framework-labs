# preshuffle_gemm.py 分析：`from_dlpack` 与动态布局

## 1. `from_dlpack` 是什么

```python
tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=1, divisibility=16)
tC = flyc.from_dlpack(C).mark_layout_dynamic(leading_dim=1, divisibility=16)
preshuffle_gemm(tA, preshuffle_B, tC, stream=torch.cuda.current_stream())
```

`flyc.from_dlpack` 的作用是把一个 PyTorch tensor 转换成 FlyDSL 编译器能理解的 **tensor 适配器 (TensorAdaptor)**，使其能作为参数传给 `@flyc.kernel` / `@flyc.jit` 函数。

### 定义位置

`python/flydsl/compiler/jit_argument.py:479`：

```python
def from_dlpack(
    tensor: torch.Tensor,
    *,
    assumed_align: Optional[int] = None,
    use_32bit_stride: bool = False,
) -> TensorAdaptor:
    return TensorAdaptor(tensor, assumed_align, use_32bit_stride, dynamic_layout=False)
```

关键区别：`from_dlpack` 创建的 TensorAdaptor 设置 `dynamic_layout=False`。与之对比，直接传 `torch.Tensor` 给 `@flyc.jit` 时会自动走 `TensorAdaptor(tensor)` 且 `dynamic_layout=True`（默认值），此时自动调用 `_mark_layout_dynamic(leading_dim=-1, divisibility=1)`——维度动态但**无对齐保证**。

### DLPack 协议

DLPack 是一个跨框架张量内存共享协议（PyTorch、TensorFlow、JAX 等通用）。`TensorAdaptor.__init__` 内部调用：

```python
dl = dlpack_tensor.__dlpack__(stream=-1)
self.tensor_adaptor = DLTensorAdaptor(dl, assumed_align, use_32bit_stride)
```

`__dlpack__()` 返回一个 DLManagedTensor capsule，内含：
- `data` — GPU 显存指针
- `shape` — 各维度大小
- `strides` — 各维度步长
- `dtype` — 元素类型
- `device` — 设备信息

`DLTensorAdaptor`（C++ 实现）解析这个 capsule，提取 data pointer、shape、strides，构建 FlyDSL 的 MLIR 类型。

## 2. `from_dlpack` vs 直接传 tensor

| | 直接传 `torch.Tensor` | `flyc.from_dlpack(tensor)` |
|---|---|---|
| `dynamic_layout` | `True`（自动标记所有维度动态，`divisibility=1`） | `False`（shape/stride 编译时烧入，需手动 `mark_layout_dynamic`） |
| 对齐信息 | 无（stride `?{i64}`） | 可通过 `divisibility=N` 传递（stride `?{i64 div=N}`） |
| 用途 | 简单场景，shape 可变但无需对齐优化 | 需精确控制哪些维度动态、且声明对齐保证 |
| 调用链 | `JitArgumentRegistry` 自动匹配 | 手动显式创建 |

## 3. `mark_layout_dynamic(leading_dim=1, divisibility=16)` 的含义

```python
tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=1, divisibility=16)
```

- **`leading_dim=1`**：声明第 1 维（K 维）的 stride 固定为 1（即 row-major 的最内层连续维），这个维度的 stride 不作为运行时参数传入 kernel。
- **`divisibility=16`**：告诉编译器，所有动态的 stride 值保证能被 16 整除。编译器利用该对齐信息，在 IR 中标注 `div=16`，使后续 pass 能推导出更强的对齐，生成更高效的 buffer_load 指令。

效果：对一个 `(M, K)` 的 tensor：
- shape `(M, K)` → 两个维度的大小都变成运行时动态值（`?`）
- stride `(K, 1)` → dim 1 的 stride 固定为 1，dim 0 的 stride 变成运行时动态值 `?{i64 div=16}`

## 4. IR 证据

### 4.1 Origin IR (`00_origin.mlir`)——`div=16` 标注

在最原始的 IR 中，可以直接看到 `from_dlpack` + `mark_layout_dynamic(divisibility=16)` 产生的类型：

```mlir
gpu.func @gemm_kernel_0(
    %arg0: !fly.memref<f16, global, (?,?):(?{i64 div=16},1)>   -- A: 动态布局，stride div=16
    %arg1: !fly.memref<f16, global, ((16,256),(8,4,128)):((8,65536),(1,128,512))>  -- B: 静态 preshuffle 布局
    %arg2: !fly.memref<f16, global, (?,?):(?{i64 div=16},1)>   -- C: 动态布局，stride div=16
    ...
```

**证据 1 — `?{i64 div=16}` 而非 `?{i64}`**

`%arg0`（矩阵 A）和 `%arg2`（矩阵 C）的 stride 类型是 `?{i64 div=16}`：
- shape 部分 `(?,?)` — 两个维度的大小都是运行时动态的
- stride 部分 `(?{i64 div=16},1)` — dim 0 的 stride 是运行时动态 i64 值且**保证被 16 整除**，dim 1 的 stride 固定为 1（`leading_dim=1` 的效果）

这个 `div=16` 正是 `mark_layout_dynamic(divisibility=16)` 注入的编译时提示。如果直接传 `torch.Tensor`（不用 `from_dlpack`），IR 中只会出现 `?{i64}`（无 `div=` 标注，即 `divisibility=1`）。

**证据 2 — `div=16` 在 layout 运算中全程传播**

```mlir
-- get_layout 提取出的 layout 保留 div=16
%5 = fly.get_layout(%arg0) : ... -> !fly.layout<(?,?):(?{i64 div=16},1)>

-- make_view 重建的 memref 也保留 div=16
%10 = fly.make_view(%9, %5) : ... -> !fly.memref<f16, #fly_rocdl.buffer_desc, (?,?):(?{i64 div=16},1)>
```

**证据 3 — 矩阵 B 是完全静态的**

`%arg1` 的类型是 `!fly.memref<f16, global, ((16,256),(8,4,128)):((8,65536),(1,128,512))>`。B 不通过 `from_dlpack` 传入，而是在 `@flyc.jit` 里通过 `fx.Tensor(fx.make_view(...))` 构造了完全静态的 preshuffle 布局，所有维度和步长在编译时已知，不需要运行时参数。

### 4.2 flat_divide 后的对齐推导——`div=16` → `div=2048`

```mlir
%27 = fly.flat_divide(%10, %26)
    : (!fly.memref<f16, #fly_rocdl.buffer_desc, (?,?):(?{i64 div=16},1)>,
       !fly.tile<[128|64]>)
    -> !fly.memref<f16, #fly_rocdl.buffer_desc,
                   (128,64,?,?):(?{i64 div=16},1,?{i64 div=2048},64)>
```

**证据 4 — `div=2048` 是 `div=16` 经过 tile 运算的结果**

`flat_divide` 将 A 的 `(?,?):(?{i64 div=16},1)` 按 tile `[128|64]` 分割后：
- 第 1、2 维是 tile 内坐标，stride 不变：`(?{i64 div=16}, 1)`
- 第 3 维是 tile 间步进（K 方向每 64 个元素跳一个 tile block），stride = `原 stride × 128 / 1 = ?{i64 div=16} × 128 = ?{i64 div=2048}`
- 第 4 维的步进是 K 内部的固定值：`64`

`div=2048 = div=16 × 128`。编译器自动推导出 tile stride 的整除性，这个信息沿 IR 传播到后续的地址计算中。

对比：如果直接传 tensor（`divisibility=1`），这里只会出现 `div=128`（来自 tile 大小本身），无法进一步收紧对齐。

矩阵 C 的 flat_divide 也类似：

```mlir
%35 = fly.flat_divide(%25, %34)
    : ... -> !fly.memref<f16, #fly_rocdl.buffer_desc,
                   (128,128,?,?):(?{i64 div=16},1,?{i64 div=2048},128)>
```

### 4.3 Signature Rewrite Pass (`01_fly_rewrite_func_signature.mlir`)

这个 pass 把高层的 `!fly.memref` 参数拆分成底层的 kernel 参数：

```mlir
gpu.func @gemm_kernel_0(
    %arg0: !fly.ptr<f16, global>,                                              -- A 的 data pointer
    %arg1: !llvm.struct<packed (struct<packed (i32, i32)>, struct<packed (i64)>)>,  -- A 的动态布局描述符
    %arg2: !fly.ptr<f16, global>,                                              -- B 的 data pointer（无描述符，静态）
    %arg3: !fly.ptr<f16, global>,                                              -- C 的 data pointer
    %arg4: !llvm.struct<packed (struct<packed (i32, i32)>, struct<packed (i64)>)>   -- C 的动态布局描述符
)
```

**布局描述符 struct 的结构**：
```
struct<packed (
    struct<packed (i32, i32)>,   -- 动态 shape: (dim0_size, dim1_size)
    struct<packed (i64)>         -- 动态 stride: (dim0_stride,)  ← 只有 1 个，dim1 stride=1 固定不传
)>
```

**证据 5 — stride 重建时保留 `div=16`**

在函数体内解包 struct 重建 layout 时：

```mlir
%5 = llvm.extractvalue %4[0] : !llvm.struct<packed (i64)>           -- 提取运行时 stride 值
%6 = fly.make_int_tuple(%5) : (i64) -> !fly.int_tuple<(?{i64 div=16},1)>  -- 标注 div=16
%7 = fly.make_layout(%3, %6) : ... -> !fly.layout<(?,?):(?{i64 div=16},1)>
```

即使 stride 值已经变成了一个运行时 i64 标量，编译器仍然记住它满足 `div=16` 的不变式。这个信息不是存在值里的，而是编码在 MLIR 类型系统中，沿着所有后续运算传播。

**证据 6 — 只有动态 tensor (A, C) 带布局描述符**

`%arg1` 和 `%arg4` 分别是 A 和 C 的运行时布局信息（各占 16 字节 = 2×i32 + 1×i64）。B（`%arg2`）只传入一个指针，因为布局完全是编译时常量。

### 4.4 ROCDL Lowering (`08_convert_fly_to_rocdl.mlir`)

ROCDL lowering 后，`!fly.ptr` 变成 `!llvm.ptr<1>`（address space 1 = global memory），布局描述符 struct 保持不变：

```mlir
gpu.func @gemm_kernel_0(
    %arg0: !llvm.ptr<1>,     -- A.data_ptr()
    %arg1: !llvm.struct<packed (struct<packed (i32, i32)>, struct<packed (i64)>)>,
    %arg2: !llvm.ptr<1>,     -- B.data_ptr()
    %arg3: !llvm.ptr<1>,     -- C.data_ptr()
    %arg4: !llvm.struct<packed (struct<packed (i32, i32)>, struct<packed (i64)>)>
)
```

## 5. 为什么需要 `from_dlpack` + `mark_layout_dynamic`

在这个 preshuffle GEMM kernel 中：

1. **矩阵 A 和 C 的 leading stride 在运行时可变**。不同调用可能传入不同 M/N/K 的矩阵。使用动态布局让同一编译后的 kernel 可以处理不同尺寸，无需重新编译。

2. **`divisibility=16` 声明对齐，使编译器生成更高效的代码**。FlyDSL 的 buffer_load 指令（`buffer_copy_128b`）需要对齐的地址。声明 stride 可被 16 整除（对 f16 而言 = 32 byte 对齐）后：
   - 入口处 stride 标注 `div=16`
   - `flat_divide` 后自动推导出 tile stride `div=2048`
   - 编译器可以省去运行时对齐检查，直接发射宽向量加载

3. **为什么不直接传 `torch.Tensor`**。直接传也能工作（`dynamic_layout=True` 自动生效），但 `divisibility` 默认为 1，编译器得不到对齐信息。`from_dlpack` + `mark_layout_dynamic(divisibility=16)` 是显式告知编译器"我保证 stride 是 16 的倍数"，换取更优的指令选择。

4. **矩阵 B 不需要 `from_dlpack`**。B 经过 preshuffle 后在 `@flyc.jit` 内部用 `fx.make_view` + `fx.make_layout` 构造了完全静态的布局 `((16,N//16),(8,4,K//32)):((8,16*K),(1,8*16,8*16*4))`，所有维度和步长在编译时已知。

## 6. 总结：数据流全链路

```
torch.Tensor A (shape=[4096,4096], stride=(4096,1), dtype=f16)
    │
    ▼ flyc.from_dlpack(A)                         — dynamic_layout=False，stride 全部烧入
TensorAdaptor (静态)
    │
    ▼ .mark_layout_dynamic(leading_dim=1, divisibility=16)
TensorAdaptor (shape 动态, dim1 stride=1 固定, dim0 stride 动态且 div=16)
    │
    ▼ @flyc.jit trace + codegen
00_origin.mlir: !fly.memref<f16, global, (?,?):(?{i64 div=16},1)>
    │                                                ↑ div=16 来自 divisibility=16
    ▼ flat_divide by tile [128|64]
00_origin.mlir: (128,64,?,?):(?{i64 div=16},1,?{i64 div=2048},64)
    │                                           ↑ div=2048 = div=16 × 128
    ▼ fly_rewrite_func_signature pass
01_*.mlir: ptr<f16> + struct{(i32,i32), (i64)}  ← 运行时传入 shape 和 stride
    │        stride 重建为 !fly.int_tuple<(?{i64 div=16},1)>
    ▼ convert_fly_to_rocdl pass
08_*.mlir: llvm.ptr<1> + struct{...}  ← 直接映射到 GPU kernel 参数
    │
    ▼ gpu_module_to_binary pass
HSACO binary: kernel 接收 {ptr, shape0, shape1, stride0} 四个 scalar 参数
```

## 7. 对齐信息传播对比

| 传入方式 | 入口 stride 类型 | flat_divide 后 tile stride |
|---|---|---|
| 直接传 `torch.Tensor`（`divisibility=1`） | `?{i64}` | `?{i64 div=128}` |
| `from_dlpack` + `divisibility=16` | `?{i64 div=16}` | `?{i64 div=2048}` |

`div=2048` 比 `div=128` 提供了 16 倍更强的对齐保证，编译器可以在地址计算中省去更多动态取模/掩码操作。

## 8. `make_tiled_mma` 的 atom_layout 和 make_tile 参数

### 8.1 两个 kernel 的 tiled_mma 对比

**preshuffle_gemm**（L237-240）：

```python
tiled_mma = fx.make_tiled_mma(
    fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.Float16)),     # MFMA atom
    fx.make_layout((1, 4, 1), (0, 1, 0)),                        # atom_layout
    fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4))),  # make_tile (K-tile)
)
```

**small_gemm_bf16**（L90-92）：

```python
tiled_mma = fx.make_tiled_mma(
    mma_atom,                                    # MFMA(16, 16, 32, BF16)
    fx.make_layout((2, 2, 1), (1, 2, 0))         # atom_layout
    # 无 make_tile 参数
)
```

### 8.2 atom_layout: 4 个 wave 怎么分工

`make_tiled_mma` 的第二个参数 `atom_layout` 定义了 256 个线程（= 4 个 wave）在 M/N/K 三个方向上怎么分配 MFMA atom。

**small_gemm: `(2, 2, 1):(1, 2, 0)`**

```
M 方向 2 个 atom × N 方向 2 个 atom = 4 个 atom
atom 编号 = m_idx*1 + n_idx*2

wave 0 → atom(0,0) → C[  0:16,  0:16]
wave 1 → atom(1,0) → C[ 16:32,  0:16]
wave 2 → atom(0,1) → C[  0:16, 16:32]
wave 3 → atom(1,1) → C[ 16:32, 16:32]

4 个 wave 覆盖 32×32 的 C，block=64×64 → M_rep=2, N_rep=2
```

IR 证据（`small_gemm_bf16_dump/00_origin.mlir` L47）：
```mlir
!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x32, (bf16, bf16) -> f32>>,
               !fly.layout<(2,2,1):(1,2,0)>>
```

frag_C shape = `((4,1), 2, 2):((1,0), 8, 4)` — M_rep=2, N_rep=2。

ISA 证据（`small_gemm_bf16_dump/22_final_isa.s`）：
```
16 条 v_mfma_f32_16x16x32_bf16
= M_rep(2) × N_rep(2) × K_rep(4) = 16  ✓
```

**preshuffle_gemm: `(1, 4, 1):(0, 1, 0)`**

```
M 方向 1 个 atom × N 方向 4 个 atom = 4 个 atom
atom 编号 = m_idx*0 + n_idx*1

wave 0 → atom(0,0) → C[ 0:16,  0:16]
wave 1 → atom(0,1) → C[ 0:16, 16:32]
wave 2 → atom(0,2) → C[ 0:16, 32:48]
wave 3 → atom(0,3) → C[ 0:16, 48:64]

4 个 wave 沿 N 方向排列，覆盖 16×64 的 C，block=128×128 → M_rep=8, N_rep=2
```

IR 证据（`preshuffle_gemm_ir/00_origin.mlir` L5）：
```mlir
!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>,
               !fly.layout<(1,4,1):(0,1,0)>,
               !fly.tile<[*|*|(4,4,2):(1,8,4)]>>
```

frag_C shape = `((4,1), 8, 2):((1,0), 8, 4)` — **M_rep=8**, N_rep=2。

### 8.3 为什么 preshuffle_gemm 用 (1,4,1) 而不是 (2,2,1)

**与 preshuffle 直接相关。**

preshuffle 的 sub_k 段是按 **N 方向 16 行连续**排列的（16 行 × 8 列 = 128 字节连续）。4 个 wave 沿 N 排列意味着：

- wave 0 负责 N[0:16]，wave 1 负责 N[16:32]，wave 2 负责 N[32:48]，wave 3 负责 N[48:64]
- 每个 wave 内的 16 个线程 load B 的同一个 sub_k 段（16 行连续）→ 完美 coalescing

如果换成 (2,2,1)，wave 0 和 wave 1 分别负责 N[0:16] 和 N[16:32]，但各自只有 8 行的数据连续——coalescing 效果打折。

IR 证据：frag_B shape 对比：

```
preshuffle_gemm frag_B: (4, 2, (2,2), 2):(1, 16, (4,8), 32)
                         ↑  ↑   ↑↑    ↑
                        val N_rep K-tile stages
  N_rep=2: 每个 wave 在 N 方向只重复 2 次（因为 atom 已覆盖 64）

small_gemm frag_B: (8, 2, 4):(1, 32, 8)
                    ↑  ↑  ↑
                   val N_rep K_rep
  N_rep=2: 每个 wave 在 N 方向重复 2 次（atom 只覆盖 32）
```

### 8.4 make_tile 第三参数：K-tile layout `(4,4,2):(1,8,4)`

`make_tile(None, None, K_tile_layout)` 的三个参数对应 M/N/K：
- `None`：M 方向不额外分 tile（使用默认按 rep 展开）
- `None`：N 方向不额外分 tile
- `fx.make_layout((4,4,2),(1,8,4))`：**K 方向按这个 layout 交错排列**

small_gemm 没有 `make_tile`，K 方向就是简单的连续排列：
```
K_rep=4 次 MFMA, 每次消耗 K=32
第 1 次: k[0:31]
第 2 次: k[32:63]
第 3 次: k[64:95]
第 4 次: k[96:127]
```

preshuffle_gemm 有 K-tile `(4,4,2):(1,8,4)`，K 方向**交错**排列（详见 preshuffle_explained.md §4）：
```
block_k_iter=0 内的 2 次 MFMA:
  MFMA #0: k[0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27]
  MFMA #1: k[4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31]
```

**交错的目的**：让每个线程两次 MFMA 需要的 B 数据在内存中连续（k[0:7]），匹配 128-bit load 和 preshuffle 的 sub_k=8 分段。

IR 证据——fly.gemm 的 frag_A/B 传入 shape：

```mlir
-- preshuffle_gemm (line 179): gemm 接收的 A 和 B 都有 K 维=2
fly.gemm(%arg3, %97,
    %195,  -- frag_A: (4, 8, 2):(1, 16, 4)   ← K dim=2 (2次MFMA)
    %197,  -- frag_B: (4, 2, 2):(1, 16, 4)   ← K dim=2 (2次MFMA)
    %97)

-- small_gemm (line 100): gemm 接收的 A 和 B 有 K 维=4
fly.gemm(%26, %72,
    %70,   -- frag_A: (8, 2, 4):(1, 32, 8)   ← K dim=4 (4次MFMA)
    %71,   -- frag_B: (8, 2, 4):(1, 32, 8)   ← K dim=4 (4次MFMA)
    %72)
```

ISA 证据——preshuffle_gemm 每次 MFMA 的 src_b 是 2 个 VGPR（4 个 f16）：

```asm
-- buffer_load 读 8 f16 到 v[88:91]
buffer_load_dwordx4 v[88:91], v76, s[16:19], 0 offen

-- 拆成 2 次 MFMA，前4和后4
v_mfma_f32_16x16x16_f16 v[12:15], v[96:97], v[88:89], v[12:15]   ; B=v[88:89] (k[0:3])
v_mfma_f32_16x16x16_f16 v[12:15], v[98:99], v[90:91], v[12:15]   ; B=v[90:91] (k[4:7])
```

### 8.5 MFMA 数量对比

| | small_gemm | preshuffle_gemm |
|---|---|---|
| MFMA atom | 16×16×32, bf16 | 16×16×16, f16 |
| atom_layout | (2,2,1) → M=2, N=2 | (1,4,1) → M=1, N=4 |
| block tile | 64×64×128 | 128×128×64 |
| make_tile | 无（K 连续） | K-tile (4,4,2):(1,8,4) 交错 |
| M_rep | 2 | 8 |
| N_rep | 2 | 2 |
| K per gemm | 4（连续） | 2（交错） |
| MFMA per fly.gemm | 2×2×4 = 16 | 8×2×2 = 32 |
| ISA 总 MFMA | 16 | 256（循环体，含 8 次 fly.gemm） |

### 8.6 总结

1. **atom_layout `(1,4,1)` vs `(2,2,1)`**：决定 4 个 wave 的空间分布。preshuffle_gemm 把 4 个 wave 全排在 N 方向，配合 preshuffle 的 N-优先内存布局实现完美 coalescing。small_gemm 把 4 个 wave 分在 M 和 N 各 2 个，对行主序矩阵更自然。

2. **make_tile K-tile `(4,4,2):(1,8,4)`**：把 K 方向的 MFMA 交错排列，使每个线程两次 MFMA 需要的 B 数据（8 个 f16）在内存中连续，匹配 preshuffle 的 sub_k=8 分段和 128-bit load。不用 preshuffle 的 kernel（如 small_gemm）不需要 K-tile 交错，K 方向按顺序即可。

3. **三者协同**：atom_layout 决定 wave 分布 → preshuffle 决定内存排列 → K-tile 决定 MFMA 执行顺序 → 128-bit load 连接内存和寄存器。改任何一个都要相应调整其他参数。
