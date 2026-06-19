# FlyDSL Kernel 如何对应 IR 行号

## 结论：默认 loc 不能用来映射 Python 源码

`00_origin.mlir` 里的 `loc("-":41:13)` **既不是 Python 源码行号，也不是 IR 文件行号**。

原因是编译管道中的 round-trip：

```
tracing
  → ops 携带正确的 Python 文件路径 + 行号
  → 例如 loc("/root/.../small_gemm_bf16.py":76:0)

jit_function.py L800:
  module = ir.Module.parse(
      module.operation.get_asm(enable_debug_info=False)
  )
  ↓ get_asm(enable_debug_info=False) 输出不含 location
  ↓ ir.Module.parse() 重新解析文本
  ↓ MLIR 用 "-" 做文件名，用中间文本的行列号做 loc
  ↓ 原始 Python 行号全部丢失

00_origin.mlir 里的 loc("-":41:13)
  → 41 是 round-trip 中间 MLIR 文本的行号
  → 和 Python 源码、最终 IR 文件行号都无关
```

### 验证

直接对比三组行号，可以确认它们互不对应：

| Python API | Python 文件行号 | IR 文件行号 | loc 报的行号 |
|---|---|---|---|
| `make_mma_atom` | 76 | 48 | 41 |
| `partition_S(bA)` | 159 | 89 | 82 |
| `fx.gemm` | 229 | 105 | 98 |

三列数字各不相同。

---

## 通用方法 1：靠 op 名称 + SSA 变量追踪（推荐）

不依赖 `loc`，靠两条线索定位：

### 1. Python API → IR op 名称映射表

| Python API | IR op 名称 |
|---|---|
| `fx.make_tile(m, k)` | `fly.static` |
| `fx.rocdl.make_buffer_tensor(T)` | `fly.rocdl.make_buffer_resource` |
| `fx.zipped_divide(T, tile)` | `fly.zipped_divide` |
| `fx.slice(T, idx)` | `fly.slice` |
| `fx.make_mma_atom(...)` | `fly.make_mma_atom` |
| `fx.make_tiled_mma(atom, layout)` | `fly.make_tiled_mma` |
| `fx.make_copy_atom(...)` | `fly.make_copy_atom` |
| `fx.make_tiled_copy_A/B/C(...)` | `fly.make_tiled_copy` |
| `thr_copy.partition_S(src)` | `fly.tiled_copy.partition_src` |
| `thr_copy.partition_D(dst)` | `fly.tiled_copy.partition_dst` |
| `thr_mma.make_fragment_A(t)` | `fly.mma.make_fragment(a, ...)` |
| `thr_mma.make_fragment_B(t)` | `fly.mma.make_fragment(b, ...)` |
| `thr_mma.make_fragment_C(t)` | `fly.mma.make_fragment(c, ...)` |
| `thr_copy.retile(frag)` | `fly.tiled_copy.retile` |
| `fx.copy(atom, src, dst)` | `fly.copy` |
| `fx.gemm(atom, C, A, B, C)` | `fly.gemm` |
| `frag.fill(0)` | `fly.fill` |

### 2. SSA 变量追踪数据流

找到 IR op 后，看它的 SSA 参数来确认对应关系。例如：

```
Python:  copy_src_A = thr_copy_A.partition_S(bA)

IR:      %67 = fly.tiled_copy.partition_src(%43, %19, %64)
              ↑ %43 = tiled_copy_A (看 make_tiled_copy 行)
              ↑ %19 = bA           (看 fly.slice 行)
              ↑ %64 = tid slice
```

### 实操步骤

```bash
# 1. dump IR
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 your_kernel.py

# 2. 找目标 op
grep "make_fragment\|partition_src\|fly.gemm\|fly.copy" \
    /root/.flydsl/debug/gemm_kernel_0/00_origin.mlir

# 3. 看结果类型拿 shape (→ 后面的部分)
#    例如 → !fly.memref<bf16, register, (8,2,1):(1,8,0)>
#    shape (8,2,1) 就是 frag_A 的 (val, M_rep, K_rep)

# 4. 看 SSA 参数追踪数据来源
#    %67 的参数 %43 是什么? grep "%43 =" 即可
```

---

## 通用方法 2：启用 debug info 保留真实 Python 行号

```bash
FLYDSL_DUMP_IR=1 \
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 your_kernel.py
```

设置 `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` 后，`jit_function.py` L800 的 round-trip
会保留 location（`get_asm(enable_debug_info=True)`），loc 恢复为完整路径：

```
loc("/root/workspace/FlyDSL/splitk_gemm/small_gemm_bf16.py":76:0)
```

注意：此模式下部分 op 的 loc 指向 **FlyDSL 框架内部**（如 `derived.py`、`primitive.py`），
而不是用户 kernel 文件。这是因为 `traced_op` 的 `_caller_location(depth=1)` 在某些
嵌套调用中落到了框架代码帧，而非用户代码帧。函数级别的 loc（如 `gpu.func`）始终指向
用户 kernel 文件。

---

## 对照示例：`small_gemm_bf16.py` 完整映射

以下是默认模式（`enable_debug_info=False`）下的完整 Python↔IR 对照，
靠 op 名称和 SSA 编号对应，不依赖 loc：

```
Python 变量         IR op                          SSA    IR行   结果 shape
──────────────────  ─────────────────────────────  ────   ────   ──────────────────
bA                  fly.slice                      %19    L41    (64,32)
bB                  fly.slice                      %22    L44    (64,32)
bC                  fly.slice                      %25    L47    (64,64)
mma_atom            fly.make_mma_atom              %26    L48    mfma<16x16x32>
tiled_mma           fly.make_tiled_mma             %30    L52    +layout<(2,2,1)>
copy_atom_ab        fly.make_copy_atom             %32    L54    buffer_copy<128>
copy_atom_c         fly.make_copy_atom             %33    L55    buffer_copy<32>
tiled_copy_A        fly.make_tiled_copy            %43    L65    TV:((16,4,2,2),(8,(1,1)))
tiled_copy_B        fly.make_tiled_copy            %53    L75    TV:((16,4,2,2),(8,(1,1)))
tiled_copy_C        fly.make_tiled_copy            %63    L85    TV:((16,8,2),((4,1),(1,1)))
copy_src_A          fly.tiled_copy.partition_src   %67    L89    ((8,1),2,1)
copy_src_B          fly.tiled_copy.partition_src   %68    L90    ((8,1),2,1)
copy_dst_C          fly.tiled_copy.partition_src   %69    L91    ((1,4),2,2)
frag_A              fly.mma.make_fragment(a)       %70    L92    (8,2,1)
frag_B              fly.mma.make_fragment(b)       %71    L93    (8,2,1)
frag_C              fly.mma.make_fragment(c)       %72    L94    ((4,1),2,2)
copy_frag_A         fly.tiled_copy.retile          %73    L95    ((8,1),2,1)
copy_frag_B         fly.tiled_copy.retile          %74    L96    ((8,1),2,1)
copy_frag_C         fly.tiled_copy.retile          %75    L97    ((1,4),2,2)
fx.copy(A)          fly.copy                       —      L98    —
fx.copy(B)          fly.copy                       —      L99    —
fx.gemm             fly.gemm                       —      L105   —
fx.copy(C)          fly.copy                       —      L106   —
```

IR 文件：`/root/.flydsl/debug/gemm_kernel_0/00_origin.mlir`

---

## 根本原因（代码级）

`python/flydsl/compiler/jit_function.py` L800：

```python
module = ir.Module.parse(
    module.operation.get_asm(enable_debug_info=env.debug.enable_debug_info)
)
```

- `env.debug.enable_debug_info` 默认 `False`（由 `FLYDSL_DEBUG_ENABLE_DEBUG_INFO` 控制）
- `get_asm(enable_debug_info=False)` 输出不含 `loc(...)` 的 MLIR 文本
- `ir.Module.parse(text)` 重新解析时，MLIR 自动给每个 op 打上 `loc("-":行:列)`
  - `"-"` 是 MLIR 对字符串输入的默认文件名
  - 行列号是 **MLIR 中间文本内的位置**，和 Python 源码无关
- 之后 `get_asm(enable_debug_info=True)` dump 时写出这些无意义的 loc
