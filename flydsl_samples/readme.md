# 一些简单的flydsl学习笔记

- 版本 Version: 0.2.0.dev635

- 部分api参考 https://github.com/ROCm/FlyDSL/tree/main/.claude/skills

- 使用skill小建议
```
cd FlyDSL
mkdir -p .claude/skills
git submodule add https://github.com/org/your-skill-name .claude/skills/your-skill-name
```
- roofline分析依据[理论上限](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350p.html)

- 和hipkittens对比

## fp8 gemm example
entry

```
python3 FlyDSL/tests/kernels/test_fp8_gemm_rowscale.py --wave_8 --preshuffle_b 

```
`--wave_8`这个输入走的8wave的版本

默认 M=N=K=8192, tile_m=tile_n=256, static_weight_scale=True


编译期常量
```
BLOCK_K	= 128
K_ITERS	 = 64
N_TILES_A/B	= 4 / 2
LDS_BLOCK_M/N =	128 / 128
a/b_lds_size  16384 元素
LDS = 16384*8/1024= 128 KB (CDNA4 160KB是够用的)
grid_x:	32 × 32 = 1024 blocks
```

### pipeline分析

### api分析

#### flyc.from_dlpack

有一些静态常量的定义
```python
 # if statement: static_weight_scale
    if static_weight_scale:
        # 用 from_dlpack 包装 → 告诉编译器这些是"静态常量"
        # 编译时可以把 B/scale_b 的指针硬编码，避免运行时传参开销
        b_flat = flyc.from_dlpack(b_flat)
        sa_flat = flyc.from_dlpack(sa_flat)
        sb_flat = flyc.from_dlpack(sb_flat)
```

#### preshuffle

```python
# in fp8_gemm_utils.py
def preshuffle_b(b_t):
    """Permute row-major ``B_T`` ``(N, K)`` for ``b_preshuffled=True``."""
    n, k = b_t.shape[-2:]
    assert n % 16 == 0 and k % 64 == 0, f"need N%16==0 and K%64==0, got N={n} K={k}"
    return b_t.reshape(n // 16, 16, k // 64, 4, 16).permute(0, 2, 3, 1, 4).contiguous()

# 调用 b 也就是[N,K] 已经被转置过了 所以
b_kernel = preshuffle_b(b_q) if b_preshuffled else b_q
```
- 原始 `[N,K]`
- reshape后:  `[N/16, 16_rows, K/64, 4_groups, 16_cols]`
- permute后:  `[N/16, K/64,  4_groups, 16_rows, 16_cols]`  ← 行和列的位置互换

### rocdl.flat_work_group_size

"rocdl.flat_work_group_size": "512,512" 是一个 ROCDL/AMDGPU 的函数属性，格式是 "min,max"。

  它告诉编译器这个 kernel 的 workgroup 大小固定为 512 个线程（最小 512，最大 512）。对应到下面 .launch(block=(512,
  1, 1)) 的配置。

https://github.com/llvm/llvm-project/blob/release/22.x/clang/include/clang/Basic/Attr.td#L2440

### xcd_remap

 xcd_remap 只在 4-wave kernel 里实现：
```python
# xcd_remap_bx_by 的核心逻辑：
_linear_id = bx * gx + by           # 原始线性 id
_wgid = (linear_id % num_xcds) * wgs_per_xcd + (linear_id // num_xcds)
# 效果：先按 XCD 轮转，再在每个 XCD 内按 M 方向排列
# → 同一 XCD 内的 block 共享 A 的行 → L2 hit 率↑
```
### 和hipkittens 对比
fp8 8wave
```
HipKittens/kernels/gemm/fp8fp32/FP8_8wave# ./tk_kernel 
Matrix dimensions: 8192x8192x8192, CUs: 256
Warmup iterations: 500, Timing iterations: 100

Running optimized kernel (matmul_device)...
Initializing 4 rotating buffer sections (640 MB total, A+B only)...
Block 0: max(a) = 4.500000, max(b) = 4.500000
Block 1: max(a) = 4.500000, max(b) = 4.500000
Block 2: max(a) = 4.500000, max(b) = 5.000000
Block 3: max(a) = 4.500000, max(b) = 4.500000
Buffer initialization complete.
Running reference kernel (matmul_device_ref)...

=== PERFORMANCE RESULTS ===
Reference kernel (matmul_device_ref):
  Kernel time (best): 9276.478 ms,  TFLOPS: 0.12
  Kernel time (avg ): 9276.478 ms,  TFLOPS: 0.12

Optimized kernel (matmul_device):
  Kernel time (best): 1.107 ms,  TFLOPS: 993.22
  Kernel time (avg ): 1.113 ms,  TFLOPS: 987.48

Speedup (best): 8379.70x
Speedup (avg ): 8331.31x

Correctness: PASSED
```


fp8 4wave
```
 ./tk_kernel 
Matrix dimensions: 8192x8192x8192, CUs: 256
Warmup iterations: 500, Timing iterations: 100

Running optimized kernel (matmul_device)...
Initializing 4 rotating buffer sections (640 MB total, A+B only)...
Block 0: max(a) = 4.500000, max(b) = 4.500000
Block 1: max(a) = 4.500000, max(b) = 4.500000
Block 2: max(a) = 4.500000, max(b) = 5.000000
Block 3: max(a) = 4.500000, max(b) = 4.500000
Buffer initialization complete.
Running reference kernel (matmul_device_ref)...

=== PERFORMANCE RESULTS ===
Reference kernel (matmul_device_ref):
  Kernel time (best): 9511.932 ms,  TFLOPS: 0.12
  Kernel time (avg ): 9511.932 ms,  TFLOPS: 0.12

Optimized kernel (matmul_device):
  Kernel time (best): 0.392 ms,  TFLOPS: 2805.98
  Kernel time (avg ): 0.424 ms,  TFLOPS: 2594.16

Speedup (best): 24274.67x
Speedup (avg ): 22442.17x

Correctness: PASSED
```