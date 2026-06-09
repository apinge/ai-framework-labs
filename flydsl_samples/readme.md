# 一些简单的flydsl学习笔记

- 版本 Version: 0.2.0.dev635

- 部分api参考 https://github.com/ROCm/FlyDSL/tree/main/.claude/skills

- 使用skill小建议
```
cd FlyDSL
mkdir -p .claude/skills
git submodule add https://github.com/org/your-skill-name .claude/skills/your-skill-name
```
- roofline分析依据[理论上限](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html)


- 和hipkittens对比

## fp8 gemm example
entry

```
python3 FlyDSL/tests/kernels/test_fp8_gemm_rowscale.py --wave_8 --preshuffle_b 

[flyc] Throughput: 421.0 us, 2611.91 TFLOPS, BW: 0.638 TB/s
```
`--wave_8`这个输入走的8wave的版本

默认 M=N=K=8192, tile_m=tile_n=256, static_weight_scale=True

roofline分析
```
FLOPs = 2 × M × N × K = 2 × 8192^3 ≈ 1.0995 × 10^12 FLOPs (约 1.1 TFLOP)

Time = 421.0 us
TFLOPS = 1.0995e12 / 421e-6 ≈ 2611 TFLOPS ✓ (与报告一致)
```
峰值算力4.6 PFLOPs， 除以下 差不多 56.7%


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


### 流水线分析1-任务分配
用 8192×8192 例子

- 整体任务分配
```
M=N=K=8192, BLOCK_M=BLOCK_N=256, BLOCK_K=128
─────────────────────────────────────────────
Grid:  (8192/256)*(8192/256)=32 × 32 = 1024 blocks（每个block算一个 BLOCK_M*BLOCK_N=256×256 的 C tile）
Block: 512 threads = 8 waves
K-iters: 8192 / 128 = 64 次 K 循环

每个 block 总工作量:
  (256, 8192)@(256,8192)
  分到 8 waves
  ───────────────
```
- 一个wave计算哪些
```
block 内的 256×256 输出 tile，8 个 wave 这样切分：

       cols 0..31  32..63  64..95  96..127 128..159 160..191 192..223 224..255
            │       │       │        │        │        │        │        │
rows 0..63  │ w0/c00│ w1/c00│ w2/c00 │ w3/c00 │ w0/c01 │ w1/c01 │ w2/c01 │ w3/c01
            ├───────┼───────┼────────┼────────┼────────┼────────┼────────┼────────
rows 64..127│ w4/c00│ w5/c00│ w6/c00 │ w7/c00 │ w4/c01 │ w5/c01 │ w6/c01 │ w7/c01
            ├───────┼───────┼────────┼────────┼────────┼────────┼────────┼────────
rows128..191│ w0/c10│ w1/c10│ w2/c10 │ w3/c10 │ w0/c11 │ w1/c11 │ w2/c11 │ w3/c11
            ├───────┼───────┼────────┼────────┼────────┼────────┼────────┼────────
rows192..255│ w4/c10│ w5/c10│ w6/c10 │ w7/c10 │ w4/c11 │ w5/c11 │ w6/c11 │ w7/c11

每个 wave 持有 4 个 c_frag (c00/c01/c10/c11)，分散在 4 个象限
单个 wave 总输出: 4 × (64×32) = 8192 个 bf16
8 waves × 8192 = 256×256 ✓
```
注意每个64X32的块 代表 8个mfma的结果(16X16), 一次k iter 要做完4个(64X32)块对应的计算
- 一个 wave 一轮 K-iter 干啥
每轮 K-iter共 64 轮(8192//128)，单个 wave 要做：
```
读寄存器 (S2R, 从LDS):
  a0_frag = 4 个 (32xfp8)  (从 a_cur0 读，覆盖A的上半64行)
  a1_frag = 4 个 (32xfp8)  (从 a_cur1 读，覆盖A的下半64行)
  b0_frag = 2 个 (32xfp8)  (从 b_cur0 读，覆盖B的左半32列)
  b1_frag = 2 个 (32xfp8)  (从 b_cur1 读，覆盖B的右半32列)

计算 (MFMA):
  c00 += a0 × b0    内含 4×2 = 8 条 MFMA
  c01 += a0 × b1    内含 8 条
  c10 += a1 × b0    内含 8 条
  c11 += a1 × b1    内含 8 条
                    ──────
                    32 MFMA / wave / K-iter

64 K-iters → 每个wave每block共 2048 MFMA
```
- 关键：LDS 是 block 共享的，8 waves 一起搬数据！
```
b_g2s.load(b_cur0, ...)  # ← 这一行由所有 512 threads(8X64) 同时执行
```

每个 wave 用 wave_id 计算自己的 LDS 写入位置：
```
step_off = wave_id * 1024 + step * 8192
```
每次 b_g2s.load 的搬运量：
```
8 waves × 64 lanes × 16 fp8/lane × 2 steps = 16384 fp8 = 一个 LDS half-tile (128×128)
```
所以"加载 b_cur0"这件事是 8 个 wave 一起完成的，单个 wave 只搬 1/8。
- Pipeline 在做什么
没有 pipeline 的"裸版"K-iter（串行）：
```
[G2S k] → [等G2S完成] → [S2R k] → [MFMA k] → [G2S k+1] → ...
                       ▲
                       这里 MFMA 单元闲着，G2S 单元也闲着
```
有 pipeline 的版本（重叠）：
```
时间轴 →
K-iter k:    [S2R][MFMA c00]  [G2S k+1]
                  [S2R][MFMA c01]    [G2S k+2 partial]
                       [S2R][MFMA c10]      [G2S k+2 cont]
                            [S2R][MFMA c11]
                            ↑ 计算单元几乎不闲
```
- 每条 MFMA 之间都插入了一次 G2S 发射，让内存单元和 MFMA 单元都满载
- [s_setprio(1/0)]fp8_gemm_8wave.py ) 锁定 MFMA 优先发射
- wait_barrier(N) 控制最多有 N 个 G2S in-flight，防止内存子系统过载

### 计算总共需要多少lds
```
单个 MFMA 16×16×128 需要:
  A: 16行 × 128K = 2048 fp8
  B: 128K × 16列 = 2048 fp8

8 waves 联合覆盖:
  A: 16 × 128 × 8 = 16384 fp8  ← 这个数字是对的！
```
对应 a_lds_size = LDS_BLOCK_M × BLOCK_K = 128 × 128 = 16384 fp8  

BLOCK_M/N 被切成两半（A0/A1, B0/B1)
```
BLOCK_M = 256 行
  ↓ 切成 A0(rows 0~127) + A1(rows 128~255)
LDS_BLOCK_M = 128

→ 需要 2 个 A buffer (A0_lds, A1_lds)
→ 需要 2 个 B buffer (B0_lds, B1_lds)
```


ping-pong 双缓冲（cur + next）
```
cur 缓冲: 当前 K-iter k 的数据 (S2R + MFMA 在用)
next 缓冲: 下一个 K-iter k+1 的数据 (G2S 正在搬)
                     ↓
              swap 后 cur ↔ next
```
完整汇总
```
单个 buffer:           16384 fp8  = 16 KB    ← 你算到这里
                          × 2  (A0 + A1)
─────────────────────────────────
A-side cur:            32768 fp8  = 32 KB
                          × 2  (B 同理)
─────────────────────────────────
所有 cur:              65536 fp8  = 64 KB    ← 你的"32KB×2"差不多到这
                          × 2  (cur + next ping-pong)
─────────────────────────────────
LDS 总量:             131072 fp8  = 128 KB  ✓
```
所以总结：你算的 32KB ×2 (A0/A1切分) ×2 (ping-pong) = 128KB

### 流水线分析2 

流水线也可以参考[HipKittens FP8_8wave](https://github.com/HazyResearch/HipKittens/blob/main/kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu)

层1: G2S(k+2) 与 MFMA(k) 并行
层2: ping-pong 双缓冲 cur/next 交替
层3: 4个 MFMA(c00/01/10/11) 与 G2S 交织



```
s_setprio(1)  ← 告诉调度器：优先发射 MFMA，别插入内存指令
  c00 = mfma(...)
s_setprio(0)  ← 恢复正常优先级，允许 G2S 指令穿插
```

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
