# moe 

从头分析MOE (Top-k Sparse MoE + Shared Expert )的原理， 以 [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)TP8 为例，介绍torch基本计算（torch reference），kernel算法优化，ROCM上各种量化的区别和联系 和在framework里的实现。

## 从数学理解到矩阵计算

Torch golden（expert 体，**不含 router**）：[`torch_moe_golden.py`](torch_moe_golden.py)

- `torch_experts` 原样摘自 `vllm/tests/kernels/utils.py`（L826–L964）
- 测试直接传入 `topk_weight` / `topk_ids`，不经 `softmax` + `topk`
- 默认 shape：`M=1024, K=4096, N=128, E=512, topk=10`（bf16）
- 运行：`PYTHONPATH=/opt/vllm python3 torch_moe_golden.py`



## 1. 符号

| 符号 | 含义 | 取值 |
|------|------|----------|
| $M$ | token 数 | 1024 |
| $K$ | hidden 维 | 4096 |
| $N$ | 每个 expert 的 intermediate（TP在这个维度切的） | 128 |
| $E$ | expert 个数 | 512 |
| $T$ | 每个 token 选的 expert 数 | 10 |

张量（与 `torch_experts` / `test_moe` 一致）：

- 输入：$`\mathbf{h}_t \in \mathbb{R}^{K}`$，$`t=0,\ldots,M-1`$（一行 hidden）
- 权重（expert $e$）：
  - $\mathbf{W}^{(1)}_e \in \mathbb{R}^{2N \times K}$ → `w1[e]`，shape `(256, 4096)`，上半是 gate、下半是 up
  - $\mathbf{W}^{(2)}_e \in \mathbb{R}^{K \times N}$ → `w2[e]`，shape `(4096, 128)`
- 路由结果（**已给定**，不算 softmax）：
  - $e_{t,k} = \mathit{topk\\_ids}[t,k]$
  - $`\alpha_{t,k} = \mathit{topk\_weights}[t,k]`$ （按行归一化，$`\sum_k \alpha_{t,k}=1`$ ）

---

## 2. 单个 expert 在算什么（SwiGLU FFN）

对 expert $e$，输入 $\mathbf{h}\in\mathbb{R}^K$：

**① Gate + Up（合并成一次线性）**

$$
\mathbf{z} = \mathbf{h}\,\mathbf{W}^{(1)\top}_e \in \mathbb{R}^{2N}
$$

拆成两半（各 $N$ 维）：

$$
\mathbf{z} = [\mathbf{g};\ \mathbf{u}],\quad \mathbf{g},\mathbf{u}\in\mathbb{R}^N
$$

**② SwiGLU 激活**

$$
\mathrm{SiLU}(\mathbf{g}) = \mathbf{g} \odot \sigma(\mathbf{g}),\quad \sigma \text{ 为 sigmoid}
$$

$$
\mathbf{a} = \mathrm{SiLU}(\mathbf{g}) \odot \mathbf{u} \in \mathbb{R}^N
$$

（代码里 `SiluAndMul`：对合并后的 $\mathbf{z}$ 做 `silu(前N) * 后N`。）

**③ Down 投影回 hidden**

$$
\mathbf{o} = \mathbf{a}\,\mathbf{W}^{(2)\top}_e \in \mathbb{R}^K
$$

记这个映射为 $\mathrm{Expert}_e(\mathbf{h})$：

$$
\mathrm{Expert}_e(\mathbf{h}) = \mathbf{W}^{(2)\top}_e \,\Big( \mathrm{SiLU}(\mathbf{g}) \odot \mathbf{u} \Big),\quad [\mathbf{g};\mathbf{u}] = \mathbf{h}\mathbf{W}^{(1)\top}_e
$$

这就是一个 **SwiGLU 版的两层 MLP**（中间维 $N$，无 bias 时与标准 SwiGLU block 一致）。

---

## 3. Top‑k Sparse MoE：token $t$ 的最终输出

对第 $t$ 个 token，选 $T=10$ 个 expert，**加权求和**：

$$
\mathbf{y}_t = \sum_{k=1}^{T} \alpha_{t,k}\,\mathrm{Expert}_{e_{t,k}}(\mathbf{h}_t) \in \mathbb{R}^{K}
$$

矩阵写法（整批 $M$ 个 token）：

$$
\mathbf{Y} \in \mathbb{R}^{M\times K},\quad \mathbf{y}_t^\top = \sum_{k=1}^{T} \alpha_{t,k}\, \mathrm{Expert}_{e_{t,k}}(\mathbf{h}_t)^\top
$$

这就是 `torch_experts` 最后做的事：先对每个 $(t,k)$ 算 expert 输出，再按 $\alpha_{t,k}$ 在 $k$ 上 `sum`（代码里先展成 `M*T` 行，乘 weight 再 `view(M,T,K).sum(dim=1)`）。



## 4. 代入数字

对某一个 $(t,k)$：

1. $`\mathbf{h}_t`$: `(4096,)`
2. $`\mathbf{z} = \mathbf{h}_t \mathbf{W}^{(1)\top}_{e_{t,k}}`$: `(256,)` = gate `(128,)` + up `(128,)`
3. $`\mathbf{a}`$: `(128,)`
4. $`\mathbf{o}_{t,k} = \mathbf{a}\mathbf{W}^{(2)\top}_{e_{t,k}}`$: `(4096,)`
5. $`\mathbf{y}_t = \sum_{k=1}^{10} \alpha_{t,k}\,\mathbf{o}_{t,k}`$: `(4096,)`

整批输出 `out`: `(1024, 4096)`。

---


在每个 token 的 hidden 向量上，对 router 已选出的 $T$ 个 expert 各做一遍 SwiGLU FFN（$K\to 2N\to N\to K$），再用 $\alpha_{t,k}$ 把这 $T$ 个 $K$ 维向量加起来，得到该 token 的 MoE 层输出。


## 5. 简单的伪代码
```python
"""
shape
hidden (1024, 4096) w1 (512, 256, 4096) w2 (512, 4096, 128)
topk_weight (1024, 10) topk_ids (1024, 10)
"""
out = torch.zeros(num_token, hidden_size) # (1024, 4096)

for t in range(num_token): # 0,1,...1023
  for k in range(topk): # 0,1,...9
    e = topk_ids[t,k] 
    gemm1_out = hidden_states[t]@ w1[e].T # (4096) @(4096,256) = (256)
    gate, up = gemm1.chunk(2, dim=-1) #(256) => (128) , (128)
    tmp = torch.nn.functional.silu(gate) * up # still (128)
    out[t] += topk_weight[t,k]*(tmp@w2[e].T) # topk_weight[t,k]是个标量，（128)@(128,4096) = (4096)

return out
```

## 算法优化 (torch)

刚才的伪代码需要循环整个 `num_token (M)` ，torch的实现一般改为循环`num_experts(E)`+ mask, vllm的参考代码和aiter和
[aiter的torch reference](https://github.com/ROCm/aiter/blob/d0c313d78eb04b495f6d126a281fe9e29a8d2d89/aiter/fused_moe.py#L1110)都采取了类似实现。

先稍微复习下torch mask的用法。
```python
import torch
mask = [True, False,False,False]
a = torch.tensor([1,2,3,4],device="cuda")
print(a[mask]) # tensor([1], device='cuda:0')
out = torch.tensor([5,6,7,8],device="cuda")
out[mask] = a[mask] 
print(out) # tensor([1, 6, 7, 8], device='cuda:0')
```
### 1. 先展平
```python
# M=1024, topk=10  →  10240 行
a = hidden.view(M, 1, K).repeat(1, topk, 1).reshape(M * topk, K)
topk_ids = topk_ids.view(-1)   # shape (10240,)
out = zeros(M * topk, K)
```
### 2. 按 expert 循环 + mask
```python
for i in range(num_experts):          # i = 0..511
    mask = (topk_ids == i)  # 哪些行选中了 expert i
    if mask.sum():
        tmp1 = a[mask] @ w1[i].T      # GEMM1，一批行一起算
        tmp2 = SiluAndMul(tmp1)         # SwiGLU
        out[mask] = tmp2 @ w2[i].T      # GEMM2
```
### 3. 加权合并会每个token
```python
(out.view(M, topk, K) * topk_weight.unsqueeze(-1)).sum(dim=1)  # → (1024, 4096)
# topk_weight 是(1024, 10) ， unsqueeze(-1) 在最后加了 1的轴 变为（1024, 10，1）
# 接下来是张量 A: (1024, 10, 4096) 和张量 B: (1024, 10,    1)的元素乘，B广播为(1024, 10, 4096)
# .sum(dim=1) 在10 这个维度求和
```
## 算法优化 (常见的kernel实现)

我们在kernel里一般不采取torch那样的循环循环`num_experts(E)`+ mask。 

一种典型的实现 为 `moe_sorting` => `GEMM1 (gate up) ` => `SwiGLU` => `GEMM2 (down)` => `atomic / reduce`，并且根据prefill和decode，可能有各种优化或者简化。 非常典型地，由于decode一次推理token较少第一步`moe_sorting`和最后一步`atomic / reduce`, 可以有所简化。 再次我们暂时只虑数据量较大的prefill阶段，介绍这几个步骤。

### 1. moe_sorting

以[aiter.fused_moe.moe_sorting](https://github.com/ROCm/aiter/blob/v0.1.14/aiter/fused_moe.py#L98)为例

```python
sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
        topk_ids, # （num_tokens, num_experts_per_tok）(1024,10) 
        topk_weight,  # （num_tokens, num_experts_per_tok） (1024,10)
        E,   # num_experts 512
        N2,     # reduce dim is same with output dim, i.e. hidden_size, 4096
        hidden_states.dtype, # (num_tokens, hidden_size）
        TILE_M, # 128 算 kernel用的
        None,
        None,
        0,
    )
# sorted_ids.shape 75766
# sorted_weights.shape, 75766
#  sorted_expert_ids.shape 592
#  num_valid_ids tensor([65536,  1024])
# cur_out torch.Size([1024, 4096]) 预分配的 MoE 输出 buffer  (num_tokens, hidden_size）
```
`cur_out`这里是结果所需要的buffer很好理解。主要看其他几个。
- `sorted_ids` 每条记录 哪个 token、第几个 topk 槽, 元素是int类型，长度为 M×TOPK + E×TILE_M − TOPK (10240 + 512×128 − 10=75766)
- `sorted_weights` 与 `sorted_ids` 对应 记录路由权重 元素为float类型
- `sorted_expert_ids` 每个 `TILE_M`维的block属于哪个expert 75766 / 128 = ⌈592.07⌉ = 592
- `num_valid_ids`有效的元素信息, 在这里包含两个元素 `tensor([65536,  1024])`,第二个是这次的`num_token`，第一个表示kernel只处理`sorted_ids[0 : max_id)` 表示上界。65536 / 128 = 512 个 block  →  正好 512 个 expert，每个 expert 占 1 个 TILE_M=128 的块（在「每个 expert 都被命中」是个均匀分布 （实际上不一定这样）
按 expert 分组、按 TILE_M 分块、带 padding」的 launch 表，处理成GPU友好的布局。