# `--moe-a2a-backend none` vs `--moe-a2a-backend mori`

`--moe-a2a-backend none` 这里sglang并不能指定nccl作为moe a2a backend，由于nccl这样的底层库本身并不实现dispatch combine这样的功能，
`--moe-a2a-backend none`为什么做ep还能work，和`--moe-a2a-backend mori`的区别是本文要讨论的。

这里先不讨论 Mori/DeepEP kernel 细节，只看 SGLang 里两种模式的基本数据流。

记号：

```text
T: 本 rank 当前要处理的 token 数
H: hidden size
K: router top-k
E: 全局 routed expert 数
ep_size: expert parallel size
E_local = E / ep_size

hidden_states: [T, H]
topk_ids:     [T, K]
topk_weights: [T, K]
```

以 Qwen3.5-397B 这个 log 为例：

```text
E = 512
ep_size = 4
E_local = 128
H = 4096
moe_intermediate_size = 1024
K = 10
```

所以 AITER fused_moe log 里看到：

```text
('gfx942', 80, 32768, 4096, 1024, 128, 9, ...)
```

可以理解成：

```text
token     = 32768
model_dim = 4096
inter_dim = 1024
expert    = 128   # 本 rank 的 local experts，不是全局 512
topk      = 9     # AITER tuning key 里的 routed topk，EP 下可能扣掉 fake/masked slot
```

这个模型的 `moe_intermediate_size` 本来就是 1024。在这组 `tp_size=4, ep_size=4` 下：

```text
moe_tp_size = tp_size / ep_size = 1
```

所以单个 expert 的 intermediate 不再按 TP 切；切的是 expert 个数：

```text
512 experts -> 每张 rank 128 experts
```

## `none`

`--moe-a2a-backend none` 是最朴素的 EP/TP MoE 路径：没有专门的 MoE all-to-all dispatch/combine backend。

它的大致流程是：

```text
1. router 得到 topk_ids / topk_weights
2. StandardDispatcher 看当前 rank 负责哪些 experts
3. 不属于本 rank 的 expert_id 被 mask/remap
4. 本 rank fused_moe 只计算本 rank local experts 的贡献
5. 最后用普通 all_reduce 把各 rank 的 partial output 加起来
```

`none` 模式可以理解为：

```text
MoE 输入 hidden_states 已经在每张 rank 上都有一份
token 不被专门 dispatch 到 expert owner rank
每张 rank 在本地筛 expert、算自己那份贡献
最后 all_reduce 合并
```

因此它更像：

```text
local expert mask + local MoE + all_reduce
```

而不是：

```text
dispatch token -> remote expert -> combine back
```

这个地方容易误解：`none` 不是“不搬 token 还算远端 expert”。它能算对，是因为每张 rank 本来就有当前 batch 的 `hidden_states [T, H]`。

举个最小例子：

```text
E = 4 experts
ep_size = 2

rank0 owns: expert0, expert1
rank1 owns: expert2, expert3

token0 topk = [expert0, expert3]
token1 topk = [expert2, expert3]
```

在 `none` 路径下，rank0 和 rank1 都有：

```text
hidden_states[token0]
hidden_states[token1]
```

注意这句话只是在解释 `none` 为什么能用 all_reduce 算对。对 `mori` / `deepep` 要分两个时刻看：

```text
dispatch 之前:
  在普通 TP/EP 配置下，每张 EP rank 通常也有当前这批 token 的 hidden_states [T, H]。
  这些 hidden_states 是 dispatch 的发送源。

dispatch 之后:
  每张 rank 收到的是 packed_recv_hidden / recv_x。
  它只包含“需要本 rank local experts 处理”的 token hidden 向量。
  它不是完整原始 [T, H]，也不是每张卡都有所有 token。
```

所以 `mori` / `deepep` 不是靠“大家都有 hidden_states，然后本地 mask，最后 all_reduce”来算对；它是把 hidden vector 按 expert owner 发过去，local expert 算完后再 combine 回原 token 位置。

如果开启 DP attention / scattered token 之类的路径，还要更小心：dispatch 前每张 attn-TP rank 可能只看到 token 的一部分，不一定有完整 batch。SGLang 的 DeepEP 路径里会额外处理这种 gather/scatter。

rank0 只算自己拥有的 experts：

```text
token0: expert0 有效，expert3 不是本 rank 的，mask 掉
token1: expert2/expert3 都不是本 rank 的，全 mask 掉

rank0_output[token0] = w0 * expert0(token0)
rank0_output[token1] = 0
```

rank1 也只算自己拥有的 experts：

```text
token0: expert3 有效，expert0 不是本 rank 的，mask 掉
token1: expert2/expert3 都有效

rank1_output[token0] = w3 * expert3(token0)
rank1_output[token1] = w2 * expert2(token1) + w3 * expert3(token1)
```

最后普通 all_reduce sum：

```text
final = all_reduce_sum(rank0_output, rank1_output)

final[token0] = w0 * expert0(token0) + w3 * expert3(token0)
final[token1] = w2 * expert2(token1) + w3 * expert3(token1)
```

所以 `none` 的通信不是消失了，而是从 MoE 前面的 token dispatch，变成了 MoE 后面的 dense output all_reduce：

```text
none = 复制/共享输入 hidden_states + 本地 expert mask + local MoE + all_reduce output
```

通信上，`none` 不使用 `--moe-a2a-backend` 里的专门 A2A 机制。底层普通 collective 在 NVIDIA 上通常是 NCCL，在 AMD ROCm 上通常也是 PyTorch 的 NCCL backend 名字，但实现是 RCCL。

所以不能写：

```bash
--moe-a2a-backend nccl
--moe-a2a-backend rccl
```

要看这个朴素路径，应显式写：

```bash
--moe-a2a-backend none
```

并在 log 里确认：

```text
moe_a2a_backend='none'
```

## `mori`

`--moe-a2a-backend mori` 是真正的 EP dispatch/combine 路径。

它的大致流程是：

```text
1. router 得到 topk_ids / topk_weights
2. Mori dispatcher 根据 expert_id 判断目标 rank
3. dispatch 把 token hidden state 发到拥有对应 expert 的 rank
4. 每张 rank 收到 packed_recv_hidden / recv_topk_ids / recv_topk_weights
5. 本 rank fused_moe 只对收到的 token 跑 local experts
6. combine 把 expert 输出送回原 token 所在位置，并按 topk_weights 加权求和
```

从“学习 EP”的角度看，`mori` 模式就是：

```text
router -> dispatch -> local experts -> combine
```

这里 combine 合并的是 expert output vectors：

```text
final[token] = sum_k topk_weights[token, k] * expert_output[token, k]
```

也就是把多个 expert 对同一个 token 的输出向量加权加起来，回到 `[T, H]`。

Mori path 的 log 会很明显，例如：

```text
MoRI MoE is enabled
[MORI init] world_size=4 rank=0 hidden_size=4096 ... num_local_experts=128 router_topk=10
```

如果看到这些，就不是 `none` 路径。

## 关键区别

| 项目 | `none` | `mori` |
|---|---|---|
| MoE A2A backend | 没有专门 backend (其实没做a2a) | Mori dispatch/combine |
| token 是否按 expert owner rank 发过去 | 否 | 是 |
| 每张 rank 算什么 | 本地 expert 对当前 token 的 partial contribution | 收到的 token 在本地 expert 上的输出 |
| 聚合方式 | 普通 all_reduce 汇总 partial output | Mori combine 送回并加权聚合 |
| SGLang dispatcher | `StandardDispatcher` | `MaybeTboDeepEPDispatcher` / Mori implementation |
| 适合理解什么 | local expert mask、remap、all_reduce | 真 EP 的 dispatch/combine |

## 疑问：既然每张卡都有 hidden_states，为什么 `none` 不一定最高效？

这个疑问很自然：

```text
如果 MoE 之前每张卡本来就有 hidden_states，
那不搬 token，直接每张卡本地算 local experts，
最后 all_reduce 一下，是不是应该比 mori/deepep 更快？
```

我的理解是：**不一定**。

原因是 `none` 不是“没有通信”，而是把通信放在 MoE 之后：

```text
none:
  省掉 MoE 前的 dispatch
  但是 MoE 后要聚合完整 output [T, H]
```

`mori` / `deepep` 是另一种代价模型：

```text
mori/deepep:
  MoE 前按 expert owner 搬 token hidden vector
  expert owner 算本地 expert
  MoE 后 combine 回原 token 位置
```

所以真正比较的不是：

```text
none = 不通信
mori/deepep = 多通信
```

而是：

```text
none = 不 dispatch token，但做 dense output reduce
mori/deepep = dispatch routed token，再 combine routed output
```

SGLang 代码里也能看到这个分叉。

`--moe-a2a-backend none` 会走 `StandardDispatcher`：

```text
sglang/srt/layers/moe/fused_moe_triton/layer.py
create_moe_dispatcher():
  a2a_backend.is_none() -> StandardDispatcher
```

`StandardDispatcher` 里面基本不搬 `hidden_states`。它主要做的是把全局 expert id remap 到本 rank 的 local expert id；不属于本 rank 的 expert 会变成无效 id：

```text
sglang/srt/layers/moe/token_dispatcher/standard.py
topk_ids = local_expert_mapping[topk_ids]
return hidden_states
```

这说明 `none` 的语义确实是：

```text
当前 rank 已经有 hidden_states
当前 rank 只算自己拥有的 experts
输出只是 partial contribution
```

然后 partial contribution 需要在 MoE 后聚合。`FusedMoE.forward_impl()` 里有 post-experts reduce 的分支：

```text
sglang/srt/layers/moe/fused_moe_triton/layer.py
if reduce_results and (moe_tp_size > 1 or moe_ep_size > 1):
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
```

有些模型/配置可能把这个 reduce 融合、下沉、或者替换成别的 collective，但核心语义还是：`none` 路径需要把各 rank 的 partial output 合成完整 output。

而 `mori` / `deepep` 会走 `MaybeTboDeepEPDispatcher` / DeepEP-style dispatcher：

```text
sglang/srt/layers/moe/fused_moe_triton/layer.py
a2a_backend.is_deepep() / is_mori() -> MaybeTboDeepEPDispatcher
```

Mori 里能看到 dispatch 输入是原始 `hidden_states/topk_weights/topk_ids`，输出变成 `packed_recv_hidden`：

```text
sglang/srt/layers/moe/token_dispatcher/moriep.py
mori_op.dispatch(hidden_states, topk_weights, scale, topk_ids)
-> packed_recv_hidden, recv_topk_weights, recv_topk_ids, packed_recv_count
```

DeepEP 里也是先根据 `topk_ids` 计算 layout，再 dispatch 成 `recv_x`：

```text
sglang/srt/layers/moe/token_dispatcher/deepep.py
get_dispatch_layout(topk_ids, ...)
buffer.dispatch(x, topk_idx=topk_ids, ...)
-> recv_x, recv_topk_ids, recv_topk_weights, num_recv_tokens_per_expert
```

所以 `mori/deepep` 的 MoE core 输入不是原始完整 `[T, H]`，而是已经按 expert owner 收到的 packed token hidden。

## Profile

`--moe-a2a-backend none` 没有combine和dispatch只有最后的rccl all reduce
![a2a backend none](./ep_a2a_backend_none.png)


`--moe-a2a-backend mori` 有dispatch和combine
![a2a backend mori](./ep_a2a_backend_mori.png)


两种情况 moe都是`aiter::fmoe_bf16_pertokenFp8_g1u1_vs_silu_1tg_ps_32x512`

但 profile 里有一个重要发现：**kernel 名字相同，不代表实际喂进去的 MoE workload 相同**。


在 `TP-1 / EP-1` 这个 rank 上，按 GPU kernel 统计：

```text
none:
  aiter::fmoe_bf16_pertokenFp8_g1u1_vs_silu_1tg_ps_32x512
  count = 60
  total = 138.825 ms
  avg   = 2313.8 us
  p50   = 2319.5 us
  所有调用都 >= 1.2 ms

mori:
  aiter::fmoe_bf16_pertokenFp8_g1u1_vs_silu_1tg_ps_32x512
  count = 300
  total = 136.395 ms
  avg   = 454.7 us
  p50   = 138.6 us
  其中 60 次是长调用，另外大量调用只有 100~300 us，甚至有接近空的调用
```

所以不能只看：

```text
两个 profile 里都出现了同名 AITER fmoe kernel
```

然后推出：

```text
它们做的是同样的 MoE 计算
```

这一步是不对的。这个 kernel 名字更像是 AITER 选到的 kernel 实现/tile，例如 `1tg_ps_32x512`，但动态输入 token 数、expert 数、valid token-expert pairs、padding、shared expert 是否融合，都可能不同。

从 PyTorch record shapes 里看，`aiter::fmoe_g1u1` 的输入也不一样。

`none` 路径里这个 rank 的形状大致是：

```text
hidden_states: [8000, 4096]
output:        [8000, 4096]
w13:           [129, 2048, 4096]
w2:            [129, 4096, 1024]
topk_weights:  [8000, 1]
```

这里的 `129` 很关键：它不是纯 routed local experts 的 `128`，而是多了一个 fused shared expert slot。

这和 SGLang 代码能对上。Qwen MoE 在普通 `none` 路径下可以把 shared expert append 到 topk 里，一起塞进 fused MoE：

```text
sglang/srt/models/qwen2_moe.py
_append_shared_to_topk_output(...)
```

并且注释里也写了 allreduce-EP path 下 shared expert 是 single global slot，每个 EP rank 都会算一份 shared output，然后靠 `1 / ep_size` scale 抵消后面的 all_reduce 重复求和。

`mori` 路径里这个 rank 的形状大致是：

```text
hidden_states: [32768, 4096]
output:        [32768, 4096]
w13:           [128, 2048, 4096]
w2:            [128, 4096, 1024]
topk_weights:  [32768, 1]
```

这里的 `32768` 不是说 `--chunked-prefill-size 8192` 没生效。server log 里已经能看到：

```text
chunked_prefill_size=8192, max_prefill_tokens=8192
[MORI init] world_size=4 ... num_max_dispatch_tokens_per_rank=8192
Prefill batch ... #new-token: 8192 ... #pending-token: 7538
Prefill batch ... #new-token: 7538 ... #pending-token: 0
```

所以 scheduler 层的 prefill chunk 确实被切成了 `8192 + 7538`。trace 里 Mori 路径的 `aiter::fmoe_g1u1` 看到 `[32768, 4096]`，更像是 EP dispatch 后给本 rank fused MoE 准备的最大接收/工作区容量：

```text
32768 = 8192 tokens per rank * 4 EP ranks
```

也就是说，Mori init 用 `num_max_dispatch_tokens_per_rank=8192` 初始化通信/dispatch 能力，但进入 local expert 计算前，backend 要给“来自所有 EP rank 的 token”预留/暴露一个合并后的 buffer。profiler record shapes 记录的是这个 fmoe 调用看到的 buffer shape，不等价于调度器一次给某个 rank 塞了 32768 个本地 prefill token。

trace 里还有一个很直接的旁证：`fmoe_g1u1` 后面马上出现把一维 buffer reshape 回 `[8192, 4096]` 的操作，说明最终 combine/返回给本 rank 的 hidden states 仍然回到本 rank 的 prefill chunk 规模。

这里是 `128`，没有那个 fused shared expert slot。SGLang 代码里也能看到，如果是 DeepEP/Mori 这类 backend 且 `moe_ep_size > 1`，Qwen shared-expert fusion 会被关掉，改走 separate shared expert MLP：

```text
sglang/srt/models/qwen2_moe.py
if enable_shared_expert_fusion and uses_per_rank_fused_shared_slots() and moe_ep_size > 1:
    enable_shared_expert_fusion = False
```

这解释了一个现象：`none` 和 `mori` 虽然都出现同名 AITER fmoe kernel，但它们的 MoE 子图并不完全等价。

```text
none:
  routed experts + fused shared expert slot
  local expert count = 129
  MoE 后还要做 EP/TP partial output reduce

mori:
  routed experts 走 dispatch/combine
  local expert count = 128
  shared expert 可能走 separate MLP
  dispatch 后的 hidden_states 是 backend 准备好的 buffer / packed view
```

因此 `none` 的同名 fmoe kernel 更慢，可能不是因为 AITER kernel 本身换了，而是因为：

```text
1. local expert 数不同：none 是 129，mori 是 128
2. none 把 shared expert 融到了 routed MoE 里，mori/DeepEP-class backend 会禁用这个 fusion
3. none 的 local mask/remap 后，AITER 仍要处理本地 topk/padding/sorting 布局
4. mori 的 dispatch/combine 已经把 token/expert 数据整理成更适合 EP 的输入形式
5. mori 路径里有很多同名 fmoe kernel 调用很短，说明同名 kernel 的动态有效工作量差异很大
```

我现在对这个 profile 的理解是：

```text
不能说：
  none 和 mori 都用了同一个 aiter::fmoe... kernel，
  所以 MoE kernel workload 一样。

应该说：
  两者最终都调用了同一种 AITER kernel 实现，
  但进入 kernel 前的数据布局、local expert 数、shared expert 处理方式、valid token-expert pairs 都不一样。
```

这也解释了为什么 `--moe-a2a-backend none` 的 MoE kernel 可能更长：它不是单纯“省掉 dispatch/combine 后的同一份计算”，而是走了另一套 `StandardDispatcher + fused shared expert + post reduce` 的数据流。

### 什么时候 `none` 可能更好？

`none` 可能在这些情况下比较有竞争力：

```text
EP size 小
top-k 比较大
每个 token 的 experts 分布到很多 rank 上
post output all_reduce 很快
dispatch/combine backend 的固定开销比较明显
```

这时“省掉 dispatch/combine”可能真的划算。

### 什么时候 `mori/deepep` 可能更好？

`mori/deepep` 可能在这些情况下更好：

```text
EP size 大
routing 比较稀疏
每个 rank 只需要处理一部分 token-expert pairs
packed 后的 expert input 更适合 grouped GEMM
dispatch/combine 可以和计算 overlap
backend 支持 fp8/fp4 dispatch、专门 buffer、专门 kernel
```

这时虽然多了 dispatch/combine，但它避免了“每张卡都围绕完整 `[T, H]` output 做 dense reduce”的代价，也让每张卡的 expert 计算更贴近自己真正拥有的 experts。

我现在的结论是：

```text
不能只看“MoE 前每张卡是否有 hidden_states”。

none 的优势:
  数据流简单，不做 token A2A dispatch。

none 的代价:
  每张卡都产生 partial output，需要 MoE 后聚合完整 [T, H]。

mori/deepep 的代价:
  MoE 前后有 dispatch/combine。

mori/deepep 的优势:
  expert owner rank 只处理发给自己的 packed routed tokens，
  通信和计算有机会被专门 backend 优化。
```

## 总结

```text
none:
  不搬 token 到 expert，大家本地算自己负责的 expert，最后 all_reduce。

mori:
  按 expert owner 搬 token，专家所在 rank 算完，再 combine 回原 token。
```

所以如果目标是先理解最 basic 的 SGLang EP，不看 Mori 通信 kernel，可以先看 `none`。

如果目标是理解真实的 production EP dispatch/combine，就看 `mori` 或 `deepep`。
