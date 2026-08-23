# EP MoE 

本文sglang代码基于
https://github.com/apinge/sglang/tree/qwen3_5_tp4_ep4_mi308x

docker base rocm/sgl-dev:v0.5.16-rocm720-mi30x-20260727

模型和config见[Qwen3.5-397B-A17B-PTPC-FP8](https://huggingface.co/sammysun0711/Qwen3.5-397B-A17B-PTPC-FP8)

model launch
```
export SGLANG_DISABLE_CUDNN_CHECK=1
#export SGLANG_USE_CUDA_IPC_TRANSPORT=1 --mm-feature-transport=cuda_ipc
export SGLANG_VLM_CACHE_SIZE_MB=8192

export SGLANG_USE_AITER=1

export USE_AITER_COMM=1
export USE_HIP_LINEAR_ATTN=1
export SGLANG_USE_AITER_NEW_CA=false
export SGLANG_USE_IPC_POOL_HANDLE_CACHE=1

export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8192

model=/models/Qwen/Qwen3.5-397B-A17B-PTPC-FP8
python3 -m sglang.launch_server \
        --port 7080 \
        --model-path ${model} \
        --tp-size 4 \
        --ep-size 4 \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --enable-multimodal \
        --trust-remote-code \
        --chunked-prefill-size 8192 \
        --mem-fraction-static 0.7 \
        --max-prefill-tokens 8192 \
        --max-running-requests 32 \
        --attention-backend aiter \
        --mm-attention-backend aiter_attn \
        --kv-cache-dtype fp8_e4m3 \
        --moe-a2a-backend mori \
        --disable-custom-all-reduce \
        --watchdog-timeout 1200 \
        --disable-radix-cache  2>&1 | tee launch_qwen3.5-397B-fp8_tp4_mori.log
```
## EP的基本概念
Expert 不再复制/切分在每个 GPU 上，而是把不同 expert 放到不同 GPU

dispatch → MoE → combine



```
                 GPU0
                  │
             Router / TopK
                  │
             expert_id
                  │
                  ▼
            ┌──────────┐
            │ Dispatch │
            └────┬─────┘
                 │
          ┌──────┼──────┐
          │      │      │
          ▼      ▼      ▼
        GPU0   GPU1   GPU2 ...
          │      │      │
          ▼      ▼      ▼
        Expert Expert Expert
          │      │      │
          └──────┼──────┘
                 │
            ┌────▼─────┐
            │  Combine │
            └────┬─────┘
                 │
                 ▼
            原 token 顺序
```

因此 combine 不只是：“把数据发回来”还涉及：
**token 的原始位置、expert assignment、top-k 权重、排序/反排序**。
这恰恰是 EP 实现非常容易出 bug 的地方

## Topy EP

[ep_dispatch_combine_topy.py](./ep_dispatch_combine_toy.py)

  TP MoE：

  token 不跨机器/卡搬家
  每张 TP rank 都有每个 expert 的一片权重
  所以主要问题是：
    把 token 按 expert 分组 + padding/alignment
    让 grouped GEMM 好算

  这就是你 readme 里 L216-L227 那种 sorted_token_ids / sorted_expert_ids / num_tokens_post_padded 的东西。

  EP MoE：

  每张 EP rank 只负责一部分 experts
  所以先要根据 topk_ids 判断：
    这个 token 的某个 expert 在哪张 rank 上
  然后 dispatch 把 token hidden state 发到那个 rank
  收到以后，本 rank 再把收到的 token 按本地 expert 分组
  然后做 expert GEMM
  最后 combine 把 expert 输出送回原 token 所在 rank，并按 topk_weight 加权求和

  所以 EP 比 TP 多了通信这层：

  router topk
    -> dispatch 到 expert owner rank
    -> 本地按 expert 分组/打包
    -> expert GEMM
    -> combine 回原 token

  重点纠正一下 toy 的简化：

  assignments.sort()
  dispatch_x = hidden_states[dispatch_token_idx]

  这个 toy 是把 EP 简化成单卡，所以它直接把每个 (token, topk_slot) 展开成一行，然后按 expert_id 排序。这个可以帮助理解 expert 计算前的分组。

  但真实 EP 里，通信阶段未必真的是“每个 token 物理复制 K 份后排序发送”。更准确是：

  一个 token 如果 top-k expert 分布在多个 rank，
  dispatch 会把这个 token 发给相关 rank。

  在某个目标 rank 内，
  这个 token 只需要参与属于本 rank 的 local experts。
  之后本地 kernel/runner 再把它展开/放到对应 expert 的连续区域。

  比如 topk=2：

  token0 -> expert1, expert5
  expert1 在 rank0
  expert5 在 rank2

  那通信上 token0 会到 rank0 和 rank2。

  如果两个 expert 都在同一个 rank：

  token1 -> expert4, expert5
  expert4/expert5 都在 rank2

  高效实现里可能只把 token1 发到 rank2 一次，然后 rank2 本地根据 recv_topk_ids 再展开给 expert4/expert5。不是一定通信复制两份。

  SGLang/Mori 里能看到这个结构：Mori dispatcher 返回 packed_recv_hidden / recv_topk_weights / recv_topk_ids / packed_recv_count，也就是“收到的 hidden states + 收到后对应的 topk 信息 + 每个 expert 收到多
  少 token”。见 sglang/python/sglang/srt/layers/moe/token_dispatcher/moriep.py:627。

  DeepGEMM 路径里还有一个 ep_scatter(...)，它用 num_recv_tokens_per_expert 把 dispatch 后的数据重新摆成 expert 连续的输入，并记录 output_index，后面 ep_gather 再按这个 index 和 topk_weights 聚合回来。见
  sglang/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:874。


  toy 里的 sort = 教学版的“按 expert 分组”。
  真实 SGLang EP 仍然需要按 expert 分组，但通常不是 Python sort；
  dispatch kernel / scatter kernel / fused_moe runner 会完成这个重排。
  EP 的新增点是：先按 expert 所在 rank 通信，再在本地按 expert 做 grouped GEMM，最后 combine 回原 token。

  combine 不是 combine dispatch 前的 hidden state，而是 combine 每个 expert 算出来的输出向量：

  final[token] = sum_k topk_weight[token, k] * expert_output[token, k]

  也就是说 combine 的对象是 [token, expert_slot, H] 这些 expert output vectors，最后回到 [T, H]。


## 为什么 EP 最自然是 All-to-All

```
                 EP MoE

GPU token
   │
   │  Dispatch
   ↓
┌─────────────────┐
│  All-to-All-v   │
└─────────────────┘
   │
   ↓
Expert tokens
   │
   │  Expert GEMM
   ↓
Expert outputs
   │
   │  Combine
   ↓
┌─────────────────┐
│  All-to-All-v   │
└─────────────────┘
   │
   ↓
Original token order
```

EP 的根本约束是：

```text
token 在任意 rank 上产生
expert 只存在于某些 rank 上
router 对每个 token 给出 top-k expert id
```

所以每个 rank 都会产生一批“要发给不同 expert owner rank”的 token。对 rank i 来说，它的 send buffer 天然是：

```text
send_to_rank_0
send_to_rank_1
...
send_to_rank_{ep_size-1}
```

对 rank j 来说，它需要从所有 rank 收到属于自己 local experts 的 token：

```text
recv_from_rank_0
recv_from_rank_1
...
recv_from_rank_{ep_size-1}
```

这正是 All-to-All / All-to-All-v (All-to-All：每个目的 rank 收到的数据量是一样的;All-to-All-v：每个目的 rank 收到的数据量可以不一样，v = variable )：

```text
每个 rank 给每个 rank 发不同内容。
```

MoE routing 是稀疏且不均匀的，不同目标 rank 的 token 数不一样，所以工程上更接近 All-to-All-v，也就是每个 peer 的 split size 可变。Mori/DeepEP 这类 EP dispatcher 做的不只是通信，还会把 token 按 expert owner 分桶、发出、接收、按 local expert 打包，并保存 combine 需要的反向 routing metadata。


## 补充：shared expert 个数和 disable-shared-experts-fusion

这个 Qwen3.5-397B config 里没有直接写 `num_shared_experts` 或 `n_shared_experts`，但有：

```json
"moe_intermediate_size": 1024,
"num_experts": 512,
"num_experts_per_tok": 10,
"shared_expert_intermediate_size": 1024
```

SGLang 对 Qwen MoE 的判断逻辑是：

```python
def get_num_shared_experts(config):
    n_shared_experts = getattr(config, "n_shared_experts", None)
    if n_shared_experts is not None:
        return n_shared_experts
    if hasattr(config, "shared_expert_intermediate_size") and config.shared_expert_intermediate_size > 0:
        return 1
    return 0
```

所以这个模型的结构语义是：

```text
512 routed experts
+ 1 shared expert MLP
```

`num_experts=512` 只表示 routed experts，不包含 shared expert。`num_experts_per_tok=10` 也只表示每个 token 选 10 个 routed experts。

`513` 只会在 shared expert 被融合进 fused MoE kernel 时出现。融合开启时，SGLang 会把 shared expert 当成额外的 fused expert slot：

```text
fused MoE num_experts = 512 + 1 = 513
fused MoE top_k       = 10 + 1 = 11
```

同时 weight loading 会把：

```text
mlp.shared_expert.*
```

映射到：

```text
mlp.experts.512.*
```

这不是 router 真的从 513 个 expert 里 top-k。router 仍然只对 512 个 routed experts 输出 logits；shared expert 的权重来自单独的 `shared_expert_gate(hidden_states) -> [num_tokens, 1]`，然后被 append 到 top-k 结果后面。

如果显式传：

```text
--disable-shared-experts-fusion
```

那么 fused MoE kernel 会回到只看 routed experts：

```text
TP / non-EP fused MoE kernel 里看到 512 experts
```

这不表示 shared expert 没了。它只是从 fused MoE kernel 里拿出来，走单独的 shared MLP：

```text
routed_out = fused_moe(hidden_states, 512 routed experts, top_k=10)
shared_out = shared_expert(hidden_states) * sigmoid(shared_expert_gate(hidden_states))
final = routed_out + shared_out
```

所以在 TP 场景里：

```text
不禁用 shared expert fusion:
  fused MoE 可能看到 513 个 expert slot

显式 --disable-shared-experts-fusion:
  fused MoE 看到 512 个 routed experts
  shared expert 单独算
```

EP Mori 的情况和这个一致，但触发原因不是用户显式传 flag，而是 SGLang 会在 Mori/DeepEP-family + `moe_ep_size > 1` 时自动关掉 Qwen shared-expert fusion：

```python
if enable_shared_expert_fusion and uses_per_rank_fused_shared_slots() and moe_ep_size > 1:
    enable_shared_expert_fusion = False
```

原因是 Qwen 这里的 shared expert fusion 语义更像“一个 global shared slot”。Mori/DeepEP 这类 backend 的 fused shared slot 是 per-rank physical slot，和这个 Qwen shared expert 的处理方式不兼容，所以 SGLang 选择回退成 separate shared expert MLP。

因此在 EP=4 的 Mori path：

```text
global routed experts = 512
local routed experts per EP rank = 512 / 4 = 128
shared expert fusion disabled
fused MoE kernel local expert count = 128
shared expert 走单独 MLP
```

而在 `--moe-a2a-backend none` 且 shared fusion 开启的 allreduce-EP path，shared expert 可能被放进 fused MoE：

```text
local routed experts = 512 / 4 = 128
fused shared expert slot = 1
fused MoE local expert count = 129
```

所以要分清两个问题：

```text
模型结构有几个 expert:
  512 routed + 1 shared

当前 fused MoE kernel 里看到几个 expert slot:
  取决于 shared expert fusion 是否开启，以及 EP backend 怎么处理 shared slot
```
