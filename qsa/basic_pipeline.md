## qsa

### indexer

| 参数 | 值 |
|---|---:|
| `indexer_n_heads` | 4 |
| `indexer_kv_heads` | 1 |
| `indexer_head_dim` | 128 |
| `indexer_budget` | 2048 |
| `indexer_compress_ratio` | 4 |
| 每个 query 选中的压缩块数 | 512 |
| 最终参与稀疏 attention 的 token 数 | 2048 |

https://github.com/huggingface/transformers/blob/v5.16.0/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L611 

注意下面hf的逻辑是不带任何cache的
```python
# 本模型：hidden_size=2560；HF eager 形状为 [B, S, 2560]。
# SGLang prefill 会把 batch 内序列打包，等价形状为 [T, 2560]，T=sum(各请求 extend_len)。
# indexer 有 4 个 Q head 和 1 个 KV head，head_dim=128：2560 -> (4 + 1) * 128 = 640。
qk = index_qk_proj(hidden_states)  # [B, S, 640]；SGLang packed 时为 [T, 640]

# index_q:     [B, S, 4, 128]，当前 token 用于召回的 4 个 index-Q head。
# raw_index_k: [B, S, 1, 128]，每 token 一个尚未压缩的 index-K。
index_q, raw_index_k = split(qk)

index_q = RMSNorm(index_q)  # [B, S, 4, 128]
index_q = RoPE(index_q)     # [B, S, 4, 128]；Qwen4-Exp 使用层本身的 MRoPE

# HF eager 原版在 Python 中显式循环 batch 和 query token。
for b in range(B):
    for t in range(S):
        # L 是 query=(b,t) 可见的 token 数；visible_tokens: [L]，元素是逻辑 token position。
        visible_tokens = causal_mask_for_query(b, t)
        # G=floor(L/4)。每行是一个完整的 4-token group：complete_blocks [G, 4]。
        # 这里区分的是完整四元组和不足四个的尾部，不是区分历史 token 和新 token。
        complete_blocks = visible_tokens[:len(visible_tokens) // 4 * 4].view(-1, 4)

        # 每 4 个 raw index-K 压成一个 block K。
        # gather 后 [G, 4, 1, 128]，沿 4 个成员平均后 block_k 为 [G, 1, 128]。
        block_k = mean(raw_index_k[b, complete_blocks], dim=1)
        block_k = RMSNorm(block_k)  # [G, 1, 128]
        block_k = RoPE(block_k)     # [G, 1, 128]；每组用起始 token 的 RoPE position

        # 每个 query=(b,t) 都要单独召回：它的 index-Q 不同，且可见长度 L 会随 t 增长。
        # h=0..3 是 4 个 index-Q head；g=0..G-1 是这个 query 可见的历史 compressed block。
        score = zeros(G)  # 当前 query 对每个 block 的最终分数：[G]
        for g in range(G):
            for h in range(4):
                per_head_score[h, g] = dot(index_q[b, t, h], block_k[g, 0])  # 标量
                score[g] += ReLU(per_head_score[h, g])
            score[g] /= sqrt(128)

        # 实际 HF 用矩阵乘法并行掉上述 g/h 循环：
        # per_head_score = index_q[b, t] @ block_k.squeeze(1).T  # [4, 128] @ [128, G] -> [4, G]
        # score = ReLU(per_head_score).sum(dim=0) / sqrt(128)    # 对 h 求和，得到 [G]

        # block_topk = indexer_budget / compress_ratio = 2048 / 4 = 512。
        selected_blocks = topk(score, 512)           # [min(G, 512)] 个 block id
        selected_tokens = flatten(selected_blocks)   # 最多 512 × 4 = 2048 个逻辑 token id
        selected_tokens += causal tail               # 再附加 0~3 个未凑满 4 的 token；最多 2051 个 id
```
`g` 和 `h` 分别是：

| 符号 | 含义 | 这个模型中的范围 |
|---|---|---|
| `g` | compressed block 的编号 | `0 ... G-1` |
| `h` | index query head 的编号 | `0 ... 3` |
```
index_q        [4, 128]
block_k        [G, 128]       # 为了方便，省略中间那个长度为 1 的 KV-head 维
```
**心智模型**

```
hidden state
   │
   ├── 主 attention 的 Q / K / V
   │       │
   │       └── 真正的 sparse attention：只看选出的 2048 个 token
   │
   └── indexer 的 index-Q / raw index-K
           │
           ├── raw index-K 每 4 个取平均
           ├── 得到 compressed index-K
           ├── index-Q × compressed index-K
           ├── top-512 个 block
           └── 展开为 2048 个 token index
```


### indexer sglang reference


```python
# sglang/python/sglang/srt/layers/attention/qsa/mqa.py
def torch_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for packed, variable-length prefill MQA."""

    # _validate_q(q)
    # _validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    # q: [M, H=4, D=128]，M 是本次 packed prefill 的 query row 数。
    # k: [N, 1, D=128]，N 是 batch 中拼接后的 compressed block 数；第二维只有 1，因为 indexer 是 MQA。
    # 这里的 k 是本次 prefill 临时 gather 后、按 request 连续拼接的逻辑 tensor，不是物理 compressed cache buffer。
    # 因而物理 slot 即使交错（A=[0,1,4]、B=[2,3]），这个 k 仍会排成 A->[0,1,2]、B->[3,4]。
    # k[:, 0] 不是 k[:0]：它取所有 N 个 block 的唯一 KV head，形状 [N, 128]，等价于 k.squeeze(1)。
    # einsum "mhd,nd->mnh" 沿 D 点积，得到每个 query row、每个 compressed block、每个 Q head 的分数 [M, N, 4]。
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    logits = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    # k 把各 request 的 compressed block 平铺到同一条轴：columns 是这个“临时 packed k”的行号 [1, N]，例如 [0, 1, ..., N-1]，不是物理 slot id。
    # row_starts / row_ends 是每个 query row 自己在这条平铺轴上的有效半开区间：[M] -> [M, 1]。
    # 广播比较后 valid 是 [M, N]；它防止某 request 的 query 给别的 request 的 block 打分，
    # 同时也屏蔽该 query 尚不可见的未来 compressed block。
    columns = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    valid = (columns >= row_starts.to(q.device).reshape(-1, 1)) & (
        columns < row_ends.to(q.device).reshape(-1, 1)
    )
    return logits.masked_fill(~valid, -float("inf"))
```

```
物理 compressed cache slot：

request A: [0, 1, 4]
request B: [2, 3]
```

这是正常的 paged KV allocator 行为。物理 slot 不是按 request 连续分配的，也不应假设连续。

但 `torch_qsa_mqa_prefill()` 里的参数 `k` 不是整个物理 compressed cache buffer。它是 SGLang 为当前 prefill 临时 gather 后、按 request 顺序重新拼接的逻辑 tensor。

上面的物理布局进入 prefill MQA 前，会变成：

```
物理 cache 读取：
A -> physical slot [0, 1, 4]
B -> physical slot [2, 3]

临时 packed k：
row 0 = A 的 physical slot 0
row 1 = A 的 physical slot 1
row 2 = A 的 physical slot 4
row 3 = B 的 physical slot 2
row 4 = B 的 physical slot 3

k.shape = [5, 1, 128]
```

因此 `columns` 的含义是：

```
columns = [0, 1, 2, 3, 4]
```

它们是这个**临时 packed `k` 的行号**，不是物理 cache slot id。

对应 row range 例如：

```
A 的 query: row_start=0, row_end=3
B 的 query: row_start=3, row_end=5
```

于是：

```
A 只允许给 packed-k 的 [0, 1, 2] 打分
B 只允许给 packed-k 的 [3, 4] 打分
```

物理 cache slot 是否交错已经不重要，因为 gather 时被重新排好了。SGLang 做这个 gather + concat 的地方是 sglang/python/sglang/srt/layers/attention/qsa/metadata.py:136：

```
compressed_buffer.index_select(0, compressed_locs)
```

对每个 request 得到一段 `part` 后，再：

```
torch.cat(parts, dim=0)
```
因此 `row_starts / row_ends` 就能用连续区间描述每个 request 在 packed `k` 里的范围。

Decode 路径不同。Decode 不会生成这个临时 packed `k`，而是直接读取：

```
compressed physical cache
+ 每个 request 的 compressed page table
+ compressed length
```

所以 decode 的 page table 才负责把 request 的逻辑 block 顺序映射到可能交错的物理 page / slot。

### sparse gqa
真正主 attention：全局 Q heads = 24全 局 KV heads = 2 head_dim = 256

| TP | `q` | `k_cache / v_cache` | 每个 KV head 对应的 Q head 数 |
|---:|---|---|---:|
| 2 | `[M, 12, 256]` | `[KV_capacity, 1, 256]` | 12 |
| 4 | `[M, 6, 256]` | `[KV_capacity, 1, 256]` | 6 |
| 8 | `[M, 3, 256]` | `[KV_capacity, 1, 256]` | 3 |


```python
# - sglang/python/sglang/srt/layers/attention/qsa/kernel.py:318
def qsa_sparse_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Device-agnostic sparse GQA reference."""

    scale = softmax_scale or q.shape[-1] ** -0.5
    outputs = []
    repeats = q.shape[1] // k_cache.shape[1] # gqa ratio
    for row in range(q.shape[0]): # 逐 query token算主 sparse attention 这里是torch ref才这样
        # token_slots 每行固定宽度为 2051，未使用的位置是 -1。
        # K_valid = 4 * min(floor(L/4), 512) + (L mod 4)：
        #   最多 512 个完整 compressed block -> 512*4=2048 个 token，
        #   再加当前不满四个、尚未被压缩也无法按 block 打分的 causal tail 0~3 个 token。
        # 注意 token_slots 此时已经不是 indexer 的 compressed block id：
        #   top-k block g -> 展开成该 request 的逻辑 token [4*g, 4*g+1, 4*g+2, 4*g+3]
        #   -> 经 token_slot_table 映射成主 KV cache 的物理 slot。
        # 所以后面的 index_select 读取的是原始主 K/V [*,1,256]，不是 compressed index-K [*,1,128]。
        valid = token_slots[row] >= 0
        slots = token_slots[row, valid].long()
        if slots.numel() == 0:
            outputs.append(torch.zeros_like(q[row]))
            continue
        # 下面这句意思沿着0 轴选择，也就是选择某些k 然后repeat是沿着dim = 1对 num_kv_heads 维度进行重复
        keys = k_cache.index_select(0, slots).repeat_interleave(repeats, dim=1) #  [K_valid, 12, 256]
        values = v_cache.index_select(0, slots).repeat_interleave(repeats, dim=1) #  [K_valid, 12, 256]
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float()) * scale #   [12, K_valid]
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(
            torch.einsum("hk,khd->hd", probabilities, values.float()).to(q.dtype)
        ) #   [12, 256]
    return torch.stack(outputs)
```

```
token_slots      [M, K]              K ≤ 2051，物理 KV slot
q[row]           [H_q, 256]
keys / values    [K_valid, H_q, 256] # gather 后把唯一 KV head repeat 到 Q heads
scores            [H_q, K_valid]
probabilities     [H_q, K_valid]
attention output  [H_q, 256]
stack 后输出      [M, H_q, 256]
```
这个2051是indexer 给每个 query 的输出是固定宽度 `2051`，但其中有效 token 数随上下文长度而变化。

```
K_valid = 4 × min(floor(L / 4), 512) + (L mod 4)
```

其中 `L` 是当前 query 实际可见的 token 数。
两部分分别是：
```
完整 block：
  最多选 512 个 block
  每个 block 有 4 个原始 token
  → 最多 512 × 4 = 2048 token

causal tail：
  L 无法被 4 整除时，最后会剩 0～3 个 token
  它们没有完整 compressed block，不能走 block score
  → 直接追加进最终 attention 候选
```

因此最大 K_valid = 2048 + 3 = 2051


