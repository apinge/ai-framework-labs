# 本文主要讨论sglang里 kv cache存储
- sglang版本在v0.5.9
- 不考虑MLA, sliding window等复杂情况
## save_kv_cache
观察结构更清晰的triton_backend `sglang/python/sglang/srt/layers/attention/triton_backend.py`

`save_kv_cache`分别出现在`forward_extend`和`forward_decode`，attention kernel调用之前。

```python
    # in forward extend
    if save_kv_cache:
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v
        )
    # ...
    self.extend_attention_fwd(
        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        k.contiguous(),
        v.contiguous(),
        o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
        forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
        forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
        self.forward_metadata.qo_indptr,
        kv_indptr,
        kv_indices,
        #...


    # IN forward decode
    if save_kv_cache:
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v
        )
    # ...
    self.decode_attention_fwd(
        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
        forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
        o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
        kv_indptr,
        kv_indices,
        # ...
    )
```
- `extend_attention_fwd`里 `k` `v`当前 extend 新产生的 token 的 KV, `k_buffer` `v_buffer` 是全局历史 KV cache pool，前面已经执行了`save_kv_buffer`，假设之前的`k_buffer`和 `v_buffer`是[A, B ,C], 当前推理是[D, E, F, G], 之前已经save 是否要需要 `k`和`v`呢，如果用的mha的是 `extend_attention.py`里的[_fwd_kernel](https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/srt/layers/attention/triton_ops/extend_attention.py#L605) 我们看得出这两份指针是分开的，也就是这次存的没有在这次用是给之后用的。Qwen3.5 27B TP2 来打印看出 `k` `v`是本次的shape  `k_buffer`和`v_buffer`是历史buffer
```
q torch.Size([80, 3072])
k torch.Size([80, 2, 256])
v torch.Size([80, 2, 256])
o torch.Size([80, 3072])
key_buffer torch.Size([2318080, 2, 256])
value_buffer torch.Size([2318080, 2, 256])
```
- `set_kv_buffer`：把当前层的 `k`, `v` 按 `out_cache_loc` scatter 进 paged k_buffer/v_buffer (`out_cache_loc` token在kv cache上的槽位)。
- `extend_attention_fwd`：用这批 k, v 做 extend 段的 attention，并从 buffer 读 prefix, 这个函数输出 attention的结果 不存KV
- `kv_indptr` `kv_indices` 这是attention kernel 定位kv cache位置指针

先看一下两个函数本身，在`forward_extend`(prefill)时， 假设enable radix cache 已经推理了[A, B ,C] 这次要推理 [D, E, F, G], 先用`set_kv_buffer` 更新 kv cache存储，一般 kv是shape是`[num_blocks, block_size, num_kv_heads, head_dim]` 再拿整
## kv cache的物理存放

根据 `MHATokenToKVPoll`(https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/srt/mem_cache/memory_pool.py#808-L825)， `k_buffer`和`v_buffer`的物理存放大约是这样`[num_slots, num_kv_heads, head_dim]`
```python
class MHATokenToKVPool(KVCache):
     def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.k_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.v_head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
    # 这里相当于拿了个指针 
    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)
```
page-size影响的是allocator
```python
    if self.page_size == 1:
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            self.max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            device=self.device,
            kvcache=self.token_to_kv_pool,
            need_sort=need_sort,
        )
    else:
        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
            self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            device=self.device,
            kvcache=self.token_to_kv_pool,
            need_sort=need_sort,
        )

```

## kv cache layout
从上面的分析看出, kv cache layout是否需要shuffle本身是为了适应kernel的实现，常见的layout一般是  key和value为`[num_tokens, num_kv_heads, head_size]`。

aiter里kernel `pa_ps`等的新layout为 `[num_blocks, num_heads, heads_size//x, block_size, x]` ，为了把sglang里存储的kv 换成新shape，需要引入`reshape_and_cache_shuffle_kernel`.


## kv cache 5d
https://github.com/sgl-project/sglang/pull/27063