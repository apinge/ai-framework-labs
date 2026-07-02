# quantization

## per token quant


`aiter.pertoken_quant`

看实现 这里$dtype\_max$ 是 目标quant_type的max

$$Y_{quant} = \frac{X_{float}}{scale} = \frac{X_{float}}{\frac{actual\_max}{dtype\_max}} = X_{float} \times \frac{dtype\_max}{actual\_max}$$

```python
def pertoken_quant(
    x,
    scale=None,
    x_scale=None,  # smooth_scale
    scale_dtype=dtypes.fp32,
    quant_dtype=dtypes.i8,
    dtypeMax=None,
):
    x = x.to(dtypes.fp32)
    if x_scale is None:
        hidden_states = x
    else:
        # smooth quant
        hidden_states = x * x_scale

    if dtypeMax is None:
        dtypeMax = get_dtype_max(quant_dtype)

    per_token_scale = scale
    if scale is None:
        # [m, 1]
        per_token_amax, _ = torch.max(
            input=torch.abs(hidden_states), dim=-1, keepdim=True
        ) # 每列做reduction  每行取个max 
        per_token_scale = per_token_amax / dtypeMax
        per_token_scale[per_token_scale == 0] = 1

    # quant hidden_states
    y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
    y_scale = per_token_scale.to(scale_dtype)
    return y, y_scale

```
torch api

```python
x = torch.tensor([[0.1, -0.5, 0.3],
                  [2.0,  1.0, -0.8]])
values, indices = torch.max(x.abs(), dim=-1, keepdim=True)
# values  = [[0.5],
#            [2.0]]
# indices = [[1],   # row0: abs 最大在 col=1 (-0.5)
#            [0]]   # row1: abs 最大在 col=0 ( 2.0)
```
## block-wise quant

in aiter `/opt/aiter/op_tests/test_moe_2stage.py`

```python
    Sdef weight_per_128x128_quant(weight, quant_dtype):
        E, dim1, dim2 = weight.shape
        weight_blocks = weight.view(
            E, dim1 // 128, 128, dim2 // 128, 128
        )  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
        weight_blocks = weight_blocks.permute(
            0, 1, 3, 2, 4
        ).contiguous()  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
        weight_blocks = weight_blocks.view(
            E, -1, 128 * 128
        )  # [E, num_blocks, 128*128]
        weight_qt, weight_scale = aiter.pertoken_quant(
            weight_blocks, quant_dtype=quant_dtype
        )
        weight_qt = weight_qt.view(
            E, dim1 // 128, dim2 // 128, 128, 128
        )  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
        weight_qt = weight_qt.permute(
            0, 1, 3, 2, 4
        ).contiguous()  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
        weight_qt = weight_qt.view(E, dim1, dim2)  # [E, dim1, dim2]
        weight_scale = weight_scale.view(
            E, dim1 // 128, dim2 // 128
        )  # [E, num_blocks_dim1, num_blocks_dim2]
        return weight_qt, weight_scale
```

`permute + contiguous() 操作：真正把内存里的数据重新排了顺序
