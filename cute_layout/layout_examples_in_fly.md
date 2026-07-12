## cases

## $(8192:4096) \circ (128,64):(1,128)$


$(8192,4096) \circ (128,64):(1,128) = ((8192:4096) \circ(128,1),(8192:4096) \circ(64,128))$

先看 $(8192:4096) \circ(128,1)$

根据[02_layout_algebra.html#composition](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#composition)

```
8192 % 128 = 128
```

$(8192:4096) \circ(128,1) =128:4096$

再看 $(8192:4096) \circ(64,128)$

```
8192/128 = 64
```
64:4096*128 

```
64%64 = 64
```
$8192:4096 \circ(64,128)=64:524288$
汇总
$(8192:4096) \circ (128,64):(1,128)= (128:4096),(64:524288)= (128,64):(4096,524288)$

#### $4096:1  \oslash 64:1$

$$4096:1  \oslash 64:1 = 4096:1 \circ (64:1,complement(64:1,4096)) = 4096:1 \circ (64:1,64:64) = 4096:1 \circ (64,64):(1,64)$$


$$4096:1 \circ (64,64):(1,64) =  4096:1 \circ 64:1,4096:1 \circ 64:64 $$


先看 $4096:1 \circ 64:1$

根据 https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#computing-composition

$$4096:1 \circ 64:1 = 64:1$$

同理

$$4096:1 \circ 64:64 = 64:64$$


所以
$$4096:1 \circ (64,64):(1,64) =  64:1,64:64= (64,64):(1,64) $$


## logical_divide(8192:4096,128:64) to flat_devide

$$logical_divide(8192:4096,128:64)  = ((128, 64), (64, 64)) : ((4096, 524288), (1, 64))$$
```
((128, 64), (64, 64)) : ((4096, 524288), (1, 64))
  mode0       mode1      mode0_strides   mode1_strides
```

zipped_divde 
```
((128, 64), (64, 64)) : ((4096, 1), (524288, 64))
  mode0       mode1      mode0_strides   mode1_strides
```

## tv layout 

```python3
thr_layout = fx.make_layout((32,8), (8, 1))
val_layout = fx.make_layout((1, 8), (1, 1))
tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)
```

从IR看 

```python3
def make_layout_tv(thr_layout, val_layout):
    layout_mn = raked_product(thr_layout, val_layout)          # Step 1
    thr_size = size(thr_layout).to_py_value()                  # = 256
    val_size = size(val_layout).to_py_value()                  # = 8
    tmp = make_layout((thr_size, val_size), (1, thr_size))     # Step 2

    layout_tv = composition(right_inverse(layout_mn), tmp)     # Step 3

    tiler_mn = int_tuple_product_each(get_shape(layout_mn)).to_py_value()  # Step 4
    return (tiler_mn, layout_tv)
```

It produces two things:

- `tile_mn` = (32, 64) — the tile shape this TV layout covers
- `tv_layout` = ((8,32), 8) : ((256,1), 32) — maps (tid, val_idx) → 1D MN offset

看下这两个结果怎么算出来的

### Step1 layout_mn = raked_product(thr_layout, val_layout)


先把logical_product 算出来
```
logical_product(thr_layout,val_layout)=((32, 8), (1, 8)):((8, 1), (256, 256))
```


根据cutlass的规则把ranked_product写出来
```C++
raked_product(block, tiler) {
  constexpr int R = max(rank(block), rank(tiler));
  auto result = logical_product(append<R>(block), append<R>(tiler));
  return zip(get<1>(result), get<0>(result));  // 注意：顺序反了！
}
```

首先logical_product结构拆成两个mode
```python
(
    Layout<(32,8),(8,1)>,      // 第一个layout mode0
    Layout<(1,8),(0,256)>      // 第二个layout mode1
)
```

于是
```python
get<0>(reulst) =  Layout<(32,8),(8,1)>
get<1>(reulst) =  Layout<(1,8),(0,256)>

zip(Layout<(1,8),(0,256)>, Layout<(32,8),(8,1)>)
```
CuTe 的 zip 是把两个 layout 对应 mode 配成 tuple

Before Zip
```
A
shape  (a0,a1)
stride (sa0,sa1)

B
shape  (b0,b1)
stride (sb0,sb1)
```
After Zip
```
shape
(
   (a0,b0),
   (a1,b1)
)

stride
(
   (sa0,sb0),
   (sa1,sb1)
)
```
```
zip(Layout<(1,8),(0,256)>, Layout<(32,8),(8,1)>)=Layout<((1,32),(8,8)),((0,8),(256,1))>
```

### Step2 make_layout((thr_size, val_size), (1, thr_size)) 
make_layout((thr_size, val_size), (1, thr_size)) = (256,8):(1,256) 似乎是在256X8的大小造一个 Identity Layout


### Step 3 right_inverse and composition


```
right_inverse(Layout(((1,32),(8,8)),((0,8),(256,1))))= (8, 256):(256, 1)
composition(Layout((8, 256),(256, 1))),Layout((256,8), (1,256))= ((8, 32), 8):((256, 1), 32)
```



得到  `tv_layout`

```
tv_layout = ((8,32), 8) : ((256,1), 32)
```

### Step4 tile_mn=  product_each(shape(layout_mn))

每个tuple里单独搞
```
shape(layout_mn) = ((1, 32), (8, 8))
product_each     = (1×32, 8×8) = (32, 64)
```

### tile_mn, tv_layout的理解

```python 
thr_layout = fx.make_layout((32,8), (8, 1))
val_layout = fx.make_layout((1, 8), (1, 1))
tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)
# tile_mn = (32,64)
# tv_layout = Layout<((8,32),8):(256,1),32>
```