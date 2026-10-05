# gemm bf16 splitk


## 问题描述

machine cdna3 (mi308x 80cu)

```python
M,N,K=32,5120,8192
TILE_M,TILE_N,TILE_K = 64,64,64
grid = ()
# for eatch workgroup

# LDS
c_lds_size = TILE_M * TILE_N * 4
@fx.struct
    class SharedStorage:
        C_lds: fx.Array[fx.Float32, c_lds_size, 16]
```

launch
```python
    @flyc.jit
    def launch(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        M: int, #真实的M 
        stream: fx.Stream
    ):
    #
    gemm_splitk(arg_c, arg_a, arg_b, M).launch(grid=(div_up(M, TILE_M), div_up(N, TILE_N), 1), block=(256, 1, 1), stream=stream)
```

each workgroup 处理
```
(64,8192)@(64,8192).T=(64,64)
```

### 输入layout 定义 


```
A: fx.make_layout((M, K), (K, 1))

C: fx.make_layout((M, N), (N, 1))

```
B 经历过[preshuffle weight](../flydsl_samples/preshuffle_explained.md)

#### preshuffle B
[preshuffle模拟](./print_preshuffle.py)

```python
def shuffle_weight(x: torch.Tensor, layout=(16, 16), use_int4=False) -> torch.Tensor:
    """Same layout as ``tests.utils.shuffle_weight``; copied to avoid ``from tests...`` (can import wrong ``tests``)."""
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    """
    x_.view(-1,
         N_dim // 16,    # 维1：沿 N 走了多少个「16 的块」
         16,             # 维2：在这 16 宽的条带里，是第几个 N
         K_dim // 32,    # 维3：沿 K 走了多少个「32 的块」
         4,              # 维4：一个 32 元素块再拆成 4 段，每段 8 个元素 (32/8=4)
         8)              # 维5：这 8 个 fp16 是内存里最细的一段
    (1, 256, 16, 128, 4, 8)
     ↑    ↑   ↑   ↑  ↑
   batch  N块 BN K块 子K 8元

    (1, 256, 128, 4, 16, 8)
    permute 之后 所以严格说：物理存储顺序往往还没变，变的是「多维下标怎么映射到线性下标」。
    contiguous() 会按 当前维度的逻辑顺序 把数据 重新排好并（通常）拷到新的一块连续存储。
所以 最终返回的 preshuffle_B：线性内存里的字节顺序 已经变了；这才是「preshuffle 真的生效」的那一步。
    """
    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    x_ = x_.view(x_type)
    x_.is_shuffled = True
    return x_

```


#### preshuffle最小例
- 原始 B 矩阵 (16×32) 

```
行 0: [  0   1   2   3   4   5   6   7 |  8   9  10  11  12  13  14  15 | 16  17  18  19  20  21  22  23 | 24  25  26  27  28  29  30  31]
行 1: [ 32  33  34  35  36  37  38  39 | 40  41  42  43  44  45  46  47 | 48  49  50  51  52  53  54  55 | 56  57  58  59  60  61  62  63]
行 2: [ 64  65  66  67  68  69  70  71 | 72  73  74  75  76  77  78  79 | 80  81  82  83  84  85  86  87 | 88  89  90  91  92  93  94  95]
行 3: [ 96  97  98  99 100 101 102 103 |104 105 106 107 108 109 110 111 |112 113 114 115 116 117 118 119 |120 121 122 123 124 125 126 127]
行 4: [128 129 130 131 132 133 134 135 |136 137 138 139 140 141 142 143 |144 145 146 147 148 149 150 151 |152 153 154 155 156 157 158 159]
行 5: [160 161 162 163 164 165 166 167 |168 169 170 171 172 173 174 175 |176 177 178 179 180 181 182 183 |184 185 186 187 188 189 190 191]
行 6: [192 193 194 195 196 197 198 199 |200 201 202 203 204 205 206 207 |208 209 210 211 212 213 214 215 |216 217 218 219 220 221 222 223]
行 7: [224 225 226 227 228 229 230 231 |232 233 234 235 236 237 238 239 |240 241 242 243 244 245 246 247 |248 249 250 251 252 253 254 255]
行 8: [256 257 258 259 260 261 262 263 |264 265 266 267 268 269 270 271 |272 273 274 275 276 277 278 279 |280 281 282 283 284 285 286 287]
行 9: [288 289 290 291 292 293 294 295 |296 297 298 299 300 301 302 303 |304 305 306 307 308 309 310 311 |312 313 314 315 316 317 318 319]
行10: [320 321 322 323 324 325 326 327 |328 329 330 331 332 333 334 335 |336 337 338 339 340 341 342 343 |344 345 346 347 348 349 350 351]
行11: [352 353 354 355 356 357 358 359 |360 361 362 363 364 365 366 367 |368 369 370 371 372 373 374 375 |376 377 378 379 380 381 382 383]
行12: [384 385 386 387 388 389 390 391 |392 393 394 395 396 397 398 399 |400 401 402 403 404 405 406 407 |408 409 410 411 412 413 414 415]
行13: [416 417 418 419 420 421 422 423 |424 425 426 427 428 429 430 431 |432 433 434 435 436 437 438 439 |440 441 442 443 444 445 446 447]
行14: [448 449 450 451 452 453 454 455 |456 457 458 459 460 461 462 463 |464 465 466 467 468 469 470 471 |472 473 474 475 476 477 478 479]
行15: [480 481 482 483 484 485 486 487 |488 489 490 491 492 493 494 495 |496 497 498 499 500 501 502 503 |504 505 506 507 508 509 510 511]
```
这里 k group是
```
  k_group 0       k_group 1       k_group 2       k_group 3
  [0 1 2 3 4 5 6 7 | 8 9 ... 15 | 16 17 ... 23 | 24 25 ... 31]
```
- preshuffle后 (16x32)

`BF16` 的 `element_size()` 是 2，所以这里 `K = 16 / 2 = 8`。该例的
变换为：

```python
(1, 1, 16, 1, 4, 8)  # (batch, N block, N, K block, K group, 8 BF16)
    -> permute(0, 1, 3, 4, 2, 5)
(1, 1, 1, 4, 16, 8)
```

也就是说，先写入所有 N 行的 `K[0:8]`，再写入所有 N 行的
`K[8:16]`、`K[16:24]` 和 `K[24:32]`。下面是最终连续内存按原来的
`(16, 32)` 形状重新解释后的样子；竖线每隔 8 个 BF16 元素：

下面注意id是之前的逻辑id 
```
物理行 0: [  0   1   2   3   4   5   6   7 | 32  33  34  35  36  37  38  39 | 64  65  66  67  68  69  70  71 |  96  97  98  99 100 101 102 103]
物理行 1: [128 129 130 131 132 133 134 135 |160 161 162 163 164 165 166 167 |192 193 194 195 196 197 198 199 | 224 225 226 227 228 229 230 231]
物理行 2: [256 257 258 259 260 261 262 263 |288 289 290 291 292 293 294 295 |320 321 322 323 324 325 326 327 | 352 353 354 355 356 357 358 359]
物理行 3: [384 385 386 387 388 389 390 391 |416 417 418 419 420 421 422 423 |448 449 450 451 452 453 454 455 | 480 481 482 483 484 485 486 487]

物理行 4: [  8   9  10  11  12  13  14  15 | 40  41  42  43  44  45  46  47 | 72  73  74  75  76  77  78  79 | 104 105 106 107 108 109 110 111]
物理行 5: [136 137 138 139 140 141 142 143 |168 169 170 171 172 173 174 175 |200 201 202 203 204 205 206 207 | 232 233 234 235 236 237 238 239]
物理行 6: [264 265 266 267 268 269 270 271 |296 297 298 299 300 301 302 303 |328 329 330 331 332 333 334 335 | 360 361 362 363 364 365 366 367]
物理行 7: [392 393 394 395 396 397 398 399 |424 425 426 427 428 429 430 431 |456 457 458 459 460 461 462 463 | 488 489 490 491 492 493 494 495]

物理行 8: [ 16  17  18  19  20  21  22  23 | 48  49  50  51  52  53  54  55 | 80  81  82  83  84  85  86  87 | 112 113 114 115 116 117 118 119]
物理行 9: [144 145 146 147 148 149 150 151 |176 177 178 179 180 181 182 183 |208 209 210 211 212 213 214 215 | 240 241 242 243 244 245 246 247]
物理行10: [272 273 274 275 276 277 278 279 |304 305 306 307 308 309 310 311 |336 337 338 339 340 341 342 343 | 368 369 370 371 372 373 374 375]
物理行11: [400 401 402 403 404 405 406 407 |432 433 434 435 436 437 438 439 |464 465 466 467 468 469 470 471 | 496 497 498 499 500 501 502 503]

物理行12: [ 24  25  26  27  28  29  30  31 | 56  57  58  59  60  61  62  63 | 88  89  90  91  92  93  94  95 | 120 121 122 123 124 125 126 127]
物理行13: [152 153 154 155 156 157 158 159 |184 185 186 187 188 189 190 191 |216 217 218 219 220 221 222 223 | 248 249 250 251 252 253 254 255]
物理行14: [280 281 282 283 284 285 286 287 |312 313 314 315 316 317 318 319 |344 345 346 347 348 349 350 351 | 376 377 378 379 380 381 382 383]
物理行15: [408 409 410 411 412 413 414 415 |440 441 442 443 444 445 446 447 |472 473 474 475 476 477 478 479 | 504 505 506 507 508 509 510 511]
```

例如，原始的 `B[1, 0:8] = [32, ..., 39]` 被放到了 preshuffle buffer
的线性偏移 `8:16`，即上图物理行 0 的第二段。内核通过 B 的特殊 layout
将该物理地址再映射回逻辑坐标 `(N=1, K=0:8)`。


### preshuffle layout

```python
arg_b = fx.Tensor(fx.make_view(
    fx.get_iter(arg_b_),
    # layout分成两部分：第一部分(m, n)描述一个wave内部划分为16行，(8列每组)x4组；第二部分为重复第一部分的次数
    # shape: (16, (8, 4)),   (N//16, K//32)
    # stride:(8,  (1, 128)), (K*16, 512))
    # 重新排列为第0维、第1维
    fx.make_layout(((16, N // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 128, 512)))
))
```
最小的例子N=16 K=32
于是
shape  = ((16, 1), (8, 4, 1))
stride = ((8, 512), (1, 128, 512))

我们根据[cute layout coalesce](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#coalesce)
逻辑坐标: (n_inner, (k_inner, k_group))
shape:    (16,   ,(8,       4))
stride:   ( 8,   ,(1,     128))

offset = n_inner * 8 + k_inner + k_group * 128
```
 n_inner    对应逻辑元素       物理 offset
  0          B[0, 0]            0
  1          B[1, 0]            8
  2          B[2, 0]            16
  3          B[3, 0]            24
  ...
  15         B[15, 0]           120

N inner = 0..15          # 逻辑上的 N 坐标
0, 8, 16, ..., 120       # 它们在 preshuffle buffer 中的地址
```


- N inner 的 stride 8，表示从一个 N 到下一个 N，跨过一小段 K[0:8]。
- K inner 的 stride 1，表示这 8 个 K 元素本身连续。
- k_group 的 stride 128，表示跳过 16 * 8 个元素，进入下一批所有 N 的 K 段。


#### flat_divide

来看三个flat_divide, 先看最简单的A和C
```mlir
 %41 = fly.flat_divide(%26, %40) : (!fly.memref<bf16, #fly_rocdl.buffer_desc, (?,8192):(8192,1)>, !fly.tile<[64|256]>) -> !fly.memref<bf16, #fly_rocdl.buffer_desc, (64,256,?,32):(8192,1,524288,256)> loc(#loc252) //A

  %49 = fly.flat_divide(%39, %48) : (!fly.memref<bf16, #fly_rocdl.buffer_desc, (?,5120):(5120,1)>, !fly.tile<[64|64]>) -> !fly.memref<bf16, #fly_rocdl.buffer_desc, (64,64,?,80):(5120,1,327680,64)> loc(#loc258) //C
```

不讨论 这里 A需要divide `fly.tile<TILE_M|4*TILE_K]>` 这是spkit k算法需要。

先复习一下`logical_divide`和`flat_divide`关系 
```
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
```

```
logical_divide(
    (?, 8192):(8192, 1),
    <64, 256>
  ) =  ((64, ?), (256, 32)):((8192, 524288), (1, 256))
```
注意<64,256>本身并没有stride,stride来自(8192,1)

我们来这样看
```
logical_divide(
    (?, 8192):(8192, 1),
    <64, 256>
)
=
(
    logical_divide(?:8192, 64:1),
    logical_divide(8192:1, 256:1)
)
logical_divide(?:8192, 64:1) = (64,rest1):(8192,64*8192) #52428=64*8192
logical_divide(8192:1, 256:1) = (256,32):(1,256) # (tile,rest2):(origin stride,...)

logical_divide
=
(64,rest1,256,32):(8192,64*8192,1,256)

=> flat divide
(65,256,rest1,32):(8192,1,64*8192,256)
```


同样对于C
```
logical_divide(
    (?,5120):(5120,1),
    <64, 64>
)
=
(
    logical_divide(?:5120, 64:1),
    logical_divide(5120:1, 64:1)
)
logical_divide(?:5120, 64:1) = (64,rest1):(5120,64*5120) # 327680=64*8192
logical_divide(5120:1, 64:1) = (64,80):(1,64)

logical_divide
=
(64,rest1,64,80):(8192,64*8192,1,64)

=> flat divide
(64,64,rest1,80):(8192,64*8192,1,256)
```

#### flat_divide B
```mlir
 %45 = fly.flat_divide(%31, %44) : (!fly.memref<bf16, #fly_rocdl.buffer_desc, ((16,320),(8,4,256)):((8,131072),(1,128,512))>, !fly.tile<[64|256]>) -> !fly.memref<bf16, #fly_rocdl.buffer_desc, ((16,4),(8,32),80,32):((8,131072),(1,128),524288,4096)> loc(#loc255)
```

```
logical_divide(
    ((16,320),(8,4,256)):((8,131072),(1,128,512)),
    <64, 256>
)
=
(
    logical_divide((16,320):(8,131072), 64:1),
    logical_divide((8,4,256):(1,128,512), 256:1)
)

logical_divide((16,320):(8,131072), 64:1)= ((16, 4), 80):((8, 131072), 524288)
logical_divide((8,4,256):(1,128,512), 256:1) = ((8, 32), 32):((1, 128), 4096)

flat_divide
= ((16,4),(8,32),80,32):((8,131072),(1,128),524288,4096)
```


### make_tiled_mma
```python
  tiled_mma = make_tiled_mma(
        mma_atom,
        # splitk, 4个wave分布在K维度
        fx.make_layout((1, 1, 4), (0, 0, 1)),
        # K维度((4*2)*4)*4，线程分布依次为：连续的4点为一个lane，跳过8点为一组并重复4组*4wave，最后为重复第二次调用
        (None, None, fx.make_layout((4, 4 * 4, 2), (1, 8, 4)))
    )
```


`(4,4*4,2):(1,8,4)` 可以参考 [preshuffle weight](../flydsl_samples/preshuffle_explained.md),那个例子只有一个wave所以是`(4,4,2):(1,8,4)`这里要考虑4waves 所以是现在这样。

```mlir
 %61 = fly.make_tiled_mma(%53, %56, %60) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (bf16, bf16) -> f32>>, !fly.layout<(1,1,4):(0,0,1)>, !fly.tile<[*|*|(4,16,2):(1,8,4)]>) -> !fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (bf16, bf16) -> f32>>, !fly.layout<(1,1,4):(0,0,1)>, !fly.tile<[*|*|(4,16,2):(1,8,4)]>> loc(#loc264)
```

### make_tiled_copy and partition_S

#### make_tiled_copy
```python
    a_tiled_thr = fx.make_tiled_copy_A(cp_atom_r, tiled_mma).get_slice(tid)
    b_tiled_thr = fx.make_tiled_copy_B(cp_atom_r, tiled_mma).get_slice(tid)

```

```mlir
%71 = fly.make_tiled_copy(%52, %62, %70) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<128>, 16>, !fly.layout<((16,16),(4,(1,2))):((1,128),(16,(0,64)))>, !fly.tile<[16:1|128:1]>) -> !fly.tiled_copy<!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<128>, 16>, !fly.layout<((16,16),(4,(1,2))):((1,128),(16,(0,64)))>, !fly.tile<[16:1|128:1]>> loc(#loc265)


 %82 = fly.make_tiled_copy(%52, %73, %81) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<128>, 16>, !fly.layout<((16,16),(4,(1,2))):((1,128),(16,(0,64)))>, !fly.tile<[16:1|128:1]>) -> !fly.tiled_copy<!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<128>, 16>, !fly.layout<((16,16),(4,(1,2))):((1,128),(16,(0,64)))>, !fly.tile<[16:1|128:1]>> loc(#loc267)
```

  make_tiled_copy_A 的 Python 实现等价于：
```python
  layout_tv = tiled_mma.tv_layout_A_tiled
  tile_size = tiled_mma.tile_size_mnk

  tile_mk = make_tile(
      make_layout(select(tile_size, [0]), 1),  # M
      make_layout(select(tile_size, [2]), 1),  # K
  )

  return make_tiled_copy(copy_atom, layout_tv, tile_mk)
```
其中 `tiled_mma.tv_layout_A_tiled`
```
// 1. 从 tiled_mma 取 A 的 thread-value layout
  %62 = fly.static :
    !fly.layout<((16,16),(4,(1,2))):((1,128),(16,(0,64)))>
```
这句是tv layout所以左边不是`(M,K)`而是 `(thread coordinate, value coordinate)` 能恰好覆盖 `A[16, 128]` 的原因。

- thread coordinate = 16 * 16 = 256 个线程
- value coordinate = 4 * 1 * 2 = 8 个 value / thread
```
  t0: 0..15
  t1: 0..15
  v0: 0..3
  v1: 0..0
  v2: 0..1
```
- stride  `q = t0 * 1 + t1 * 128 + v0 * 16 + v1 * 0 + v2 * 64` 里的 q 不是 A 的全局内存地址，而是基础 (16,128) copy tile 中的逻辑线性坐标。 这个基础 tile 用 M-fastest 的坐标解释
```
 q = m + 16 * k
 =>
 m = t0
 k = 8 * t1 + v0 + 4 * v2
```
**256 个线程排成： 16 个 M 行 x 16 个 K[8] chunk**
`tile_size = tiled_mma.tile_size_mnk`
```
tile_size=tile_size_mnk = (M=16, N=16, K=128)
```
#### partition_S

```python
a_tensor_thr = a_tiled_thr.partition_S(a_tile)
b_tensor_thr = b_tiled_thr.partition_S(b_tile)
```

```mlir
%84 = fly.tiled_copy.partition_src(%71, %43, %72) : (!fly.tiled_copy<!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<128>, 16>, !fly.layout<((16,16),(4,(1,2))):((1,128),(16,(0,64)))>, !fly.tile<[16:1|128:1]>>, !fly.memref<bf16, #fly_rocdl.buffer_desc, (64,256,32):(8192,1,256)>, !fly.int_tuple<?>) -> !fly.memref<bf16, #fly_rocdl.buffer_desc, ((8,1),4,2,32):((1,0),131072,128,256)> loc(#loc269)

%85 = fly.tiled_copy.partition_src(%82, %47, %83) : (!fly.tiled_copy<!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<128>, 16>, !fly.layout<((16,16),(4,(1,2))):((1,128),(16,(0,64)))>, !fly.tile<[16:1|128:1]>>, !fly.memref<bf16, #fly_rocdl.buffer_desc, ((16,4),(8,32),32):((8,131072),(1,128),4096)>, !fly.int_tuple<?>) -> !fly.memref<bf16, #fly_rocdl.buffer_desc, ((8,1),4,2,32):((1,0),131072,2048,4096)> loc(#loc270)
```

```
 a_tiled_thr.partition_S(a_tile)
    a_tiled_thr = %71 + %72
                  |     |
                  |     +-- get_slice(tid) 的 tid
                  +-------- A 的 tiled_copy
    a_tile      = %43 #a_tile就是flat_divide后的结果分到每个workgroup的那个

  b_tiled_thr.partition_S(b_tile)
    b_tiled_thr = %82 + %83
                  |     |
                  |     +-- get_slice(tid) 的 tid
                  +-------- B 的 tiled_copy
    b_tile      = %47
```

**`partition_S` 的意思是：把整个 `a_tile` / `b_tile` 按 %71 / %82 的 TV 分工以及当前 `tid` 切出“当前 `thread` 应读的 `source view`**。

 所以 %84 和 %85 是 `a_tensor_thr`、`b_tensor_thr`：
```mlir
  %84 = a_tensor_thr
  %85 = b_tensor_thr
```

两者 shape 一样((8,1), 4, 2, 32)

但物理 stride 不一样：
```
A: ((1,0), 131072, 128, 256)
B: ((1,0), 131072, 2048, 4096)
```
这个 stride 差异正是普通 row-major A 和 preshuffle B 的区别。

##### partition_S A

先看下%43这句怎么来的来自这句`a_tile = fx.flat_divide(a_tensor, fx.make_tile(TILE_M, TILE_K * 4))[None, None, blk_x, None]`
```
   %43 = fly.slice(%41, %42) : (!fly.memref<bf16, #fly_rocdl.buffer_desc, (64,256,?,32):(8192,1,524288,256)>, !fly.int_tuple<(*,*,?,*)>) -> !fly.memref<bf16, #fly_rocdl.buffer_desc, (64,256,32):(8192,1,256)> loc(#loc253)
```
所以%43每个workgroup的A成了(64,256,32):(8192,1,256), 分别代表 (m_in_tile, k_in_256, k_tilA) 

%84 就是 a_tensor_thr，它不是新的数据拷贝，而是：在 %43 = a_tile 中，为当前 tid 建立一个“这个 thread 应读取哪些 A 元素”的 view。

**partition_S**后 ((8, 1), 4,      2,   32):((1, 0), 131072, 128, 256)
```
(v, dummy, m_rep, k_rep, k_tile)

  v:      0..7
  dummy:  0..0
  m_rep:  0..3
  k_rep:  0..1
  k_tile: 0..31
```

- 131072: m_rep 每加 1，跨过 16 行 A： 16 * 原始 M stride 8192 = 131072。
- 256:  k_tile 每加 1，跨过 flat_divide 的一个 K=256 tile：
- 128: k_rep 每加 1，跨过一个基础 copy tile 的 K=128?

** 还差一个关键部分：当前 tid 负责哪一个基础 (M=16, K=128) 区域内的 8 元素？**

理解一下%84
`(8,1),4,2,32):((1,0),131072,128,256)`
- `(8,1)`  当前 thread 一次读取连续的 8 个 BF16。
- 4, 在 block 内跨 4 个 M=16 分块，合计覆盖 M=64。
- 2, 在当前 K=256 范围内跨 2 个 K=128 分块。
- 32,  遍历完整 K=8192 的 32 个 K=256 分块。

##### partition_S B
先看%47 = b_tile，分到每个workgroup ((16,4),(8,32),32):((8,131072),(1,128),4096)

shape:((n0, n1), (k0, kg), k_tile)=((16, 4), (8, 32), 32)
stride:((8, 131072), (1, 128), 4096)

对应的逻辑坐标为：

N = n0 + 16 * n1
K = k0 + 8 * kg + 256 * k_tile
其中
- n0:     当前 16-N block 内的 N，范围 0..15
- n1:     当前 N=64 tile 内第几个 N=16 block，范围 0..3

- k0:     一个 thread vector 内部的 K，范围 0..7
- kg:     当前 K=256 tile 内第几个 K[8] chunk，范围 0..31
- k_tile: 整个 K=8192 内第几个 K=256 tile，范围 0..31

接着 partition_S 要把这个基础 tile 扩展到整个 B tile：
```
  n_rep:  0..3  # N=64 中的四个 N=16 block
  k_rep:  0..1  # K=256 中的两个 K=128 block
  k_tile: 0..31 # 完整 K=8192 中的 32 个 K=256 block
```
得到
```
%85 shape  = ((8,1), 4,      2,    32)
%85 stride = ((1,0), 131072, 2048, 4096)
```
### .make_fragment_A

```python
a_frag = tiled_mma.make_fragment_A(a_tile[None, None, 0])
b_frag = tiled_mma.make_fragment_B(b_tile[None, None, 0])
```
这样理解
```mlir
%43 = a_tile，shape 是 (M=64, K=256, K_outer=32)
%47 = b_tile，shape 是 (N=64, K=256, K_outer=32)
```
`[None, None, 0]`只取 K_outer = 0 的第一段 K=256,所以切下来后A和B的shape都是(64,256)

ir是
```
 %88 = fly.mma.make_fragment(a, %61, %87) : (!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (bf16, bf16) -> f32>>, !fly.layout<(1,1,4):(0,0,1)>, !fly.tile<[*|*|(4,16,2):(1,8,4)]>>, !fly.memref<bf16, #fly_rocdl.buffer_desc, (64,256):(8192,1)>) -> !fly.memref<bf16, register, (4,4,(2,2)):(1,16,(4,8))> loc(#loc272)

 %91 = fly.mma.make_fragment(b, %61, %90) : (!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (bf16, bf16) -> f32>>, !fly.layout<(1,1,4):(0,0,1)>, !fly.tile<[*|*|(4,16,2):(1,8,4)]>>, !fly.memref<bf16, #fly_rocdl.buffer_desc, ((16,4),(8,32)):((8,131072),(1,128))>) -> !fly.memref<bf16, register, (4,4,(2,2)):(1,16,(4,8))> loc(#loc274)
```
因此 %88、%91 的 result type 相同， 不是因为 A 和 B 的全局存储相同，而是因为它们都要交给同一个 MFMA(16x16x16) tiled MMA，并按同一套 MFMA register fragment 规则解释。A/B 的物理地址差异仍保留在输入 %87 与 %90 的 stride 中。

 %88 / %91:根据 tiled_mma 的 MFMA thread-value 规则，生成每一个 thread 所需的 register fragment view，描述的是**当前 thread 的 64 个 BF16 register slot 怎么编号**。

 这个(4,4,(2,2))和后面的对应

```python
 for sub_k in range_constexpr(2):
    x.gemm(mma_atom, c_frag, b_frag[None, None, (None, sub_k)], a_frag[None, None, (None, sub_k)], c_frag)
```
### make_rmem_tensor

```mlir
 // fx.make_layout(...)
%94 = fly.make_layout(%92, %93) : (!fly.int_tuple<((4,1),4,4)>, !fly.int_tuple<((1,0),16,4)>) -> !fly.layout<((4,1),4,4):((1,0),16,4)> loc(#loc275)
// fx.make_rmem_tensor(...)
%95 = fly.memref.alloca(%94) : (!fly.layout<((4,1),4,4):((1,0),16,4)>) -> !fly.memref<f32, register, ((4,1),4,4):((1,0),16,4)> loc(#loc276)
```
它是当前 thread 的 64 个 f32 accumulator register, %95的shape理解为`[value, n_req, m_req]`

### retile

```python
a_frag_retile = a_tiled_thr.retile(a_frag)
b_frag_retile = b_tiled_thr.retile(b_frag)
```

```mlir
 %97 = fly.tiled_copy.retile(%71, %88) : (!fly.tiled_copy<!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<128>, 16>, !fly.layout<((16,16),(4,(1,2))):((1,128),(16,(0,64)))>, !fly.tile<[16:1|128:1]>>, !fly.memref<bf16, register, (4,4,(2,2)):(1,16,(4,8))>) -> !fly.memref<bf16, register, ((8,1),4,2):((1,0),16,8)> loc(#loc279)

%98 = fly.tiled_copy.retile(%82, %91) : (!fly.tiled_copy<!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<128>, 16>, !fly.layout<((16,16),(4,(1,2))):((1,128),(16,(0,64)))>, !fly.tile<[16:1|128:1]>>, !fly.memref<bf16, register, (4,4,(2,2)):(1,16,(4,8))>) -> !fly.memref<bf16, register, ((8,1),4,2):((1,0),16,8)> loc(#loc280)
```

```
%71 = A 的 tiled_copy
%82 = B 的 tiled_copy

%88 = a_frag，MFMA register layout
%91 = b_frag，MFMA register layout

%97 = a_frag_retile，copy-compatible register view
%98 = b_frag_retile，copy-compatible register view
```
retile 不会搬运或转换 BF16 数据，它只重解释同一批 64 个 register 元素的索引方式， **把 MFMA fragment 的 register view，改成和 copy source 拥有相同 shape 的 register destination view**

source 和 destination 的 stride 可以、也应该不同。

  三者分别是：
```
  A source:
  shape  = ((8,1), 4,      2)
  stride = ((1,0), 131072, 128)

  B source:
  shape  = ((8,1), 4,      2)
  stride = ((1,0), 131072, 2048)

  register destination:
  shape  = ((8,1), 4,  2)
  stride = ((1,0), 16, 8)

  它们共享同一组逻辑坐标：

  (v, dummy, m_rep, k_rep)

  v:      0..7
  dummy:  0
  m_rep:  0..3
  k_rep:  0..1

  所以 fly.copy 可以按相同坐标逐项配对：

  src(v, dummy, m_rep, k_rep)
    ->
  dst(v, dummy, m_rep, k_rep)

  但同一个坐标在 source 和 destination 中的地址不同。

  例如取：

  (v=3, dummy=0, m_rep=2, k_rep=1)

  A 的 global-memory 相对元素地址：

  3 * 1 + 0 * 0 + 2 * 131072 + 1 * 128
  = 262275

  B 的 preshuffle global-memory 相对元素地址：

  3 * 1 + 0 * 0 + 2 * 131072 + 1 * 2048
  = 264195 
```
这里来自`%85 -> ((8,1),4,2,32):((1,0),131072,2048,4096)`
```
  register destination 中的 register 编号：

  3 * 1 + 0 * 0 + 2 * 16 + 1 * 8
  = 43

  因此这次 copy 的含义是：

  A:
  global buffer 的 offset 262275
    -> 当前 thread 的 register 43

  B:
  preshuffle buffer 的 offset 264195
    -> 当前 thread 的 register 43
```


为什么 destination stride 是((1,0),16,8)它只是把 64 个 register 排成易于 vector copy 的次序：


### fly.memref.load_vec
```python
acc_init = c_frag.load()
```
```MLIR
%99 = fly.memref.load_vec(%95) : (!fly.memref<f32, register, ((4,1),4,4):((1,0),16,4)>) -> vector<64xf32> loc(#loc281)
```
%99 的作用不是再次初始化，而是把 register memref %95 中的 64 个 f32 读成一个 SSA vector，作为循环携带状态的初始值：

### Loop

**编译期展开**
```python
  for k in range_constexpr(0, K // TILE_K // 4):
      ...
      fx.gemm(..., c_frag)
```
**保留为运行时**
```
  for k, state in range(loop_start, loop_end, loop_step, init=[acc_init]):
      c_frag.store(state[0])
      ...
      results = yield [c_frag.load()]
```

### Loop2
注意下面的loop先传B再传A
```
D[n, m] = Σk B[n, k] * A[m, k]
```
```python
a_frag_retile = a_tiled_thr.retile(a_frag)
b_frag_retile = b_tiled_thr.retile(b_frag)
# loop
    fx.copy(cp_atom_r, a_tensor_thr[None, None, None, k_i32], a_frag_retile)
    fx.copy(cp_atom_r, b_tensor_thr[None, None, None, k_i32], b_frag_retile)
    for sub_k in range_constexpr(TILE_K // 32):
        fx.gemm(mma_atom, c_frag, b_frag[None, None, (None, sub_k)], a_frag[None, None, (None, sub_k)], c_frag) 
```
#### make_fragment和.retile的区别
为什么 拷贝的是`a_frag_retile` 执行mfma的是`a_frag`因为 **`a_frag_retile` 和 `a_frag` 不是两份 register 数据, 它们是同一批 register 的两个 layout view, a_frag_retile 是为了让 BufferCopy128b 能方便地写入；a_frag 是为了让 MFMA 能按硬件要求读取。**。

 IR 数据流正好说明这一点：

```mlir
// 1. 先创建 MFMA 视角的 register fragment
  %88 = fly.mma.make_fragment(a, %61, %87)
    -> !fly.memref<bf16, register, (4,4,(2,2)):(1,16,(4,8))>

  // 2. 对 %88 做 retile，得到 copy 视角
  %97 = fly.tiled_copy.retile(%71, %88)
    -> !fly.memref<bf16, register, ((8,1),4,2):((1,0),16,8)>
```

可以类比为同一段 64 个 BF16 register。这里的 64 是 64 个逻辑 BF16元素；实际硬件上可能将两个 BF16 打包进一个 32-bit VGPR，不能把它直接等同为 64 个物理 VGPR。

本 kernel 的一个 outer K iteration 处理 A 的 `(M, K) = (64, 256)`。
`atom_layout = (1, 1, 4)` 使 4 个 wave 沿 K 分工，因此每个 wave 负责
`A[64, 64]`。对某一个 lane：

```
M 方向: 64 / 16 = 4 个 M rep
K 方向: 64 / 16 = 4 个 MFMA K rep
MFMA(16x16x16) 的 A operand: 每 lane 每次需要 4 个 BF16

4 * 4 * 4 = 64 BF16 / lane
```

注意 N 方向的 4 个 rep 会增加 MFMA 的执行次数，但不会增加 A fragment的数据量，因为同一份 A 会复用到不同的 N block。若按 M/N/K 三个方向数概念上的 MFMA tile，则每个 wave 覆盖 `4 * 4 * 4 = 64` 个 `16x16x16`tile；上面的 64 BF16/lane 只计 A 所需的数据，未把 N rep 重复算入。

`a_frag_retile` 和 `a_frag` 以不同的二维网格解释完全相同的`r[0] ... r[63]`。为了便于画图，省略 shape 为 1、stride 为 0 的 dummy维度。

```
copy view: ((8,1), 4, 2):((1,0), 16, 8)

              K rep 0             K rep 1
M rep 0    r[ 0] ... r[ 7]     r[ 8] ... r[15]
M rep 1    r[16] ... r[23]     r[24] ... r[31]
M rep 2    r[32] ... r[39]     r[40] ... r[47]
M rep 3    r[48] ... r[55]     r[56] ... r[63]

每个 cell 是连续 8 个 BF16，正好是 BufferCopy128b 的 16-byte vector。copy 的地址公式是: reg = v + 16 * m_rep + 8 * k_rep。
```

```
MFMA view: (4, 4, (2,2)):(1, 16, (4,8))

              K=(k0,k1)             K=(k0,k1)             K=(k0,k1)             K=(k0,k1)
                 (0,0)                 (1,0)                 (0,1)                 (1,1)
M rep 0    r[ 0] ... r[ 3]     r[ 4] ... r[ 7]     r[ 8] ... r[11]     r[12] ... r[15]
M rep 1    r[16] ... r[19]     r[20] ... r[23]     r[24] ... r[27]     r[28] ... r[31]
M rep 2    r[32] ... r[35]     r[36] ... r[39]     r[40] ... r[43]     r[44] ... r[47]
M rep 3    r[48] ... r[51]     r[52] ... r[55]     r[56] ... r[59]     r[60] ... r[63]

每个 cell 是一个 MFMA operand 的 4 个 BF16。MFMA 的地址公式是:
reg = val + 16 * m_rep + 4 * k0 + 8 * k1。
```

因此，copy view 的一个 8-BF16 vector 恰好由 MFMA view 的两个 4-BF16cell 组成：

```
copy K rep 0: r[0:8]   = MFMA (k0,k1) = (0,0) 和 (1,0)
copy K rep 1: r[8:16]  = MFMA (k0,k1) = (0,1) 和 (1,1)
```

`retile` 不搬运数据，只是把这同一批 register 从左图的 copy 坐标改用右图的 MFMA 坐标解释。`fx.copy` 经由左图写入，`fx.gemm` 经由右图读取。


还有一点值得注意的是a_frag_retile `((8,1),4,2):((1,0),16,8)`和a_frag`(4,4,(2,2)):(1,16,(4,8))`本身的cosize都为64，并没有hole， 它们都是不同的坐标到同一段 register storage 的映射。

```
  copy view:
  (v, dummy, m_rep, k_rep)
    -> r[v + 16*m_rep + 8*k_rep]

  MFMA view:
  (val, m_rep, k0, k1)
    -> r[val + 16*m_rep + 4*k0 + 8*k1]

  两式的结果都只会落在：

  r[0] ... r[63]
```
而这里的 global-memory source view，虽然也有 64 个逻辑元素，即
```mlir
//A
%174 = fly.slice(%84, %173) : (!fly.memref<bf16, #fly_rocdl.buffer_desc, ((8,1),4,2,32):((1,0),131072,128,256)>, !fly.int_tuple<(*,*,*,?)>) -> !fly.memref<bf16, #fly_rocdl.buffer_desc, ((8,1),4,2):((1,0),131072,128)> loc(#loc285)
//B
%176 = fly.slice(%85, %175) : (!fly.memref<bf16, #fly_rocdl.buffer_desc, ((8,1),4,2,32):((1,0),131072,2048,4096)>, !fly.int_tuple<(*,*,*,?)>) -> !fly.memref<bf16, #fly_rocdl.buffer_desc, ((8,1),4,2):((1,0),131072,2048)> loc(#loc287)
```
### 循环完 reduce之前
C的结果还在`c_frag`  [value, N_rep, M_rep]
### C LDS

#### C lds
```python
c_lds = fx.make_view(lds.C_lds.ptr,
    fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((TILE_M * 4, TILE_N), order=(1, 0))))
```

`c_lds shape = (wave_count * TILE_M, TILE_N)= (4 * 64, 64)` 但第一维 256 不是 GEMM 的真实 M=256。它是把 split-K 的 4 份 partial C 沿第一维堆叠：
```
  c_lds[  0: 64, 0:64] = wave 0 的 C partial，shape 64x64
  c_lds[ 64:128, 0:64] = wave 1 的 C partial，shape 64x64
  c_lds[128:192, 0:64] = wave 2 的 C partial，shape 64x64
  c_lds[192:256, 0:64] = wave 3 的 C partial，shape 64x64
```
后面的 SwizzleType(3,3,3) 只是把这个原始 offset 做 LDS bank-friendly 的重新映射；逻辑上仍然可以先按上面的 (256,64):(64,1) 来理解。

#### c_tiled_lds = make_tiled_copy...
```python
c_tiled_lds = make_tiled_copy(cp_atom_lds,
          # 线程划分: 16x4，4 wave切分m方向64行，每个线程写入n方向连续4个点
          fx.make_layout(((16, 4, 4), 4), ((1, 256, 16), 64)),
          fx.make_tile(16 * 4, 16))
```
先不要把它看成 C 在 LDS 中的最终 memory layout。它是一个 thread-value layout （真正的 LDS 地址还要再和 L178-L179 的 c_lds composed/swizzle layout 组合。）

((16, 4, 4), 4):(( 1,256,16),64)
```
(t0, t1, t2) # 第一个大组是 thread 坐标： 
t0: 0..15
t1: 0..3
t2: 0..3
v: 0..3 # 第二个大组是每 thread 的 value 坐标：第二个大组是每 thread 的 value 坐标：
q =t0 * 1+ t1 * 256+ t2 * 16+ v  * 64
(M, N) = (64, 16)
q = m + 64 * n
m = t0 + 16 * t2
n = 4 * t1 + v
```
- t0: M 内的第 0..15 行。
- t2:再选择第几个 M=16 block：M = [0:16]、[16:32]、[32:48]、[48:64]。
- t1:选择 N 方向第几个 4-element group：N = [0:4]、[4:8]、[8:12]、[12:16]。
- v:当前 thread 写 N 方向连续 4 个 f32。

注意这步能够得到当前 thread 的 copy layout，是因为 `c_tiled_lds` 是一个 `TiledCopy`，它内部同时保存：
1. copy atom:UniversalCopy128b，4 个 f32 / 16 bytes。
2. TV layout:哪个 thread 负责哪个逻辑 C 坐标。
3. tiler: 当前 copy 基本 tile 是 64 x 64。

第二个 `c_tiled_lds` 的 **TV layout** 是：

TiledCopy 的 TV layout 给出 thread-to-coordinate 映射，get_slice(tid) 固定当前 thread，partition_S/D 再同具体 memory view 组合。
#### c_tiled_lds.get_slice(tid).partition_S(c_lds)
```python
  c_tensor_thr_lds_r = c_tiled_lds.get_slice(tid).partition_S(c_lds)
```

```mlir
// L193-L196: 第二个 c_tiled_lds
  %125 = fly.make_tiled_copy(%113, %123, %124)
    : (...) -> !fly.tiled_copy<
        !fly.copy_atom<!fly.universal_copy<128>, 32>,
        !fly.layout<((16,4,4),(4,4)):((256,1,4),(64,16))>,
        !fly.tile<[64|64]>
      >

  // L198: get_slice(tid) 中的 tid
  %126 = fly.make_int_tuple(%0)
    : (i32) -> !fly.int_tuple<?>

  // L198: partition_S(c_lds)
  %127 = fly.tiled_copy.partition_src(%125, %112, %126)
    : (
        !fly.tiled_copy<...>,
        !fly.memref<f32, shared,
          S<3,3,3> o 0 o (256,64):(64,1)
        >,
        !fly.int_tuple<?>
      )
   -> !fly.memref<f32, shared,
        S<3,3,3> o ?{div=4} o ((4,4),4,1):((1,1024),4096,0)
      >

```

#### fly.make_fragment_like bf16->f32

```python
  c_tensor_thr_lds_r = c_tiled_lds.get_slice(tid).partition_S(c_lds)
  c_frag_reduce =fx.make_fragment_like(c_tensor_thr_lds_r)
```
```mlir
  %127 = fly.tiled_copy.partition_src(%125, %112, %126) : (!fly.tiled_copy<!fly.copy_atom<!fly.universal_copy<128>, 32>, !fly.layout<((16,4,4),(4,4)):((256,1,4),(64,16))>, !fly.tile<[64|64]>>, !fly.memref<f32, shared, S<3,3,3> o 0 o (256,64):(64,1), align<16>>, !fly.int_tuple<?>) -> !fly.memref<f32, shared, S<3,3,3> o ?{div=4} o ((4,4),4,1):((1,1024),4096,0), align<16>> loc(#loc313)
  // 但重新给出适合 register 保存的紧密 layout：
  %128 = fly.make_fragment_like(%127) : (!fly.memref<f32, shared, S<3,3,3> o ?{div=4} o ((4,4),4,1):((1,1024),4096,0), align<16>>) -> !fly.memref<f32, register, ((4,4),4,1):((1,4),16,0)> loc(#loc314)
```

对应注意 
```python
  c_frag_bf16 = fx.make_fragment_like(c_frag_reduce[(None, 0), None, None], dtype=fx.BFloat16)
```
```mlir
   %154 = fly.make_fragment_like(%153, bf16) : (!fly.memref<f32, register, (4,4,1):(1,16,0)>) -> !fly.memref<bf16, register, (4,4,1):(1,4,0)> loc(#loc322)
```