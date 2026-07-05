## CuTe Layout Algebra

主要基于[CuTe Layout Algebra - NVIDIA Documentation Hub](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)和[CuTe Layout Representation and Algebra - arxiv paper](https://arxiv.org/pdf/2603.02298)
## concatenation

```
B = 4:2
C = (2,3):(1,8)
R = concantenation(B,C)=(4:2,(2,3):(1,8))=(4,(2,3)):(2,(1,8))
```

## by-mode operations
> We often call these “by-mode operations," and every operation that is defined in this section (coalesce, composition, complement, logical divide, etc) can also be applied by-mode. This approach 

$$A \star \langle B, C \rangle = (A_0, A_1) \star \langle B, C \rangle = (A_0 \star B, \; A_1 \star C)$$

右边等式类似分配律的计算，数学上看上去似乎是[Direct product](https://en.wikipedia.org/wiki/Direct_product)

我们看paper3.2的计算
```
coalesce((2, (1, 6)) : (1, (6, 2))) = coalesce((2 : 1, (1, 6) : (6, 2)))
= (coalesce(2 : 1), coalesce((1, 6) : (6, 2)))
= (2 : 1, 6 : 2)
= (2, 6) : (1, 2).
```
## coalesce
基本上 [CuTe Layout Algebra - Coalesce](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#coalesce)的几种计算方法。

> 1. s0:d0  ++  _1:d1  =>  s0:d0. Ignore modes with size static-1. 
> 
> 2. _1:d0  ++  s1:d1  =>  s1:d1. Ignore modes with size static-1. 
>
> 3. s0:d0  ++  s1:s0*d0  =>  s0*s1:d0. If the second mode’s stride is the product of the first mode’s size and stride, then they can be combined. 
>
> 4. s0:d0  ++  s1:d1  =>  (s0,s1):(d0,d1). Else, nothing can be done and they must be treated separately. 

然后我们来看下下面这个例子
$$(2,(1,6)):(1,(6,2)) ;\longrightarrow; \text{shape}=(2,1,6),\ \text{stride}=(1,6,2)$$
**从左到右**逐个合并/丢弃逐维处理，规则：

1. size = 1 的维度 → 直接丢弃
2. 相邻两维若 $d_{next} = s_{cur} \times d_{cur}$ → 合并

我们先把 mode1 那个1:6直接丢弃 剩下(2,6):(1,2)这个满足规则2，合并成为12:1

## left problems
 
- Complement and Product

> Complement of A = (2,2):(4,1) under 6*4 = 24 is A* = (2,3):(2,8).


## Ref



- [CuTe Layout Algebra - NVIDIA Documentation Hub](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)


- [^CuTe Layout Representation and Algebra - arxiv paper](https://arxiv.org/pdf/2603.02298)

- [pycute](https://github.com/NVIDIA/cutlass/tree/main/python/pycute)






