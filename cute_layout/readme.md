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
coalesce((2, (1, 6)) : (1, (6, 2)), ⟨∗, ∗⟩) = coalesce((2 : 1, (1, 6) : (6, 2)), ⟨∗, ∗⟩)
= (coalesce(2 : 1, ∗), coalesce((1, 6) : (6, 2), ∗))
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




## injective
$$f(x_1) = f(x_2) \implies x_1 = x_2$$
一个 Layout 是 injective（单射）的，意思是：不同的逻辑坐标，一定映射到不同的内存偏移量（没有两个坐标指向同一块内存地址）。
(2, 3) : (1, 0) 这个就不是injective

## composition
$B = s : d$（单模式，前面已假设）
$A = a : b$（也是单模式整数布局，这是本段新增的进一步简化）

先来看下面两个例子
### A is integral
$A = 8:3$ 且 $B = 4:2$ 手动计算 $R(c) = A(B(c))$

A
```
0 3 6 9 12 15 18 21
```
B
```
0 2 4 6
```
计算

```
R(0)=A(B(0))=A(0)=0
R(1)=A(B(1))=A(2)=6
R(2)=A(B(2))=A(4)=12
R(3)=A(B(3))=A(6)=18
```
### A is multimodal

$$A = (4, 3) : (2, 8) \qquad B = s : d = 3 : 2$$
print下 A和B
```
A
Layout((4, 3):(2, 8)) shape(size)->coshape(cosize) (4, 3)(12)->23(23) rank:2 depth:1
   0:      0,     8,    16,
   1:      2,    10,    18,
   2:      4,    12,    20,
   3:      6,    14,    22,
B
Layout(3:2) shape(size)->coshape(cosize) 3(3)->5(5) rank:1 depth:0
     0,     2,     4,
```

```
R(0)=A(B(0))=A(0)=A(0,0)=0
R(1)=A(B(1))=A(2)=A(2,0)=4
R(2)=A(B(2))=A(4)=A(0,1)=8
```

### (6,2):(8,2) o (4,3):(3,1)
https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#example-1-worked-example-of-calculating-a-composition

```
R = A o B
  = (6,2):(8,2) o (4,3):(3,1)
  = ((6,2):(8,2) o 4:3, (6,2):(8,2) o 3:1)
```

#### 先计算 (6,2):(8,2) o 4:3
```
(6,2)/3  = (2,2)

(8,2) => (3*8,2)
```

把剩下的shape(2,2) 去mod
```
(2,2) % 4 = (2,2)
```

第一个是
(2,2):(24,2)

#### 再计算 (6,2):(8,2) o 3:1
```
(6,2)/1 = (6,2)

(8,2)也不动
```

```
(6,2) %3  = (3,1) 
```
这里取余够了 剩下的维度1

(3,1):(8,2) coalesce以后 3:8

#### 汇总
```
(6,2):(8,2) o (4,3):(3,1) = ((2,2):(24,2)),(3:8) = ((2,2),3):((24,2),8)
```

### 20:2  o  (5,4):(4,1)

```
20:2  o  (5,4):(4,1) 
= (20:2  o (5,4)):(20:2 o (4,1) )
```

按这个trival的公式算 R = A o B = a:b o s:d = s:(b*d)

```
20:2 o (5,4)

= 5:8
```

```
20:2 o (4,1)

= 4:2
```

(5,4):(8,2)

= 4:8
### (10,2):(16,4)  o  (5,4):(1,5)

```
(10,2):(16,4)  o  (5,4):(1,5)
= (10,2):(16,4)  o  (5,1), (10,2):(16,4)o(4,5)
```

```
(10,2):(16,4) o (5,1) 

(10,2)/1 = (10,2)
(16,4)不动
(10,2)%5 = (5,1)  # 注意后面是1
(5,1):(16,4)


(10,2):(16,4)o(4,5) 

(10,2)/5  = (2,2)

(16,4)=>(16*5,4)

(2,2)%4 = (2,2)

(2,2):(80,4)
```

finally
```
(5,1):(16,4),(2,2):(80,4)
```
```
(5,16),(2,2):(80,4)
```
=>
```
(5,(2,2)):(16,(80,4))
```
### By-mode Composition

注意这里尖括号的分配律是 类似direct product
```
A ⋆ ⟨B,C⟩ = (A0,A1) ⋆ ⟨B,C⟩ = (A0 ⋆ B,A1 ⋆ C)
```

**这里注意R = AoB，R的shape和B一样的，或者说R是B的refinement**




## Ref



- [CuTe Layout Algebra - NVIDIA Documentation Hub](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)


- [CuTe Layout Representation and Algebra - arxiv paper](https://arxiv.org/pdf/2603.02298)

- [pycute](https://github.com/NVIDIA/cutlass/tree/main/python/pycute)






