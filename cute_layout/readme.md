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

# complement

## complement(4:2, 24)
```
A= 4:2
codomain(A) = {0, 2, 4, 6}
```
现在要让她覆盖 0,1,2,...,23

洞①(内部空隙): 1, 3, 5    ← stride=2 跳过的
洞②(远方):     8~23       ← A 到6就停了

先用2:1 补第一种洞
补完以后 8:1 然后重复三次
{0,8,16} => 3:8

所以 concate
(2,3):(1,8)

验证
layout = (4,(2,3)):(2,(1,8))

```
  a = Layout((4,(2,3)),(2,(1,8))).show()
Layout((4, (2, 3)):(2, (1, 8))) shape(size)->coshape(cosize) (4, (2, 3))(24)->24(24) rank:2 depth:2
   0:      0,     1,     8,     9,    16,    17,
   1:      2,     3,    10,    11,    18,    19,
   2:      4,     5,    12,    13,    20,    21,
   3:      6,     7,    14,    15,    22,    23,
```
24覆盖完了


但注意这里验证用的是concat 并不是local_product 这两个logical_product是(4, (2, 3)):(2, (1, 32))

## complement((2,4):(1,6), 24) 

A = (2,4):(1,6)
codomain(A) = {0,1, 6,7, 12,13, 18,19}
```
空洞 2 3 4 5
8 9 10 11
14 15 16 17
20 21 22 23
```
看规律：每 6 个一组，A 占了前 2 个 {0,1}，漏了后 4 个 {2,3,4,5}

complement = 3:2  →  {0, 2, 4}
A部分 → {0,1, 6,7, 12,13, 18,19}
R = 3:2 每次 +0/+2/+4:

完整 layout = ((2,4), 3) : ((1,6), 2)

## complement((2,2):(1,6), 24) 

codomain(A) = 0,1,6,7

洞① 每组内: {2,3,4,5}

补上去 就是 3:2

补完就是 0~11
第二个洞② 远方:   12~23  这是 2:12

complement((2,2):(1,6), 24) = (3,2):(2,12)

## 总结
complement 的任务 = 找出 A 漏掉的所有洞，用最简洁、递增stride的方式补齐


洞可能有两种：
┌─────────────────────────────────────┐
│  洞①【内部空隙】stride跳过的小格子      │
│  洞②【远方】A的最大范围之外的区域       │
└─────────────────────────────────────┘

[CuTe Layout Algebra - NVIDIA Documentation Hub](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)
也给了三个原则
> The size (and cosize) of R is bounded by size(M).

> R is ordered. That is, the strides of R are positive and increasing. This means that R is unique.

> A and R have disjoint codomains. R attempts to “complete” the codomain of A.


# devide

$$A \oslash B := A \circ (B,B^*)$$

先把坐标空间按照 B 和 complement(B) 分解成两个独立方向，再用 A 去composition(解释)这个新的二维（或多维）坐标。

所以 logical_divide 与我们理解的**除法**有很大区别。


比如
```python
A=logical_divide(Layout(24,1),Layout(4,1)).show()

A_ = composition(Layout(24,1), concatenation(Layout(4,1),complement(Layout(4,1),24))).show()

assert A==A_
```

这个式子正好说明了logical_devicde(24:1, 4:1)，我们刚才算了 complement(4,1,24)=6:4

于是
$$A \oslash B = 24:1 \circ (4:1,6:4) $$


logical_divide(Layout(24,1), Layout(4,1)) = (4, 6):(1, 4)
## Tile and REST

zd和ld的区别 如何区分 Tile和Rest呢
```
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
```

从这个例子出发

```
logical_divide(Layout(24,1), Layout(4,1)) = (4, 6):(1, 4)
(4,     6)   :   (1,     4)
 ↑       ↑         ↑       ↑
mode0   mode1     mode0   mode1

get<0> = 4:1   ← 这是 TILE（瓦片/分块内部）
get<1> = 6:4   ← 这是 REST（剩余/分块之间）

logical_divide:  (TILE, REST)              ← 本例已经是这样
zipped_divide:   ((所有TILE), (所有REST))   ← 把tile们打包，rest们打包
```

zipped_divide(Layout(24,1), Layout(4,1)) = (4:1,6:4) = (4,6):(1,4)本例结果是一样的

## logical Devide and Zipped Devide

这个例子也是一致的
```
A = Layout((4,2,3),(2,1,8))
B = Layout(4,2)
print(logical_divide(A,B))
print(zipped_divide(A,B),)
```
我们看一个更复杂的例子

```
A = 4:8
B = <2:1, 4:1>

A⋆⟨B,C⟩=(A0​,A1​)⋆⟨B,C⟩=(A0​⋆B,A1​⋆C)

A 怎么拆成两个mode呢

我们拆出来  coalesce的逆运算
A= 4:8 = (4,8):(1,4)

分配律

(logical_devide(4:1,2:1),logical_devide(8:4,4:1))

先算第一个

logical_devide(4:1,2:1) = composition(4:1,concatenation(2:1,complement(2:1,cosize(4:1))))
= concatenation(2:1,2:2) = (2,2):(1.2)


第二个
logical_devide(8:4,4:1)= composition(8:4,concatenation(4:1,complement(4:1,32)))
=composition(8:4,concatenation(4:1,omplement(4:1,32)))=composition(8:4,(4:1,8,4)) = composition(8:4,(4,8),(1,4))
= (4, 8):(4, 16)

result = (2,2),(4,8):(1,2),(4,16)

```

logical_devide((4,8),<2:1, 4:1>) =  (2,2),(4,8):(1,2),(4,16)

```
logical_divide = ((TileM, RestM), (TileN, RestN))
               = ((2,      2),    (4,      8))

结构：
  mode0 = (TileM, RestM) = (2,2)   ← M维的tile和rest抱一起
  mode1 = (TileN, RestN) = (4,8)   ← N维的tile和rest抱一起
```

**这里 complement到底怎么算有点问题**

A = 4:8
B = <2:1, 4:1>
logical_divide(A,B) = ((2, 2), (4, 2)):((1, 2), (4, 16))
zipped_divide(A,B)  = ((2, 4), (2, 2)):((1, 4), (2, 16))


### A 是什么？

```
A = (4,8):(1,4)   ← 默认列优先(column-major)

地址 = m*1 + n*4,  m∈[0,4), n∈[0,8)

画成 4×8 网格（每格是内存地址）：
        n=0  n=1  n=2  n=3  n=4  n=5  n=6  n=7
m=0      0    4    8   12   16   20   24   28
m=1      1    5    9   13   17   21   25   29
m=2      2    6   10   14   18   22   26   30
m=3      3    7   11   15   19   23   27   31
```

### tiler B 的含义

```
<2:1, 4:1>
  ↑    ↑
 M维   N维
每2个  每4个
切一块 切一块

→ 把 4×8 的矩阵切成 2×4 的小瓦片
```

---

## 真实结果对照

```python
logical_divide(A,B) = ((2, 2), (4, 2)):((1, 2), (4, 16))
zipped_divide(A,B)  = ((2, 4), (2, 2)):((1, 4), (2, 16))
```

我们逐个拆解验证。

---

## 第一步：算出四个基本零件（Tile 和 Rest）

### M 维：4 被 tiler 2:1 切

```
TileM = 2:1        （块内：走2步，stride=1）→ {0,1}
RestM = complement(2:1, 4) = 2:2   （块间：2块，stride=2）→ {0,2}
```

### N 维：8 被 tiler 4:1 切

```
原始N维在A里是 8:4（因为A的N维stride=4）
TileN = 4:4        （块内：走4步，stride=4）→ {0,4,8,12}
RestN = complement → 2:16   （块间：2块，stride=16）→ {0,16}
```

**四个零件**：
```
TileM = 2:1     RestM = 2:2
TileN = 4:4     RestN = 2:16
```

---

## 第二步：验证 logical_divide（按维度组织）

```
logical_divide = ((TileM,RestM), (TileN,RestN))
```

代入零件：
```
= ((2:1, 2:2), (4:4, 2:16))
= ((2,2):(1,2),  (4,2):(4,16))
= ((2, 2), (4, 2)):((1, 2), (4, 16))   ✓ 和代码一致！
```

**结构解读**：
```
((2, 2), (4, 2)) : ((1, 2), (4, 16))
  └M维┘  └N维┘      └M维┘  └N维┘

mode0 = (2,2):(1,2)   ← M维的 (TileM, RestM) 抱团
mode1 = (4,2):(4,16)  ← N维的 (TileN, RestN) 抱团
```

---

## 第三步：验证 zipped_divide（按角色组织）

```
zipped_divide = ((TileM,TileN), (RestM,RestN))
```

代入零件：
```
= ((2:1, 4:4), (2:2, 2:16))
= ((2,4):(1,4),  (2,2):(2,16))
= ((2, 4), (2, 2)):((1, 4), (2, 16))   ✓ 和代码一致！
```

**结构解读**：
```
((2, 4), (2, 2)) : ((1, 4), (2, 16))
  └Tile┘  └Rest┘     └Tile┘  └Rest┘

mode0 = (2,4):(1,4)   ← (TileM,TileN) = 一块完整的2×4瓦片！
mode1 = (2,2):(2,16)  ← (RestM,RestN) = 瓦片怎么排布（2×2=4块）
```

---

## 第四步：对比两者（拉链效果）

```
四个零件：
  TileM=2:1   TileN=4:4   RestM=2:2   RestN=2:16

logical_divide  按【维度】分组：
  ((TileM,RestM), (TileN,RestN))
  = ((2:1,2:2),   (4:4,2:16))
     └─M维─┘       └─N维─┘

zipped_divide   按【角色】分组：
  ((TileM,TileN), (RestM,RestN))
  = ((2:1,4:4),   (2:2,2:16))
     └─Tile─┘      └─Rest─┘

┌─────────────────────────────────────────┐
│  拉链动作：                                │
│  从 logical 里把 TileN 从第2组"拉"到第1组  │
│  把 RestM 从第1组"拉"到第2组               │
│  → 变成 zipped                            │
└─────────────────────────────────────────┘
```
## 第五步：直观理解 zipped 的两个 mode

### mode0 = `(2,4):(1,4)` —— 一块瓦片长啥样

```
一块 2×4 的瓦片，内部地址：
地址 = m*1 + n*4,  m∈[0,2), n∈[0,4)

        n=0  n=1  n=2  n=3
m=0      0    4    8   12
m=1      1    5    9   13

这就是【左上角第一块瓦片】的内容！
```

### mode1 = `(2,2):(2,16)` —— 瓦片如何排布

```
2×2 = 4 块瓦片，每块的【起始地址】：
地址 = i*2 + j*16,  i∈[0,2), j∈[0,2)

        j=0   j=1
i=0      0    16
i=1      2    18

意思是4块瓦片的左上角分别在 0, 2, 16, 18
```
原 4×8 矩阵被切成 4 块 2×4 瓦片：

```
        n=0  n=1  n=2  n=3 │ n=4  n=5  n=6  n=7
      ┌───────────────────┼───────────────────┐
m=0   │  0    4    8   12  │  16   20   24   28 │
m=1   │  1    5    9   13  │  17   21   25   29 │
      ├───────────────────┼───────────────────┤
m=2   │  2    6   10   14  │  18   22   26   30 │
m=3   │  3    7   11   15  │  19   23   27   31 │
      └───────────────────┴───────────────────┘
        瓦片(0,0)起点0       瓦片(0,1)起点16
        瓦片(1,0)起点2       瓦片(1,1)起点18
```

**zipped_divide 的两个 mode 正好对应**：
- **mode0** `(2,4):(1,4)` = 每块瓦片内部的 8 个地址模式
- **mode1** `(2,2):(2,16)` = 4 块瓦片的起点排布 {0,2,16,18}


## Logical Devide 2-D Example

类似direct product这个规律
A ⋆ ⟨B,C⟩ = (A0,A1) ⋆ ⟨B,C⟩ = (A0 ⋆ B,A1 ⋆ C)

用到例子上
```
A = (9,(4,8)):(59,(13,1))
B = <3:3, (2,4):(1,8)>

A = (mode0, mode1) = ((9:59),(4,8):(13,1))
Then 

logical_devide(A,B) = (logical_devicde((9:59),(3:3)),logical_devide((4,8):(13,1),(2,4):(1,8) ))
```

注意 (3,(2,4)):(3,(1,8)) 不同于 <3:3, (2,4):(1,8)>,第一个是Layout第二个是tuple



# product
$$A \otimes B := (A, A^* \circ B)$$

这里有个有趣的问题 complment(A,M),M 哪儿来的



## logical_product((2,2):(4,1),6:1)

M = size(A) × size(B) =24

> Complement of A = (2,2):(4,1) under 6*4 = 24 is A* = (2,3):(2,8).
> Composition of A* = (2,3):(2,8) with B = 6:1 is then (2,3):(2,8).
> Concatenation of (A,A* o B) = ((2,2),(2,3)):((4,1),(2,8)).


## logical_product((2,2):(4,1),(4,2):(2,1))

```

A = Layout((2,2),(4,1))
B = Layout((4,2),(2,1))

R = composition( complement(A,4*8),B)
print("R",R)
concat(A,R) 
```
这里R同样是M = size(A) × size(B) =24
## Ref



- [CuTe Layout Algebra - NVIDIA Documentation Hub](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)


- [CuTe Layout Representation and Algebra - arxiv paper](https://arxiv.org/pdf/2603.02298)

- [pycute](https://github.com/NVIDIA/cutlass/tree/main/python/pycute)






