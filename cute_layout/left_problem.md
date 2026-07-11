# complement


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
