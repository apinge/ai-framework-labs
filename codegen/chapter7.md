# LLVM 地址空间、虚拟地址与 HIP/ROCm

本节记录阅读 *LLVM Code Generation: A Deep Dive into Compiler Backend Development* 中这句话时产生的问题：

> Most CPUs only have one address space.

这里的 **address space** 是 LLVM IR 中指针类型携带的语义信息。它和操作系统讨论的“进程虚拟地址空间”相关，但不是同一个概念。要理解 HIP/ROCm 的代码生成，必须把这两层分开。

## 最初的问题

1. 为什么书中要强调 CPU 和 GPU 的不同？
2. LLVM IR 阶段的地址是否全都是虚拟地址？
3. MMU 是什么，是否只有 CPU 才有？
4. 在 HIP/ROCm 的 AMDGPU 目标上，LLVM 的 address space 实际表示什么？

## 两个同名但不同层次的概念

### 虚拟地址与 MMU

虚拟地址（virtual address, VA）回答的是：**一个地址数值最终如何对应到物理内存？**

以 CPU 上的用户程序为例，指针中常保存的 `0x00007f12_3456_7000` 是进程的虚拟地址。MMU（Memory Management Unit，内存管理单元）依据页表，把它翻译成物理内存地址，并检查读、写、执行等权限。TLB 是这类页表翻译的缓存。

```text
CPU 指针值（虚拟地址）
          |
          v
       CPU MMU / TLB
          |
          v
物理内存页与访问权限
```

现代 GPU 也不例外。AMD GPU 在访问 global memory 时也有虚拟内存、页表和 TLB 一类的机制。ROCm 可以将 VRAM、以及可供 GPU 访问的系统内存映射到 GPU 可见的虚拟地址范围。因此，MMU 并非 CPU 独有。

不过，LDS（`local`）和每线程私有存储（`private`）不能简单理解为“又一套进程虚拟地址”。它们的寻址还依赖 work-group 或 work-item 的上下文。

### LLVM IR 的 address space

LLVM 的 `addrspace(N)` 回答的是：**此指针指向哪一类内存，它能被谁看见，应使用什么寻址和访问语义？**

它是指针类型的一部分。例如：

```llvm
; 默认地址空间，编号 0 可以省略
%x = load i32, ptr %p

; 明确指向地址空间 1
%y = load i32, ptr addrspace(1) %q
```

因此，不能笼统地说“LLVM IR 中的所有地址都是虚拟地址”。LLVM IR 建模的是**目标机的指针和内存语义**。某个 address space 中的指针可能是经 MMU 翻译的虚拟地址，也可能是 LDS 内的偏移、每线程 scratch 的偏移，或其他目标特定表示。

## 为什么普通 CPU 通常只需一个 LLVM 地址空间

在 x86-64、AArch64 等常见 CPU 的普通 C/C++ 程序中，栈、堆、全局变量和 `mmap` 得到的内存，通常都可由同一种普通指针和同一类 load/store 指令访问：

```text
普通 CPU 指针
      |
      v
当前进程的一套虚拟地址规则
      |
      v
MMU 翻译到物理内存
```

它们当然有不同的分配来源和生命周期，但对 LLVM CPU 后端而言，通常无需把“这是栈地址”或“这是堆地址”编码为不同的指针 address space。默认的 `addrspace(0)` 已足够表达普通内存指针。

这正是书中 “Most CPUs only have one address space” 的核心意思：对于普通 CPU 程序，目标机提供给编译器的普通可寻址内存模型大多是统一的。

## AMDGPU 为什么需要多个 LLVM 地址空间

HIP kernel 同时面对几种硬件语义不同的内存。LLVM 后端必须一直保留这种差别，才能选择正确的指令、缓存和同步规则，并做正确的别名分析。

AMDGPU 后端的主要约定如下：

| LLVM address space | 名称 | 含义 |
| --- | --- | --- |
| `addrspace(0)` | flat / generic | 可能指向多类可由 flat 指令处理的内存；在 AMDGPU 上并非单纯的 CPU 默认指针。 |
| `addrspace(1)` | global | GPU 全局可访问内存；一般通过 GPU 虚拟地址访问，底层可以是 VRAM 或映射给 GPU 的系统内存。 |
| `addrspace(3)` | local | 一个 work-group 共享的 LDS。 |
| `addrspace(5)` | private | 每一个 work-item / thread 独有的存储，通常关联 scratch。 |

这些编号由 LLVM AMDGPU 后端定义于 `llvm/Support/AMDGPUAddrSpace.h`。

### `local` 不是普通 global virtual address

考虑两个 work-group 都访问 local 地址偏移 `0`：

```text
work-group 0: local offset 0  -> work-group 0 自己的 LDS
work-group 1: local offset 0  -> work-group 1 自己的 LDS
```

同样的数值不是同一块存储，因为 work-group 上下文参与了地址解释。由此可见，`addrspace(3)` 的价值不在于给地址加一个编号，而在于表达“这是 work-group 范围共享的 LDS”。

`private` 也类似。同一个 private 偏移对两个 GPU 线程会表示不同的位置，因为线程身份参与解释。

## 放进 HIP 代码中理解

下面的 kernel 参数指向 global memory：

```cpp
__global__ void saxpy(const float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  y[i] += x[i];
}
```

AMDGPU LLVM IR 中，参数可能以接近下面的形式表示（具体签名会随 Clang/ROCm 版本和优化阶段变化）：

```llvm
define amdgpu_kernel void @saxpy(
    ptr addrspace(1) %x,
    ptr addrspace(1) %y) {
  ; ...
}
```

`addrspace(1)` 告诉后端：这是 global memory。该地址仍可能是 GPU 虚拟地址，但 LLVM 还知道访问语义是 global，而不是 LDS 或每线程私有存储。

再看 shared memory：

```cpp
__global__ void copy_tile(const float* input, float* output) {
  __shared__ float tile[256];
  tile[threadIdx.x] = input[threadIdx.x];
  __syncthreads();
  output[threadIdx.x] = tile[threadIdx.x];
}
```

概念上，其中的 `tile` 会使用 `addrspace(3)`：

```llvm
@tile = addrspace(3) global [256 x float] undef
```

它要求后端使用 LDS 对应的访问方式；`__syncthreads()` 的同步含义也正是围绕同一 work-group 的这块 local memory 建立的。

## `flat` / generic 指针为什么存在

CPU 上 `addrspace(0)` 多半就是默认普通指针。AMDGPU 上，`addrspace(0)` 称为 flat 或 generic：它可以表示由 flat 指令处理的多类内存。编译器如果之后能够证明一个 generic 指针实际只可能指向 global memory，就可以将其收窄或提升到更具体的 address space，并产生更合适的指令。

当指针确实需要跨 address space 转换时，LLVM 用 `addrspacecast` 表示这个语义：

```llvm
%generic = addrspacecast ptr addrspace(1) %global_ptr to ptr
```

这不是普通的 `bitcast`。`bitcast` 只改变类型视图；`addrspacecast` 告诉后端，源和目标指针属于不同的内存语义，转换可能需要目标机专门的规则或指令。

## 结论

“Most CPUs only have one address space” 并不是说 CPU 没有虚拟内存，也不是说 GPU 没有 MMU。它是在解释 LLVM 后端为何需要 address space 这一类型信息：

- 普通 CPU 程序中的内存大多可通过一套统一的普通指针语义访问，因此通常以 `addrspace(0)` 建模。
- AMDGPU 的 global、local/LDS 和 private/scratch 内存具有不同的可见范围、地址解释和访问指令；LLVM 需要用不同的 `addrspace(N)` 保留这些区别。
- 在 ROCm 中，global memory 可以走 GPU 虚拟地址翻译；但这与 LLVM 用 `addrspace(1)` 标记它是 global memory 是两个不同层次的信息。

## 向量、连续内存与端序

阅读 address space 段落后，书中紧接着有如下提醒：

> 把 vector 当成 C 风格数组来理解很方便，但两者并不完全等价。数组中的字节序列通常可假定在内存中连续；vector 不一定如此，具体取决于目标端序。后面的 Data layout 一节会进一步说明。

这句话很容易被理解成“`<4 x i32>` 的四个元素会随机散落在内存中”。这不是它对 x86 或 AMDGPU 的实际含义。更准确的理解如下。

### vector 首先是 SIMD 值，不是数组对象

LLVM 的 `<4 x i32>` 是一个四 lane 的 SIMD 值。它常常一直保存在 X86 的 XMM/YMM/ZMM 寄存器或 AMDGPU 的 VGPR 中，根本还没有一个对应的内存对象。只有执行 `load`、`store`、寄存器溢出（spill），或需要按位重解释时，才必须讨论它的内存布局。

对字节大小的元素，LLVM LangRef 规定 vector 元素在内存中的布局通常和数组一样。因此，下面这个常见例子可以按连续数组理解：

```llvm
; %v 的 lane 0, 1, 2, 3 分别对应 %p 起始处的四个连续 i32
%v = load <4 x i32>, ptr %p, align 16
```

```text
地址:  %p            %p + 4        %p + 8        %p + 12
内容:  lane 0         lane 1        lane 2        lane 3
```

因此，在我们主要关心的 `float`、`half`、`i8`、`i16`、`i32`、`i64` 向量 load/store 中，可以把 `<N x T>` 的元素顺序理解为数组 `T[N]` 的顺序：lane 0 位于最低地址，元素依次向高地址排列。

### 书中提醒的真正边界：按位重解释

vector 的数组类比在“把整条向量当成一个连续的位串，再改用另一类型解释”时不再充分。LLVM 的 `bitcast` 不改变任何 bit，语义相当于：

1. 把源值按目标的内存布局存入内存；
2. 从同一批 bit 以目标类型读回。

因此，vector lane 在这个连续位串中谁处在高位、谁处在低位，取决于端序。

```llvm
%word = bitcast <2 x i8> <i8 0x11, i8 0x22> to i16
```

| 目标端序 | `%word` 的 `i16` 数值 |
| --- | --- |
| little-endian | `0x2211` |
| big-endian | `0x1122` |

原因是 little-endian 将 lane 0 放入结果整数的最低有效位；big-endian 则放入最高有效位。注意，`bitcast` 本身没有生成任何数据重排指令；变化的是 LLVM 对同一 bit pattern 的语义解释。

### 小于一个字节的元素才会真正破坏“字节数组”直觉

例如 `<4 x i4>` 的每一个 lane 只有 4 bit，无法为每个 lane 分配一个独立、可按字节寻址的位置。LLVM 会将 lane 无填充地打包，端序决定它们拼成整数时的次序：

```llvm
%x = bitcast <4 x i4> <i4 1, i4 2, i4 3, i4 5> to i16
```

| 目标端序 | `%x` |
| --- | --- |
| little-endian | `0x5321` |
| big-endian | `0x1235` |

若总 bit 数不是字节宽度的整数倍，目标还可能对存储大小进行填充；填充落在哪一侧并没有统一的跨目标约定。这才是“不能简单假定 vector 是一串连续字节”的重点。

### 对 x86 和 AMDGPU 的结论

x86-64 是 little-endian；HIP/ROCm 使用的 AMDGPU 目标也是 little-endian。两者都在 module 的 `target datalayout` 字符串开头以 `e` 表示这一点，例如：

```llvm
target datalayout = "e-..."
```

所以，对这两个目标：

- 常见的 `<N x float>`、`<N x i32>` 等向量 load/store 可以按连续数组的元素顺序推理。
- `<2 x i8> <0x11, 0x22>` `bitcast` 为 `i16` 时，结果按 little-endian 解释为 `0x2211`。
- 不要把这种结论移植到 big-endian target；更不要依据它推断 `<N x i1>`、`<N x i4>` 等非字节元素 vector 的字节布局。
- HIP kernel 的 SIMD/SIMT 执行、global/LDS/private address space，与此问题正交。这里讨论的是**一个 LLVM vector 值的 lane 和 bit 在内存中的排列**，不是不同 GPU 内存区域的选择。

### 书中后续 Data layout 一节的要点

后文说明 `target datalayout` 是 module 级字符串，用于描述目标的内存布局约定，包括：类型大小和 ABI 对齐、不同 address space 的指针大小、原生整数宽度，以及端序。各字段用 `-` 分隔。

与当前问题直接相关的内容有：

- 开头的 `e` 表示 little-endian，`E` 表示 big-endian。
- `v<size>:<abi>[:<pref>]` 描述给定总位宽 vector 的 ABI 对齐与首选对齐。例如 vector 的总位宽为 128 bit 时，目标可要求 128-bit 对齐。
- data layout 会影响 `load`/`store` 的默认对齐、优化代价模型和代码生成；它并非注释。
- data layout 还可指定 program、`alloca` 和 global object 的默认 address space，但这只影响之后由 LLVM API 新建的对象；文本 IR 中裸写的 `ptr` 仍然是 `addrspace(0)`。
- 不应为了把一份 IR 强行交给另一 target 而随意手改 data layout。已有 IR 中显式写出的 `align`、地址空间、ABI 已经携带原 target 的决定，单独替换字符串无法把它们自动改正确。

实践中，分析 x86 或 HIP/ROCm 的 `.ll` 文件时，先查看 module 顶部的 `target triple` 与 `target datalayout`；然后只在遇到 vector `bitcast`、按字节访问、非自然对齐 load/store 或跨 target 移植 IR 时，才需要深入追踪端序和 vector 内存布局。

# GPU Address Space

在 HIP/ROCm 的 AMD GPU 上，主要是这些：
```
 LLVM IR         GPU 内存含义          HIP 中的直觉对应
━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 addrspace(1)    global memory         hipMalloc 得到、所有 work-group 可访问的全局显存/映射内存
──────────────  ────────────────────  ───────────────────────────────────────────────────────────
 addrspace(3)    local memory / LDS    __shared__，一个 block/work-group 内共享
──────────────  ────────────────────  ───────────────────────────────────────────────────────────
 addrspace(5)    private memory        每个 GPU 线程各自的局部变量、scratch
──────────────  ────────────────────  ───────────────────────────────────────────────────────────
 addrspace(0)    flat / generic        暂时未确定具体类别，或可统一访问的指针
```
所以书中强调 CPU 与 GPU 的差异，是因为下面两行在 GPU 上的含义很不一样：
```C++
__shared__ float tile[256];  // LDS：一个 block 共享
float temp;                  // private：每个线程一份
```
tile[0] 对同一个 block 内所有线程而言是同一个位置；而 temp 即使每个线程都在“自己的 offset 0”，实际也是每线程不同的位置。

LLVM 必须在 IR 中保留这种区别，才能：

对 tile 生成 LDS 的读写指令；
对 global 指针生成访问全局显存的指令；
知道哪些内存能在 block 内共享、哪些不能；
做正确的同步和别名优化。

更准确地说，书里不是在断言“所有 GPU 都严格只有这几种地址空间”，而是在说：GPU 这种目标机常有多种语义不同的存储区域，因此 LLVM 需要 address space 机制；普通 CPU 通常不需要区分这么多种。

## `undef` 与 `poison`

书中说，除 `label` 和 `void` 外，所有 LLVM 类型都可以使用常量 `undef` 与 `poison`，例如 `i32 undef`、`ptr poison`。这里的“所有类型”包括整数、浮点数、指针和 vector；它说的是 LLVM IR 的常量和值语义，并不表示源语言中应主动写出这两个值。

这两个词最容易被误读为同一种“坏值”，但它们表达的约束完全不同。

| 值 | 核心含义 | 直接使用本身 | 优化器可作的假设 |
| --- | --- | --- | --- |
| `undef` | 一个未指定的 bit pattern；程序对任意取值都应成立。 | 不自动产生未定义行为。 | 可在每次使用时选择对优化有利的值。 |
| `poison` | 某个受约束操作已经违反前提，得到的无效值。 | 在大多数普通计算中传播为 poison。 | 一旦流入控制流、解引用等敏感位置，程序具有 UB。 |

### `undef`：不关心值，不等于 undefined behavior

`undef` 的含义不是“程序已经发生 C/C++ 未定义行为”，而是“这里允许是任意 bit pattern”。例如：

```llvm
%r = add i32 %x, undef
```

结果不能被依赖为一个确定值，但这条 `add` 自身仍有定义。LLVM 可以针对某一次使用把 `undef` 视为任何适合优化的整数。

一个重要后果是：同一个 `undef` 在不同使用点不保证相同。下面的结果仍是 `undef`，不能折叠为零：

```llvm
%u = add i32 0, undef
%r = xor i32 %u, %u       ; 结果是 undef，不保证为 0
```

直觉上，两个 `%u` 看起来是同一个 SSA 名称；但 `undef` 表示该值的每次观察都可选取任意值。因此，用 `undef` 写分支条件也不安全：

```llvm
br i1 undef, label %then, label %else ; UB
```

原因不是 `undef` 本身立刻导致 UB，而是控制流必须选择一个明确的方向；LLVM 可以假设这个无定义控制流不会实际执行。

### `poison`：先传播，进入敏感位置时触发 UB

`poison` 常由“带额外语义保证”的指令违反保证而产生。例如 `nsw` 表示 signed no-wrap；若加法发生有符号溢出，结果不是环绕后的整数，而是 poison：

```llvm
%p = add nsw i32 2147483647, 1 ; %p 是 poison
%q = add i32 %p, 1            ; %q 仍是 poison
```

这项设计使 LLVM 可以放心地推测执行和重排纯计算：产生 poison 时不必立刻终止程序，poison 先经由绝大多数指令传播。它真正到达必须具有具体、有效操作数的位置时，立即产生 undefined behavior。典型敏感位置包括：

- `br` 或 `switch` 的条件；
- `load`、`store` 和其他内存操作的地址；
- 整数除法或取余的除数；
- 间接 `call` 的被调函数指针；
- 带 `noundef` 约束的调用参数或返回值。

例如：

```llvm
%p = add nsw i32 %x, 1
%is_negative = icmp slt i32 %p, 0
br i1 %is_negative, label %a, label %b
```

若 `%x` 为 `INT_MAX`，`%p` 和 `%is_negative` 都是 poison；poison 被作为分支条件使用，于是整个执行是 UB。于是优化器可以基于“执行能到达这个分支就说明 `%x + 1` 没有 signed overflow”做推理。

书中的“poison 不应被读取”是便于入门的说法。形式上，`add poison, 1` 这种纯计算是允许的，结果仍是 poison；真正不可接受的是把它用于需要有效值的操作数位置。另一个细节是，将 poison **作为要写入的数值** `store i32 %poison, ptr %out` 本身允许；之后从这块内存读回相同 poisoned bits 仍得到 poison。把 poison **作为地址** `load i32, ptr %poison_ptr` 才是立即 UB。

### `freeze`：将不确定或 poisoned 值固定为一次有效选择

`freeze` 可以阻止 `undef` 或 poison 的继续传播，并产生一个任意但确定的、定义良好的值。相同执行中该 `freeze` 结果的每次使用都相同：

```llvm
%flag = freeze i1 undef
br i1 %flag, label %then, label %else ; 合法；方向可以不确定，但已固定
```

同样，`freeze i32 %poison` 会得到一个任意合法的 `i32`，从而可以安全地继续参与普通计算或控制流。它不是“恢复原本正确的结果”，而是明确告诉 LLVM：此处接受任意一个值，但此后必须把它当作一个真实、稳定的值。

### 对 x86 与 AMDGPU 的影响

`undef`、poison 和 `freeze` 是 LLVM IR 的通用语义；在选择 x86 后端还是 AMDGPU 后端之前就已经成立。它们不因小端序、GPU 的 SIMT 执行或 `global`/`local`/`private` address space 而改变。

对 HIP/ROCm 特别值得注意的是：若 `ptr addrspace(1)`、`ptr addrspace(3)` 或 `ptr addrspace(5)` 的指针值为 poison，把它交给对应 address space 的 `load`/`store` 解引用一样是 UB。不同 address space 只决定访问哪类 GPU 内存，不能让 poison 指针变得可用。

读 IR 时可以采用下面的判断顺序：

1. 找到可能产生 poison 的操作，例如 `nsw`/`nuw` 算术、带严格前提的转换或 `inbounds` 地址计算。
2. 沿 SSA use-def 链检查它是否传播到 branch、内存地址、除数、间接调用或 `noundef` 边界。
3. 若中间有 `freeze`，从该点起按一个未知但确定的普通值处理。
4. 看到 `undef` 时，不要假定它在多次使用中相同，也不要把它当作“随机但固定”的值；需要固定值时同样使用 `freeze`。

## 用 `getelementptr` 计算结构体字段地址

书中用下面的类型和指令说明如何访问嵌套结构体字段：

```llvm
%my.type = type { i32, { ptr, half }, { i32, i1, i1 } }

%addr_half_field = getelementptr inbounds %my.type, ptr %dst,
                                     i64 0, i32 1, i32 1
```

这条指令的结果是：**`%my.type` 对象中，第二个字段里的第二个字段（`half`）的地址。** 它只做地址计算，不会从 `%dst` 读取任何字节；要取出该 `half` 值，还需后续的 `load half, ptr %addr_half_field`。

### 先展开 `%my.type`

```text
%my.type = {
  i32,                  ; 字段 0
  { ptr, half },        ; 字段 1
    ├─ ptr              ; 字段 1.0
    └─ half             ; 字段 1.1  <- 目标
  { i32, i1, i1 }       ; 字段 2
}
```

`getelementptr`（常简写为 GEP）的第一个类型参数 `%my.type` 很重要。LLVM 现在使用 opaque pointer，`ptr` 本身不再携带“它指向什么元素类型”的信息；因此 GEP 需要这个类型来知道每一步该如何计算大小、对齐和字段偏移。

### 逐个解释索引

可把原指令读作下面的伪 C++，但不要把它理解成真的解引用：

```cpp
&dst[0].field1.field1
```

其中 `dst` 在类型意义上被当作指向 `%my.type` 的地址。

| 指令片段 | 含义 | 得到的子对象类型 |
| --- | --- | --- |
| `%my.type, ptr %dst` | 以 `%my.type` 作为从 `%dst` 开始解释内存布局的根类型。 | `%my.type` |
| `i64 0` | 在连续的 `%my.type` 对象中取第 0 个，即 `&dst[0]`。这一步保留在根对象上。 | `%my.type` |
| `i32 1` | 取结构体的字段 1；结构体字段从 0 编号，所以这是第二个字段 `{ ptr, half }`。 | `{ ptr, half }` |
| `i32 1` | 再取该内部结构体的字段 1，即第二个字段 `half`。 | `half` |

因此 `%addr_half_field` 的 LLVM 类型仍是 `ptr`，但它在语义上指向一个 `half`，且地址空间与 `%dst` 相同。

第一个 `i64 0` 经常令初学者疑惑：它不是“选第一个结构体字段”。GEP 的第一个索引总是先把基指针看作某个连续元素序列中的位置；只有第二个索引开始才是在根结构体中选择字段。若第一项写 `i64 1`，含义是跳过一个完整的 `%my.type`，再从下一个结构体对象中选择字段。

结构体字段索引必须是编译期常量 `i32`，因为 LLVM 需要在 IR 中知道选择哪个字段及其静态偏移。数组、vector 和首个“连续对象”索引可以是运行时整数值；其位宽应符合当前 target 的 data layout 约定。

### 这条 GEP 计算的偏移由 DataLayout 决定

GEP 不是简单地把 `1 + 1` 加到地址上。它会跳过前面字段的**实际布局大小**，其中包括 ABI 对齐插入的 padding。

以常见 64-bit x86 的布局为直觉示例，`ptr` 通常要求 8-byte 对齐，可能得到如下布局：

```text
%dst + 0   : i32                 (4 bytes)
%dst + 4   : padding             (4 bytes，使下一字段按 8 对齐)
%dst + 8   : 内层 ptr            (8 bytes)
%dst + 16  : half                (2 bytes)  <- %addr_half_field
```

于是这个特定 x86 示例通常计算出 `%dst + 16`。不过 `16` 不是 IR 语义中写死的结论：pointer 的大小与对齐、结构体尾部 padding、以及某个 address space 的 pointer 布局都由 module 的 `target datalayout` 决定。

在 AMDGPU 上尤其不要手算后假设和 x86 永远相同。`ptr addrspace(1)`、`ptr addrspace(3)` 等可能有目标特定的指针表示或对齐。GEP 会保留 `%dst` 的 address space，并使用该 target/data layout 的规则计算字段偏移；后端再决定访问 global、LDS 或 private memory 的指令。

### `inbounds` 加了什么约束

`inbounds` 不会让 GEP 读取内存，也不会做运行时边界检查。它是 IR 对优化器作出的保证：这次地址计算的结果必须仍落在同一个已分配对象内，或恰好位于该对象末尾；计算过程不能发生不符合该对象范围的地址溢出。

若这个保证在某次执行中不成立，带 `inbounds` 的 GEP 结果是 poison。后续若把该 poison 指针用于 `load`、`store`、分支等敏感操作，便会产生 UB。这正是前一节 poison 规则与 GEP 相接的地方。

如果无法证明索引不会越过对象，IR 可以省略 `inbounds`：普通 GEP 仍然只计算地址，但不会附带这项“同一对象内”的优化承诺。只有在确实满足条件时，前端或优化 pass 才应加上 `inbounds`。

### 与 C/C++ 的对应和读 IR 的方法

该 GEP 对应的源级意图可以写成：

```cpp
struct Inner { void *p; _Float16 h; };
struct MyType { int x; Inner inner; struct { int a; bool b; bool c; } last; };

auto *addr_half_field = &dst[0].inner.h;
```

但 LLVM IR 不承诺 C++ 源码中字段名、`bool` 的具体宽度或原始 typedef 一定仍存在；唯一可靠的依据是 GEP 前的 LLVM 类型和 module 的 data layout。

读任意 GEP 时，按以下顺序展开即可：

1. 找到第一个类型参数，画出它的 array/struct 嵌套形状。
2. 将第一个索引解释为“跳过多少个完整根对象”。
3. 之后每遇到 struct 索引，就按从 0 开始的字段号选择成员；每遇到 array/vector 索引，就按元素大小乘索引。
4. 查 data layout 处理 size、alignment 和 padding；不要直接用源码的 `sizeof` 猜测。
5. 最后检查是否存在 `inbounds`，并把它视为可能产生 poison 的额外范围保证。

## 静态数组类型、实际分配大小与 `[0 x T]`

书中指出：array type 的元素数量虽然写死在类型中，但 LLVM 不会限制对该数组使用的索引；请求静态范围外的地址在语法上仍然合法。问题只在实际分配对象不够大、而程序真的访问该地址时出现。由此，`[0 x type]` 可以用来表达运行时大小的数组。

### 类型中的长度不等于一次地址计算的边界检查

```llvm
%array4 = type [4 x i32]

; 语法合法：只计算第 10 个 i32 的地址
%p10 = getelementptr [4 x i32], ptr %base, i64 0, i64 10
```

`[4 x i32]` 的类型布局描述了四个连续 `i32`。但上面的 GEP 依然合法，因为 GEP 只计算 `%base + 10 * sizeof(i32)`，并不读取或写入该位置，也不插入运行时数组边界检查。

接下来若执行访问，实际分配大小才决定程序是否合法：

```llvm
%value = load i32, ptr %p10
```

若 `%base` 指向的实际对象确实只有四个 `i32`，此读取越界，不能作为合法程序执行。反过来，若 `%base` 来自一个更大的分配对象，而 `[4 x i32]` 只是用来描述其中的起始布局，静态类型中的 `4` 本身不足以判断这次访问是否越界。

这说明 LLVM 中有两件不同的事：

| 层次 | 回答的问题 |
| --- | --- |
| array type，例如 `[4 x i32]` | 从这个类型视角看，元素如何连续排列、每一步索引应跨过多少字节。 |
| 实际 allocated object | 当前指针实际指向多少可访问存储；它决定某次 `load`/`store` 是否有效。 |

### `[0 x T]` 表示运行时决定的尾部数组

`[0 x T]` 不表示“自动分配了一个可增长数组”，也不表示数组永远为空。它表示：类型的固定部分到此结束，后面是否有多少个 `T` 元素由实际分配大小决定。

它很适合表达 C 的 flexible array member：

```c
struct Packet {
  int length;
  char data[];
};
```

对应的 LLVM 类型可近似写为：

```llvm
%Packet = type { i32, [0 x i8] }
```

运行时通常会为 header 和数据一起分配足够空间：

```text
allocation size = sizeof(Packet header) + length * sizeof(i8)
```

随后可通过 GEP 计算动态尾部中第 `%i` 个元素的地址：

```llvm
; 概念上对应 &packet->data[i]
%data_i = getelementptr inbounds %Packet, ptr %packet,
                              i64 0, i32 1, i64 %i
```

索引含义是：

1. `i64 0` 选当前 `%Packet` 对象；
2. `i32 1` 选其第 2 个字段 `[0 x i8]`；
3. `i64 %i` 在该运行时尾部数组中选择第 `%i` 个 `i8`。

LLVM 类型不会记录 `length` 与 `%i` 的关系。前端、运行时或优化器必须根据分配大小、控制流条件和已有的范围信息保证实际访问不越界。

### 与 `inbounds` 的关系

不带 `inbounds` 的 GEP 可以形成对象范围以外的地址；GEP 本身不会解引用它。带 `inbounds` 的 GEP 则承诺：每一步地址计算和最终结果都在同一个实际分配对象内，或位于该对象末尾。违反承诺时，GEP 的结果为 poison。

因此，上一例中的 `inbounds` 是在说：执行到这里时，`%packet` 的真实分配空间足以覆盖所求的 `%data_i` 地址。它不是运行时检查；若这个事实无法保证，生成 IR 时应省略 `inbounds` 或先在控制流中完成适当的范围检查。

如果真的写出：

```llvm
%empty = alloca [0 x i32]
```

那么该 `alloca` 本身只分配零个数组元素。`[0 x i32]` 的动态尾部含义只有在指针实际来自更大的对象，例如 `malloc(header_size + count * sizeof(i32))`、调用方提供的变长缓冲区或某个包含该字段的扩展对象时才成立。

### 对 x86 与 AMDGPU 的结论

数组索引、GEP、`[0 x T]` 和 `inbounds` 的上述语义属于目标无关的 LLVM IR 规则，在 x86 和 AMDGPU 上相同。差别只会出现在 data layout 决定的元素大小、字段 padding、pointer index width，以及最终使用哪条 host 或 GPU 内存访问指令。

在 HIP/ROCm 里，`[0 x T]` 仍然只是动态布局的描述，不会自行决定数据位于 global、LDS/local 或 private memory。该存储区域取决于指针的 address space 和实际分配方式；例如一个 `ptr addrspace(3)` 指向的对象可有动态尾部，但它仍是 work-group 的 LDS 对象。

## 例子：向嵌套数组整体写入零初始化值

书中的示例是：

```llvm
define void @useOfArrayType(ptr %dst) {
  store [12 x [36 x i32]] zeroinitializer, ptr %dst
  ret void
}
```

它的作用是：**将 `%dst` 指向的内存视为一个 12 × 36 的 `i32` 数组，并把整个数组写成零。**

### 先读数组类型

```llvm
[12 x [36 x i32]]
```

从最内层向外读：

```text
i32             一个 32-bit 整数
[36 x i32]      36 个连续 i32：一行
[12 x [36 x i32]]
                12 行，每行 36 个 i32：一个二维数组
```

它和 C/C++ 中下面的对象布局相对应：

```cpp
int array[12][36];
```

共有 `12 * 36 = 432` 个 `i32`。在常见的 x86-64 和 AMDGPU data layout 中，`i32` 为 4 bytes，数组元素没有额外字段 padding，因此这次 store 覆盖 `432 * 4 = 1728` bytes 的连续存储。

### `zeroinitializer` 是整个聚合值的零值

`zeroinitializer` 是 LLVM 的常量，表示“这个类型的递归零初始化值”。在本例中，它等价于概念上的：

```text
{
  {0, 0, ..., 0},  // 共 36 个 i32
  ...              // 共 12 行
}
```

对于这个纯 `i32` 数组，它就是 432 个整数零。对于更复杂的 aggregate，`zeroinitializer` 会递归地产生各成员的零值，例如整数为 `0`、浮点为 `0.0`、指针为 `null`、嵌套数组和结构体继续递归初始化。

### 为什么 `%dst` 只有 `ptr`

LLVM 使用 opaque pointer 后，`%dst` 的类型只有 `ptr`，不再写成旧式的 `[12 x [36 x i32]]*`。这不代表 store 不知道布局：

```llvm
store [12 x [36 x i32]] zeroinitializer, ptr %dst
      ^^^^^^^^^^^^^^^^^
      写入值的类型提供大小、元素布局和对齐语义
```

`store` 左侧值的类型告诉 LLVM：应当从 `%dst` 开始写入一个完整的 `[12 x [36 x i32]]`。因此，调用者必须保证 `%dst` 指向至少 1728 bytes 的有效、可写内存，并满足该 store 所需的对齐条件。

此函数没有自行分配数组；`%dst` 是由调用者传进来的地址。若传入的是不足大小的对象，或指向不可写内存，后续 store 的行为不合法。

### 这不是 432 条显式 `store`

IR 中只有一条 aggregate `store`，其语义是一次写入整个复合值。它并不规定最终机器码必须真的是一条硬件指令，也不规定必须展开成 432 条标量 store。

后续优化和后端可按 target、对齐、地址空间、对象是否可见等条件将其降成不同形式：

- x86 上可能成为对齐的 SIMD store 序列、普通标量 store 序列，或等价的内存清零操作；
- AMDGPU 上若 `%dst` 指向 global memory，可能使用 global-memory store；若指向 LDS，则使用 local/LDS store；
- 若值未被观察到，优化器也可能删除整次初始化。

因此读 LLVM IR 时，应先把它理解为“写入 1728 bytes 的全零 aggregate”，不要过早把它等同于某一种循环或某条特定汇编指令。

### 如果改为逐元素访问

若要访问第 `row` 行、第 `col` 列，GEP 需要逐层穿过两个数组：

```llvm
; 概念上对应 &array[row][col]
%element = getelementptr [12 x [36 x i32]], ptr %dst,
                          i64 0, i64 %row, i64 %col
```

第一个 `i64 0` 仍表示“从当前完整二维数组对象开始”；`%row` 在外层 12 元素数组中选择一行，`%col` 在该行的 36 元素数组中选择一个 `i32`。若 GEP 带 `inbounds`，则执行路径必须保证 `0 <= row < 12` 和 `0 <= col < 36`，以及 `%dst` 指向完整的实际对象。

### 对 HIP/ROCm 的补充

示例中的 `ptr %dst` 默认是 `addrspace(0)`。若该 store 是 GPU device IR 中写入某个明确的地址空间，会写作例如：

```llvm
store [12 x [36 x i32]] zeroinitializer, ptr addrspace(1) %dst ; global
```

或使用 `ptr addrspace(3)` 表示 LDS/local memory。数组的 12 × 36 布局与 `zeroinitializer` 的含义不变；改变的是 `%dst` 指向哪类 GPU 内存、有效分配范围以及后端选择的实际访存指令。

## 阅读 `@foo`：SSA 编号、隐式 basic block 与控制流

书中 “Walking through an example” 使用的函数经过排版后可读作：

```llvm
define i32 @foo(i32, i32, i32 %arg) {
entry:
  %myid = add i32 %0, %1
  %31 = mul i32 %myid, 2
  %45 = shl i32 %31, 5
  %"00~random~00" = udiv i32 %45, %arg
  br label %46

  br label %47

47:
  ret i32 %"00~random~00"
}
```

这个例子最重要的目的不是算法，而是把 function、argument、basic block、SSA 值、指令、类型、terminator 和命名规则放进同一段 IR 中，并展示隐式编号怎样让手读 IR 变困难。

### 它计算什么

前三个参数都是 `i32`。前两个没有显式名字，所以 LLVM 将其编号为 `%0` 和 `%1`；第三个显式命名为 `%arg`。

算术数据流为：

```text
%myid              = %0 + %1
%31                = %myid * 2
%45                = %31 << 5
%"00~random~00"   = unsigned_divide(%45, %arg)
```

概念上接近：

```cpp
uint32_t foo(uint32_t a, uint32_t b, uint32_t arg) {
  return (((a + b) * 2) << 5) / arg;
}
```

`udiv` 是无符号除法；若 `%arg` 为零，执行该 `udiv` 是 UB。`add`、`mul` 和 `shl` 没有 `nsw` 或 `nuw` 标记，因此不能从它们推导“不发生有符号/无符号回绕”的额外保证。

### 三种 SSA 值命名方式

```llvm
%myid                ; 普通、可读的名称
%31                  ; 数字形式的未命名临时值
%"00~random~00"     ; 可使用自由字符的带引号名称
```

数字名称并不表示值有特别的运行时含义，只是 LLVM 的 value 编号。它们和 basic block 的隐式编号、以及未命名函数参数共享同一个函数内计数序列。

### `%46` 到底在哪里

最容易困惑的部分是：

```llvm
  br label %46

  br label %47
```

`br label %46` 引用一个 basic block。该 block 在文本中没有显式写出 `46:` 标签；它从下一条 `br label %47` 开始。因为这个 block 的标签被省略，LLVM 自动使用下一个可用的未命名编号，恰好是 `46`。

换成把标签全部写出的等价形式，就是：

```llvm
entry:
  ; ... 计算 %"00~random~00"
  br label %46

46:
  br label %47

47:
  ret i32 %"00~random~00"
```

控制流图为：

```text
entry  -->  %46  -->  %47  -->  ret
```

`%46` 没有 UB、异常或特殊控制流含义；它只是一个只包含无条件跳转的中转 block。`%47` 也不是因编号而“正常”，只是恰好包含 `ret`。真正可能发生 UB 的地方是前面的 `udiv` 除数为零。

LLVM parser 允许先引用一个稍后才出现的 basic block，因此 `br label %46` 可以先出现；解析到前一 block 的 terminator 后，下一条指令自然开始新的 basic block。由于没有写标签，parser 使用自动编号 `46`，并将此前的前向引用解析到它。

### 为什么作者认为这种写法难读

若想知道 `%46` 是谁，必须同时追踪：

- 未命名参数 `%0`、`%1`；
- 数字形式的 instruction result，例如 `%31`、`%45`；
- 未显式标出的 basic block；
- 可能存在的前向 branch 引用。

稍微插入、删除或移动一条未命名指令，都可能改变后续自动编号。手写或手改 `.ll` 时，应优先显式命名参数、关键 SSA 值和 basic block。LLVM 的 `instnamer` pass 也可为未命名 instruction result 生成可读名称；这会显著降低人工阅读和修改 CFG 时出错的概率。

### 与 x86 和 AMDGPU 的关系

本例中的 function、SSA、隐式 block、`br`、`ret`、`udiv` 和 UB 语义都是 target-independent 的 LLVM IR 规则。无论随后选择 x86 还是 AMDGPU 后端，`%46` 都是同一个中转 basic block。

target triple、data layout、ABI、function attribute 和指令选择才会使同一类 IR 在 x86 与 HIP/ROCm AMDGPU 上生成不同机器码。此例没有出现这些 target-specific 信息，因此它适合先学习 LLVM IR 的通用控制流与命名规则。

## 整数运算的 `nsw` 与 `nuw`：把“不回绕”写进 IR

书中说：如果需要表达自己希望保留的溢出/下溢语义，就必须使用 No Signed Wrap（`nsw`）或 No Unsigned Wrap（`nuw`）标记。这里的“capture the semantics”并不是要求后端插入溢出检查，而是把前端已知的、不回绕前提写进 LLVM IR，供优化器和后端使用。

### 不带标记时：按 N-bit bit pattern 模运算

对普通整数 `add`、`sub`、`mul` 等指令，若没有 `nsw`/`nuw`，LLVM 只关心 N 个结果 bit：

```llvm
%r = add i32 %a, %b
```

该结果按模 \(2^{32}\) 计算。也就是说，`0xffffffff + 1` 的结果 bit pattern 是 `0`。LLVM 不从这条指令本身推断 `%a + %b` 在有符号或无符号意义下没有溢出。

同一组 bit 可按 signed 或 unsigned 解读：

```text
i32 bit pattern 0xffffffff
signed interpretation:    -1
unsigned interpretation:  4294967295
```

因此 `add i32` 没有“默认 signed”或“默认 unsigned”的算术语义；签名主要在 `sdiv`/`udiv`、`icmp slt`/`icmp ult`、`sext`/`zext` 等需要解释大小关系或扩展规则的指令中才显式出现。

### `nsw`：承诺有符号结果可表示

```llvm
%r = add nsw i32 %a, %b
```

`nsw` 表示这次加法按有符号二补码解释时不会回绕。换言之，数学结果必须处于 `[-2^31, 2^31 - 1]`。例如：

```text
2147483647 + 1
```

若在带 `nsw` 的 `i32 add` 中发生，`%r` 不是环绕为 `-2147483648`，而是 **poison**。

类似地，对 `sub nsw`，`INT_MIN - 1` 会产生 poison；对允许该标记的乘法或左移，标记表达相应的有符号不回绕保证。

### `nuw`：承诺无符号结果不回绕

```llvm
%r = add nuw i32 %a, %b
```

`nuw` 表示按无符号解释时结果必须仍处于 `[0, 2^32 - 1]`。例如：

```text
4294967295 + 1
```

在带 `nuw` 的 `i32 add` 中会得到 poison，而不是环绕到 `0`。

对于 `sub nuw`，它等价于承诺 `%a >= %b` 的无符号前提；若 `%a < %b`，无符号下溢发生，结果为 poison。

两个标记可同时出现：

```llvm
%r = add nuw nsw i32 %a, %b
```

这表示有符号和无符号两种解释下都不能发生回绕。只要任一承诺被违反，结果就是 poison。

### 为什么这是优化信息

考虑：

```llvm
%next = add nsw i32 %i, 1
%is_increasing = icmp sgt i32 %next, %i
```

因为 `nsw` 排除了 `%i == INT_MAX` 时的有符号回绕，优化器可推导 `%next > %i` 恒为真，并将 `%is_increasing` 折叠为 `true`。

若没有 `nsw`，当 `%i == INT_MAX` 时结果会按 bit pattern 环绕成 `INT_MIN`，比较为假；该优化不再合法。

这也解释了为什么标记必须真实可靠：错误地加上 `nsw`/`nuw` 会把原本可能发生回绕的执行路径变成 poison，之后一旦 poison 到达分支、内存地址等敏感位置，程序就出现 UB。

### 它不等于硬件溢出检查

```llvm
%r = add nsw i32 %a, %b
```

通常仍可在 x86 或 AMDGPU 上降低成普通的一条整数加法指令；`nsw` 不要求生成“检查 overflow 后报错”的指令序列。它的作用主要是约束 IR 语义、允许优化器据此变换程序。

若源程序需要在运行时检测溢出，前端或程序必须显式产生检查，例如使用 LLVM 的 `llvm.sadd.with.overflow` 一类 intrinsic，或比较计算结果并分支；仅加 `nsw` 并不会保留一个可观察的 overflow flag。

### 如何从 C/C++ 与 HIP 代码理解

在典型的 C/C++ 语义中，有符号溢出是 UB，因此 Clang 在符合条件时能够用 `nsw` 把这一前提带入 IR。无符号整数回绕则是 C/C++ 明确定义的行为，所以不能仅因源类型是 `unsigned` 就随意加 `nuw`；只有额外的范围证明表明不会回绕时才可以加。

HIP device code同样先被编译为 LLVM IR，因此这套语义在 host x86 和 device AMDGPU 上相同。两者的后端可能使用不同的机器加法指令、寄存器和调度方式，但都必须遵守：普通 `add` 可回绕，`add nsw`/`add nuw` 违规产生 poison。

### 读 IR 时的规则

1. 先看整数指令是否带 `nsw`、`nuw`，不要从变量名或源码类型猜测。
2. 无标记时，按 N-bit 模运算理解结果；不要擅自假设“不会 overflow”。
3. 带标记时，将违反的路径视为产生 poison，并继续检查 poison 是否被 `freeze` 截断或流入敏感操作。
4. `nsw`/`nuw` 只可用于 LangRef 允许的指令和位置；最常见的是 `add`、`sub`、`mul`、`shl`，以及具有专门不回绕语义的 GEP 标记。

## 例子：`llvm.vector.reduce.add` 对一个向量做水平求和

书中用下面两条 IR 展示 generic intrinsic 如何保留“水平求和”的意图：

```llvm
%vec = load <4 x i32>, ptr %arg
%hadd = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %vec)
```

### 第一条：从内存载入一个四 lane 向量

```llvm
%vec = load <4 x i32>, ptr %arg
```

`<4 x i32>` 是四个 `i32` lane 组成的 LLVM vector 值。该 load 从 `%arg` 开始读取四个连续的 32-bit 整数，概念上得到：

```text
内存地址:  %arg        %arg + 4    %arg + 8    %arg + 12
向量 lane: lane 0       lane 1      lane 2      lane 3
```

若内存内容依次为 `10, 20, 30, 40`，则：

```llvm
%vec = <i32 10, i32 20, i32 30, i32 40>
```

这里的 `ptr %arg` 是 opaque pointer；向量 load 的类型 `<4 x i32>` 才告诉 LLVM 要读取的元素类型、总大小和 vector 布局。调用者必须保证 `%arg` 指向至少 16 bytes 的可读连续内存，并满足这次 load 的对齐要求。

### 第二条：把四个 lane 归约成一个标量

```llvm
%hadd = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %vec)
```

`llvm.vector.reduce.add.v4i32` 是 LLVM 提供的 **通用 vector reduction intrinsic**：

```text
v4i32   = 输入是 4 个 i32 lane 的向量
reduce  = 将多个 lane 缩减为一个值
add     = 用整数加法组合 lane
返回值  = 一个 i32 标量
```

因此其语义是：

```text
%hadd = lane0 + lane1 + lane2 + lane3
```

对上面的 `10, 20, 30, 40`，结果为：

```llvm
%hadd = i32 100
```

这称为 horizontal reduction（横向归约）：数据流在**同一个 vector 值的 lane 之间**横向聚合。它不是把不同 CPU 线程相加，也不是把 AMDGPU wavefront 中不同 work-item 的值相加。

在没有额外 `nsw`/`nuw` 语义的这个整数 reduction 中，加法按 `i32` 的模 \(2^{32}\) bit pattern 语义理解；它没有隐含的 signed overflow 检查。

### 为什么不直接写四次 `extractelement` 和 `add`

等价的普通 IR 大致可以写成：

```llvm
%v0 = extractelement <4 x i32> %vec, i32 0
%v1 = extractelement <4 x i32> %vec, i32 1
%v2 = extractelement <4 x i32> %vec, i32 2
%v3 = extractelement <4 x i32> %vec, i32 3
%s01 = add i32 %v0, %v1
%s23 = add i32 %v2, %v3
%hadd = add i32 %s01, %s23
```

这种展开形式表达的是“若干 extract 与 scalar add”，而不是“这是一个向量水平求和”。中间优化可能改写、打散或混合这些指令，导致后端难以可靠地重新识别原始意图。

intrinsic 把意图直接保留在 IR 中。它的名称以 `llvm.` 开头，但没有 `x86.` 或 `amdgcn.` 前缀，所以它是 generic intrinsic，不绑定某个后端。x86、AMDGPU 和其他支持该 intrinsic 的后端都可各自决定最合适的 lowering。

### `tail call` 不改变求和含义

`tail` 修饰的是 **call**，不是 `%hadd` 的类型，也不是“向量末尾的元素”。它是一个 tail-call optimization（尾调用优化）提示：若调用确实位于函数的尾部、ABI 和目标条件允许，后端可以让被调用者直接返回到当前函数的调用者。`tail` 本身不自动插入 `ret`，也不保证最终一定采用这种 lowering；LLVM LangRef 明确规定它是可被忽略的 hint。

普通调用的控制流和调用栈可理解为：

```text
caller  -- call f -->  f  -- call g -->  g
caller  <-- ret f  --   f  <-- ret g  --
```

若 `f` 对 `g` 的调用是尾调用，`f` 不需要在 `g` 返回后恢复并继续执行。后端可以把它变成：

```text
caller  -- call f -->  f  -- jump g -->  g
caller  <--------------------- ret g --
```

也就是说，`g` 直接返回到 `caller`，`f` 的调用帧可被复用或移除。x86 上常见的机器码差别是将 `call g; ret` 优化为跳转到 `g`；这样可减少一次调用帧和返回操作。

一个典型的普通函数尾调用例子是：

```llvm
define i32 @f(i32 %x) {
  %r = tail call i32 @g(i32 %x)
  ret i32 %r
}
```

`@f` 在调用 `@g` 后只做 `ret %r`，所以具备尾调用的形状。是否最终真的采用 tail-call lowering，仍取决于目标 ABI、调用约定、参数/返回值如何传递、栈布局以及后端是否支持；`tail` 表示允许或请求这种优化，并不是无条件保证。即使未被降低为跳转，带 `tail` 的 call 仍要求被调用者不去访问调用者的 `alloca`、`va_arg` 或普通 `byval` 存储，因为尾调用实现可能已经释放或复用调用者的栈帧。

LLVM 还有两个相关标记：

| 标记 | 含义 |
| --- | --- |
| 无标记 `call` | 普通调用；后端可自行决定优化。 |
| `tail call` | 调用点具有尾调用语义/资格，后端在条件允许时可进行 tail-call lowering。 |
| `musttail call` | 强制尾调用；IR 必须满足更严格的 ABI 和签名规则，不能正常实现尾调用时就是无效 IR。 |
| `notail call` | 禁止将该调用变成尾调用。 |

回到书中的：

```llvm
%hadd = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %vec)
```

`tail` 不改变“输入四个 `i32`，输出它们的和”这一 intrinsic 语义。更重要的是，`llvm.vector.reduce.add` 是一个 LLVM intrinsic，通常在中后端被识别并直接展开或选择为目标操作，并不真的生成对某个普通函数 `@llvm.vector.reduce.add.v4i32` 的运行时调用。因此在这个示例中，`tail` 是附带的 call 标记；理解 vector reduction 时可先忽略它。

LLVM intrinsic 在中后端常会直接被识别、展开或选择为目标操作，不应将这条文字上的 `call` 简单理解成必然发生一次普通函数调用。

### x86 与 AMDGPU 的实际关联

这两条 IR 的语义在 x86 和 AMDGPU 上相同，但 lowering 可以不同：

- 在 x86 上，`%vec` 很可能映射到 SIMD 寄存器；后端可使用 shuffle/add 序列，或在可用 ISA 特性下使用合适的水平加法指令。
- 在 AMDGPU 上，`%vec` 表示**单个 work-item 内**的四 lane LLVM vector 值，常会映射到多个 VGPR 分量或被标量化；归约仍只在这个 work-item 的四个值之间进行。
- 若需求是将一个 wavefront 或整个 block/work-group 的线程值相加，需要 AMDGPU 的 cross-lane 操作、wave intrinsic、LDS 加同步等机制；`llvm.vector.reduce.add` 不完成这类跨线程归约。

因此，看到 `llvm.vector.reduce.add` 时，应先问“它在归约一个 LLVM vector 的 lane 吗？”；只有答案为是时，这个 intrinsic 才与问题匹配。不要把 IR vector 的 lane 与 GPU 的线程 lane 混为一谈。

## `target datalayout = "i32:8"`：类型大小不等于对齐要求

书中先给出一个没有 data layout 的 load：

```llvm
define i32 @foo(ptr %src) {
  %res = load i32, ptr %src
  ret i32 %res
}
```

再加入：

```llvm
target datalayout = "i32:8"
```

并观察到经过 LLVM 处理后，load 会写成：

```llvm
%res = load i32, ptr %src, align 1
```

这里最重要的是：`i32:8` **没有把 `i32` 变成 8 bit，也没有改变 `i32` 的大小。** `i32` 始终是 32 bit，即 4 bytes。`8` 描述的是该 target 对 `i32` 的 ABI alignment，单位也是 bit：8 bit = 1 byte。

### 大小与对齐是两条不同的规则

| 属性 | 对 `i32` 的含义 | 本例中的值 |
| --- | --- | --- |
| bit width / store size | 存储一个值本身需要多少 bit/byte。 | 32 bit = 4 bytes |
| alignment | 地址必须是多大的倍数，编译器才可假定按该边界对齐。 | `i32:8` 时为 8 bit = 1 byte |

因此下列两种说法可同时成立：

```text
i32 的数据本体有 4 bytes。
i32 可以从任意 1-byte 对齐的地址开始存放或读取。
```

例如地址 `0x1001` 不是 4 的倍数，但它是 1 的倍数。`align 1` 表示 `%src` 最多只能保证这种 1-byte 对齐；LLVM 不能假定它有 4-byte 或 16-byte 对齐。

data layout 字段的一般形式是：

```text
i<size>:<abi>[:<preferred>]
```

所以：

```text
i32:8      i32 的 ABI 对齐是 8 bit = 1 byte
i32:32     i32 的 ABI 对齐是 32 bit = 4 bytes
i32:128    i32 的 ABI 对齐是 128 bit = 16 bytes
```

书中前一个 `i32:128` 示例因此得到 `align 16`；本例 `i32:8` 则得到 `align 1`。这些只改变对齐约束，不改变 `i32` 的 4-byte 元素大小和 GEP 中 `i32` 元素的步长。

### `align 1` 不是“只读一个字节”

```llvm
%res = load i32, ptr %src, align 1
```

仍然读取完整的 4-byte `i32`。`align 1` 只是在说起始地址的对齐保证很弱。

在某些 CPU 上，未对齐的 4-byte load 仍可由一条普通机器指令完成，只是可能较慢或跨越缓存线；在另一些 target 或某些 GPU memory access 形式上，硬件/ABI 对齐限制更严格，后端可能要将一次 `i32` load 合法化为多个较小的访问。LLVM 必须尊重 `align 1`，即使运行时某些地址恰好更对齐，也不能在未证明时依赖它。

### 书中说的 “alignment requirements have spread all over the IR”

这一段是在警告跨 target 复用 `.ll` 的一个陷阱。

data layout 最初可以为未显式标注对齐的内存操作补出默认值。例如原始 IR：

```llvm
%res = load i32, ptr %src
```

在 `i32:8` 的 data layout 下处理后，可能被固化为：

```llvm
%res = load i32, ptr %src, align 1
```

这时 `align 1` 已经写在每一条相关 `load`、`store`、`alloca` 或 global object 上，成为 IR 的显式事实。之后即使删除、替换或“丢掉” module 顶部的：

```llvm
target datalayout = "i32:8"
```

然后交给另一个 target，LLVM 也不会回头自动把已写出的 `align 1` 改成该 target 的自然对齐。data layout 是默认规则；指令上的 `align` 是已经具体化的承诺。

这就是书中 “this data layout may have spread the alignment requirements all over the IR” 的意思：**旧 target 的布局决定已经渗入许多 IR 指令和对象，单改一行 data layout 字符串无法撤销它们。**

### 为什么跨 target 会造成更差 codegen

假设源 IR 在 `i32:8` 下已得到：

```llvm
%res = load i32, ptr %src, align 1
```

然后把 module 改交给一个希望至少 4-byte 对齐才能高效或合法地完成 `i32` load 的 target。后端看到的仍是 `align 1`，只能按“地址可能是 `0x1001`”来生成安全代码。它不能擅自把这次访问提升为 `align 4`。

可能的后果是：

- 不能使用只适用于自然对齐地址的访存形式；
- 需要生成多次更小的 byte/halfword 访问，或额外的拼接代码；
- 失去向量化、合并访存或对齐 load 的机会；
- 在 AMDGPU 上，global、LDS 或 private memory 的特定 access legalization 可能更保守。

反方向也危险：若旧 IR 显式写了过强的 `align 16`，但新的 target 或真实分配只保证 4-byte 对齐，则保留 `align 16` 会成为错误的 IR 承诺，可能导致错误优化或不正确访问。

### 对 x86 与 HIP/ROCm 的实践结论

x86-64 与 AMDGPU 的正常 Clang/ROCm 编译不会把普通 `i32` 的 data layout 设成这个教学用的 `i32:8`；该字符串是用来放大展示 alignment 的作用。真实模块应保留由相应 frontend/target 生成的 `target triple` 和 `target datalayout`。

若确实需要将 IR 从 x86 迁移到 AMDGPU，或反向迁移，不要只替换 `target triple` 和 data layout 字符串。还要审查并重新生成或合法化已经存在的：

- `load` / `store` 的 `align`；
- `alloca` 和 global 的对齐与 address space；
- pointer width / index width 假设；
- ABI 已经反映到函数签名和属性中的决定。

最可靠的做法是从源代码或更高层、尚未携带旧 target ABI/layout 决定的 IR 重新针对目标编译。

## `0x04030201` 与 little-endian：x86 和 AMDGPU 的字节顺序

书中用 32-bit 十六进制值 `0x04030201` 说明端序。这里要先区分两件事：

- `0x04030201` 是**一个整数值的文字写法**；从左到右的 `04 03 02 01` 是从最高有效字节（MSB）到最低有效字节（LSB）的通常书写顺序。
- endian 描述的是该数值**存入按 byte 编址的内存后**，这些字节按地址从低到高怎样排列。

若 `%p` 指向一块可存放 `i32` 的内存，并执行：

```llvm
store i32 0x04030201, ptr %p
```

在 little-endian target 上，内存是：

```text
地址             %p + 0    %p + 1    %p + 2    %p + 3
地址方向         最低地址                              最高地址
存放的 byte      0x01      0x02      0x03      0x04
字节意义         最低有效 byte                         最高有效 byte
```

所以“`01` 是低地址，`04` 是高地址”这个直觉是对的，但更准确地说：`0x01` 和 `0x04` 是 **byte 值**，不是地址本身。真实地址可能是 `0x7fff...`、GPU VA 中的某个地址或 LDS offset；这里只讨论相对于 `%p` 的偏移。

从寄存器中书写或阅读数值时，仍写作：

```text
i32 value = 0x04030201
```

它的 bit pattern 从最高有效端到最低有效端可写为：

```text
00000100  00000011  00000010  00000001
   0x04      0x03      0x02      0x01
```

little-endian 只是在 store 时先把最低有效的 `0x01` 放到最低地址；随后同一 target 上的 `load i32, ptr %p` 会自动按相同规则重新组合，仍得到数值 `0x04030201`。因此普通的 `i32` load/store 不需要程序员手动 byte-swap。

### 对 x86-64 与 HIP/ROCm AMDGPU 的结论

x86-64 是 little-endian。ROCm 使用的 AMDGPU/AMDGPU（amdgcn）目标也是 little-endian。两者的 LLVM module data layout 通常以：

```llvm
target datalayout = "e-..."
```

开头，其中 `e` 表示 little-endian。因此对我们关心的两个目标，以上表格的内存字节顺序相同：

```text
store i32 0x04030201, ptr %p

低地址 --------------------------------------------------------> 高地址
                 01        02        03        04
```

在 x86 上，类似 `mov eax, dword ptr [p]` 的 32-bit load 会把这四个 bytes 解释为 `EAX = 0x04030201`。在 AMDGPU 上，对 global、flat、LDS/local 或 private memory 的 dword load 也遵从该 target 的 little-endian 内存表示；区别是地址空间和选用的访存指令，不是这四个 byte 的顺序。

### 什么时候端序会真的显露出来

若程序只用同一 target 上的 `i32` store 和 `i32` load，端序通常不可见。它会在以下情况变得重要：

- 把 `i32` 当作四个 `i8` 分别读取、写入或通过网络/文件传输；
- 按字节构造整数，例如 `input[i] << (8 * i)`；
- 用 `bitcast` 在 vector、整数和 byte 序列之间重解释；
- 将二进制数据在 little-endian 与 big-endian target 之间共享。

例如在 little-endian x86 或 AMDGPU 上：

```llvm
%first_byte = load i8, ptr %p
```

读取的是 `0x01`，因为它位于 `%p + 0`。若想读取最高有效 byte `0x04`，则需要访问 `%p + 3`，或先加载 `i32` 再右移 24 bit。

## 例子：从 byte 序列构造 little-endian 与 big-endian 整数

书中接着使用两个 C/C++ 函数说明：当程序手动把 byte 序列拼成整数时，端序假设已经成为源程序逻辑，随后会直接进入 LLVM IR 的 `shl`、`or` 等数据流。

```cpp
int buildIntLittleEndian(const char *input) {
  int res = 0;
  for (int i = 0; i < sizeof(res); ++i)
    res |= input[i] << (8 * i);
  return res;
}

int buildIntBigEndian(const char *input) {
  int res = 0;
  for (int i = 0; i < sizeof(res); ++i) {
    res <<= 8;
    res |= input[i];
  }
  return res;
}
```

两者都从 `input` 读取四个 bytes，并产生一个 32-bit 整数；区别是 `input[0]` 被当作最低有效 byte 还是最高有效 byte。

### 用同一组输入比较

假设外部 byte 序列为：

```text
input[0] = 0x01
input[1] = 0x02
input[2] = 0x03
input[3] = 0x04
```

#### `buildIntLittleEndian`

它将 `input[0]` 放到最低 8 bit，之后每个 byte 向更高有效位移动：

```text
res = (0x01 <<  0)
    | (0x02 <<  8)
    | (0x03 << 16)
    | (0x04 << 24)

    = 0x04030201
```

```text
输入 byte:    01        02        03        04
整数位段:  [31:24]   [23:16]   [15:8]    [7:0]
结果数值:                 0x04030201
```

这与 x86-64 和 ROCm AMDGPU 的 native little-endian 内存表示一致：若将数值 `0x04030201` 存入内存，低地址到高地址也正是 `01 02 03 04`。

#### `buildIntBigEndian`

它每次先把已有结果左移 8 bit，再把新读入的 byte 放进最低 8 bit：

```text
初始：0x00000000

读 0x01： (0x00000000 << 8) | 0x01 = 0x00000001
读 0x02： (0x00000001 << 8) | 0x02 = 0x00000102
读 0x03： (0x00000102 << 8) | 0x03 = 0x00010203
读 0x04： (0x00010203 << 8) | 0x04 = 0x01020304
```

所以同一组输入在这个函数中被解释为：

```text
input bytes: 01 02 03 04
结果数值:     0x01020304
```

这里 `input[0] = 0x01` 被当作最高有效 byte，符合 big-endian 数据格式的约定。

### 这不是在切换 CPU/GPU 的端序

即使在 little-endian 的 x86 或 AMDGPU 上，`buildIntBigEndian` 也会正确返回数值 `0x01020304`。只是若这个结果随后在同一 little-endian target 上存为 `i32`，其内存 bytes 会是：

```text
低地址 -------------------------------------------------> 高地址
04                     03        02        01
```

因此，这两个函数是在把**外部 byte 序列的格式**转换成程序中的整数数值，并没有改变机器的 native endianness。它们常用于读取网络协议、文件格式或设备寄存器数据。

这也正是书中提到“端序会泄漏进 IR”的原因：前端编译这两个函数后，得到的是不同的 load、shift、OR 数据流。即使之后只修改 module 的 `target datalayout`，也不会自动把 little-endian 拼装算法改成 big-endian 拼装算法；这种语义已经由指令本身表达出来。

### 实际 C/C++ 代码的 `char` 注意事项

书中代码用于说明端序，但实际代码宜使用 `uint8_t` 或 `unsigned char`，并在移位前转换到无符号 32-bit 类型。因为某些平台的 `char` 是 signed，`input[i] >= 0x80` 时可能先发生符号扩展，得到错误结果。

```cpp
uint32_t buildIntLittleEndian(const uint8_t *input) {
  return uint32_t(input[0])
       | (uint32_t(input[1]) << 8)
       | (uint32_t(input[2]) << 16)
       | (uint32_t(input[3]) << 24);
}
```
