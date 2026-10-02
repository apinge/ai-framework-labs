# 第 8 章：现有 Pass 概览

本章开始讨论 LLVM middle end 已有的 LLVM-IR-to-LLVM-IR transformation（通常称为 pass）。书的核心建议是：在为 x86、AMDGPU 或其他 target 写后端前，先确认 middle end 是否已有可复用的通用优化；例如无需为每个后端重新实现 `mem2reg`、常量折叠或公共子表达式消除。

## `opt --passes=gvn`：使用新 Pass Manager 运行 GVN

书中的命令是：

```bash
${LLVM_INSTALL_DIR}/bin/opt --passes=gvn input.bc -S -o -
```

它的意思是：让 LLVM 的 `opt` 工具读取 `input.bc`，只运行一次名为 `gvn` 的 middle-end pass，然后将转换后的 LLVM IR 以文本形式输出到终端。

各部分含义如下：

| 片段 | 含义 |
| --- | --- |
| `${LLVM_INSTALL_DIR}/bin/opt` | 已安装或构建出的 LLVM `opt` 可执行文件。 |
| `--passes=gvn` | 使用 **new Pass Manager** 的 pipeline 语法，选择 `gvn` pass。 |
| `input.bc` | 输入 bitcode 文件；也可在合适场景传入文本 `.ll`。 |
| `-S` | 以人可读的 textual LLVM IR 输出，而不是 bitcode。 |
| `-o -` | 输出目标为标准输出，因此结果直接显示在终端。 |

若想保存为文件，可改写为：

```bash
opt --passes=gvn input.bc -S -o output.ll
```

### GVN 是什么

GVN 是 Global Value Numbering，中文可理解为“全局值编号”。这里的 global 不是指 LLVM 的 `@global` 变量，而是指它能跨 basic block 分析一个函数中的等价计算。

它给每个已知值或表达式分配一个抽象编号；若两个表达式在当前位置必然计算出相同的值，GVN 可以复用先前结果、删除后一个冗余计算。

例如：

```llvm
entry:
  %a = add i32 %x, %y
  br label %next

next:
  %b = add i32 %x, %y
  %r = mul i32 %b, 2
  ret i32 %r
```

`entry` 支配 `next`，且两次 `add` 的操作码、类型和操作数相同。因此 `%b` 与 `%a` 有相同的值编号，GVN 可将其化简为：

```llvm
entry:
  %a = add i32 %x, %y
  br label %next

next:
  %r = mul i32 %a, 2
  ret i32 %r
```

随后死代码删除可移除第二次 `add`。在满足别名分析和中间没有修改相关内存的条件时，GVN 也能消除冗余 load。它不能仅因两条指令文本相同就盲目合并：控制流中的支配关系、可能产生副作用的 call、内存写入、volatile/atomic 语义，以及 `poison` 等都必须保持正确。

`gvn` 是 standard compiler optimization，意思是它属于许多编译器都会采用的、目标无关的通用优化思路。它的 IR 语义对 x86 和 AMDGPU 相同；最终是否值得执行及其收益，会受后续 target code generation 影响。

### 什么是 New Pass Manager

Pass Manager 是 LLVM 中负责安排 pass 执行、提供 analysis 结果并在 IR 改动后使失效 analysis 重新计算的框架。LLVM 目前同时保留两套接口：

| 框架 | 典型命令形式 | 说明 |
| --- | --- | --- |
| Legacy Pass Manager | `opt -gvn input.bc -S -o -` | 较旧的 pass 管理接口；历史上也能驱动部分 backend/machine pass。 |
| New Pass Manager | `opt --passes=gvn input.bc -S -o -` | 当前推荐的 pipeline 语法；以 pass 名称组成可组合的管线。 |

书中特别强调：`opt --help` 列出的 `--gvn`、`--aa`、`--aarch64-O0-prelegalizer-combiner` 等选项主要来自 legacy 接口的目录。名称出现在这里，不保证它能作为 new Pass Manager 的 `--passes=<name>` 运行。

对 `gvn`，两套接口都存在，所以：

```bash
opt --passes=gvn input.bc -S -o -
```

可以工作。LLVM 源码中 new Pass Manager 的注册位置是：

```text
llvm/lib/Passes/PassRegistry.def
FUNCTION_PASS_WITH_PARAMS("gvn", "GVNPass", ...)
```

它表明 `gvn` 是 function pass：`opt` 会对 module 中的每个函数分别运行该 pass。

### 为什么书中接着让我们使用 `--print-passes`

下面的命令才是查看 new Pass Manager 可用 pass 的可靠来源：

```bash
opt --print-passes
```

它会按 Module、CGSCC、Function、Loop、Machine 等层级列出可放入 `--passes=` pipeline 的 pass。研究 middle end 时，应优先关注 Module、CGSCC、Function 和 Loop 层级；Machine 层级属于更靠近 backend 的 machine IR pass。

书中用两个反例说明不能只相信 `--help`：

```text
aarch64-O0-prelegalizer-combiner
```

显然是 AArch64 machine pass，不能直接作为 new PM middle-end pipeline 中的 `--passes=aarch64-O0-prelegalizer-combiner` 使用；而：

```text
aa
```

虽然是 target-independent alias analysis，也不是一个可直接以 `--passes=aa` 单独运行的 new PM transformation pipeline。analysis 和 transformation pass 的注册、运行方式不同。

因此，发现未知 pass 的正确工作流是：

1. 用 `opt --help` 搜索 pass 名称和简短说明；
2. 用 `opt --print-passes` 验证该名称是否被 new Pass Manager 支持，以及它属于哪个 IR 层级；
3. 用 `--passes=<pipeline>` 运行它；
4. 在源码的 `llvm/lib/Passes/PassRegistry.def` 搜索该名称，再沿着对应 pass class 找到实现和测试。

### 与 HIP/ROCm 的关系

`gvn` 本身不关心最终会生成 x86 代码还是 AMDGPU 代码。HIP 编译通常会分别产生 host 侧 x86 IR 与 device 侧 AMDGPU IR；若两边的 IR 满足 GVN 的语义前提，它都可以消除重复表达式。

但 GVN 不会将 GPU 的 global、LDS/local、private memory 混为一谈。对 load 的优化仍需要正确的 address space、别名分析和内存依赖信息。后续章节遇到 AMDGPU 专用 machine pass 时，再把它们与这种 target-independent middle-end pass 区分开。

## `Machine function analyses (WIP)` 是谁输出的

在 FlyDSL README 的源码构建步骤中：

```bash
bash scripts/build_llvm.sh -j64
bash scripts/build.sh -j64
```

第一条命令先构建 LLVM/MLIR，并在：

```text
/root/workspace/llvm-project/build-flydsl/bin/opt
```

生成 `opt`。第二条命令才构建 FlyDSL 自己的 C++ dialect、MLIR pass、Python binding 和 `fly-opt` 等工具。

因此执行：

```bash
./opt --print-passes
```

看到：

```text
Machine function analyses (WIP):
```

不是 FlyDSL 额外加的一组 pass，也不是 FlyDSL 的 “work in progress” 标记。它是 **upstream LLVM 自己的 `opt` 输出**。

当前源码中的直接证据是：

```text
llvm/lib/Passes/PassBuilder.cpp
```

其中打印了这个固定标题，并从：

```text
llvm/include/llvm/Passes/MachinePassRegistry.def
```

收集名称。该注册表中列出的都是 LLVM 自己的 machine-function analysis，例如：

```text
live-intervals
live-reg-matrix
machine-dom-tree
machine-loops
machine-uniformity
slot-indexes
virtregmap
```

### `MachineFunction` 与 `WIP` 分别是什么意思

`MachineFunction` 是 LLVM backend 使用的、低于 LLVM IR 的函数表示。大致流水线位置为：

```text
LLVM IR
  -> instruction selection
Machine IR / MachineFunction
  -> register allocation, scheduling, prologue/epilogue, instruction relaxation
machine code
```

所以这些 analysis 是为 backend 的 Machine IR pass 提供信息的。例如 `live-intervals` 分析虚拟寄存器的活跃区间，供寄存器分配使用；`machine-loops` 在 Machine Basic Block CFG 上识别循环。

标题中的 `WIP` 是 LLVM 对 **New Pass Manager 对 Machine IR pipeline 的支持仍在持续完善** 的标记。它不表示下面每个 analysis 都不可用，也不表示这些 analysis 是 FlyDSL 临时实现；而是提醒使用者，这一部分的 new-PM machine-pass 接口尚未像 LLVM-IR middle-end pipeline 一样稳定、完整。

### 为什么它会出现在 FlyDSL 的 `opt` 中

FlyDSL 的 `scripts/build_llvm.sh` 会：

1. 从 upstream `llvm-project` 获取 README 所指定的 LLVM revision；
2. 构建 `mlir;clang;lld`，并启用 `X86;NVPTX;AMDGPU` target；
3. 生成 `build-flydsl/bin/opt`。

该脚本应用的 FlyDSL patch 只修改 MLIR 的 `ROCDL/Target.cpp` 中调用 `ld.lld` 时传入的 `argv[0]`，没有修改 `PassBuilder.cpp`、`MachinePassRegistry.def` 或 machine pass 注册。因此本次看到的 `Machine function analyses (WIP)` 来自 LLVM 本身。

FlyDSL 自己注册的是 MLIR pass，入口是 `fly-opt` 和 FlyDSL 的 Python/C API 注册函数；它们不会自动变成 LLVM `opt --print-passes` 中的 `MachineFunction` analysis。除非将来有人直接修改 LLVM 源码或给 `opt` 加载专门的 LLVM pass plugin，否则这两套 pass 注册表是分开的。

## `Coroutines`：将可暂停函数 lower 为状态机

书中将 Coroutines 描述为“围绕 coroutine lowering 的 transformation”。这里的 “a function that can be resumed or suspended without locking the current thread” 应理解为：**coroutine 暂停时不会阻塞当前线程**，不是指 mutex 的加锁。

例如 C++20：

```cpp
task f() {
  int x = 1;
  co_await async_read();
  use(x);
}
```

执行到 `co_await` 时，`f` 必须把 `x` 和“下次应从哪里继续”的状态保存下来，然后把控制权还给调用者或 runtime；当前线程可以继续执行其他任务。异步操作完成后，runtime 再恢复该 coroutine。

普通函数的栈帧在函数返回后不能再保存这种状态，因此 LLVM coroutine lowering 会创建一个 **coroutine frame**。最初，前端将 coroutine 表示为普通 LLVM function 加上 `llvm.coro.*` intrinsics，例如：

```llvm
llvm.coro.id
llvm.coro.begin
llvm.coro.save
llvm.coro.suspend
llvm.coro.resume
llvm.coro.destroy
```

随后 pass 将其改写为普通的函数、控制流、内存访问和间接调用。一般结果是：

```text
原 coroutine function
          |
          v
ramp function：首次调用时运行到第一个 suspend
resume function：之后恢复执行
destroy function：销毁已暂停 coroutine
coroutine frame：保存跨 suspend 存活的局部值与恢复位置
```

LLVM 文档明确说明，最一般的 lowering 会将原函数改写为 ramp function，并将剩余部分拆到一个或多个 resume function；跨 suspend 必须保留的状态放入 coroutine frame。

当前源码中主要 pass 的职责如下：

| Pass | 作用 |
| --- | --- |
| `coro-early` | 先处理部分 coroutine intrinsic，例如将 `coro.resume` / `coro.destroy` 改写为经由 frame 中函数地址的调用。 |
| `coro-split` | 核心拆分：生成 ramp、resume、destroy 等 function，并构造 coroutine frame。 |
| `coro-elide` | 若 frame 不逃逸，或内联后不再需要动态分配，则消除 allocation。 |
| `coro-cleanup` | 清理剩余 intrinsic、无用控制流和 lowering 的中间痕迹。 |

相关源码与文档：

```text
llvm/docs/Coroutines.md
llvm/lib/Transforms/Coroutines/CoroEarly.cpp
llvm/lib/Transforms/Coroutines/CoroSplit.cpp
llvm/lib/Transforms/Coroutines/CoroElide.cpp
llvm/lib/Transforms/Coroutines/CoroCleanup.cpp
```

为什么书中说 lowering 通常高度依赖源语言：LLVM 并不知道 C++ 的 `promise_type`、`co_await` awaiter 协议，或 Swift async context 的高层语义。前端必须先选择 ABI 并产生相应的 `llvm.coro.*` intrinsic 模式；LLVM 再做底层状态机转换。

这不是普通 HIP kernel 的主线。它更接近 C++20 coroutine、Swift async 和异步 runtime 的编译支持。

## `HipStdPar`：将 C++ 标准并行算法 offload 到 AMD GPU

`HipStdPar` 不是手写：

```cpp
__global__ void kernel(...)
```

的编译路径。它面向 C++17 标准并行算法，例如：

```cpp
std::find(std::execution::par_unseq,
          values.begin(), values.end(), 42);
```

在 Clang 使用：

```bash
clang++ --hipstdpar --offload-arch=gfx942 ...
```

编译时，受支持的 `std::execution::par_unseq` 算法可通过 HIP 基础设施和 rocThrust / rocPRIM 等加速库 offload 到 AMD GPU。若库不支持某算法，则可回退到 host CPU。

`--hipstdpar` 的 Clang 选项说明就是：

```text
Enable HIP acceleration for standard parallel algorithms
```

它会切换到 HIP 编译/链接基础设施，并隐式包含用于将标准算法转发到 accelerator library 的头文件；这不意味着用户可以把它当作普通 HIP 源码并任意使用 `__device__`、`__global__` 等语言扩展。

### 当前源码中实际存在的三个 HipStdPar module pass

书中只概括为 “Transformations to enable the HIP C++ standard parallelism support”。当前源码的注册表显示它包含三个 module pass：

| `opt --passes=` 名称 | Pass | 作用 |
| --- | --- | --- |
| `hipstdpar-select-accelerator-code` | `HipStdParAcceleratorCodeSelectionPass` | 从 `amdgpu_kernel` 出发沿直接 call graph 收集可达函数，只留下设备端真正需要执行的代码。 |
| `hipstdpar-interpose-alloc` | `HipStdParAllocationInterpositionPass` | 将 `malloc`、`free`、`new`、`delete` 等改写到 `__hipstdpar_*` accelerator-aware runtime 函数。 |
| `hipstdpar-math-fixup` | `HipStdParMathFixupPass` | 将部分 `cmath` / LLVM math intrinsic 调用改写到 `__hipstdpar_*` 转发层。 |

#### 1. Accelerator code selection

`--hipstdpar` 要支持的是未显式标注 `__device__` 的普通 C++ 标准算法调用。device module 因而可能包含许多用户本来不希望在 GPU 上运行的函数，其中甚至可能有 AMDGPU backend 不支持的构造。

该 pass 将 `amdgpu_kernel` 视为 accelerator 执行根，沿**直接**调用边计算可达函数，再移除其余函数。若 module 没有 kernel，它会清空该 device module。这保证只有标准算法实现所需的 kernel 及其调用链继续送往 AMDGPU backend。

当前实现对 indirect call 有已知限制：间接调用的真实目标可能无法被完整识别，因此不能仅依赖这一 pass 将任意 C++ 程序安全 offload。

#### 2. Allocation interposition

有 HMM（异构内存管理、透明按需分页）的系统中，普通 `malloc`/`new` 内存可直接被 accelerator 访问时，通常不需要该步骤。

若系统或 accelerator 不支持这种机制，`hipstdpar-interpose-alloc` 可将：

```text
malloc  -> __hipstdpar_malloc
free    -> __hipstdpar_free
new     -> __hipstdpar_operator_new
delete  -> __hipstdpar_operator_delete
```

等调用改写给 runtime。runtime 再使用 accelerator-aware 分配机制，例如 managed-memory 对应实现，保证标准算法传入的数据可被 GPU 访问。

#### 3. Math fixup

不同 accelerator 对 `<cmath>` 函数及其 LLVM intrinsic lowering 的支持可能不同。该 pass 把已知有问题的 math libcall 或 intrinsic，例如部分 `acosh`、`erf`、`lgamma`、`log1p` 等，替换为 `__hipstdpar_*` 函数。HIPStdPar runtime 可在该统一接口后提供适合设备的实现。

### 与我们研究 AMDGPU backend 的关系

`HipStdPar` 比 Coroutines 更接近 HIP/ROCm，因为它会产生 AMDGPU device IR；但它解决的是 **C++ 标准并行算法的自动 offload**，不是 FlyDSL 直接生成或手写 HIP kernel 的常用途径。

学习 backend 时，重点是理解它如何在 middle end 清理 device module、确保内存可访问、补齐数学库调用；而后面的 AMDGPU backend 才负责将剩余 device IR lower 为 GPU Machine IR 和 ISA。

相关源码和文档：

```text
llvm/lib/Transforms/HipStdPar/HipStdPar.cpp
llvm/include/llvm/Transforms/HipStdPar/HipStdPar.h
llvm/lib/Passes/PassRegistry.def
clang/docs/HIPSupport.md
clang/include/clang/Options/Options.td
```

## Verifier：检查 pass 是否产生了非法 IR

书中将 verifier 列为 helper pass。它不做优化，也不修改 IR；它的职责是检查 LLVM IR 是否仍满足 LLVM IR 的结构和语义不变量，即 IR 是否 **well formed**。

例如 verifier 会检查：

- 一个 SSA 值的定义是否支配所有使用点；
- 指令的操作数类型是否正确，例如 `fadd` 的两个输入是否同为浮点类型；
- basic block 是否有合法 terminator，`phi` 的 incoming block/value 是否与 CFG 对应；
- `call`、intrinsic、attribute、address space、global initializer 等是否满足 LLVM IR 规则；
- module 中函数、类型和全局对象之间的引用关系是否有效。

它检查的是“这份 IR 能否作为合法 LLVM IR 继续交给后续 pass / backend”，不检查以下内容：

- 程序算法是否符合产品需求；
- 数组访问是否在运行时一定不越界；
- 是否存在源语言层面的 data race；
- 某个优化是否带来预期性能；
- 代码是否生成了用户想要的 x86 或 AMDGPU 指令。

### 为什么它对 pass 开发很重要

假设一条 pipeline 有三个 transformation：

```text
Pass A -> Pass B -> Pass C -> backend 崩溃或报 “Broken module”
```

最终失败并不能直接说明是哪一条 pass 破坏了 IR。若启用逐 pass 验证：

```text
Pass A -> verify OK
Pass B -> verifier fails
```

就能立即将问题缩小为 “Pass B 的输出非法”，而不是去怀疑后面所有 pass 或 backend。这正是书中所说 verifier “invaluable to narrow down when a transformation modified the IR in an illegal way” 的含义。

一个典型错误是自定义 pass 移动了定义，却遗漏更新某个 use：

```llvm
entry:
  br label %use

use:
  %r = add i32 %v, 1      ; 使用 %v
  ret i32 %r

def:
  %v = add i32 %x, %y    ; %v 的定义不支配 %use
  br label %use
```

`%v` 不是沿所有到达 `%use` 的路径都先定义，因此违反 SSA dominance。verifier 会拒绝这份 IR；它不会尝试猜测开发者原本希望怎样修 CFG 或插入 `phi`。

### `llc` 与 `-disable-verify`

`llc` 将 LLVM IR 送入 instruction selection、Machine IR、寄存器分配、调度等 codegen pipeline，生成汇编或目标文件。书中说明 `llc` 默认会在 codegen 的若干位置启用验证。

```bash
llc input.ll -o output.s
```

这意味着 backend pass 若破坏了某个受验证的 IR/Machine IR 不变量，通常会在更接近错误产生处报错，而不必等到最终汇编阶段。

```bash
llc -disable-verify input.ll -o output.s
```

会关闭这类标准验证。该选项只适合临时诊断 verifier 自身相关问题；它不能修复非法 IR，反而可能让错误继续传播为 assertion、崩溃或错误代码生成。

### `opt -verify-each`：每一条 transformation 后插入验证

`opt` 的选项：

```bash
opt --passes='mem2reg,gvn' -verify-each input.ll -S -o output.ll
```

含义是：在 `mem2reg` 后运行 verifier，再在 `gvn` 后运行 verifier。当前 `opt` 源码中该选项的描述就是：

```text
Verify after each transform
```

因此，开发新的 LLVM-IR pass 或调试某条 pass pipeline 时，`-verify-each` 应是首选开关。若还需要看失败前的 IR，可结合：

```bash
opt --passes='...' -verify-each -print-after-all input.ll -disable-output
```

其中 `-print-after-all` 输出每个 pass 后的 IR，`-disable-output` 表示不写最终文件。先由 verifier 找到失败 pass，再看该 pass 前后的打印结果，通常就能定位错误变换。

### New Pass Manager 中的对应实现

`-verify-each` 在 new Pass Manager 中不是把 `verify` 文本 pass 手动塞进每条 pipeline，而是通过标准 instrumentation 实现。`opt` 构造：

```cpp
StandardInstrumentations SI(Context, DebugLogging, VerifyEachPass);
SI.registerCallbacks(PIC, &MAM);
```

当 `VerifyEachPass` 为 `true` 时，instrumentation 在每个 transform 后调用相应 verifier。自定义 new-PM driver 或 out-of-tree 工具也可采用同样方式。

### 对 x86 与 AMDGPU backend 的关系

LLVM IR verifier 是 target-independent：它同样检查将要交给 x86 backend 的 IR 和将要交给 AMDGPU backend 的 IR。它会理解 pointer address space 等 LLVM 语义，但不会替我们判断一个 AMDGPU global-memory access 是否有好性能，或一个 LDS 使用是否超过硬件资源预算。

进入 Machine IR 后，LLVM 还会使用 machine-level verification 来检查寄存器、Machine Basic Block、指令约束等 backend 不变量。对于 AMDGPU backend 开发，这类验证尤其有价值，因为错误可能发生在 instruction selection、register allocation 或 target-specific lowering，而不再是最初的 LLVM IR。

相关源码：

```text
llvm/lib/IR/Verifier.cpp
llvm/include/llvm/IR/Verifier.h
llvm/tools/opt/optdriver.cpp          (-verify-each)
llvm/tools/opt/NewPMDriver.cpp        (StandardInstrumentations)
llvm/tools/llc/llc.cpp                (-disable-verify, -verify-each)
llvm/lib/Passes/StandardInstrumentations.cpp
```

## Analysis、`TargetIRAnalysis` 与 `TargetTransformInfo`

书中在介绍 Analysis passes 时提到：analysis 可以提供算法所需的信息，例如 dominance tree、alias information、loop information、block frequency、scalar evolution，以及 target-specific knowledge。Analysis pass 本身通常不改写 IR；它计算或缓存事实，供 transformation pass 查询。

### transformation 与 analysis 的区别

```text
transformation pass
  输入 LLVM IR
  -> 改写 LLVM IR
  例：GVN、mem2reg、loop unroll、vectorizer

analysis pass
  输入 LLVM IR
  -> 计算并返回信息
  例：DominatorTree、LoopInfo、AliasAnalysis、TargetIRAnalysis
```

书中当前这一段主要讨论 middle-end 的 LLVM-IR-to-LLVM-IR transformation。`transformation` 这个词在全书可以泛指“改变某层 IR 的 pass”，范围可以比 middle end 更宽；但 `TargetTransformInfo` 在这里的职责是帮助 middle-end 优化作出 target-aware 决策。

### `TargetIRAnalysis` 不是一种特殊运行行为

`TargetIRAnalysis` 是 New Pass Manager 中一个 analysis pass 的 C++ 类型/注册名。它不是某条会执行的目标指令，不是某个优化，也不是“LLVM 的特定行为”。

它做的事是：针对当前函数 `F`，结合当前 TargetMachine、target triple、CPU、target features 和 backend 提供的模型，产生对应的 `TargetTransformInfo` 结果。

书中的 new-PM 用法：

```cpp
TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(F);
```

应读作：

```text
从 FunctionAnalysisManager（FAM）取得
“函数 F 的 TargetIRAnalysis 结果”，
该结果对象的类型是 TargetTransformInfo（TTI）。
```

Legacy Pass Manager 中，获取同一类结果的形式是：

```cpp
TargetTransformInfo &TTI =
    getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
```

三层名称的关系如下：

```text
TargetIRAnalysis
  New Pass Manager 的 analysis pass；负责取得/缓存结果
          |
          v
TargetTransformInfo（TTI）
  transformation pass 实际查询的统一接口
          |
          v
TargetTransformInfoImpl
  X86、AMDGPU 等 backend 提供的具体成本模型和查询实现
```

### TTI 包含什么 information

TTI 不是单一静态表，而是一个查询接口。middle-end pass 通过它询问“对当前 target 来说，某种 IR 操作的成本、能力和偏好是什么”。常见查询类别包括：

| 类别 | TTI 可回答的问题 |
| --- | --- |
| 标量指令成本 | `i32 add`、`f32 mul`、`div`、cast 等操作代价大致如何？ |
| 向量指令成本 | `<4 x float>` 操作是否高效？应使用多宽的 vector？ |
| 内存访问成本 | 特定宽度/对齐/address space 的 load/store、gather/scatter 有多贵？ |
| intrinsic 与调用 | 某个 LLVM intrinsic 是否有高效 target lowering，调用成本如何？ |
| 循环优化 | 展开、interleave、vectorize、reduction 是否值得？ |
| 寄存器资源 | 更宽 vector 或更多展开会不会造成过高寄存器压力？ |
| target 特性 | 当前 function 的 `target-cpu` / `target-features` 是否支持某类操作？ |
| target 特定约束 | 某些 IR 操作在当前 target 上的 legality、preferred lowering 或代价。 |

例如 vectorizer 考虑把四个标量浮点加法改成一个 `<4 x float>` 操作时，可以询问：

```text
四次 scalar fadd 的成本是多少？
一次 <4 x float> fadd 的成本是多少？
相应 vector load/store 的成本是多少？
这样做是否会造成不合适的寄存器压力？
```

TTI 的结果帮助 vectorizer 决定是否向量化、使用什么 vector width，或保持标量代码。

### x86 与 AMDGPU 会给出不同答案

同一条 LLVM IR 指令的语义不因 target 改变，但 TTI 给出的成本模型可以不同：

```text
同一函数 F
  |
  +-- target = x86-64, features = +avx2
  |      -> 宽 SIMD vector 可能很有利
  |
  +-- target = amdgcn-amd-amdhsa
         -> 还需考虑 VGPR 压力、wavefront 执行、
            uniform/divergent 值、global/LDS/private address space 访存
```

因此 TTI 的 information 不是运行时 profile 数据，也不是程序输入数据。它是：

```text
当前 target 对某种 LLVM IR 操作的
成本、能力、偏好和限制。
```

TTI 不亲自生成 x86 或 AMDGPU 指令，也不亲自改 IR；它让 target-independent middle-end transformation 能够避免作出明显不适合当前机器的决定。

## Alias Analysis：用多个分析按成本逐步判断内存是否重叠

书中举的例子说明：Alias Analysis（AA）通常不是一个孤立算法，而是多个 alias analysis provider 组合起来的查询系统。它要回答的核心问题是：

```text
两个指针所代表的内存访问，是否可能访问同一段内存？
```

典型结果可概念化为：

| 结果 | 含义 |
| --- | --- |
| `NoAlias` | 两个访问不可能重叠。 |
| `MayAlias` | 现有信息不足，可能重叠，必须保守处理。 |
| `MustAlias` | 两个访问确定是同一位置或等价位置。 |
| `PartialAlias` | 两个访问可能部分重叠。 |

### 书中的两个分析层次

书中假设 AA 依次使用两类分析：

1. **基于类型的 alias analysis**：利用 strict aliasing 规则排除明显不可能重叠的访问；
2. **基于范围的 alias analysis**：保守估算指针可能到达的内存区间，判断两个区间是否相交。

第一类通常更便宜，因此先执行。若它已经能够回答 `NoAlias`，第二类更昂贵的范围分析就没有必要运行。

例如源级概念上有：

```cpp
float *fp;
int *ip;
```

在 C/C++ strict aliasing 规则下，正常的 `float` 对象不能被当作 `int` 对象读写，反之亦然。因此，在没有特殊例外的前提下：

```text
*fp 与 *ip 不会是同一个合法对象访问
=> 可得出 NoAlias
=> 不需要继续做范围分析
```

这里的目的不是“类型不同就必然物理上指向不同地址”。它们当然可能来自同一块 raw allocation，甚至数值地址相同；结论是：若程序真的以不允许的类型别名方式访问，则它已经违反源语言 strict aliasing 语义。编译器可以基于“只需保持定义良好的程序行为”来假定这种非法重叠不会发生。

### LLVM 中这通常依赖 TBAA，不是 opaque pointer 的指针类型

现代 LLVM IR 使用 opaque pointer：

```llvm
ptr %p
ptr %q
```

文本中的 `ptr` 不带旧式的 `float*` 或 `int*` pointee type。因此 LLVM 不能只看指针类型字符串来实现这类判断。

前端通常通过 TBAA（Type-Based Alias Analysis）metadata，将源语言的类型别名事实附在 load/store 上。概念示例：

```llvm
%f = load float, ptr %fp, !tbaa !float_type
%i = load i32,   ptr %ip, !tbaa !int_type
```

AA 利用这些 metadata 以及语言规则判断两次内存访问是否可以重叠。以下情形会使“`float*` 与 `int*` 不 alias”的简化说法失效或需要额外处理：

- `char` / `unsigned char` / `std::byte` 按 byte 访问对象；
- `memcpy`、序列化、合法的对象表示访问；
- union、语言扩展或明确的 type-punning 规则；
- 前端没有提供可信 TBAA metadata；
- 源语言本身不是采用 C/C++ strict aliasing 的语言。

因此，TBAA 提供的是“在特定语言语义和 metadata 前提下可利用的事实”，不是任意 LLVM 指针之间的绝对物理定律。

### 范围分析如何补充类型分析

若类型分析不能排除 alias，例如两个指针都访问 `float`，AA 可以进一步利用 base object、GEP offset、访问大小、已知边界等信息推断可达范围。

```text
%a 可访问：base + [0, 64)
%b 可访问：base + [128, 192)

区间不相交
=> NoAlias
```

反过来：

```text
%a 可访问：base + [0, 64)
%b 可访问：base + [32, 96)

区间相交
=> 不能回答 NoAlias；可能是 PartialAlias 或 MayAlias
```

真实 LLVM 中，这类能力通常由 BasicAA、底层对象识别、GEP offset 推理、`MemoryLocation` 的访问大小、范围/捕获信息及其他 AA provider 共同完成。书中所说 “range analysis” 是这种“保守估算 pointer 可达内存”的高层概括，不必将它理解为唯一某个名为 RangeAnalysis 的 pass。

### 为什么要先用便宜分析

AA 查询在许多优化中出现得非常频繁，例如：

- GVN 想删除重复 load；
- LICM 想将 load 移到 loop 外；
- vectorizer 想确认两个数组访问可安全并行化；
- DSE 想删除被后续 store 覆盖的旧 store；
- MemorySSA 想找某次 load 可能看到的 store。

如果每次都先运行最复杂的 pointer range 或 memory-dependence 推理，编译时间会迅速上升。组合 AA 的常见策略是：

```text
cheap analysis
  -> 已证明 NoAlias：立即结束
  -> 仍是 MayAlias：继续询问更精细分析
  -> 仍无法证明：保守地保留 MayAlias
```

这就是书中“第一种分析足够回答 pointers 是否 overlap 时，第二种不使用”的意思。保守性很关键：无法证明不重叠时，优化器必须当作可能重叠，宁可错过优化，也不能重排出错误程序。

### 对 x86 与 AMDGPU 的关系

TBAA、GEP offset 和访问范围等 AA 语义首先在 LLVM IR 层成立，对 x86 和 AMDGPU 都有用。AMDGPU 还带来不同 address space：例如 global、LDS/local 与 private memory 具有不同可见范围，target-specific AA 可利用这种事实进一步证明某些访问不 alias。

但 address space 不会取代一般 alias analysis。两个 `ptr addrspace(1)` 都是 global pointer 时，仍需通过对象、offset、TBAA、call effect 和内存依赖等信息判断是否重叠。

## Block Frequency Info：预计每个 basic block 会执行多少次

书中的 `BlockFrequencyAnalysis` 产生 `BlockFrequencyInfo`（简称 BFI）。这里的 frequency 不是 CPU/GPU 时钟频率，也不是某条机器指令的延迟；它表示：

```text
在一次函数调用的预期执行中，
某个 basic block 相对于其他 block 大约会被走到多少次。
```

例如：

```cpp
for (int i = 0; i < n; ++i) {
  sum += a[i];
}
return sum;
```

对应的 CFG 中，loop header、body 和 latch 大约会执行 `n` 次，`entry` 与 `exit` 通常只执行一次：

```text
entry             ~ 1 次
loop.header       ~ n + 1 次
loop.body         ~ n 次
loop.latch        ~ n 次
exit              ~ 1 次
```

因此即使没有真实 profile，分析也能知道 loop body 比 exit 更“热”。

### BFI 从哪里得到估计

BlockFrequencyAnalysis 会结合 CFG 边的概率推导各 block 的相对频率。概率来源可以包括：

- 静态 branch heuristic；
- `!prof` branch weight metadata；
- PGO（profile-guided optimization）采集到的真实执行 profile；
- loop backedge 与控制流结构。

例如：

```llvm
br i1 %cond, label %likely, label %unlikely, !prof !0
!0 = !{!"branch_weights", i32 900, i32 100}
```

这表达 `likely` 边相对更常走。若该 branch 所在 block 本身很热，BFI 会将较高频率传播到 `likely` block，将较低频率传播到 `unlikely` block。

没有 PGO 时，BFI 是**静态估计**，不应将其误解为精确运行时计数；有 PGO 时，它可与入口计数结合，得到更接近实际执行次数的 profile count。无论何种来源，优化器仍将它视为概率/频率信息，而不是必须在每次运行都成立的事实。

### 为什么优化器需要它

很多优化在 hot path 上值得付出更高代码体积或编译时间代价，在 cold path 上则不值得。BFI 常用于：

- block / function layout：让常走路径在机器码中更连续，减少 branch 和 I-cache 压力；
- inlining：更积极地内联 hot call site，对 cold call site 更保守；
- loop 与 vectorization 决策：优先优化预期高频执行的 loop；
- code sinking、outlining、cold block splitting：将罕见错误处理路径放到较远位置；
- profile-guided 的优化收益估算。

因此 BFI 回答的不是“这个 block 能不能执行”，而是：

```text
如果必须在两个合法优化中选择，
优化哪一条路径更可能带来总体收益？
```

### New Pass Manager、Legacy PM 与命令行名称

书中给出的对应关系是：

| Pass Manager | Analysis pass | 结果对象 |
| --- | --- | --- |
| New Pass Manager | `BlockFrequencyAnalysis` | `BlockFrequencyInfo` |
| Legacy Pass Manager | `BlockFrequencyInfoWrapperPass` | `BlockFrequencyInfo` |

在 new PM 中，function pass 获取结果的模式与 TTI 相同：

```cpp
BlockFrequencyInfo &BFI = FAM.getResult<BlockFrequencyAnalysis>(F);
```

`opt --print-passes` 中的名称是 `block-freq`。因为它是 analysis，若要在 pipeline 中显式要求并打印，书中建议使用 `require` 关键字和：

```text
print<block-freq>
```

即先要求分析结果可用，再由 printer 输出结果。

### LLVM IR 与 Machine IR 各有一份频率信息

书中特别指出，这个分析在 Machine IR 层也存在：

```text
LLVM IR CFG
  -> BlockFrequencyAnalysis / BlockFrequencyInfo

instruction selection 后的 Machine CFG
  -> MachineBlockFrequencyInfo
```

IR 层 BFI 服务于 middle-end 的 inlining、loop、代码布局等决策。进入 backend 后，Machine IR 的 block frequency 可继续帮助 x86 或 AMDGPU 的 machine scheduler、block placement、寄存器分配和 target-specific codegen 决策。

对 AMDGPU 而言，BFI 仍描述一个 kernel/control-flow 中各 Machine Basic Block 的预计执行频率；它不是 wavefront 数量、work-group 数量或 GPU occupancy。那些是不同层次的执行模型信息。

## Dominator Tree：控制流中“到达 B 前必经 A”的关系

书中说 `DominatorTreeAnalysis` 计算 dominator tree information，结果对象是 `DominatorTree`。它回答的核心问题是：

```text
basic block A 是否支配（dominate）basic block B？
```

定义为：若从函数入口 `entry` 到达 `B` 的**每一条**控制流路径都必须先经过 `A`，那么 `A` dominates `B`。

```text
A dominates B
<=> 所有 entry -> B 的路径都经过 A
```

这是一种“必经关系”，不是执行次数、branch 概率或代码在文本中的先后顺序。

### 一个 CFG 例子

```text
             entry
               |
              cond
             /    \
          then    else
             \    /
              join
               |
              exit
```

在这个 CFG 中：

```text
entry dominates cond, then, else, join, exit
cond  dominates then, else, join, exit
then  does not dominate join
else  does not dominate join
join  dominates exit
```

`then` 不支配 `join`，因为存在：

```text
entry -> cond -> else -> join
```

这条路径到达 `join` 时没有经过 `then`。

### 为什么称为 tree

所有 block 都被 `entry` 支配；此外，每个可达 block 都有一个最近的严格支配者，称为 immediate dominator（`idom`）。把每个 block 连到它的 immediate dominator，就得到 dominator tree：

```text
entry
  |
 cond
 / | \
then else join
            |
           exit
```

上图只展示支配关系树，不表示 CFG 的所有 branch edge。CFG 与 dominator tree 是不同图：

```text
CFG             描述控制可能怎样流动
Dominator tree  描述到达某处必须经过谁
```

### 它与 SSA 的关系

SSA 的重要不变量是：一个普通 SSA 定义必须支配它的所有 use。

```llvm
entry:
  %x = add i32 %a, %b
  br label %use

use:
  %y = mul i32 %x, 2
  ret i32 %y
```

`entry` 支配 `use`，所以 `%x` 的定义支配 `%x` 的 use，IR 合法。

若定义只位于 `then`，却在 `join` 中直接使用：

```text
entry -> cond -> then -> join
             \-> else -> join
```

则 `then` 不支配 `join`。因为从 `else` 到 `join` 的路径没有定义 `%x`，需要 `phi` 合并两个路径上的值，而不能直接使用只在 `then` 定义的值。

`phi` 的 incoming value 有特殊规则：它不要求支配 `phi` 所在 block，而要在对应的 predecessor edge 上可用。这正是 `phi` 能在 `join` 合并 `then` 和 `else` 结果的原因。

### 哪些优化需要它

许多 transformation 都需要利用或维护 dominance：

- GVN：只有前一次计算支配当前位置时，结果才可在当前位置安全复用；
- LICM：将 loop 内计算移到 loop preheader 时，必须确保定义仍支配所有 use；
- code motion、common subexpression elimination、值传播；
- 插入新 SSA 值、重写 CFG、构造或修复 `phi`；
- verifier：检查 SSA definition 是否支配普通 use。

因此 DominatorTree 是 LLVM 中使用极广的 analysis。它位于 IR library 而不是普通 Analysis library，书中解释的原因是它与 SSA 性质紧密绑定。

### API、Pass Manager 与命令行名称

| Pass Manager | Analysis | 结果 |
| --- | --- | --- |
| New Pass Manager | `DominatorTreeAnalysis` | `DominatorTree` |
| Legacy Pass Manager | `DominatorTreeWrapperPass` | `DominatorTree` |

new PM function pass 中通常这样取得结果：

```cpp
DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);

bool IsDominated = DT.dominates(A, B);
```

`A`、`B` 可以是 basic block、instruction 等可比较的 CFG/IR 实体。`opt --print-passes` 中的名称是 `domtree`；书中还指出可用：

```text
print<domtree>
verify<domtree>
```

分别打印或验证这个分析结果。

### 修改 CFG 后不要继续相信旧的树

若 pass 添加、删除或重连 branch edge，原来的 DominatorTree 可能已经过期。此时必须：

- 让 Pass Manager 使该 analysis 失效并在下次查询时重算；或
- 在适用时使用 `DomTreeUpdater` 维护更新后的树。

继续拿旧 tree 判断 dominance，可能导致错误的 code motion 或非法 SSA。书中因此特别建议在自行维护 CFG 变换时利用 `DomTreeUpdater`。

### Post-dominator 与 Machine IR 对应物

post-dominator 反过来回答：从 `A` 出发到达函数退出的每条路径是否都会经过 `B`。其 analysis 名称是 `PostDominatorTreeAnalysis`，命令行名称是 `postdomtree`。

instruction selection 后，Machine IR 也有相同概念：`MachineDominatorTree` 与 `MachinePostDominatorTree`。它们用于 x86、AMDGPU 等 backend 的 Machine Basic Block CFG；定义仍是控制流上的必经关系，不会因为 target 不同而改变。

## InstCombine 与 target-specific IR：canonical form 是 middle-end 的边界

书中提醒：当我们开始为了某个 target 而修改 IR 时，要注意 `instcombine` 可能会将 IR 改回 LLVM 的 canonical form。这里不是说 InstCombine 产生错误，而是说它的目标是服务**通用 middle-end**，并非保存某个 target 的局部偏好。

### 书中的 `sub` 与 `add + neg` 例子

源级表达式：

```cpp
a = b - c;
```

可以用两种语义相同的 LLVM IR 表示：

```llvm
; Version 1：LLVM 的 canonical form
%a = sub i64 %b, %c
```

```llvm
; Version 2：先取负，再加
%neg_c = sub i64 0, %c
%a = add i64 %b, %neg_c
```

对普通 `i64` 的模运算语义而言，两者都计算 `b - c`。但 Version 1 更短、更直接，也是 LLVM middle-end 偏好的 canonical representation。因此 InstCombine 会倾向于把 Version 2 重新化简为 Version 1。

这对通用优化很有价值：若所有“减法”都尽量写成 `sub`，GVN、reassociate、constant folding、pattern matching 等 pass 只需主要处理一种形态，而不必同时处理 `sub b, c` 与 `add b, (neg c)` 的所有变体。

### 为什么 target-specific pass 可能不希望被改回去

设想一个假想 target 没有 native subtraction instruction，却有高效的 negate 和 add：

```text
目标机支持：NEG、ADD
目标机不支持：SUB
```

对于这种机器：

```llvm
%neg_c = sub i64 0, %c
%a = add i64 %b, %neg_c
```

可能比保留 `sub` 更接近它最终需要的形态。一个 target-specific transformation 若特意把 `sub` 改写成这两个操作，随后再运行 InstCombine，InstCombine 会基于通用 canonicalization 把它改回：

```llvm
%a = sub i64 %b, %c
```

于是 target pass 的选择被“撤销”。这正是书中所说 “instcombine may put the IR back in the canonical form”。

注意：x86 和 AMDGPU 都有高效的整数减法 lowering，这个 `SUB` 缺失的 target 只是解释 canonicalization 与 target 偏好可能冲突的教学例子。即使 target IR 中保持 `sub`，x86 或 AMDGPU backend 仍可在 instruction selection 时选择最合适的机器指令序列。

### 典型 pipeline 为什么在 target-specific 阶段前大量运行 InstCombine

一个简化的典型 pipeline 可写成：

```text
frontend LLVM IR
       |
       v
generic middle-end optimizations
  InstCombine <-> GVN <-> SimplifyCFG <-> loop/vector passes
       |
       | 此处 IR 保持通用 canonical form
       v
target-specific IR transforms / target lowering
       |
       | 此处开始保留 target 的专门表达
       v
instruction selection -> Machine IR -> target machine code
```

middle-end 中常反复运行 InstCombine，是因为其他通用 pass 会制造冗余、常量表达式或非 canonical 变体；InstCombine 负责周期性清理，使后续通用优化面对较少的 IR 形态。

一旦 pipeline 进入 target-specific 阶段，通常不再把**通用** InstCombine 放在 target pass 后面。否则它可能：

- 把为特定指令选择准备的等价形态改回通用形态；
- 隐藏 target pass 想保留的 pattern；
- 让后端不得不再次重建刚被 canonicalization 抹掉的信息。

这是一条典型 pipeline 设计原则，不是“LLVM 此后绝对不能运行任何 InstCombine”的语言规则。某个 target 可以在明确知道 rewrite 不会冲突时安排局部 InstCombine，或注册针对该 target 的后续组合逻辑；但必须验证 canonicalization 不会破坏 target-specific intent。

### target-specific constructs 不只指 `sub` 改写

这里的 constructs 可包括：

- target intrinsic，例如 `llvm.amdgcn.*`、`llvm.x86.*`；
- 为特定 address space、memory operation、vector idiom 设计的 IR pattern；
- target-specific function attribute、calling convention 或 metadata；
- 后端自己识别的指令选择前 pattern。

InstCombine 不会任意删除所有 target intrinsic；但若 target pass 只是用普通 LLVM 指令拼出一个等价计算，InstCombine 可能识别并 canonicalize 它。因此问题不在“IR 是否 target-specific”这个标签，而在于该表达是否落入 InstCombine 已知的通用 rewrite pattern。

### 若 target-specific 阶段仍想做局部简化

书中给出的建议是：不要重新运行完整 InstCombine，而可直接调用 Analysis library 中的 `simplifyXXXInst` helper，例如适合当前操作的 `simplifyAddInst`、`simplifySubInst`、`simplifyCmpInst` 等。

这样 target pass 可以只采用所需的、可控的局部代数化简，而不触发完整 InstCombine 的大量 canonicalization 和 compile-time 成本。

### 实操判断规则

当实现一个 LLVM IR pass 时，可按以下顺序判断：

1. 若 pass 仍属于通用 middle-end，优先生成 canonical LLVM IR，并允许后续 InstCombine 清理。
2. 若 pass 已表达确定的 target-specific 选择，确认后面是否仍会运行 InstCombine；若会，检查它是否会逆转该选择。
3. 若后端本来就能从 canonical IR 选择正确指令，优先保留 canonical IR，让 instruction selection 处理 target 差异。
4. 若确实必须在 LLVM IR 层保留 target 特有 pattern，将该 pass 安排在最后一次通用 InstCombine 之后，并以 IR test 验证 pattern 存活到目标 lowering。

### 例子：32-bit pointer target、显式 `trunc` 与 `trunc(zext(x))` 消除

书中紧接着给出一个看似反直觉的 canonicalization：当 target 只支持 32-bit pointer 时，InstCombine 会将 `inttoptr` 中**隐含**的位宽截断显式写入 IR。

原始 IR：

```llvm
%b = load i64, ptr %x
%c = inttoptr i64 %b to ptr
```

在一个 pointer 宽度为 32 bit 的 target 上，`inttoptr i64 %b to ptr` 必须只使用 `%b` 的低 32 bit；高 32 bit 无法放入 pointer 表示。因此该转换隐含：

```text
取 %b 的低 32 bit
再把这 32 bit 解释为 pointer
```

InstCombine 将它改写为：

```llvm
%b = load i64, ptr %x
%b32 = trunc i64 %b to i32
%c = inttoptr i32 %b32 to ptr
```

这个 rewrite 的第一目的不是直接提高运行时性能，而是让 `inttoptr` 成为位宽中立的操作：所有改变 bit width 的动作都在明确的 `trunc` / `zext` / `sext` 指令中出现。这样，后续优化能够看见并匹配这些转换。

#### `zext` 是什么

`zext` 是 zero extension（零扩展）：把较窄整数扩展为较宽整数，原值放在低位，新增加的高位全部补零。

```llvm
%wide = zext i32 %x to i64
```

例如：

```text
%x    = i32 0xAABBCCDD
%wide = i64 0x00000000AABBCCDD
```

它和 `sext`（sign extension，符号扩展）不同：`sext` 会把原 i32 的最高位复制到新增高位；`zext` 永远补零。

#### 为什么 `trunc i64 (zext i32 %x to i64) to i32` 是 no-op

若 `%b` 的来源恰好是：

```llvm
%b = zext i32 %x to i64
```

再结合前面显式化的 32-bit pointer 截断：

```llvm
%b32 = trunc i64 %b to i32
```

整个表达式就是：

```llvm
%b32 = trunc i64 (zext i32 %x to i64) to i32
```

逐 bit 看：

```text
%x     : i32 = 0xAABBCCDD

zext
%b     : i64 = 0x00000000AABBCCDD

trunc
%b32   : i32 = 0xAABBCCDD
              = %x
```

所以：

```text
trunc i64 (zext i32 %x to i64) to i32
== %x
```

InstCombine 可以直接将 `%b32` 的所有 use 替换为 `%x`，删除中间的 `zext` 与 `trunc`，最后得到：

```llvm
%c = inttoptr i32 %x to ptr
```

书中所说 “can now be simplified into a no-op” 指的是这对 `zext` 与 `trunc` 的组合可被消除；**不是说 `inttoptr` 本身是 no-op，也不是生成一条 CPU `nop` 指令。**

#### 这不是内存节省

`zext` 与 `trunc` 操作的是 SSA value：

```llvm
%wide = zext ...
%back = trunc ...
```

它们没有分配 heap/stack，也没有 `load` / `store`，因而不涉及“少占多少内存”。删除它们的收益主要是：

- 少两个 LLVM IR instruction 与 SSA temporary；
- 数据流和后续 pattern matching 更简单；
- 减少潜在的虚拟寄存器压力和机器级转换操作；
- 让最终 backend 更容易直接看到正确位宽的值。

如果 `%b` 是任意 `i64`，而非由 `zext i32` 得到，则：

```llvm
%b32 = trunc i64 %b to i32
```

不能消除，因为 `%b` 的高 32 bit 可能任意，但低 32 bit 是真正要保留的结果。只有“先无损扩宽同一个 `i32`，再截回原宽度”的特定模式才是 identity。

#### 这个例子与 target-specific 信息的关系

该例同时说明两个层次：

```text
target-specific fact
  当前 target 的 pointer 宽度是 32 bit
       |
       v
InstCombine 的通用 canonicalization
  将隐式 i64 -> pointer 截断显式写成 trunc i64 -> i32
       |
       v
其他通用优化
  看见 trunc(zext(%x)) 后将其折叠为 %x
```

因此 InstCombine 主要是通用 middle-end canonicalization pass，但并非完全不了解 target；它可利用 DataLayout 等 target 事实，例如 pointer size。它仍不负责保留某个 backend 的专属 instruction-selection pattern，这正是前一节“target-specific IR 应位于最后一次通用 InstCombine 之后”的原因。

## `mem2reg`：将局部内存访问提升为 SSA 值

书中说 `mem2reg` 不是 canonicalization pass，却把它放在 canonicalization 一节，是因为它产生的 IR 是“任何合理优化”的起点。这里的 pass 是 `PromotePass`（legacy PM 中为 `PromoteLegacyPass`），命令行名称是 `mem2reg`。

它的核心工作是：

```text
promotable local alloca
  + store
  + load
        |
        v
SSA definition
  + direct use
  + phi（控制流汇合时）
```

### 一个局部变量的例子

源代码：

```cpp
int foo(int x, bool cond) {
  int a = 1;
  if (cond)
    a = x;
  return a + 2;
}
```

前端在较早阶段可以产生以下内存形式的 IR：

```llvm
entry:
  %a = alloca i32
  store i32 1, ptr %a
  br i1 %cond, label %then, label %merge

then:
  store i32 %x, ptr %a
  br label %merge

merge:
  %value = load i32, ptr %a
  %result = add i32 %value, 2
  ret i32 %result
```

这份 IR 的语义没有问题，但要优化 `%value`，其他 pass 需要回答：

```text
这个 load 会读到 entry 的 store，还是 then 的 store？
then 路径是否走过？
是否存在另一个可能 alias 到 %a 的指针写入该位置？
```

即使这里 `%a` 是显而易见的局部对象，通用内存优化仍需围绕 store/load、MemorySSA 或 Alias Analysis 推理。

运行 `mem2reg` 后，等价的 SSA 形式是：

```llvm
entry:
  br i1 %cond, label %then, label %merge

then:
  br label %merge

merge:
  %a.value = phi i32 [ 1, %entry ], [ %x, %then ]
  %result = add i32 %a.value, 2
  ret i32 %result
```

现在变量 `a` 的值直接由 `phi` 表达：

```text
从 entry 直接到 merge：a = 1
从 then 到 merge：        a = x
```

优化器不再需要通过内存位置找值；`%a.value` 的定义、使用与控制流选择全都在 SSA 图中显式可见。

### 为什么这使优化容易得多

提升前：

```text
store -> memory location -> load
```

提升后：

```text
SSA definition -> direct use
```

这给 middle end 带来：

- **直接 def-use chain**：从一个值可直接枚举所有 use；
- **dominance 性质**：普通 SSA definition 必须支配其 use，值的可用范围更容易判断；
- **更少的 alias 查询**：对于已提升局部变量，不再需判断多个 pointer 是否写到了它；
- **更简单的常量传播、GVN、DCE、循环优化和 code motion**；
- **更少的显式 load/store**，从而暴露更多算术和控制流优化机会。

书中 “without this transformation, all optimizations would need to track memory locations and use alias analyses” 的意思是：若每个局部变量都保持 memory form，每个优化都要面对“某次 load 究竟从哪次 store 取值”的 memory-dependence 问题。现代 SSA-based compiler 会尽早把能提升的局部变量移出这个问题域。

“stuck with a compiler technology of another age” 是作者的强调说法：没有 SSA promotion 当然仍能写编译器，但大量优化会更复杂、更慢且难以组合。

### `mem2reg` 不是物理寄存器分配

名称容易误导：

```text
mem2reg 的 reg
  = LLVM IR 中的 SSA virtual value
  ≠ x86 的 RAX/XMM
  ≠ AMDGPU 的 SGPR/VGPR
```

`mem2reg` 在 LLVM IR middle end 运行。真正将 SSA value 分配到 x86 通用寄存器/SIMD 寄存器，或 AMDGPU SGPR/VGPR，并处理 spill 的 register allocation，发生在后面 Machine IR backend 阶段。

### 它只能提升 promotable 的局部对象

`mem2reg` 不是任意内存消除器。它主要适用于函数内、地址不逃逸、以简单 load/store 访问的局部 `alloca`。以下对象通常不能直接提升：

- `malloc`、`new`、`hipMalloc` 等动态分配；
- 地址传给未知函数、存入全局变量或以其他方式逃逸的 `alloca`；
- 需要复杂 pointer arithmetic 或动态索引的数组/结构体访问；
- `volatile`、atomic 或具有必须保留可观察内存语义的访问；
- 跨函数共享的 global、LDS 或 GPU global memory 对象。

这也是为什么 HIP/ROCm kernel 中的普通线程局部标量常能很早变成 SSA value，而 global memory、LDS/shared memory 与需要真正地址的 private scratch 对象仍需由后续 AMDGPU backend 处理。

## LCSSA、exit `phi` 与 AMDGPU 的 uniform induction variable

书中用下面的 loop 说明 LCSSA（Loop Closed SSA）为何不仅方便通用 loop transformation，也能为 target-specific lowering 提供有价值的边界信息：

```llvm
define i64 @def_in_loop_use_outside(i64 %src, i64 %upper_bound) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv_plus_1, %loop ]
  %iv_plus_1 = add i64 %iv, 1
  %cond = icmp ult i64 %iv_plus_1, %upper_bound
  br i1 %cond, label %loop, label %end

end:
  %tmp = add i64 %iv_plus_1, %src
  %res = add i64 %tmp, %iv_plus_1
  ret i64 %res
}
```

### induction variable 是什么

induction variable（归纳变量）是循环中每轮按可预测规则推进、通常控制循环进度的变量。源级最常见形式是：

```cpp
for (int i = 0; i < n; ++i)
```

其中 `i` 是 induction variable。

在上面的 LLVM IR 中，induction variable 由一对 SSA 值表示：

```llvm
%iv = phi i64 [ 0, %entry ], [ %iv_plus_1, %loop ]
%iv_plus_1 = add i64 %iv, 1
```

其含义为：

```text
首次进入 loop：%iv = 0
后续每轮进入 loop：%iv = 上一轮的 %iv_plus_1
本轮：%iv_plus_1 = %iv + 1
```

`%iv` 是当前迭代值，`%iv_plus_1` 是递增后的下一迭代值。书中的 loop 在每轮先加一、再比较。下面是一个与其中的 `icmp ult` 具有相同比较语义的**源级草图**：

```cpp
uint64_t iv = 0;
uint64_t iv_plus_1;
do {
  iv_plus_1 = iv + 1;
  iv = iv_plus_1;
} while (iv_plus_1 < upper_bound);
```

这里的 `uint64_t` 不是从 SSA 名称 `%iv` 或 `%upper_bound` 本身推导出来的。LLVM 的整数类型是 signless：

```llvm
%iv          ; 名称没有 signed/unsigned 信息
i64          ; i64 只表示 64 个 bit，不表示 signed i64 或 unsigned i64
```

因此，不能只根据：

```llvm
%iv = phi i64 ...
```

断言源代码一定写了 `uint64_t`，也不能断言一定写了 `int64_t`。前端在降低 C/C++ 时会把 signed/unsigned 的**每次使用语义**编码到具体指令中。

在本例中：

```llvm
%cond = icmp ult i64 %iv_plus_1, %upper_bound
```

`ult` 表示 unsigned less-than，因此仅能确定：这一处比较将两个 `i64` bit pattern 按无符号整数解释。它可能来自普通的：

```cpp
uint64_t iv_plus_1;
uint64_t upper_bound;
iv_plus_1 < upper_bound;
```

也可能来自 signed 值经过显式 cast 后的无符号比较。IR 已不保留“变量在源代码中声明时带的是 signed 还是 unsigned”这一全局标签。

常见的 signed/unsigned 语义在 LLVM IR 中通过使用点表达：

| 源级所需语义 | LLVM IR 例子 |
| --- | --- |
| 有符号比较 | `icmp slt i64 %a, %b` |
| 无符号比较 | `icmp ult i64 %a, %b` |
| 有符号除法 / 余数 | `sdiv` / `srem` |
| 无符号除法 / 余数 | `udiv` / `urem` |
| 有符号扩展 | `sext i32 %x to i64` |
| 无符号扩展 | `zext i32 %x to i64` |
| 算术右移 | `ashr` |
| 逻辑右移 | `lshr` |
| 有符号 / 无符号整数转浮点 | `sitofp` / `uitofp` |

普通：

```llvm
%iv_plus_1 = add i64 %iv, 1
```

只计算 64-bit 的结果 bit pattern；它本身没有 signed 或 unsigned 加法版本。若源语义还要求“不发生有符号回绕”或“不发生无符号回绕”，前端或优化器可附加：

```llvm
add nsw i64 %iv, 1 ; signed no-wrap
add nuw i64 %iv, 1 ; unsigned no-wrap
```

书中示例的 `add i64` 不带 `nsw` 或 `nuw`，所以不能从该 add 推断出 source-level signed overflow 的承诺；它按 64-bit modulo arithmetic 的 IR 语义工作。

#### 用本机 `hipcc` 验证 source signedness 如何落入 AMDGPU IR

为确认这不是只停留在规则层面的说法，我们用本机 ROCm `hipcc` 将两个最小 HIP C++ kernel 编译为 AMDGPU LLVM IR。两者的循环形状相同，唯一差别是 source type：

```cpp
extern "C" __global__
void unsigned_loop(uint64_t *out, uint64_t bound) {
  uint64_t iv = 0;
  do {
    iv = iv + 1;
  } while (iv < bound);
  out[0] = iv;
}

extern "C" __global__
void signed_loop(int64_t *out, int64_t bound) {
  int64_t iv = 0;
  do {
    iv = iv + 1;
  } while (iv < bound);
  out[0] = iv;
}
```

使用等价于：

```bash
hipcc -x hip --offload-arch=gfx942 --cuda-device-only \
  -O0 -S -emit-llvm source.hip -o output.ll
```

得到的关键 IR 为：

```llvm
; uint64_t 版本
%next = add i64 %iv, 1
%cond = icmp ult i64 %next, %bound
```

```llvm
; int64_t 版本
%next = add nsw i64 %iv, 1
%cond = icmp slt i64 %next, %bound
```

这验证了三个结论：

1. 两个 source type 都降低为 `i64`；
2. unsigned/signed 比较分别落到 `ult`/`slt`；
3. signed C++ 加法带 `nsw`，因为 signed overflow 在普通 C++ 语义下是 UB；unsigned 加法不带 `nuw`，因为无符号回绕是定义良好的行为。

在 `-O1` 下，这两个小循环还会被进一步折叠为不同 intrinsic：

```llvm
; unsigned_loop
%r = tail call i64 @llvm.umax.i64(i64 %bound, i64 1)

; signed_loop
%r = tail call i64 @llvm.smax.i64(i64 %bound, i64 1)
```

这再次说明 signedness 没有从优化中消失；它被保存在比较、算术 flag 和选择的 intrinsic 中，而非保存在 `%iv` 的静态类型里。

### 为什么 LLVM 将整数设计为 signless

这套设计起初容易让 C/C++ 开发者困惑，因为源语言中：

```cpp
int64_t
uint64_t
```

是不同类型，影响比较、除法、右移和 overflow 行为。但 LLVM IR 处在比 C++ 更低的抽象层：`i64` 首先是一组 64 bit。

例如同一 bit pattern：

```text
0xffffffffffffffff
```

既可按 signed `i64` 解释为 `-1`，也可按 unsigned `i64` 解释为 `18446744073709551615`。硬件寄存器通常没有“这个寄存器永远是 signed”的标签；许多操作，例如 `add`、`sub`、`and`、`or`、`xor`，对两种解释执行完全相同的 bit 运算。

真正依赖解释方式的是具体操作。例如，同一个 `%x` 可以合理地同时参与：

```llvm
%is_negative = icmp slt i64 %x, 0
%is_large = icmp ugt i64 %x, 100
```

若 `%x` 永久带一个 signed 或 unsigned 类型，第二种解释需要制造许多没有 bit 变化的类型转换；而 LLVM 只需让每条操作选择所需语义。

这也减少了 IR type 和 rewrite rule 的重复。若存在独立的 `si64` 与 `ui64`，很多 bit-level 等价的操作都需要两套类型规则；signless `i64` 只在 signedness 真正影响结果时，通过 `slt`/`ult`、`sdiv`/`udiv`、`sext`/`zext`、`ashr`/`lshr`、`nsw`/`nuw` 等明确区分。

我的判断是，这是一个合理的 IR 设计边界：前端负责将 C/C++ 类型系统和 overflow 规则分解为精确的操作级语义，middle end 和 backend 处理统一的 bit-vector 值。代价是读 IR 时不能把 `i64` 自动翻译成“有符号 long”或“无符号 long”，必须查看其 use。

调试信息可能仍记录原始 C++ 声明类型，例如 DWARF 中的 signed/unsigned 编码；但它可以被剥离，也不作为优化器的语义依据。`!tbaa`、`!range` 等 metadata 可提供别名或范围事实，也不取代具体指令的 signed/unsigned 语义。

### 原始 IR 的 loop boundary 问题

在原始形式中：

```llvm
%iv_plus_1 = add i64 %iv, 1  ; 定义在 loop 内
```

却直接在 `end` 中使用：

```llvm
%tmp = add i64 %iv_plus_1, %src
```

因此从 `%iv_plus_1` 的 def-use chain 出发，会同时看到 loop 内 use 和 loop 外 use。对想只处理 loop 内部、或想对 loop exit 做特殊 lowering 的 pass，这种直接跨边界 use 不方便。

### LCSSA 如何加入 exit `phi`

LCSSA 将 `end` 改为：

```llvm
end:
  %iv_plus_1.lcssa = phi i64 [ %iv_plus_1, %loop ]
  %tmp = add i64 %iv_plus_1.lcssa, %src
  %res = add i64 %tmp, %iv_plus_1.lcssa
  ret i64 %res
```

这条 `phi` 应读作：

```text
若控制流从 %loop 进入 %end，
则 %iv_plus_1.lcssa 的值为 %iv_plus_1。
```

`end` 在该例中只有一个 predecessor，即 `%loop`，所以 `phi` 只有一个 incoming pair。它在普通单线程语义上看起来等价于复制：

```text
%iv_plus_1.lcssa = %iv_plus_1
```

它仍然有结构价值。LCSSA 规定：

```text
loop 内定义的值
  不能在 loop 外直接使用
  必须先通过 loop exit block 的 phi 导出
```

这样 loop 外部只使用 `%iv_plus_1.lcssa`，而 loop 内的 `%iv_plus_1` 不再直接有外部 use。exit `phi` 相当于 loop 的显式输出接口。

`phi` 不访问内存、不分配对象，也不是普通算术指令。它在控制流进入 basic block 时，按实际 predecessor 选择对应 incoming value；所有 `phi` 必须位于 basic block 的开头。

### 为什么这对 AMDGPU 有额外价值

LLVM IR 本身描述一个标量函数，并没有写“这个值是 SGPR 还是 VGPR”。AMDGPU backend 的 uniformity analysis、lowering 和寄存器分配会判断：一个 value 在某个程序点是否对 wave 内所有 active lane 相同。

书中假设 `%iv` 由常量初始化，并每轮加相同常量：

```text
所有 active lane：初始 iv = 0
所有 active lane：每轮 iv = iv + 1
```

因此 loop **内部**的 `%iv` 与 `%iv_plus_1` 是 wave-uniform。AMDGPU 可在适合的 subtarget/代码形态下，把它们放在 scalar register（SGPR）中，并以 scalar ALU 指令计算，而不是让每个 lane 各占一个 VGPR。

这里的 uniform 范围是 **wavefront**，不是整个 workgroup：

```text
workgroup
├─ wave 0：有自己的一份 SGPR 状态
├─ wave 1：有自己的一份 SGPR 状态
├─ wave 2：有自己的一份 SGPR 状态
└─ wave 3：有自己的一份 SGPR 状态
```

若整个 workgroup 的 induction value 在逻辑上相同，这当然足以让每个 wave 都使用 SGPR；但后端实际需要的条件只是“同一个 wave 的 active lane 相同”。SGPR 不是 workgroup 中所有线程共用的一份寄存器，也不能作为 wave 间通信媒介。

不同 wave 可以被硬件独立调度。例如同一时刻：

```text
wave 0 的逻辑 s_iv = 37
wave 1 的逻辑 s_iv = 12
wave 2 暂未调度
wave 3 的逻辑 s_iv = 55
```

这不表示同一个 SGPR 被不同线程竞争写入；每个 resident wave 有独立的寄存器上下文。对同一 wave 而言，scalar 指令只执行一次，所有 active lane 使用该单一 SGPR 值。

### divergent upper bound 为什么让 loop 外值变成 per-lane 值

书中进一步假设 `%upper_bound` 可以随线程/lane 而异。例如：

```text
lane 0：upper_bound = 2
lane 1：upper_bound = 5
lane 2：upper_bound = 3
```

执行过程可概念化为：

```text
loop 内当前 scalar iv_plus_1 = 1：三个 lane 都继续
loop 内当前 scalar iv_plus_1 = 2：lane 0 离开，lane 1、2 继续
loop 内当前 scalar iv_plus_1 = 3：lane 2 离开，lane 1 继续
loop 内当前 scalar iv_plus_1 = 5：lane 1 离开
```

循环内，仍 active 的 lane 共享当前 iteration count，因此计数器可保持 uniform/SGPR。可是每个 lane 离开时需要保留不同的最终值：

```text
lane 0 的 loop 输出 = 2
lane 1 的 loop 输出 = 5
lane 2 的 loop 输出 = 3
```

这时 `%iv_plus_1.lcssa` 在 loop 外是 divergent 的 per-lane 结果，通常需要 VGPR 或等价的 per-lane 表示。

LCSSA `phi` 没有直接执行“SGPR 转 VGPR”的硬件操作；它将“loop 内 uniform value”与“loop 外每 lane 的 exit value”之间的 SSA 边界明确表达出来。AMDGPU lowering 可以利用这个边界：在每个 lane 从 loop exit edge 离开时，在其 active mask 下捕获当前 scalar iteration value，形成供 loop 外使用的 per-lane value。

若 `%upper_bound` 也被证明 uniform，则所有 lane 同时退出，`%iv_plus_1.lcssa` 也可保持 uniform；此时没有必然要转为 VGPR 的理由。实际 SGPR/VGPR 选择仍取决于后端分析、register pressure、子目标能力和完整 use-def 图。

### workgroup 级通信仍要使用 LDS

若需求是：

```text
wave 0 计算动态值
  -> wave 1、2、3 读取同一值
```

不能通过 SGPR 实现。通常需要：

```text
wave 0 写入 LDS
  -> workgroup barrier
  -> 其他 wave 从 LDS 读取
```

对 loop counter、kernel 参数等计算便宜的 uniform 值，每个 wave 独立在自己的 SGPR 中重算通常更合适。

# CGSCC 在inline里的作用


先把函数调用关系画成有向图：

```
A -> B
A -> C
B -> D
C -> D
```

箭头表示：

```
caller -> callee
```

这里没有递归，因此每个函数自己就是一个 SCC：

```
{A}, {B}, {C}, {D}
```

inliner 会先从叶子开始：

```
D
↑
B, C
↑
A
```

也就是先处理 `D`，再处理调用 `D` 的 `B` 和 `C`，最后才处理 `A`。

这样做的直觉是：

> 先让更底层的 callee 完成内联和优化，再判断它是否值得被上层 caller 内联。

例如 `B` 先把 `D` 内联或优化后，`B` 的体积、常量信息、调用数量都会变化；之后判断是否把 `B` 内联进 `A` 会更准确。

***

“强连通”处理的是递归或互相递归。

```
foo -> bar
 ^      |
 |______|
```

从 `foo` 能到 `bar`，也从 `bar` 能回到 `foo`。因此：

```
{foo, bar}
```

是一个 CGSCC。

它们不能被简单排成：

```
先 foo 后 bar
```

或：

```
先 bar 后 foo
```

因为两者互相依赖。LLVM 将它们视为一个整体区域来处理。

自递归也是 SCC：

```
int fact(int n) {
  return n <= 1 ? 1 : n * fact(n - 1);
}
```

调用图：

```
fact -> fact
```

因此：

```
{fact}
```

是一个带自环的 CGSCC。

***

可以把调用图先压缩成“每个 SCC 一个节点”的 DAG：

```
原 call graph：

A -> B -> D
 \    \
  -> C -> D

压缩 SCC 后仍是 DAG：

{A} -> {B} -> {D}
  \      \
   -> {C} -> {D}
```

若有递归：

```
main -> foo <-> bar -> helper
```

则压缩后：

```
{main} -> {foo, bar} -> {helper}
```

这时 inliner 的处理顺序接近：

```
{helper}
    ↑
{foo, bar}
    ↑
{main}
```

***

书中这句话：

> This particular inliner uses the regions formed by the call graph strongly connected component (CGSCC) to determine the order in which inline decisions are made, starting from the leaves of the call graph and moving up to their parents.

可以翻成：

> 这个 inliner 将调用图按强连通分量分组，以此决定内联决策顺序。它先处理调用图底部的 callee，再逐步处理调用它们的 caller；互相递归的一组函数作为一个整体处理。

这里的 `region` 不是内存区域，而是：

```
调用图中一组互相可达的函数
```

也就是一个 CGSCC。

## Induction Variables Simplification：为什么循环 IR 能变成 `umax`

书中的 `IndVarSimplifyPass`（命令行名 `indvars`）专门处理 induction variable 及其派生值。它不一定删除整个 loop；它的目标是把“每轮更新 induction variable 才能得到的值”改写成更容易被其他优化使用的表达式。

书中的输入 IR 是：

```llvm
define i64 @foo(i64 %src, i64 %ub) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv1, %loop ]
  %iv1 = add i64 %iv, 1
  %cond = icmp ult i64 %iv1, %ub
  br i1 %cond, label %loop, label %end

end:
  %tmp = add i64 %iv1, %src
  %res = add i64 %tmp, %iv1
  ret i64 %res
}
```

### 先看这个 loop 实际算了什么

`%iv` 初始为 `0`。每轮：

```text
%iv1 = %iv + 1
若 %iv1 <u %ub，继续下一轮
否则离开 loop
```

`<u` 来自 `icmp ult`，所以比较是 unsigned。这个 loop 的退出值可直接列出来：

| `%ub` | `%iv1` 依次取得的值 | 离开 loop 时的最终 `%iv1` |
| --- | --- | --- |
| `0` | `1` | `1` |
| `1` | `1` | `1` |
| `2` | `1, 2` | `2` |
| `5` | `1, 2, 3, 4, 5` | `5` |

因此最终值是：

```text
final_iv1 = unsigned_max(%ub, 1)
```

也就是 LLVM intrinsic：

```llvm
%umax = call i64 @llvm.umax.i64(i64 %ub, i64 1)
```

书中所谓 “loop trip count can be determined statically” 容易被理解为“编译期已经知道具体循环次数”。这里更准确的意思是：编译器可以从 induction variable、步长 `+1`、初值 `0` 和 exit condition 推导出一个**符号公式**。`%ub` 仍然是运行时参数，但最终 `%iv1` 可在不执行 loop 的情况下由 `%ub` 表示。

用接近源代码的写法，原函数相当于：

```cpp
uint64_t foo(uint64_t src, uint64_t ub) {
  uint64_t iv = 0;
  uint64_t iv1;
  do {
    iv1 = iv + 1;
    iv = iv1;
  } while (iv1 < ub);

  return iv1 + src + iv1;
}
```

在该 loop 没有任何其他副作用时，可改成：

```cpp
uint64_t foo(uint64_t src, uint64_t ub) {
  uint64_t final_iv1 = std::max(ub, uint64_t{1});
  return final_iv1 + src + final_iv1;
}
```

### 为什么变换后的 IR 中还留着一个假 loop

书中展示的 IndVarSimplify 输出是：

```llvm
define i64 @foo(i64 %src, i64 %ub) {
entry:
  br label %loop

loop:
  br i1 false, label %loop, label %end

end:
  %umax = call i64 @llvm.umax.i64(i64 %ub, i64 1)
  %tmp = add i64 %umax, %src
  %res = add i64 %tmp, %umax
  ret i64 %res
}
```

`loop` 中的：

```llvm
br i1 false, label %loop, label %end
```

表示回边永远不走，控制流总是进入 `%end`。它是这个 pass 改写后的临时 CFG 形态：IndVarSimplify 已删除 loop 内所有有意义的计算，并将循环结果搬到 `%end`，但它不负责做所有死 block / CFG 清理。

后续的 `SimplifyCFG`、DCE 或类似 pass 可以继续将它化简为：

```llvm
entry:
  br label %end
```

甚至把不再需要的 `%loop` block 完全删除。LLVM pipeline 中常将这种职责拆给多个小 pass：一个 pass 专注数学和 induction-variable 推导，另一个 pass 专注 CFG 清理。

### 为什么这个优化合法

原 loop 唯一可观察到的效果是计算最终 `%iv1`：

```text
没有 load/store
没有 call
没有 volatile/atomic
没有可能被外部观察的 side effect
```

因此，只要新表达式在所有合法输入上产生同样的 `%iv1`，就可以删除实际迭代。

如果 loop body 有副作用，例如：

```llvm
call void @log_iteration(i64 %iv1)
store i64 %iv1, ptr %p
```

则不能将 loop 简单替换为 `umax`，因为每轮执行本身已可被观察。IndVarSimplify 仍可能简化 induction variable 的形式，但不会以这种方式删除副作用。

### `llvm.umax.i64` 不是普通外部函数调用

```llvm
%umax = call i64 @llvm.umax.i64(i64 %ub, i64 1)
```

文本写作 `call`，但 `llvm.umax.i64` 是 LLVM intrinsic。它表达 unsigned max 语义，后续可以被继续优化或由 backend lower 为 target 适合的 compare/select、max 指令或其他序列。

这里使用 unsigned max 是由原始：

```llvm
icmp ult i64 %iv1, %ub
```

决定的。若 exit compare 是 signed `<`，相应的闭式表达和 intrinsic 会不同；不能机械地把任意 induction loop 都改写为 `umax`。

### TargetLibraryInfo 与 TargetTransformInfo 的作用

书中指出该 pass 使用两类 target-aware analysis：

- `TargetLibraryInfo`：判断相关 library call 是否有副作用、是否可安全移除或改写；
- `TargetTransformInfo`：比较原 loop 与新表达式在当前 target 上的成本，决定改写是否划算。

所以这个 pass 不是只做数学恒等式证明。它先证明语义等价，再根据 target 的成本模型决定是否采用某种具体改写。

对 x86 和 AMDGPU，这个 IR-level 证明相同；但 `llvm.umax.i64` 的 lowering 成本、寄存器压力和最终机器指令可以不同，因此 TTI 参与决策。

## Loop Strength Reduction：将 `A[i]` 的地址计算改成适合 target 的形态

书中的 `LoopStrengthReducePass`，命令行名为 `loop-reduce`。它处理的是 loop 内由 induction variable 驱动的地址计算。

“strength reduction” 的一般含义是：用语义相同、硬件成本更低或更容易组合的计算替换原计算。例如：

```text
x * 2   ->   x << 1
```

但 Loop Strength Reduction 的重点通常不是普通算术本身，而是：

```text
数组元素的地址如何由 loop index 计算出来
```

### `A[i]` 在机器地址层面是什么意思

假设 C/C++ 有：

```cpp
int64_t value = A[i];
```

源级的数组下标表达式是：

```cpp
A[i]
```

按 C/C++ 指针算术的语义，它等价于：

```cpp
*(A + i)
```

但机器内存按 byte 编址。若 `A` 是 `int64_t*`，一个元素占 8 bytes，因此实际访问地址是：

```text
address(A[i]) = base_address(A) + i * sizeof(int64_t)
              = base_address(A) + i * 8
```

例如：

```text
A 的 base address = 0x1000
i = 0  -> A[0] 位于 0x1000
i = 1  -> A[1] 位于 0x1008
i = 2  -> A[2] 位于 0x1010
i = 3  -> A[3] 位于 0x1018
```

所以书中说每前进一个元素，地址会前进多个 bytes；这个 `* 8` 就是 addressing mode 中的 scaling factor。

### 书中的输入 IR

书中示例遍历 `%arg` 指向的 `i64` 数组：若在 index 小于 `%ub` 的范围中找到第一个非零元素，就返回其 index；若没有找到则返回 `-1`。

核心 loop body 是：

```llvm
bb4:
  %i5 = getelementptr inbounds i64, ptr %arg, i64 %idx
  %i6 = load i64, ptr %i5
  %i7 = icmp ne i64 %i6, 0
  br i1 %i7, label %bb10, label %bb8
```

其中：

```llvm
%i5 = getelementptr inbounds i64, ptr %arg, i64 %idx
```

语义上是：

```text
%i5 = %arg + %idx * sizeof(i64)
    = %arg + %idx * 8
```

`getelementptr` 在 LLVM IR 中保留“按 `i64` 元素索引”的抽象语义；后端最终仍需形成 byte address。

### Loop Strength Reduction 后的 IR

书中展示的相关改写是：

```llvm
bb4:
  %0 = shl i64 %idx, 3
  %scevgep = getelementptr i8, ptr %arg, i64 %0
  %i6 = load i64, ptr %scevgep
```

逐项对应：

```llvm
%0 = shl i64 %idx, 3
```

表示：

```text
%0 = %idx << 3
   = %idx * 8
```

随后：

```llvm
%scevgep = getelementptr i8, ptr %arg, i64 %0
```

因为 GEP 的元素类型现在是 `i8`，每个元素正好是 1 byte：

```text
%scevgep = %arg + %0 * sizeof(i8)
          = %arg + %0 * 1
          = %arg + %idx * 8
```

因此改写前后访问的地址完全一样：

```text
getelementptr i64, %arg, %idx
==
getelementptr i8, %arg, (%idx << 3)
```

### 为什么看起来多了指令，却可能更好

表面上：

```llvm
getelementptr i64, ptr %arg, i64 %idx
```

是一条 IR 指令；改写后却有：

```llvm
shl
getelementptr i8
```

因此不能只按 LLVM IR 指令数量判断成本。Loop Strength Reduction 的目的，是将地址表达式变成后端容易识别和匹配的形式。

很多 target 的 load/store addressing mode 能直接处理类似：

```text
base + index * scale + constant_offset
```

其中 `scale` 常是 `1`、`2`、`4`、`8`。对于 `i64` 数组，`%idx << 3` 明确表达了 scale 为 8；后端可能把 shift、base add 和 load 合并进一条或少量目标指令，而不需要生成独立 multiply。

书中称 shift 的 strength 比 multiply 低，是因为对于 2 的幂常量：

```text
idx * 8
==
idx << 3
```

shift 常更便宜，或更容易折叠到目标机寻址模式中。现代硬件上常量乘法有时也很便宜，因此这个 pass 不假设 shift 永远胜出；它通过 target 的成本与寻址能力决定具体形态。

### 这个 pass 还可能做什么

更一般地，LSR 可能将：

```text
base + i * stride
```

变为递推地址：

```text
p = base
每轮：使用 p，然后 p = p + stride
```

或者将多个共享同一 induction variable 的地址表达式组合成可复用 base/offset 形式。书中这个例子选择显式 `shl + byte GEP`，并不表示所有 loop 都会改成指针递推。

### 书中输出里其他变化的原因

书中 Loop Strength Reduction 输出还出现：

- `%idx.lcssa1 = phi ...`：将 loop 内 `%idx` 通过 LCSSA 形式导出，方便 exit block 使用；
- `bb10split` 与 `bb4.bb10_crit_edge`：分裂 critical edge，让插入新的计算或 `phi` incoming value 更安全、更简单；
- `getelementptr i8`：将 scaling factor 从 GEP 元素类型中取出，显式表示为 shift。

这些变化是为 address-expression rewrite 和 SSA/CFG 维护服务，核心语义仍是“找到第一个非零 `i64` 元素并返回 index”。

### TargetLoweringInfo 与 TTI 为什么参与

书中指出这个 pass 使用：

- `TargetLoweringInfo`：检查某种地址模式和相应操作在 target 上是否合法；
- `TargetTransformInfo`：评估原始与新地址表达式的成本，决定改写是否划算。

因此它需要一个实际 target triple。没有 target，pass 无法可靠判断：

```text
这个 target 支不支持 base + index * 8？
应保留 element-scaled GEP，还是显式 shift？
一次额外 add/shl 是否值得？
```

对 x86，典型寻址模式天然接近 `base + index * {1,2,4,8} + offset`。对 AMDGPU，global/LDS/private address space 以及 scalar/vector address 的 lowering 规则不同；LSR 仍会尝试产生适合后续 AMDGPU codegen 的地址计算形态，但最终是否融合为特定 ISA 指令取决于 subtarget、address space、uniformity 和后端选择。

### 实操结论

读 `A[i]` 时，应先把它翻译为：

```text
base + index * element_size
```

Loop Strength Reduction 的工作不是改变访问哪个元素，而是改变这条 byte address 计算在 IR 中的表达方式，让后续 target-specific codegen 能选择更高效的寻址模式。

## Loop Unrolling：复制循环体，用更少的迭代完成同样工作

书中的 `LoopUnrollPass`，命令行名为 `loop-unroll`。loop unrolling 的含义是：将原 loop body 复制多份，使新 loop 的每一轮完成原 loop 的多轮工作。

原始概念代码：

```cpp
for (int i = 0; i < n; ++i)
  body(i);
```

若 unroll factor 是 2，概念上可变成：

```cpp
int i = 0;
for (; i + 1 < n; i += 2) {
  body(i);
  body(i + 1);
}
if (i < n)
  body(i);  // 处理奇数个元素留下的 remainder
```

新 loop 的每轮做两次原 body，循环控制、branch 和 induction-variable update 的次数大约减半。

### 书中的输入：在前三个元素中找第一个非零值

书中将之前“查找数组第一个非零元素”的上界固定为常量 `3`：

```llvm
bb3:
  %idx = phi i64 [ 0, %bb ], [ %i9, %bb8 ]
  %i = icmp slt i64 %idx, 3
  br i1 %i, label %bb4, label %bb10

bb4:
  %i5 = getelementptr inbounds i64, ptr %arg, i64 %idx
  %i6 = load i64, ptr %i5
  %i7 = icmp ne i64 %i6, 0
  br i1 %i7, label %bb10, label %bb8

bb8:
  %i9 = add nsw i64 %idx, 1
  br label %bb3
```

源级意图接近：

```cpp
int64_t foo(const int64_t *arg) {
  for (int64_t idx = 0; idx < 3; ++idx) {
    if (arg[idx] != 0)
      return idx;
  }
  return -1;
}
```

因为 trip count 已知为 3，unroller 可以做 full unroll：不保留实际循环，直接写出三次访问和检查。

概念上，结果变成：

```cpp
int64_t foo(const int64_t *arg) {
  if (arg[0] != 0)
    return 0;
  if (arg[1] != 0)
    return 1;
  if (arg[2] != 0)
    return 2;
  return -1;
}
```

这仍严格保留原来“返回**第一个**非零元素 index”的顺序。

### 书中输出为什么有 `bb4.1`、`bb4.2` 和 `phi`

书中输出的片段类似：

```llvm
bb4:
  %i6 = load i64, ptr %arg
  %i7 = icmp ne i64 %i6, 0
  br i1 %i7, label %bb10, label %bb8

bb4.1:
  %i5.1 = getelementptr inbounds i64, ptr %arg, i64 1
  %i6.1 = load i64, ptr %i5.1
  %i7.1 = icmp ne i64 %i6.1, 0
  br i1 %i7.1, label %bb10, label %bb8.1
```

`bb4` 是原 body 的第 0 次展开，`bb4.1` 是第 1 次展开，后面还有对应第 2 次展开的 block。每个 block 都检查各自固定 index 的元素：

```text
bb4    -> arg[0]
bb4.1  -> arg[1]
bb4.2  -> arg[2]
```

任何一次发现非零元素，都直接跳到 `%bb10`。因为 `%bb10` 现在可能从多个展开副本进入，它需要 `phi` 记录究竟命中了哪个 index：

```llvm
bb10:
  %res = phi i64 [ 0, %bb4 ],
                  [ 1, %bb4.1 ],
                  [ 2, %bb4.2 ],
                  [ -1, %no_match_path ]
  ret i64 %res
```

它等价于：

```text
若从第 0 次检查命中，返回 0
若从第 1 次检查命中，返回 1
若从第 2 次检查命中，返回 2
若全都未命中，返回 -1
```

书中完整输出里的 `.3` block、`unreachable` 和额外 CFG block 是 unroller 为保持循环 exit/边界条件语义产生的中间结构。后续 `SimplifyCFG`、DCE 等 pass 往往还能进一步清理它们；理解 full unroll 的重点是“原 loop body 被按每个已知 index 展开”，而不是 block 名称本身。

### full unroll 与 partial unroll

| 形式 | 条件 | 结果 |
| --- | --- | --- |
| Partial unroll | trip count 未知、过大，或不值得完全复制 | 复制 body 若干份，仍保留 loop 和 remainder handling。 |
| Full unroll | trip count 很小且已知，或成本模型认为可接受 | 删除 loop，写成有限个 straight-line / branch block。 |

若 trip count 是常量 3，full unroll 很自然。若 trip count 是运行时 `n`，通常只能 partial unroll，除非分析能证明更强的范围信息。

### 为什么 unrolling 可能更快

unrolling 的直接收益包括：

- 减少 loop branch、比较和 induction-variable update；
- 让相邻迭代的 load、算术和 store 同时出现在更大的 basic block 中；
- 暴露更多 instruction-level parallelism，帮助 scheduler 重排独立操作；
- 暴露连续内存访问和多个相似算术操作，帮助后续 vectorizer；
- 让常量 index、constant GEP offset、dead branch 等进一步折叠。

它不是 vectorization：unrolling 只是复制标量 loop body。vectorizer 若随后把多份标量操作合成一个 vector operation，才是 SIMD/vectorization。

### 为什么不能无限展开

复制代码也有成本：

- machine code 变大，增加 I-cache 压力；
- 更多同时存活的值可能增加寄存器压力；
- 更多 register spill 可能反而变慢；
- compile time 增加；
- 对含复杂控制流、调用或异常处理的 loop，展开收益可能很低。

AMDGPU 上尤其要关注 VGPR 使用量。过度 unroll 会让一个 work-item 同时保留更多值，增加 VGPR pressure；若超过某个资源阈值，wave occupancy 可能下降，最终比保留 loop 更慢。

### TTI 如何参与决策

书中指出 `LoopUnrollPass` 使用 `TargetTransformInfo` 估计 unroll 成本并决定 factor。target 还可通过：

```cpp
TargetTransformInfo::getUnrollingPreferences
```

调整阈值与偏好。

因此，不存在一个对所有 target 都正确的 “unroll factor = 4” 或 “一定 full unroll” 规则：

```text
x86：可能考虑 branch 成本、SIMD 宽度、I-cache、寄存器数量
AMDGPU：还会考虑 VGPR/SGPR pressure、occupancy、wave 执行和访存隐藏延迟
```

Loop unrolling 的正确性由原 loop 的控制流、trip count、依赖和副作用保证；TTI 主要决定这样做在当前 target 上是否值得。

### `phi` 中的 `%i9` 为何可以在文本后面定义

书中未展开前的 loop header 包含：

```llvm
bb3:
  %idx = phi i64 [ 0, %bb ], [ %i9, %bb8 ]
  %i = icmp slt i64 %idx, 3
  br i1 %i, label %bb4, label %bb10
```

而 `%i9` 在文本后面的 loop latch block 中定义：

```llvm
bb8:
  %i9 = add nsw i64 %idx, 1
  br label %bb3
```

这不是“先声明 `%i9`、以后再赋值”，也不是未定义值。LLVM IR 中 `phi` 的每个 incoming pair：

```llvm
[ value, predecessor ]
```

应读作：

```text
若控制流从 predecessor 进入当前 block，
则 phi 的结果取 value。
```

因此：

```llvm
%idx = phi i64 [ 0, %bb ], [ %i9, %bb8 ]
```

表示：

```text
从 %bb  进入 %bb3：%idx = 0
从 %bb8 进入 %bb3：%idx = %i9
```

第一次进入 loop 时，控制流为：

```text
%bb -> %bb3
```

所以：

```text
%idx = 0
```

随后执行 loop body，并到达 `%bb8`：

```text
%bb3 -> %bb4 -> %bb8
%i9 = %idx + 1
%bb8 -> %bb3
```

第二次进入 `%bb3` 时，实际 predecessor 已经是 `%bb8`，而 `%i9` 已在 `%bb8` 中算出：

```text
第一次：%idx = 0
          %i9 = 0 + 1

第二次：%idx = %i9 = 1
          %i9 = 1 + 1

第三次：%idx = %i9 = 2
```

### 文本顺序不等于 SSA 定义顺序

LLVM parser 允许 forward reference，所以 `%i9` 可以在文本中先被写进 `phi`，再在后面的 `%bb8` 定义。IR 是否合法由 CFG 和 SSA 规则决定，而不是单纯由文本行号决定。

普通 SSA use 需要 definition 支配 use。但 `phi` 是特殊情况：它的 incoming value 被视为在对应 predecessor edge 上使用。因此：

```llvm
[ %i9, %bb8 ]
```

合法的条件是：

```text
%bb8 是 %bb3 的实际 predecessor
%i9 在 %bb8 跳到 %bb3 前已经定义
```

这正是当前 IR 所满足的关系。

以下情形则会非法：

```text
整个函数中根本没有定义 %i9
  -> parser / verifier：use of undefined value

%bb8 不是 %bb3 的 predecessor
  -> verifier：phi incoming block 与 CFG 不匹配

%i9 不能在 %bb8 -> %bb3 这条 edge 上保证有效
  -> verifier：违反 phi 的 SSA/dominance 规则
```

所以阅读 `phi` 时，先看 `[value, predecessor]` pair 和 CFG edge；不要按文字中 value 定义出现的先后顺序判断它是否已定义。

## Load/Store Vectorizer：`<2 x i64>`、`extractelement` 与 lane index

书中 LoadStoreVectorizer 的输入是两个相邻 `i64` load 与两个相邻 `i64` store：

```llvm
define void @bar(ptr %src, ptr %dst) {
  %v0 = load i64, ptr %src
  %src1 = getelementptr i64, ptr %src, i64 1
  %v1 = load i64, ptr %src1
  store i64 %v0, ptr %dst
  %dst1 = getelementptr i64, ptr %dst, i64 1
  store i64 %v1, ptr %dst1
  ret void
}
```

它的源级意图接近：

```cpp
dst[0] = src[0];
dst[1] = src[1];
```

因为元素是 `i64`，两个连续元素占：

```text
2 * 8 bytes = 16 bytes = 128 bits
```

vectorizer 将两次连续 load 合并为：

```llvm
%1 = load <2 x i64>, ptr %src
```

`<2 x i64>` 的含义是：

```text
一个 vector value
├─ lane 0：一个 i64，对应 src[0]
└─ lane 1：一个 i64，对应 src[1]
```

它不是“两个 `i64` 放进一个 `i32`”。这个 vector 总宽度是：

```text
2 lanes * 64 bits/lane = 128 bits
```

### `extractelement` 中的 `i32 0` 是索引，不是元素类型

书中的优化后 IR 有：

```llvm
%v01 = extractelement <2 x i64> %1, i32 0
%v12 = extractelement <2 x i64> %1, i32 1
```

`extractelement` 的一般形态是：

```llvm
%scalar = extractelement <N x element_type> %vector, integer %lane_index
```

所以第一条应读作：

```text
从 %1 这个 <2 x i64> vector 中，
取 lane 编号 0，
结果赋给 %v01。
```

类型关系是：

```text
%1    : <2 x i64>
i32 0 : lane index 的整数常量
%v01 : i64
```

也就是说，`i32 0` 只是“取第 0 个元素”的下标，类似 C++ 的：

```cpp
v[0]
```

它并不表示 `%v01` 是 `i32`，也不表示向量只能存一个 `i32`。

对应关系为：

```text
%1 = <i64 src[0], i64 src[1]>

extractelement %1, 0  ->  i64 src[0]
extractelement %1, 1  ->  i64 src[1]
```

### 为什么又 `insertelement` 回去

书中接着有：

```llvm
%2 = insertelement <2 x i64> poison, i64 %v01, i32 0
%3 = insertelement <2 x i64> %2, i64 %v12, i32 1
store <2 x i64> %3, ptr %dst
```

`insertelement` 的操作数顺序固定是：

```llvm
%result = insertelement <vector-type> %old_vector,
                         <element-type> %new_element,
                         <integer-type> %lane_index
```

因此第一条：

```llvm
%2 = insertelement <2 x i64> poison, i64 %v01, i32 0
```

每个部分的角色是：

| 片段 | 含义 |
| --- | --- |
| `<2 x i64>` | 结果和第一个操作数都是两个 `i64` lane 的 vector。 |
| `poison` | 初始 vector；概念上它的两个 lane 都尚未有可安全使用的值。 |
| `i64 %v01` | 要写入某个 lane 的 scalar 元素，类型必须正好是 vector element type `i64`。 |
| `i32 0` | lane index；表示覆盖 lane 0。它只是索引，类型 `i32` 与 lane 内数据宽度无关。 |

所以第一条的逐 lane 效果是：

```text
初始 poison vector：< poison, poison >
在 index 0 插入 %v01：%2 = < %v01, poison >
```

第二条：

```llvm
%3 = insertelement <2 x i64> %2, i64 %v12, i32 1
```

以 `%2` 作为旧 vector，保留 `%2` 的所有 lane，只覆盖 index 1：

```text
%2 的 lane 0 保留：%v01
%2 的 lane 1 被覆盖：%v12

%3 = < %v01, %v12 >
```

因此 `%3` **确实依赖 `%2`**。若第二条又从 `poison` 开始：

```llvm
; 错误的构造方式：会丢失 lane 0
%3 = insertelement <2 x i64> poison, i64 %v12, i32 1
```

那么结果会是：

```text
< poison, %v12 >
```

而不是需要的：

```text
< %v01, %v12 >
```

`poison` 在这里不是一块已分配但未初始化的内存，也不是“随机数”。它是 LLVM IR 的特殊值，表示该 lane 在被安全地使用前没有有效语义。`insertelement` 的规则是：结果继承旧 vector 的所有 lane，只在指定 index 用新 scalar 覆盖。因此，从 `poison` vector 开始、逐个插入所有 lane，是构造完整 vector 的标准写法。

本例中：

```text
%2：lane 1 仍为 poison，但 %2 只作为下一条 insertelement 的输入
%3：lane 0、lane 1 都已被有效 i64 覆盖
```

所以 `%3` 可以安全地作为完整 `<2 x i64>` value 写入 `%dst`。若只构造了 `< %v01, poison >` 就把它当作一个已定义的完整 vector 使用，poison 会传播；store 本身可以保存 poison，但后续读取/控制流等使用不会得到一个可依赖的已定义 lane 值，也无法实现这里应有的数组复制语义。

因为两个 lane 最终都已填入有效 `i64`，所以：

```text
%3 = < src[0], src[1] >
```

最终：

```llvm
store <2 x i64> %3, ptr %dst
```

一次写入两个连续 `i64`：

```text
dst[0] = src[0]
dst[1] = src[1]
```

在这个被裁剪的简单示例中，`extractelement` 再 `insertelement` 的结果实际上重建了原 vector `%1`。后续 InstCombine 有机会继续看出：

```text
%3 == %1
```

并把它简化为：

```llvm
%1 = load <2 x i64>, ptr %src
store <2 x i64> %1, ptr %dst
```

vectorizer 产生显式 extract/insert 的原因是它需要保留 lane 与原 scalar load/store 的映射，并适用于更一般的情形：中间 lane 可能被计算、重排、掩码选择或与其他 scalar value 组合。后续 canonicalization pass 可再清理在简单场景中变得冗余的构造。

## Loop Vectorizer：一轮处理八个数组元素

`LoopVectorizePass`，命令行名 `loop-vectorize`，与前面的 Load/Store Vectorizer 和 SLP Vectorizer 不同：它沿着 loop 的 induction variable，将**不同迭代**中的标量操作合并成一次 vector 操作。

书中的输入 loop 核心是：

```llvm
bb3:
  %idx = phi i64 [ 0, %bb ], [ %i14, %bb4 ]
  %i = icmp ne i64 %idx, 24
  br i1 %i, label %bb4, label %bb15

bb4:
  %i5 = getelementptr inbounds i16, ptr %arg1, i64 %idx
  %i6 = load i16, ptr %i5
  %i7 = sext i16 %i6 to i32
  %i8 = getelementptr inbounds i16, ptr %arg2, i64 %idx
  %i9 = load i16, ptr %i8
  %i10 = sext i16 %i9 to i32
  %i11 = add nsw i32 %i7, %i10
  %i12 = trunc i32 %i11 to i16
  %i13 = getelementptr inbounds i16, ptr %arg, i64 %idx
  store i16 %i12, ptr %i13
  %i14 = add nsw i64 %idx, 1
  br label %bb3
```

源级意图近似：

```cpp
for (int64_t i = 0; i != 24; ++i) {
  int32_t sum = int32_t(arg1[i]) + int32_t(arg2[i]);
  arg[i] = int16_t(sum);
}
```

也就是：

```text
arg[i] = arg1[i] + arg2[i]，共处理 24 个 i16 元素
```

### vectorized loop 的主要部分

书中 vectorizer 生成：

```llvm
vector.body:
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = add i64 %index, 0

  %1 = getelementptr inbounds i16, ptr %arg1, i64 %0
  %2 = getelementptr inbounds i16, ptr %1, i32 0
  %wide.load = load <8 x i16>, ptr %2

  %3 = getelementptr inbounds i16, ptr %arg2, i64 %0
  %4 = getelementptr inbounds i16, ptr %3, i32 0
  %wide.load1 = load <8 x i16>, ptr %4

  %5 = add <8 x i16> %wide.load, %wide.load1

  %6 = getelementptr inbounds i16, ptr %arg, i64 %0
  %7 = getelementptr inbounds i16, ptr %6, i32 0
  store <8 x i16> %5, ptr %7

  %index.next = add nuw i64 %index, 8
  %8 = icmp eq i64 %index.next, 24
  br i1 %8, label %middle.block, label %vector.body
```

`<8 x i16>` 表示八个 16-bit lane：

```text
8 lanes * 16 bits = 128 bits = 16 bytes
```

第一轮 `%index = 0` 时：

```text
%wide.load  = < arg1[0], arg1[1], ..., arg1[7] >
%wide.load1 = < arg2[0], arg2[1], ..., arg2[7] >

%5 = < arg1[0] + arg2[0],
       arg1[1] + arg2[1],
       ...,
       arg1[7] + arg2[7] >
```

然后一次 vector store 写入：

```text
arg[0] 到 arg[7]
```

第二轮 `%index = 8` 时处理：

```text
arg[8] 到 arg[15]
```

第三轮 `%index = 16` 时处理：

```text
arg[16] 到 arg[23]
```

因此：

```text
原 loop：24 次标量迭代，每轮处理 1 个元素
vector loop：3 次向量迭代，每轮处理 8 个元素
```

### 为什么 `%index.next` 加 8

标量 loop 中：

```llvm
%i14 = add nsw i64 %idx, 1
```

每轮前进一个元素。vectorized loop 每轮已处理 8 个元素，因此：

```llvm
%index.next = add nuw i64 %index, 8
```

这里的 `nuw` 表达：在该 loop 的已知范围中，`%index + 8` 不会发生 unsigned wrap。索引依次为：

```text
0 -> 8 -> 16 -> 24
```

因为 `24` 恰好能被 vector width `8` 整除，比较：

```llvm
icmp eq i64 %index.next, 24
```

在第三轮后为 true，vector loop 退出，不需要处理 scalar remainder。

若 trip count 不是 8 的整数倍，例如 26，vectorizer 通常生成：

```text
vector loop：处理 0..23
scalar remainder loop：处理 24..25
```

`vector.ph`、`middle.block` 等 block 就是这种 vector loop、退出和可能的 remainder loop 之间的通用 CFG 脚手架。书中输出被裁剪，未展示全部部分。

### 为什么 scalar 的 `sext + add i32 + trunc` 变为 `add <8 x i16>`

原标量计算是：

```llvm
%i7 = sext i16 %i6 to i32
%i10 = sext i16 %i9 to i32
%i11 = add nsw i32 %i7, %i10
%i12 = trunc i32 %i11 to i16
```

`sext` 把两个 `i16` 的 signed 值扩展到 `i32` 后相加；最后 `trunc` 只保存结果低 16 bit。两个 signed `i16` 的和范围在：

```text
-65536 到 65534
```

不会让 `i32` 溢出，因此这里的 `nsw` 不会改变合法结果。最终只写低 16 bit 时，结果 bit pattern 与直接做 16-bit 加法相同：

```llvm
%5 = add <8 x i16> %wide.load, %wide.load1
```

该 vector add 是逐 lane 的：

```text
lane k 的结果 = arg1[index + k] + arg2[index + k]
```

这不是把八个值相加为一个值，也不是 reduction。

### `%0 = add %index, 0` 与两层 GEP 为什么看似多余

书中输出有：

```llvm
%0 = add i64 %index, 0
%1 = getelementptr inbounds i16, ptr %arg1, i64 %0
%2 = getelementptr inbounds i16, ptr %1, i32 0
```

在该裁剪示例中：

```text
%0 等于 %index
GEP ..., 0 等于原地址
```

它们看起来可以删掉。vectorizer 常先生成统一的 vector loop 地址/CFG 形态，以便处理更复杂的 offset、runtime alias check、alignment、remainder 与 predication 情况；后续 InstCombine、SimplifyCFG 等 pass 可继续清理本例中变成 identity 的操作。

因此，vectorizer 输出不一定是最终最精简 IR，而是后续优化 pipeline 的输入。

#### 逐条解读 `%1` 和 `%2` 两次 GEP

书中的代码是：

```llvm
%1 = getelementptr inbounds i16, ptr %arg1, i64 %0
%2 = getelementptr inbounds i16, ptr %1, i32 0
%wide.load = load <8 x i16>, ptr %2
```

第一条：

```llvm
%1 = getelementptr inbounds i16, ptr %arg1, i64 %0
```

应按 byte address 读作：

```text
%1 = address(%arg1) + %0 * sizeof(i16)
   = address(%arg1) + %0 * 2 bytes
```

其中：

```text
i16  是 GEP 的元素类型，决定每个 index 单位的 stride 为 2 bytes
i64  是 index operand %0 的整数类型，不能当作乘数或 element size
```

因此，若第一轮：

```text
%0 = %index = 0
```

则：

```text
%1 = %arg1 + 0 * 2 = %arg1
```

第二轮：

```text
%0 = %index = 8
```

则：

```text
%1 = %arg1 + 8 * 2 bytes
   = &arg1[8]
```

第二条：

```llvm
%2 = getelementptr inbounds i16, ptr %1, i32 0
```

是：

```text
%2 = address(%1) + 0 * sizeof(i16)
   = address(%1)
```

所以在当前示例中 `%2` 与 `%1` 指向同一个地址：

```text
%2 == %1 == &arg1[index]
```

它不是 load，也不会从 `%1` 所指向的内存读取一个指针；GEP 只计算地址。由于 LLVM 使用 opaque pointer，`%1` 和 `%2` 的静态类型都只是 `ptr`，GEP 的第一个类型参数 `i16` 提供“按 i16 元素跨度进行地址计算”的信息。

随后：

```llvm
%wide.load = load <8 x i16>, ptr %2
```

从这个相同起始地址读取八个连续 `i16`：

```text
< arg1[index + 0], arg1[index + 1], ..., arg1[index + 7] >
```

`inbounds` 不会做运行时 bounds check；它是前端/vectorizer 对 LLVM 作出的保证，表示这些地址计算位于同一个分配对象的有效范围内。若该承诺在执行时不成立，结果会成为 poison。这里它表达 vectorized load 仍然访问原标量 loop 已合法访问的数组范围。

第二个零偏移 GEP 是 vectorizer 统一生成地址形式留下的中间 identity。在更复杂的 vector loop 中，这个位置可能承载 lane offset、runtime alignment、masked/remainder access 等处理；对当前常量零偏移案例，后续 InstCombine 可以将 `%2` 替换为 `%1`。

### `noalias` 为什么重要

函数参数写作：

```llvm
define void @foo(ptr noalias %arg,
                 ptr noalias %arg1,
                 ptr noalias %arg2)
```

`noalias` 表示在该函数调用范围内，这些 pointer 所访问的对象不重叠。它帮助 vectorizer证明：写入 `arg[i]` 不会修改未来将从 `arg1[i+1]` 或 `arg2[i+1]` 读取的数据。

若 `arg`、`arg1`、`arg2` 可能 alias，标量 loop 的严格逐次执行顺序可能可观察，vectorizer 必须进行 runtime alias check、选择更保守策略，或放弃向量化。

### x86/AArch64 与 AMDGPU 的区别

书中 target triple 是 AArch64；`<8 x i16>` 是 128-bit，适合该 target 的 SIMD 宽度。Loop vectorizer 通过 TargetLibraryInfo、TTI 和 TargetLowering 检查这种 vector type 与 memory operation 是否合适。

在 AMDGPU 上，LLVM vector value 仍表示**一个 work-item 内**的八个 lane，不是八个 GPU thread。AMDGPU 已通过 wavefront 并行执行多个 work-item；对某个 loop 再做 loop vectorization 是否有收益，需要 AMDGPU 的 TTI 权衡寄存器压力、VGPR 使用、memory access、address space 和 subtarget 特性。后端也可能将 `<8 x i16>` 分解为多个操作，而非一条 128-bit vector ALU 指令。

### 为什么合并连续访问有价值

输入有四次标量内存指令：

```text
load src[0]
load src[1]
store dst[0]
store dst[1]
```

优化后在 IR 层的关键内存操作是：

```text
load  <2 x i64> src
store <2 x i64> dst
```

若 target 支持这种 128-bit vector memory operation，后端可以减少指令数和 memory-system 请求次数，或至少以更适合硬件的成组访问方式生成代码。

这不是无条件保证。`TargetTransformInfo` 会检查 target 支持哪些 vector type；若目标无法高效处理 `<2 x i64>`，后端仍可能拆回多个 scalar memory access。

在 x86/AArch64 上，向量 load/store 往往映射到 SIMD 寄存器和宽内存操作。对 AMDGPU，这个 `<2 x i64>` 是单个 work-item 中的两个 vector lane，不是两个 GPU thread；global、LDS/private address space、alignment、uniformity 和具体 subtarget 决定后端能否或应否将它合并为某种 AMDGPU memory instruction。
