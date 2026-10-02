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

`%iv` 是当前迭代值，`%iv_plus_1` 是递增后的下一迭代值。书中的 loop 在每轮先加一、再比较，因此概念上接近：

```cpp
uint64_t iv = 0;
uint64_t iv_plus_1;
do {
  iv_plus_1 = iv + 1;
  iv = iv_plus_1;
} while (iv_plus_1 < upper_bound);
```

注意原 IR 使用 `icmp ult`，即 unsigned comparison；普通 `add i64` 没有 `nsw`/`nuw`，保留 64-bit modulo arithmetic 语义。

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
