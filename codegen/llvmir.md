## language Ref
https://llvm.org/docs/LangRef.html

> no alias
> This indicates that memory locations accessed via pointer values based on the argument or return value are not also accessed, during the execution of the function, via pointer values not based on the argument or return value. This guarantee only holds for memory locations that are modified, by any means, during the execution of the function. If there are other accesses not based on the argument or return value, the behavior is undefined. The attribute on a return value also has additional semantics, as described below. Both the caller and the callee share the responsibility of ensuring that these requirements are met. For further details, please see the discussion of the NoAlias response in alias analysis.

> noalias also applies to accesses from other threads, unless they happen-before function entry, or function exit happens-before them. This means that conflicting concurrent accesses from other threads either need to be based on the noalias pointer, or else be appropriately synchronized outside the function.

## Type
https://llvm.org/doxygen/classllvm_1_1Type.html