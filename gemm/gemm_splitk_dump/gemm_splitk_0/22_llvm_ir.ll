; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@__shared_alloc_0 = external dso_local addrspace(3) global [65536 x i8], align 16

define amdgpu_kernel void @gemm_splitk_0(ptr addrspace(1) %0, <{ <{ i32 }> }> %1, ptr addrspace(1) %2, <{ <{ i32 }> }> %3, ptr addrspace(1) %4, <{ <{ i32 }> }> %5, i32 %6) #0 !dbg !3 !reqd_work_group_size !6 {
  %8 = call range(i32 0, 256) i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
  %9 = sext i32 %8 to i64, !dbg !7
  %10 = trunc i64 %9 to i32, !dbg !8
  %11 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !11
  %12 = sext i32 %11 to i64, !dbg !11
  %13 = trunc i64 %12 to i32, !dbg !12
  %14 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !11
  %15 = sext i32 %14 to i64, !dbg !11
  %16 = trunc i64 %15 to i32, !dbg !12
  %17 = sub i32 %6, 1, !dbg !13
  %18 = mul i32 %17, 8192, !dbg !13
  %19 = add i32 %18, 8192, !dbg !13
  %20 = mul i32 %19, 2, !dbg !13
  %21 = sext i32 %20 to i64, !dbg !13
  %22 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %2, i16 0, i64 %21, i32 159744), !dbg !13
  %23 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %4, i16 0, i64 83886080, i32 159744), !dbg !14
  %24 = mul i32 %17, 5120, !dbg !15
  %25 = add i32 %24, 5120, !dbg !15
  %26 = mul i32 %25, 2, !dbg !15
  %27 = sext i32 %26 to i64, !dbg !15
  %28 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %0, i16 0, i64 %27, i32 159744), !dbg !15
  %29 = mul i32 %13, 524288, !dbg !16
  %30 = mul i32 %16, 524288, !dbg !17
  %31 = mul i32 %13, 327680, !dbg !18
  %32 = mul i32 %16, 64, !dbg !18
  %33 = add i32 %31, %32, !dbg !18
  %34 = srem i32 %10, 16, !dbg !19
  %35 = sdiv i32 %10, 16, !dbg !19
  %36 = mul i32 %34, 8192, !dbg !19
  %37 = mul i32 %35, 8, !dbg !19
  %38 = add i32 %36, %37, !dbg !19
  %39 = add i32 %29, %38, !dbg !19
  %40 = mul i32 %34, 8, !dbg !20
  %41 = mul i32 %35, 128, !dbg !20
  %42 = add i32 %40, %41, !dbg !20
  %43 = add i32 %30, %42, !dbg !20
  br label %44, !dbg !21

44:                                               ; preds = %48, %7
  %45 = phi i64 [ %292, %48 ], [ 0, %7 ], !dbg !22
  %46 = phi <64 x float> [ %291, %48 ], [ zeroinitializer, %7 ], !dbg !23
  %47 = icmp slt i64 %45, 32, !dbg !21
  br i1 %47, label %48, label %293, !dbg !21

48:                                               ; preds = %44
  %49 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !24
  %50 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !24
  %51 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !24
  %52 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !24
  %53 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !24
  %54 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !24
  %55 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !24
  %56 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !24
  %57 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 32, i32 33, i32 34, i32 35>, !dbg !24
  %58 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 36, i32 37, i32 38, i32 39>, !dbg !24
  %59 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 40, i32 41, i32 42, i32 43>, !dbg !24
  %60 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 44, i32 45, i32 46, i32 47>, !dbg !24
  %61 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 48, i32 49, i32 50, i32 51>, !dbg !24
  %62 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 52, i32 53, i32 54, i32 55>, !dbg !24
  %63 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 56, i32 57, i32 58, i32 59>, !dbg !24
  %64 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 60, i32 61, i32 62, i32 63>, !dbg !24
  %65 = trunc i64 %45 to i32, !dbg !25
  %66 = mul i32 %65, 256, !dbg !26
  %67 = add i32 %39, %66, !dbg !26
  %68 = mul i32 %67, 2, !dbg !27
  %69 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %22, i32 %68, i32 0, i32 0), !dbg !27
  %70 = bitcast i128 %69 to <8 x bfloat>, !dbg !27
  %71 = add i32 %67, 131072, !dbg !27
  %72 = mul i32 %71, 2, !dbg !27
  %73 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %22, i32 %72, i32 0, i32 0), !dbg !27
  %74 = bitcast i128 %73 to <8 x bfloat>, !dbg !27
  %75 = add i32 %67, 262144, !dbg !27
  %76 = mul i32 %75, 2, !dbg !27
  %77 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %22, i32 %76, i32 0, i32 0), !dbg !27
  %78 = bitcast i128 %77 to <8 x bfloat>, !dbg !27
  %79 = add i32 %67, 393216, !dbg !27
  %80 = mul i32 %79, 2, !dbg !27
  %81 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %22, i32 %80, i32 0, i32 0), !dbg !27
  %82 = bitcast i128 %81 to <8 x bfloat>, !dbg !27
  %83 = add i32 %67, 128, !dbg !27
  %84 = mul i32 %83, 2, !dbg !27
  %85 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %22, i32 %84, i32 0, i32 0), !dbg !27
  %86 = bitcast i128 %85 to <8 x bfloat>, !dbg !27
  %87 = add i32 %67, 131200, !dbg !27
  %88 = mul i32 %87, 2, !dbg !27
  %89 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %22, i32 %88, i32 0, i32 0), !dbg !27
  %90 = bitcast i128 %89 to <8 x bfloat>, !dbg !27
  %91 = add i32 %67, 262272, !dbg !27
  %92 = mul i32 %91, 2, !dbg !27
  %93 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %22, i32 %92, i32 0, i32 0), !dbg !27
  %94 = bitcast i128 %93 to <8 x bfloat>, !dbg !27
  %95 = add i32 %67, 393344, !dbg !27
  %96 = mul i32 %95, 2, !dbg !27
  %97 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %22, i32 %96, i32 0, i32 0), !dbg !27
  %98 = bitcast i128 %97 to <8 x bfloat>, !dbg !27
  %99 = mul i32 %65, 4096, !dbg !28
  %100 = add i32 %43, %99, !dbg !28
  %101 = mul i32 %100, 2, !dbg !29
  %102 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %23, i32 %101, i32 0, i32 0), !dbg !29
  %103 = bitcast i128 %102 to <8 x bfloat>, !dbg !29
  %104 = add i32 %100, 131072, !dbg !29
  %105 = mul i32 %104, 2, !dbg !29
  %106 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %23, i32 %105, i32 0, i32 0), !dbg !29
  %107 = bitcast i128 %106 to <8 x bfloat>, !dbg !29
  %108 = add i32 %100, 262144, !dbg !29
  %109 = mul i32 %108, 2, !dbg !29
  %110 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %23, i32 %109, i32 0, i32 0), !dbg !29
  %111 = bitcast i128 %110 to <8 x bfloat>, !dbg !29
  %112 = add i32 %100, 393216, !dbg !29
  %113 = mul i32 %112, 2, !dbg !29
  %114 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %23, i32 %113, i32 0, i32 0), !dbg !29
  %115 = bitcast i128 %114 to <8 x bfloat>, !dbg !29
  %116 = add i32 %100, 2048, !dbg !29
  %117 = mul i32 %116, 2, !dbg !29
  %118 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %23, i32 %117, i32 0, i32 0), !dbg !29
  %119 = bitcast i128 %118 to <8 x bfloat>, !dbg !29
  %120 = add i32 %100, 133120, !dbg !29
  %121 = mul i32 %120, 2, !dbg !29
  %122 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %23, i32 %121, i32 0, i32 0), !dbg !29
  %123 = bitcast i128 %122 to <8 x bfloat>, !dbg !29
  %124 = add i32 %100, 264192, !dbg !29
  %125 = mul i32 %124, 2, !dbg !29
  %126 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %23, i32 %125, i32 0, i32 0), !dbg !29
  %127 = bitcast i128 %126 to <8 x bfloat>, !dbg !29
  %128 = add i32 %100, 395264, !dbg !29
  %129 = mul i32 %128, 2, !dbg !29
  %130 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %23, i32 %129, i32 0, i32 0), !dbg !29
  %131 = bitcast i128 %130 to <8 x bfloat>, !dbg !29
  %132 = shufflevector <8 x bfloat> %103, <8 x bfloat> %103, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %133 = shufflevector <8 x bfloat> %70, <8 x bfloat> %70, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %134 = bitcast <4 x bfloat> %132 to <4 x i16>, !dbg !30
  %135 = bitcast <4 x bfloat> %133 to <4 x i16>, !dbg !30
  %136 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %134, <4 x i16> %135, <4 x float> %49, i32 0, i32 0, i32 0), !dbg !30
  %137 = shufflevector <8 x bfloat> %74, <8 x bfloat> %74, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %138 = bitcast <4 x bfloat> %137 to <4 x i16>, !dbg !30
  %139 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %134, <4 x i16> %138, <4 x float> %53, i32 0, i32 0, i32 0), !dbg !30
  %140 = shufflevector <8 x bfloat> %78, <8 x bfloat> %78, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %141 = bitcast <4 x bfloat> %140 to <4 x i16>, !dbg !30
  %142 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %134, <4 x i16> %141, <4 x float> %57, i32 0, i32 0, i32 0), !dbg !30
  %143 = shufflevector <8 x bfloat> %82, <8 x bfloat> %82, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %144 = bitcast <4 x bfloat> %143 to <4 x i16>, !dbg !30
  %145 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %134, <4 x i16> %144, <4 x float> %61, i32 0, i32 0, i32 0), !dbg !30
  %146 = shufflevector <8 x bfloat> %107, <8 x bfloat> %107, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %147 = bitcast <4 x bfloat> %146 to <4 x i16>, !dbg !30
  %148 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %147, <4 x i16> %135, <4 x float> %50, i32 0, i32 0, i32 0), !dbg !30
  %149 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %147, <4 x i16> %138, <4 x float> %54, i32 0, i32 0, i32 0), !dbg !30
  %150 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %147, <4 x i16> %141, <4 x float> %58, i32 0, i32 0, i32 0), !dbg !30
  %151 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %147, <4 x i16> %144, <4 x float> %62, i32 0, i32 0, i32 0), !dbg !30
  %152 = shufflevector <8 x bfloat> %111, <8 x bfloat> %111, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %153 = bitcast <4 x bfloat> %152 to <4 x i16>, !dbg !30
  %154 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %153, <4 x i16> %135, <4 x float> %51, i32 0, i32 0, i32 0), !dbg !30
  %155 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %153, <4 x i16> %138, <4 x float> %55, i32 0, i32 0, i32 0), !dbg !30
  %156 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %153, <4 x i16> %141, <4 x float> %59, i32 0, i32 0, i32 0), !dbg !30
  %157 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %153, <4 x i16> %144, <4 x float> %63, i32 0, i32 0, i32 0), !dbg !30
  %158 = shufflevector <8 x bfloat> %115, <8 x bfloat> %115, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %159 = bitcast <4 x bfloat> %158 to <4 x i16>, !dbg !30
  %160 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %159, <4 x i16> %135, <4 x float> %52, i32 0, i32 0, i32 0), !dbg !30
  %161 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %159, <4 x i16> %138, <4 x float> %56, i32 0, i32 0, i32 0), !dbg !30
  %162 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %159, <4 x i16> %141, <4 x float> %60, i32 0, i32 0, i32 0), !dbg !30
  %163 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %159, <4 x i16> %144, <4 x float> %64, i32 0, i32 0, i32 0), !dbg !30
  %164 = shufflevector <8 x bfloat> %103, <8 x bfloat> %103, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %165 = shufflevector <8 x bfloat> %70, <8 x bfloat> %70, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %166 = bitcast <4 x bfloat> %164 to <4 x i16>, !dbg !30
  %167 = bitcast <4 x bfloat> %165 to <4 x i16>, !dbg !30
  %168 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %166, <4 x i16> %167, <4 x float> %136, i32 0, i32 0, i32 0), !dbg !30
  %169 = shufflevector <8 x bfloat> %74, <8 x bfloat> %74, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %170 = bitcast <4 x bfloat> %169 to <4 x i16>, !dbg !30
  %171 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %166, <4 x i16> %170, <4 x float> %139, i32 0, i32 0, i32 0), !dbg !30
  %172 = shufflevector <8 x bfloat> %78, <8 x bfloat> %78, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %173 = bitcast <4 x bfloat> %172 to <4 x i16>, !dbg !30
  %174 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %166, <4 x i16> %173, <4 x float> %142, i32 0, i32 0, i32 0), !dbg !30
  %175 = shufflevector <8 x bfloat> %82, <8 x bfloat> %82, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %176 = bitcast <4 x bfloat> %175 to <4 x i16>, !dbg !30
  %177 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %166, <4 x i16> %176, <4 x float> %145, i32 0, i32 0, i32 0), !dbg !30
  %178 = shufflevector <8 x bfloat> %107, <8 x bfloat> %107, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %179 = bitcast <4 x bfloat> %178 to <4 x i16>, !dbg !30
  %180 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %179, <4 x i16> %167, <4 x float> %148, i32 0, i32 0, i32 0), !dbg !30
  %181 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %179, <4 x i16> %170, <4 x float> %149, i32 0, i32 0, i32 0), !dbg !30
  %182 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %179, <4 x i16> %173, <4 x float> %150, i32 0, i32 0, i32 0), !dbg !30
  %183 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %179, <4 x i16> %176, <4 x float> %151, i32 0, i32 0, i32 0), !dbg !30
  %184 = shufflevector <8 x bfloat> %111, <8 x bfloat> %111, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %185 = bitcast <4 x bfloat> %184 to <4 x i16>, !dbg !30
  %186 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %185, <4 x i16> %167, <4 x float> %154, i32 0, i32 0, i32 0), !dbg !30
  %187 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %185, <4 x i16> %170, <4 x float> %155, i32 0, i32 0, i32 0), !dbg !30
  %188 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %185, <4 x i16> %173, <4 x float> %156, i32 0, i32 0, i32 0), !dbg !30
  %189 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %185, <4 x i16> %176, <4 x float> %157, i32 0, i32 0, i32 0), !dbg !30
  %190 = shufflevector <8 x bfloat> %115, <8 x bfloat> %115, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %191 = bitcast <4 x bfloat> %190 to <4 x i16>, !dbg !30
  %192 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %191, <4 x i16> %167, <4 x float> %160, i32 0, i32 0, i32 0), !dbg !30
  %193 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %191, <4 x i16> %170, <4 x float> %161, i32 0, i32 0, i32 0), !dbg !30
  %194 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %191, <4 x i16> %173, <4 x float> %162, i32 0, i32 0, i32 0), !dbg !30
  %195 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %191, <4 x i16> %176, <4 x float> %163, i32 0, i32 0, i32 0), !dbg !30
  %196 = shufflevector <8 x bfloat> %119, <8 x bfloat> %119, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %197 = shufflevector <8 x bfloat> %86, <8 x bfloat> %86, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %198 = bitcast <4 x bfloat> %196 to <4 x i16>, !dbg !30
  %199 = bitcast <4 x bfloat> %197 to <4 x i16>, !dbg !30
  %200 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %198, <4 x i16> %199, <4 x float> %168, i32 0, i32 0, i32 0), !dbg !30
  %201 = shufflevector <8 x bfloat> %90, <8 x bfloat> %90, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %202 = bitcast <4 x bfloat> %201 to <4 x i16>, !dbg !30
  %203 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %198, <4 x i16> %202, <4 x float> %171, i32 0, i32 0, i32 0), !dbg !30
  %204 = shufflevector <8 x bfloat> %94, <8 x bfloat> %94, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %205 = bitcast <4 x bfloat> %204 to <4 x i16>, !dbg !30
  %206 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %198, <4 x i16> %205, <4 x float> %174, i32 0, i32 0, i32 0), !dbg !30
  %207 = shufflevector <8 x bfloat> %98, <8 x bfloat> %98, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %208 = bitcast <4 x bfloat> %207 to <4 x i16>, !dbg !30
  %209 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %198, <4 x i16> %208, <4 x float> %177, i32 0, i32 0, i32 0), !dbg !30
  %210 = shufflevector <8 x bfloat> %123, <8 x bfloat> %123, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %211 = bitcast <4 x bfloat> %210 to <4 x i16>, !dbg !30
  %212 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %211, <4 x i16> %199, <4 x float> %180, i32 0, i32 0, i32 0), !dbg !30
  %213 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %211, <4 x i16> %202, <4 x float> %181, i32 0, i32 0, i32 0), !dbg !30
  %214 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %211, <4 x i16> %205, <4 x float> %182, i32 0, i32 0, i32 0), !dbg !30
  %215 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %211, <4 x i16> %208, <4 x float> %183, i32 0, i32 0, i32 0), !dbg !30
  %216 = shufflevector <8 x bfloat> %127, <8 x bfloat> %127, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %217 = bitcast <4 x bfloat> %216 to <4 x i16>, !dbg !30
  %218 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %217, <4 x i16> %199, <4 x float> %186, i32 0, i32 0, i32 0), !dbg !30
  %219 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %217, <4 x i16> %202, <4 x float> %187, i32 0, i32 0, i32 0), !dbg !30
  %220 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %217, <4 x i16> %205, <4 x float> %188, i32 0, i32 0, i32 0), !dbg !30
  %221 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %217, <4 x i16> %208, <4 x float> %189, i32 0, i32 0, i32 0), !dbg !30
  %222 = shufflevector <8 x bfloat> %131, <8 x bfloat> %131, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %223 = bitcast <4 x bfloat> %222 to <4 x i16>, !dbg !30
  %224 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %223, <4 x i16> %199, <4 x float> %192, i32 0, i32 0, i32 0), !dbg !30
  %225 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %223, <4 x i16> %202, <4 x float> %193, i32 0, i32 0, i32 0), !dbg !30
  %226 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %223, <4 x i16> %205, <4 x float> %194, i32 0, i32 0, i32 0), !dbg !30
  %227 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %223, <4 x i16> %208, <4 x float> %195, i32 0, i32 0, i32 0), !dbg !30
  %228 = shufflevector <8 x bfloat> %119, <8 x bfloat> %119, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %229 = shufflevector <8 x bfloat> %86, <8 x bfloat> %86, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %230 = bitcast <4 x bfloat> %228 to <4 x i16>, !dbg !30
  %231 = bitcast <4 x bfloat> %229 to <4 x i16>, !dbg !30
  %232 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %230, <4 x i16> %231, <4 x float> %200, i32 0, i32 0, i32 0), !dbg !30
  %233 = shufflevector <8 x bfloat> %90, <8 x bfloat> %90, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %234 = bitcast <4 x bfloat> %233 to <4 x i16>, !dbg !30
  %235 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %230, <4 x i16> %234, <4 x float> %203, i32 0, i32 0, i32 0), !dbg !30
  %236 = shufflevector <8 x bfloat> %94, <8 x bfloat> %94, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %237 = bitcast <4 x bfloat> %236 to <4 x i16>, !dbg !30
  %238 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %230, <4 x i16> %237, <4 x float> %206, i32 0, i32 0, i32 0), !dbg !30
  %239 = shufflevector <8 x bfloat> %98, <8 x bfloat> %98, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %240 = bitcast <4 x bfloat> %239 to <4 x i16>, !dbg !30
  %241 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %230, <4 x i16> %240, <4 x float> %209, i32 0, i32 0, i32 0), !dbg !30
  %242 = shufflevector <8 x bfloat> %123, <8 x bfloat> %123, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %243 = bitcast <4 x bfloat> %242 to <4 x i16>, !dbg !30
  %244 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %243, <4 x i16> %231, <4 x float> %212, i32 0, i32 0, i32 0), !dbg !30
  %245 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %243, <4 x i16> %234, <4 x float> %213, i32 0, i32 0, i32 0), !dbg !30
  %246 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %243, <4 x i16> %237, <4 x float> %214, i32 0, i32 0, i32 0), !dbg !30
  %247 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %243, <4 x i16> %240, <4 x float> %215, i32 0, i32 0, i32 0), !dbg !30
  %248 = shufflevector <8 x bfloat> %127, <8 x bfloat> %127, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %249 = bitcast <4 x bfloat> %248 to <4 x i16>, !dbg !30
  %250 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %249, <4 x i16> %231, <4 x float> %218, i32 0, i32 0, i32 0), !dbg !30
  %251 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %249, <4 x i16> %234, <4 x float> %219, i32 0, i32 0, i32 0), !dbg !30
  %252 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %249, <4 x i16> %237, <4 x float> %220, i32 0, i32 0, i32 0), !dbg !30
  %253 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %249, <4 x i16> %240, <4 x float> %221, i32 0, i32 0, i32 0), !dbg !30
  %254 = shufflevector <8 x bfloat> %131, <8 x bfloat> %131, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %255 = bitcast <4 x bfloat> %254 to <4 x i16>, !dbg !30
  %256 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %255, <4 x i16> %231, <4 x float> %224, i32 0, i32 0, i32 0), !dbg !30
  %257 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %255, <4 x i16> %234, <4 x float> %225, i32 0, i32 0, i32 0), !dbg !30
  %258 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %255, <4 x i16> %237, <4 x float> %226, i32 0, i32 0, i32 0), !dbg !30
  %259 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %255, <4 x i16> %240, <4 x float> %227, i32 0, i32 0, i32 0), !dbg !30
  %260 = shufflevector <4 x float> %232, <4 x float> %232, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %261 = shufflevector <64 x float> %260, <64 x float> zeroinitializer, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %262 = shufflevector <4 x float> %244, <4 x float> %244, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %263 = shufflevector <64 x float> %262, <64 x float> %261, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 0, i32 1, i32 2, i32 3, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %264 = shufflevector <4 x float> %250, <4 x float> %250, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %265 = shufflevector <64 x float> %264, <64 x float> %263, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 0, i32 1, i32 2, i32 3, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %266 = shufflevector <4 x float> %256, <4 x float> %256, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %267 = shufflevector <64 x float> %266, <64 x float> %265, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 0, i32 1, i32 2, i32 3, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %268 = shufflevector <4 x float> %235, <4 x float> %235, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %269 = shufflevector <64 x float> %268, <64 x float> %267, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 0, i32 1, i32 2, i32 3, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %270 = shufflevector <4 x float> %245, <4 x float> %245, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %271 = shufflevector <64 x float> %270, <64 x float> %269, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 0, i32 1, i32 2, i32 3, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %272 = shufflevector <4 x float> %251, <4 x float> %251, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %273 = shufflevector <64 x float> %272, <64 x float> %271, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 0, i32 1, i32 2, i32 3, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %274 = shufflevector <4 x float> %257, <4 x float> %257, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %275 = shufflevector <64 x float> %274, <64 x float> %273, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 0, i32 1, i32 2, i32 3, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %276 = shufflevector <4 x float> %238, <4 x float> %238, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %277 = shufflevector <64 x float> %276, <64 x float> %275, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 0, i32 1, i32 2, i32 3, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %278 = shufflevector <4 x float> %246, <4 x float> %246, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %279 = shufflevector <64 x float> %278, <64 x float> %277, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 0, i32 1, i32 2, i32 3, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %280 = shufflevector <4 x float> %252, <4 x float> %252, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %281 = shufflevector <64 x float> %280, <64 x float> %279, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 0, i32 1, i32 2, i32 3, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %282 = shufflevector <4 x float> %258, <4 x float> %258, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %283 = shufflevector <64 x float> %282, <64 x float> %281, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 0, i32 1, i32 2, i32 3, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %284 = shufflevector <4 x float> %241, <4 x float> %241, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %285 = shufflevector <64 x float> %284, <64 x float> %283, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 0, i32 1, i32 2, i32 3, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %286 = shufflevector <4 x float> %247, <4 x float> %247, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %287 = shufflevector <64 x float> %286, <64 x float> %285, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 0, i32 1, i32 2, i32 3, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %288 = shufflevector <4 x float> %253, <4 x float> %253, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %289 = shufflevector <64 x float> %288, <64 x float> %287, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 0, i32 1, i32 2, i32 3, i32 124, i32 125, i32 126, i32 127>, !dbg !31
  %290 = shufflevector <4 x float> %259, <4 x float> %259, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !31
  %291 = shufflevector <64 x float> %290, <64 x float> %289, <64 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %292 = add i64 %45, 1, !dbg !21
  br label %44, !dbg !21

293:                                              ; preds = %44
  %294 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %295 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %296 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !32
  %297 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !32
  %298 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !32
  %299 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !32
  %300 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !32
  %301 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !32
  %302 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 32, i32 33, i32 34, i32 35>, !dbg !32
  %303 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 36, i32 37, i32 38, i32 39>, !dbg !32
  %304 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 40, i32 41, i32 42, i32 43>, !dbg !32
  %305 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 44, i32 45, i32 46, i32 47>, !dbg !32
  %306 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 48, i32 49, i32 50, i32 51>, !dbg !32
  %307 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 52, i32 53, i32 54, i32 55>, !dbg !32
  %308 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 56, i32 57, i32 58, i32 59>, !dbg !32
  %309 = shufflevector <64 x float> %46, <64 x float> %46, <4 x i32> <i32 60, i32 61, i32 62, i32 63>, !dbg !32
  %310 = mul i32 %34, 64, !dbg !33
  %311 = srem i32 %35, 4, !dbg !33
  %312 = sdiv i32 %35, 4, !dbg !33
  %313 = mul i32 %311, 4, !dbg !33
  %314 = add i32 %310, %313, !dbg !33
  %315 = mul i32 %312, 1024, !dbg !33
  %316 = add i32 %314, %315, !dbg !33
  %317 = and i32 %314, 448, !dbg !34
  %318 = lshr i32 %317, 3, !dbg !34
  %319 = xor i32 %314, %318, !dbg !34
  %320 = add i32 %319, %315, !dbg !34
  %321 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %320, !dbg !34
  store <4 x float> %294, ptr addrspace(3) %321, align 16, !dbg !34
  %322 = add i32 %320, 4096, !dbg !34
  %323 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %322, !dbg !34
  store <4 x float> %298, ptr addrspace(3) %323, align 16, !dbg !34
  %324 = add i32 %320, 8192, !dbg !34
  %325 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %324, !dbg !34
  store <4 x float> %302, ptr addrspace(3) %325, align 16, !dbg !34
  %326 = add i32 %320, 12288, !dbg !34
  %327 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %326, !dbg !34
  store <4 x float> %306, ptr addrspace(3) %327, align 16, !dbg !34
  %328 = add i32 %316, 16, !dbg !34
  %329 = and i32 %328, 448, !dbg !34
  %330 = lshr i32 %329, 3, !dbg !34
  %331 = xor i32 %328, %330, !dbg !34
  %332 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %331, !dbg !34
  store <4 x float> %295, ptr addrspace(3) %332, align 16, !dbg !34
  %333 = add i32 %316, 4112, !dbg !34
  %334 = and i32 %333, 448, !dbg !34
  %335 = lshr i32 %334, 3, !dbg !34
  %336 = xor i32 %333, %335, !dbg !34
  %337 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %336, !dbg !34
  store <4 x float> %299, ptr addrspace(3) %337, align 16, !dbg !34
  %338 = add i32 %316, 8208, !dbg !34
  %339 = and i32 %338, 448, !dbg !34
  %340 = lshr i32 %339, 3, !dbg !34
  %341 = xor i32 %338, %340, !dbg !34
  %342 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %341, !dbg !34
  store <4 x float> %303, ptr addrspace(3) %342, align 16, !dbg !34
  %343 = add i32 %316, 12304, !dbg !34
  %344 = and i32 %343, 448, !dbg !34
  %345 = lshr i32 %344, 3, !dbg !34
  %346 = xor i32 %343, %345, !dbg !34
  %347 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %346, !dbg !34
  store <4 x float> %307, ptr addrspace(3) %347, align 16, !dbg !34
  %348 = add i32 %316, 32, !dbg !34
  %349 = and i32 %348, 448, !dbg !34
  %350 = lshr i32 %349, 3, !dbg !34
  %351 = xor i32 %348, %350, !dbg !34
  %352 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %351, !dbg !34
  store <4 x float> %296, ptr addrspace(3) %352, align 16, !dbg !34
  %353 = add i32 %316, 4128, !dbg !34
  %354 = and i32 %353, 448, !dbg !34
  %355 = lshr i32 %354, 3, !dbg !34
  %356 = xor i32 %353, %355, !dbg !34
  %357 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %356, !dbg !34
  store <4 x float> %300, ptr addrspace(3) %357, align 16, !dbg !34
  %358 = add i32 %316, 8224, !dbg !34
  %359 = and i32 %358, 448, !dbg !34
  %360 = lshr i32 %359, 3, !dbg !34
  %361 = xor i32 %358, %360, !dbg !34
  %362 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %361, !dbg !34
  store <4 x float> %304, ptr addrspace(3) %362, align 16, !dbg !34
  %363 = add i32 %316, 12320, !dbg !34
  %364 = and i32 %363, 448, !dbg !34
  %365 = lshr i32 %364, 3, !dbg !34
  %366 = xor i32 %363, %365, !dbg !34
  %367 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %366, !dbg !34
  store <4 x float> %308, ptr addrspace(3) %367, align 16, !dbg !34
  %368 = add i32 %316, 48, !dbg !34
  %369 = and i32 %368, 448, !dbg !34
  %370 = lshr i32 %369, 3, !dbg !34
  %371 = xor i32 %368, %370, !dbg !34
  %372 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %371, !dbg !34
  store <4 x float> %297, ptr addrspace(3) %372, align 16, !dbg !34
  %373 = add i32 %316, 4144, !dbg !34
  %374 = and i32 %373, 448, !dbg !34
  %375 = lshr i32 %374, 3, !dbg !34
  %376 = xor i32 %373, %375, !dbg !34
  %377 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %376, !dbg !34
  store <4 x float> %301, ptr addrspace(3) %377, align 16, !dbg !34
  %378 = add i32 %316, 8240, !dbg !34
  %379 = and i32 %378, 448, !dbg !34
  %380 = lshr i32 %379, 3, !dbg !34
  %381 = xor i32 %378, %380, !dbg !34
  %382 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %381, !dbg !34
  store <4 x float> %305, ptr addrspace(3) %382, align 16, !dbg !34
  %383 = add i32 %316, 12336, !dbg !34
  %384 = and i32 %383, 448, !dbg !34
  %385 = lshr i32 %384, 3, !dbg !34
  %386 = xor i32 %383, %385, !dbg !34
  %387 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %386, !dbg !34
  store <4 x float> %309, ptr addrspace(3) %387, align 16, !dbg !34
  fence syncscope("workgroup") release, !dbg !35
  call void @llvm.amdgcn.s.barrier(), !dbg !35
  fence syncscope("workgroup") acquire, !dbg !35
  %388 = mul i32 %34, 4, !dbg !36
  %389 = mul i32 %35, 64, !dbg !36
  %390 = add i32 %388, %389, !dbg !36
  %391 = and i32 %390, 448, !dbg !37
  %392 = lshr i32 %391, 3, !dbg !37
  %393 = xor i32 %390, %392, !dbg !37
  %394 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %393, !dbg !37
  %395 = load <4 x float>, ptr addrspace(3) %394, align 16, !dbg !37
  %396 = add i32 %393, 1024, !dbg !37
  %397 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %396, !dbg !37
  %398 = load <4 x float>, ptr addrspace(3) %397, align 16, !dbg !37
  %399 = add i32 %393, 2048, !dbg !37
  %400 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %399, !dbg !37
  %401 = load <4 x float>, ptr addrspace(3) %400, align 16, !dbg !37
  %402 = add i32 %393, 3072, !dbg !37
  %403 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %402, !dbg !37
  %404 = load <4 x float>, ptr addrspace(3) %403, align 16, !dbg !37
  %405 = add i32 %393, 4096, !dbg !37
  %406 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %405, !dbg !37
  %407 = load <4 x float>, ptr addrspace(3) %406, align 16, !dbg !37
  %408 = add i32 %393, 5120, !dbg !37
  %409 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %408, !dbg !37
  %410 = load <4 x float>, ptr addrspace(3) %409, align 16, !dbg !37
  %411 = add i32 %393, 6144, !dbg !37
  %412 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %411, !dbg !37
  %413 = load <4 x float>, ptr addrspace(3) %412, align 16, !dbg !37
  %414 = add i32 %393, 7168, !dbg !37
  %415 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %414, !dbg !37
  %416 = load <4 x float>, ptr addrspace(3) %415, align 16, !dbg !37
  %417 = add i32 %393, 8192, !dbg !37
  %418 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %417, !dbg !37
  %419 = load <4 x float>, ptr addrspace(3) %418, align 16, !dbg !37
  %420 = add i32 %393, 9216, !dbg !37
  %421 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %420, !dbg !37
  %422 = load <4 x float>, ptr addrspace(3) %421, align 16, !dbg !37
  %423 = add i32 %393, 10240, !dbg !37
  %424 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %423, !dbg !37
  %425 = load <4 x float>, ptr addrspace(3) %424, align 16, !dbg !37
  %426 = add i32 %393, 11264, !dbg !37
  %427 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %426, !dbg !37
  %428 = load <4 x float>, ptr addrspace(3) %427, align 16, !dbg !37
  %429 = add i32 %393, 12288, !dbg !37
  %430 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %429, !dbg !37
  %431 = load <4 x float>, ptr addrspace(3) %430, align 16, !dbg !37
  %432 = add i32 %393, 13312, !dbg !37
  %433 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %432, !dbg !37
  %434 = load <4 x float>, ptr addrspace(3) %433, align 16, !dbg !37
  %435 = add i32 %393, 14336, !dbg !37
  %436 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %435, !dbg !37
  %437 = load <4 x float>, ptr addrspace(3) %436, align 16, !dbg !37
  %438 = add i32 %393, 15360, !dbg !37
  %439 = getelementptr float, ptr addrspace(3) @__shared_alloc_0, i32 %438, !dbg !37
  %440 = load <4 x float>, ptr addrspace(3) %439, align 16, !dbg !37
  %441 = shufflevector <4 x float> %395, <4 x float> %395, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !38
  %442 = shufflevector <16 x float> %441, <16 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !38
  %443 = shufflevector <4 x float> %407, <4 x float> %407, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !38
  %444 = shufflevector <16 x float> %443, <16 x float> %442, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 0, i32 1, i32 2, i32 3, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !38
  %445 = shufflevector <4 x float> %419, <4 x float> %419, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !38
  %446 = shufflevector <16 x float> %445, <16 x float> %444, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 28, i32 29, i32 30, i32 31>, !dbg !38
  %447 = shufflevector <4 x float> %431, <4 x float> %431, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !38
  %448 = shufflevector <16 x float> %447, <16 x float> %446, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 0, i32 1, i32 2, i32 3>, !dbg !38
  %449 = shufflevector <4 x float> %398, <4 x float> %398, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !39
  %450 = shufflevector <16 x float> %449, <16 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !39
  %451 = shufflevector <4 x float> %410, <4 x float> %410, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !39
  %452 = shufflevector <16 x float> %451, <16 x float> %450, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 0, i32 1, i32 2, i32 3, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !39
  %453 = shufflevector <4 x float> %422, <4 x float> %422, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !39
  %454 = shufflevector <16 x float> %453, <16 x float> %452, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 28, i32 29, i32 30, i32 31>, !dbg !39
  %455 = shufflevector <4 x float> %434, <4 x float> %434, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !39
  %456 = shufflevector <16 x float> %455, <16 x float> %454, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %457 = fadd <16 x float> %448, %456, !dbg !40
  %458 = shufflevector <4 x float> %401, <4 x float> %401, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !39
  %459 = shufflevector <16 x float> %458, <16 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !39
  %460 = shufflevector <4 x float> %413, <4 x float> %413, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !39
  %461 = shufflevector <16 x float> %460, <16 x float> %459, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 0, i32 1, i32 2, i32 3, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !39
  %462 = shufflevector <4 x float> %425, <4 x float> %425, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !39
  %463 = shufflevector <16 x float> %462, <16 x float> %461, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 28, i32 29, i32 30, i32 31>, !dbg !39
  %464 = shufflevector <4 x float> %437, <4 x float> %437, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !39
  %465 = shufflevector <16 x float> %464, <16 x float> %463, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %466 = fadd <16 x float> %457, %465, !dbg !40
  %467 = shufflevector <4 x float> %404, <4 x float> %404, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !39
  %468 = shufflevector <16 x float> %467, <16 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !39
  %469 = shufflevector <4 x float> %416, <4 x float> %416, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !39
  %470 = shufflevector <16 x float> %469, <16 x float> %468, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 0, i32 1, i32 2, i32 3, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !39
  %471 = shufflevector <4 x float> %428, <4 x float> %428, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !39
  %472 = shufflevector <16 x float> %471, <16 x float> %470, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 28, i32 29, i32 30, i32 31>, !dbg !39
  %473 = shufflevector <4 x float> %440, <4 x float> %440, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !39
  %474 = shufflevector <16 x float> %473, <16 x float> %472, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %475 = fadd <16 x float> %466, %474, !dbg !40
  %476 = fadd <16 x float> %475, splat (float 4.591770e-41), !dbg !40
  %477 = bitcast <16 x float> %476 to <16 x i32>, !dbg !43
  %478 = lshr <16 x i32> %477, splat (i32 16), !dbg !40
  %479 = trunc <16 x i32> %478 to <16 x i16>, !dbg !44
  %480 = bitcast <16 x i16> %479 to <16 x bfloat>, !dbg !44
  %481 = shufflevector <16 x bfloat> %480, <16 x bfloat> %480, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %482 = shufflevector <16 x bfloat> %480, <16 x bfloat> %480, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %483 = shufflevector <16 x bfloat> %480, <16 x bfloat> %480, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !45
  %484 = shufflevector <16 x bfloat> %480, <16 x bfloat> %480, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !45
  %485 = mul i32 %35, 5120, !dbg !46
  %486 = add i32 %388, %485, !dbg !46
  %487 = add i32 %33, %486, !dbg !46
  %488 = mul i32 %487, 2, !dbg !47
  %489 = bitcast <4 x bfloat> %481 to i64, !dbg !47
  call void @llvm.amdgcn.raw.ptr.buffer.store.i64(i64 %489, ptr addrspace(8) %28, i32 %488, i32 0, i32 0), !dbg !47
  %490 = add i32 %487, 81920, !dbg !47
  %491 = mul i32 %490, 2, !dbg !47
  %492 = bitcast <4 x bfloat> %482 to i64, !dbg !47
  call void @llvm.amdgcn.raw.ptr.buffer.store.i64(i64 %492, ptr addrspace(8) %28, i32 %491, i32 0, i32 0), !dbg !47
  %493 = add i32 %487, 163840, !dbg !47
  %494 = mul i32 %493, 2, !dbg !47
  %495 = bitcast <4 x bfloat> %483 to i64, !dbg !47
  call void @llvm.amdgcn.raw.ptr.buffer.store.i64(i64 %495, ptr addrspace(8) %28, i32 %494, i32 0, i32 0), !dbg !47
  %496 = add i32 %487, 245760, !dbg !47
  %497 = mul i32 %496, 2, !dbg !47
  %498 = bitcast <4 x bfloat> %484 to i64, !dbg !47
  call void @llvm.amdgcn.raw.ptr.buffer.store.i64(i64 %498, ptr addrspace(8) %28, i32 %497, i32 0, i32 0), !dbg !47
  ret void, !dbg !48
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) readnone, i16, i64, i32) #2

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i64(i64, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #5

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16>, <4 x i16>, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #6

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nofree nounwind willreturn }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #6 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test_gemm.py", directory: "/models/toqiu/ai-framework-labs/gemm")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "gemm_splitk_0", linkageName: "gemm_splitk_0", scope: !1, file: !1, line: 3112, type: !4, scopeLine: 3112, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DISubroutineType(cc: DW_CC_normal, types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 77, column: 10, scope: !3)
!8 = !DILocation(line: 77, column: 10, scope: !9, inlinedAt: !10)
!9 = distinct !DILexicalBlockFile(scope: !3, file: !1, discriminator: 0)
!10 = !DILocation(line: 3112, column: 27, scope: !3)
!11 = !DILocation(line: 78, column: 4, scope: !3)
!12 = !DILocation(line: 78, column: 4, scope: !9, inlinedAt: !10)
!13 = !DILocation(line: 96, column: 15, scope: !9, inlinedAt: !10)
!14 = !DILocation(line: 97, column: 15, scope: !9, inlinedAt: !10)
!15 = !DILocation(line: 98, column: 15, scope: !9, inlinedAt: !10)
!16 = !DILocation(line: 100, column: 13, scope: !9, inlinedAt: !10)
!17 = !DILocation(line: 102, column: 13, scope: !9, inlinedAt: !10)
!18 = !DILocation(line: 104, column: 13, scope: !9, inlinedAt: !10)
!19 = !DILocation(line: 119, column: 19, scope: !9, inlinedAt: !10)
!20 = !DILocation(line: 120, column: 19, scope: !9, inlinedAt: !10)
!21 = !DILocation(line: 164, column: 8, scope: !3)
!22 = !DILocation(line: 126, column: 22, scope: !3)
!23 = !DILocation(line: 126, column: 61, scope: !3)
!24 = !DILocation(line: 165, column: 12, scope: !9, inlinedAt: !10)
!25 = !DILocation(line: 166, column: 20, scope: !9, inlinedAt: !10)
!26 = !DILocation(line: 167, column: 31, scope: !9, inlinedAt: !10)
!27 = !DILocation(line: 167, column: 12, scope: !9, inlinedAt: !10)
!28 = !DILocation(line: 168, column: 31, scope: !9, inlinedAt: !10)
!29 = !DILocation(line: 168, column: 12, scope: !9, inlinedAt: !10)
!30 = !DILocation(line: 170, column: 16, scope: !9, inlinedAt: !10)
!31 = !DILocation(line: 171, column: 29, scope: !9, inlinedAt: !10)
!32 = !DILocation(line: 172, column: 8, scope: !9, inlinedAt: !10)
!33 = !DILocation(line: 186, column: 25, scope: !9, inlinedAt: !10)
!34 = !DILocation(line: 187, column: 4, scope: !9, inlinedAt: !10)
!35 = !DILocation(line: 188, column: 4, scope: !9, inlinedAt: !10)
!36 = !DILocation(line: 198, column: 25, scope: !9, inlinedAt: !10)
!37 = !DILocation(line: 200, column: 4, scope: !9, inlinedAt: !10)
!38 = !DILocation(line: 201, column: 10, scope: !9, inlinedAt: !10)
!39 = !DILocation(line: 203, column: 15, scope: !9, inlinedAt: !10)
!40 = !DILocation(line: 388, column: 19, scope: !41, inlinedAt: !10)
!41 = distinct !DILexicalBlockFile(scope: !3, file: !42, discriminator: 0)
!42 = !DIFile(filename: "functools.py", directory: "/usr/lib/python3.12")
!43 = !DILocation(line: 210, column: 11, scope: !9, inlinedAt: !10)
!44 = !DILocation(line: 210, column: 10, scope: !9, inlinedAt: !10)
!45 = !DILocation(line: 212, column: 4, scope: !9, inlinedAt: !10)
!46 = !DILocation(line: 220, column: 21, scope: !9, inlinedAt: !10)
!47 = !DILocation(line: 221, column: 4, scope: !9, inlinedAt: !10)
!48 = !DILocation(line: 829, column: 8, scope: !9, inlinedAt: !10)
