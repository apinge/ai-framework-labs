; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

define amdgpu_kernel void @unpack_mfma_0(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) #0 !dbg !3 !reqd_work_group_size !6 {
  %4 = call range(i32 0, 64) i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
  %5 = sext i32 %4 to i64, !dbg !7
  %6 = trunc i64 %5 to i32, !dbg !7
  %7 = mul i32 %6, 4, !dbg !8
  %8 = getelementptr i32, ptr addrspace(1) %1, i32 %7, !dbg !9
  %9 = load <4 x i32>, ptr addrspace(1) %8, align 4, !dbg !10
  %10 = extractelement <4 x i32> %9, i64 0, !dbg !11
  %11 = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %10, i1 false), !dbg !12
  %12 = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %10, i1 true), !dbg !13
  %13 = bitcast <2 x float> %11 to <2 x i32>, !dbg !14
  %14 = lshr <2 x i32> %13, splat (i32 16), !dbg !15
  %15 = trunc <2 x i32> %14 to <2 x i16>, !dbg !18
  %16 = bitcast <2 x float> %12 to <2 x i32>, !dbg !19
  %17 = lshr <2 x i32> %16, splat (i32 16), !dbg !20
  %18 = trunc <2 x i32> %17 to <2 x i16>, !dbg !21
  %19 = extractelement <2 x i16> %15, i64 0, !dbg !22
  %20 = extractelement <2 x i16> %15, i64 1, !dbg !23
  %21 = extractelement <2 x i16> %18, i64 0, !dbg !24
  %22 = extractelement <2 x i16> %18, i64 1, !dbg !25
  %23 = insertelement <4 x i16> poison, i16 %19, i64 0, !dbg !26
  %24 = insertelement <4 x i16> %23, i16 %20, i64 1, !dbg !26
  %25 = insertelement <4 x i16> %24, i16 %21, i64 2, !dbg !26
  %26 = insertelement <4 x i16> %25, i16 %22, i64 3, !dbg !26
  %27 = extractelement <4 x i32> %9, i64 1, !dbg !11
  %28 = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %27, i1 false), !dbg !12
  %29 = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %27, i1 true), !dbg !13
  %30 = bitcast <2 x float> %28 to <2 x i32>, !dbg !14
  %31 = lshr <2 x i32> %30, splat (i32 16), !dbg !15
  %32 = trunc <2 x i32> %31 to <2 x i16>, !dbg !18
  %33 = bitcast <2 x float> %29 to <2 x i32>, !dbg !19
  %34 = lshr <2 x i32> %33, splat (i32 16), !dbg !20
  %35 = trunc <2 x i32> %34 to <2 x i16>, !dbg !21
  %36 = extractelement <2 x i16> %32, i64 0, !dbg !22
  %37 = extractelement <2 x i16> %32, i64 1, !dbg !23
  %38 = extractelement <2 x i16> %35, i64 0, !dbg !24
  %39 = extractelement <2 x i16> %35, i64 1, !dbg !25
  %40 = insertelement <4 x i16> poison, i16 %36, i64 0, !dbg !26
  %41 = insertelement <4 x i16> %40, i16 %37, i64 1, !dbg !26
  %42 = insertelement <4 x i16> %41, i16 %38, i64 2, !dbg !26
  %43 = insertelement <4 x i16> %42, i16 %39, i64 3, !dbg !26
  %44 = extractelement <4 x i32> %9, i64 2, !dbg !11
  %45 = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %44, i1 false), !dbg !12
  %46 = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %44, i1 true), !dbg !13
  %47 = bitcast <2 x float> %45 to <2 x i32>, !dbg !14
  %48 = lshr <2 x i32> %47, splat (i32 16), !dbg !15
  %49 = trunc <2 x i32> %48 to <2 x i16>, !dbg !18
  %50 = bitcast <2 x float> %46 to <2 x i32>, !dbg !19
  %51 = lshr <2 x i32> %50, splat (i32 16), !dbg !20
  %52 = trunc <2 x i32> %51 to <2 x i16>, !dbg !21
  %53 = extractelement <2 x i16> %49, i64 0, !dbg !22
  %54 = extractelement <2 x i16> %49, i64 1, !dbg !23
  %55 = extractelement <2 x i16> %52, i64 0, !dbg !24
  %56 = extractelement <2 x i16> %52, i64 1, !dbg !25
  %57 = insertelement <4 x i16> poison, i16 %53, i64 0, !dbg !26
  %58 = insertelement <4 x i16> %57, i16 %54, i64 1, !dbg !26
  %59 = insertelement <4 x i16> %58, i16 %55, i64 2, !dbg !26
  %60 = insertelement <4 x i16> %59, i16 %56, i64 3, !dbg !26
  %61 = extractelement <4 x i32> %9, i64 3, !dbg !11
  %62 = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %61, i1 false), !dbg !12
  %63 = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %61, i1 true), !dbg !13
  %64 = bitcast <2 x float> %62 to <2 x i32>, !dbg !14
  %65 = lshr <2 x i32> %64, splat (i32 16), !dbg !15
  %66 = trunc <2 x i32> %65 to <2 x i16>, !dbg !18
  %67 = bitcast <2 x float> %63 to <2 x i32>, !dbg !19
  %68 = lshr <2 x i32> %67, splat (i32 16), !dbg !20
  %69 = trunc <2 x i32> %68 to <2 x i16>, !dbg !21
  %70 = extractelement <2 x i16> %66, i64 0, !dbg !22
  %71 = extractelement <2 x i16> %66, i64 1, !dbg !23
  %72 = extractelement <2 x i16> %69, i64 0, !dbg !24
  %73 = extractelement <2 x i16> %69, i64 1, !dbg !25
  %74 = insertelement <4 x i16> poison, i16 %70, i64 0, !dbg !26
  %75 = insertelement <4 x i16> %74, i16 %71, i64 1, !dbg !26
  %76 = insertelement <4 x i16> %75, i16 %72, i64 2, !dbg !26
  %77 = insertelement <4 x i16> %76, i16 %73, i64 3, !dbg !26
  %78 = mul i32 %6, 8, !dbg !27
  %79 = getelementptr bfloat, ptr addrspace(1) %0, i32 %78, !dbg !28
  %80 = load <4 x bfloat>, ptr addrspace(1) %79, align 2, !dbg !29
  %81 = bitcast <4 x bfloat> %80 to <4 x i16>, !dbg !30
  %82 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %26, <4 x i16> %81, <4 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !31
  %83 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %60, <4 x i16> %81, <4 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !31
  %84 = getelementptr bfloat, ptr addrspace(1) %79, i32 4, !dbg !28
  %85 = load <4 x bfloat>, ptr addrspace(1) %84, align 2, !dbg !29
  %86 = bitcast <4 x bfloat> %85 to <4 x i16>, !dbg !30
  %87 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %43, <4 x i16> %86, <4 x float> %82, i32 0, i32 0, i32 0), !dbg !31
  %88 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %77, <4 x i16> %86, <4 x float> %83, i32 0, i32 0, i32 0), !dbg !31
  %89 = getelementptr float, ptr addrspace(1) %2, i32 %78, !dbg !32
  store <4 x float> %87, ptr addrspace(1) %89, align 4, !dbg !33
  %90 = getelementptr float, ptr addrspace(1) %89, i32 4, !dbg !32
  store <4 x float> %88, ptr addrspace(1) %90, align 4, !dbg !33
  ret void, !dbg !34
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32, i1 immarg) #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16>, <4 x i16>, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #3

attributes #0 = { "amdgpu-flat-work-group-size"="64,64" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "fp8_unpack_mfma_repro.py", directory: "/models/toqiu/ai-framework-labs/compiler_issues")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "unpack_mfma_0", linkageName: "unpack_mfma_0", scope: !1, file: !1, line: 19, type: !4, scopeLine: 19, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DISubroutineType(cc: DW_CC_normal, types: !5)
!5 = !{}
!6 = !{i32 64, i32 1, i32 1}
!7 = !DILocation(line: 21, column: 11, scope: !3)
!8 = !DILocation(line: 23, column: 12, scope: !3)
!9 = !DILocation(line: 23, column: 8, scope: !3)
!10 = !DILocation(line: 24, column: 6, scope: !3)
!11 = !DILocation(line: 28, column: 51, scope: !3)
!12 = !DILocation(line: 28, column: 21, scope: !3)
!13 = !DILocation(line: 29, column: 21, scope: !3)
!14 = !DILocation(line: 50, column: 14, scope: !3)
!15 = !DILocation(line: 388, column: 19, scope: !16, inlinedAt: !14)
!16 = distinct !DILexicalBlockFile(scope: !3, file: !17, discriminator: 0)
!17 = !DIFile(filename: "functools.py", directory: "/usr/lib/python3.12")
!18 = !DILocation(line: 50, column: 13, scope: !3)
!19 = !DILocation(line: 51, column: 14, scope: !3)
!20 = !DILocation(line: 388, column: 19, scope: !16, inlinedAt: !19)
!21 = !DILocation(line: 51, column: 13, scope: !3)
!22 = !DILocation(line: 52, column: 42, scope: !3)
!23 = !DILocation(line: 52, column: 49, scope: !3)
!24 = !DILocation(line: 52, column: 56, scope: !3)
!25 = !DILocation(line: 52, column: 63, scope: !3)
!26 = !DILocation(line: 52, column: 23, scope: !3)
!27 = !DILocation(line: 60, column: 16, scope: !3)
!28 = !DILocation(line: 60, column: 12, scope: !3)
!29 = !DILocation(line: 61, column: 10, scope: !3)
!30 = !DILocation(line: 68, column: 24, scope: !3)
!31 = !DILocation(line: 64, column: 16, scope: !3)
!32 = !DILocation(line: 77, column: 21, scope: !3)
!33 = !DILocation(line: 77, column: 8, scope: !3)
!34 = !DILocation(line: 19, scope: !3)
