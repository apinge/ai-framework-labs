	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	unpack_mfma_0
	.p2align	8
	.type	unpack_mfma_0,@function
unpack_mfma_0:
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02
	.cfi_undefined 16
	.file	1 "/models/toqiu/ai-framework-labs/compiler_issues" "fp8_unpack_mfma_repro.py"
	.loc	1 21 11 prologue_end
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_load_dwordx2 s[2:3], s[0:1], 0x10
	.loc	1 23 8
	v_lshlrev_b32_e32 v1, 4, v0
	.loc	1 52 23
	s_mov_b32 s0, 0x7060302
	.loc	1 77 21
	v_lshlrev_b32_e32 v0, 5, v0
	.loc	1 24 6
	s_waitcnt lgkmcnt(0)
	global_load_dwordx4 v[2:5], v1, s[6:7]
	.loc	1 61 10
	global_load_dwordx4 v[18:21], v1, s[4:5]
	.loc	1 29 21
	s_waitcnt vmcnt(1)
	v_cvt_pk_f32_fp8_sdwa v[6:7], v2 src0_sel:WORD_1
	.loc	1 28 21
	v_cvt_pk_f32_fp8_e32 v[8:9], v2
	v_cvt_pk_f32_fp8_e32 v[10:11], v4
	.loc	1 29 21
	v_cvt_pk_f32_fp8_sdwa v[12:13], v4 src0_sel:WORD_1
	.loc	1 28 21
	v_cvt_pk_f32_fp8_e32 v[14:15], v3
	.loc	1 29 21
	v_cvt_pk_f32_fp8_sdwa v[2:3], v3 src0_sel:WORD_1
	.loc	1 52 23
	v_perm_b32 v7, v7, v6, s0
	v_perm_b32 v6, v9, v8, s0
	v_perm_b32 v13, v13, v12, s0
	v_perm_b32 v12, v11, v10, s0
	.loc	1 29 21
	v_cvt_pk_f32_fp8_sdwa v[16:17], v5 src0_sel:WORD_1
	.loc	1 52 23
	v_perm_b32 v3, v3, v2, s0
	v_perm_b32 v2, v15, v14, s0
	.loc	1 28 21
	v_cvt_pk_f32_fp8_e32 v[14:15], v5
	.loc	1 64 16
	s_waitcnt vmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[6:9], v[6:7], v[18:19], 0
	.loc	1 52 23
	v_perm_b32 v17, v17, v16, s0
	v_perm_b32 v16, v15, v14, s0
	.loc	1 64 16
	v_mfma_f32_16x16x16_bf16 v[10:13], v[12:13], v[18:19], 0
	v_mfma_f32_16x16x16_bf16 v[2:5], v[2:3], v[20:21], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[16:17], v[20:21], v[10:13]
	.loc	1 77 8
	s_nop 5
	global_store_dwordx4 v0, v[2:5], s[2:3]
	global_store_dwordx4 v0, v[6:9], s[2:3] offset:16
	.loc	1 19 0
	s_endpgm
.Ltmp0:
.Lfunc_end0:
	.size	unpack_mfma_0, .Lfunc_end0-unpack_mfma_0
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel unpack_mfma_0
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 22
		.amdhsa_next_free_sgpr 8
		.amdhsa_accum_offset 24
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text

	.set .Lunpack_mfma_0.num_vgpr, 22
	.set .Lunpack_mfma_0.num_agpr, 0
	.set .Lunpack_mfma_0.numbered_sgpr, 8
	.set .Lunpack_mfma_0.num_named_barrier, 0
	.set .Lunpack_mfma_0.private_seg_size, 0
	.set .Lunpack_mfma_0.uses_vcc, 0
	.set .Lunpack_mfma_0.uses_flat_scratch, 0
	.set .Lunpack_mfma_0.has_dyn_sized_stack, 0
	.set .Lunpack_mfma_0.has_recursion, 0
	.set .Lunpack_mfma_0.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1
	.byte	17
	.byte	0
	.byte	37
	.byte	14
	.byte	19
	.byte	5
	.byte	3
	.byte	14
	.byte	16
	.byte	23
	.byte	27
	.byte	14
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0
.Ldebug_info_start0:
	.short	4
	.long	.debug_abbrev
	.byte	8
	.byte	1
	.long	.Linfo_string0
	.short	2
	.long	.Linfo_string1
	.long	.Lline_table_start0
	.long	.Linfo_string2
	.quad	.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"MLIR"
.Linfo_string1:
	.asciz	"fp8_unpack_mfma_repro.py"
.Linfo_string2:
	.asciz	"/models/toqiu/ai-framework-labs/compiler_issues"
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 64
    .name:           unpack_mfma_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 64
      - 1
      - 1
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         unpack_mfma_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     22
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
