	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	gemm_splitk_0
	.p2align	8
	.type	gemm_splitk_0,@function
gemm_splitk_0:
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02
	.cfi_undefined 16
	.file	1 "/models/toqiu/ai-framework-labs/gemm" "test_gemm.py"
	.loc	1 77 10 prologue_end
	s_load_dword s14, s[0:1], 0x2c
	s_load_dwordx2 s[4:5], s[0:1], 0x10
	s_load_dwordx2 s[8:9], s[0:1], 0x20
.Ltmp0:
	.loc	1 119 19
	v_and_b32_e32 v67, 15, v0
	v_lshrrev_b32_e32 v1, 4, v0
	.loc	1 96 15
	s_waitcnt lgkmcnt(0)
	s_add_i32 s14, s14, -1
	.loc	1 100 13
	s_lshl_b32 s12, s2, 19
	.loc	1 119 19
	v_lshlrev_b32_e32 v2, 13, v67
	v_lshlrev_b32_e32 v68, 3, v1
	.loc	1 96 15
	s_lshl_b32 s6, s14, 14
	s_mov_b32 s7, 0x27000
	v_lshlrev_b32_e32 v69, 4, v0
.Ltmp1:
	.loc	1 164 8
	v_mov_b32_e32 v14, 0
	v_or3_b32 v2, s12, v2, v68
.Ltmp2:
	.loc	1 96 15
	s_addk_i32 s6, 0x4000
	s_and_b32 s5, s5, 0xffff
	.loc	1 97 15
	s_mov_b32 s10, 0x5000000
	s_mov_b32 s11, s7
	s_and_b32 s9, s9, 0xffff
.Ltmp3:
	.loc	1 164 8
	v_lshl_or_b32 v66, s3, 20, v69
	v_lshlrev_b32_e32 v70, 1, v2
	s_mov_b32 s12, 0xfffc0000
	s_mov_b32 s13, -1
	v_mov_b32_e32 v15, v14
	v_mov_b32_e32 v16, v14
	v_mov_b32_e32 v17, v14
	v_mov_b32_e32 v30, v14
	v_mov_b32_e32 v31, v14
	v_mov_b32_e32 v32, v14
	v_mov_b32_e32 v33, v14
	v_mov_b32_e32 v18, v14
	v_mov_b32_e32 v19, v14
	v_mov_b32_e32 v20, v14
	v_mov_b32_e32 v21, v14
	v_mov_b32_e32 v6, v14
	v_mov_b32_e32 v7, v14
	v_mov_b32_e32 v8, v14
	v_mov_b32_e32 v9, v14
	v_mov_b32_e32 v2, v14
	v_mov_b32_e32 v3, v14
	v_mov_b32_e32 v4, v14
	v_mov_b32_e32 v5, v14
	v_mov_b32_e32 v42, v14
	v_mov_b32_e32 v43, v14
	v_mov_b32_e32 v44, v14
	v_mov_b32_e32 v45, v14
	v_mov_b32_e32 v34, v14
	v_mov_b32_e32 v35, v14
	v_mov_b32_e32 v36, v14
	v_mov_b32_e32 v37, v14
	v_mov_b32_e32 v22, v14
	v_mov_b32_e32 v23, v14
	v_mov_b32_e32 v24, v14
	v_mov_b32_e32 v25, v14
	v_mov_b32_e32 v10, v14
	v_mov_b32_e32 v11, v14
	v_mov_b32_e32 v12, v14
	v_mov_b32_e32 v13, v14
	v_mov_b32_e32 v54, v14
	v_mov_b32_e32 v55, v14
	v_mov_b32_e32 v56, v14
	v_mov_b32_e32 v57, v14
	v_mov_b32_e32 v46, v14
	v_mov_b32_e32 v47, v14
	v_mov_b32_e32 v48, v14
	v_mov_b32_e32 v49, v14
	v_mov_b32_e32 v38, v14
	v_mov_b32_e32 v39, v14
	v_mov_b32_e32 v40, v14
	v_mov_b32_e32 v41, v14
	v_mov_b32_e32 v26, v14
	v_mov_b32_e32 v27, v14
	v_mov_b32_e32 v28, v14
	v_mov_b32_e32 v29, v14
	v_mov_b32_e32 v62, v14
	v_mov_b32_e32 v63, v14
	v_mov_b32_e32 v64, v14
	v_mov_b32_e32 v65, v14
	v_mov_b32_e32 v58, v14
	v_mov_b32_e32 v59, v14
	v_mov_b32_e32 v60, v14
	v_mov_b32_e32 v61, v14
	v_mov_b32_e32 v50, v14
	v_mov_b32_e32 v51, v14
	v_mov_b32_e32 v52, v14
	v_mov_b32_e32 v53, v14
.LBB0_1:
.Ltmp4:
	.loc	1 168 12
	v_add_u32_e32 v71, s12, v66
	v_add_u32_e32 v72, 0x40000, v71
	v_add_u32_e32 v73, 0x80000, v71
	v_add_u32_e32 v74, 0xc0000, v71
	v_add_u32_e32 v75, 0x100000, v71
	buffer_load_dwordx4 v[76:79], v72, s[8:11], 0 offen
	buffer_load_dwordx4 v[80:83], v73, s[8:11], 0 offen
	buffer_load_dwordx4 v[84:87], v74, s[8:11], 0 offen
	buffer_load_dwordx4 v[88:91], v75, s[8:11], 0 offen
	.loc	1 167 12
	v_or_b32_e32 v72, 0x40000, v70
	v_or_b32_e32 v73, 0x80000, v70
	v_or_b32_e32 v74, 0xc0000, v70
	buffer_load_dwordx4 v[92:95], v70, s[4:7], 0 offen
	buffer_load_dwordx4 v[96:99], v72, s[4:7], 0 offen
	buffer_load_dwordx4 v[100:103], v73, s[4:7], 0 offen
	buffer_load_dwordx4 v[104:107], v74, s[4:7], 0 offen
	.loc	1 168 12
	v_add_u32_e32 v72, 0x41000, v71
	v_add_u32_e32 v73, 0x81000, v71
	v_add_u32_e32 v74, 0xc1000, v71
	v_add_u32_e32 v71, 0x101000, v71
	buffer_load_dwordx4 v[108:111], v72, s[8:11], 0 offen
	buffer_load_dwordx4 v[112:115], v73, s[8:11], 0 offen
	buffer_load_dwordx4 v[116:119], v74, s[8:11], 0 offen
	buffer_load_dwordx4 v[120:123], v71, s[8:11], 0 offen
	.loc	1 167 12
	v_or_b32_e32 v72, 0x80100, v70
	v_or_b32_e32 v73, 0xc0100, v70
	v_or_b32_e32 v71, 0x40100, v70
	buffer_load_dwordx4 v[124:127], v70, s[4:7], 0 offen offset:256
	buffer_load_dwordx4 v[128:131], v71, s[4:7], 0 offen
	buffer_load_dwordx4 v[132:135], v72, s[4:7], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[72:75], v73, s[4:7], 0 offen
.Ltmp5:
	.loc	1 164 8
	s_add_u32 s12, s12, 0x2000
	s_addc_u32 s13, s13, 0
	s_cmp_lg_u64 s[12:13], 0
	v_add_u32_e32 v70, 0x200, v70
.Ltmp6:
	.loc	1 170 16
	s_waitcnt vmcnt(11)
	v_mfma_f32_16x16x16_bf16 v[30:33], v[76:77], v[92:93], v[30:33]
	s_waitcnt vmcnt(10)
	v_mfma_f32_16x16x16_bf16 v[42:45], v[76:77], v[96:97], v[42:45]
	s_waitcnt vmcnt(9)
	v_mfma_f32_16x16x16_bf16 v[54:57], v[76:77], v[100:101], v[54:57]
	s_waitcnt vmcnt(8)
	v_mfma_f32_16x16x16_bf16 v[62:65], v[76:77], v[104:105], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[80:81], v[92:93], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[80:81], v[96:97], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[80:81], v[100:101], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[80:81], v[104:105], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[84:85], v[92:93], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[84:85], v[96:97], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[84:85], v[100:101], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[84:85], v[104:105], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[88:89], v[92:93], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[88:89], v[96:97], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[88:89], v[100:101], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[88:89], v[104:105], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[78:79], v[94:95], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[78:79], v[98:99], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[78:79], v[102:103], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[78:79], v[106:107], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[82:83], v[94:95], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[82:83], v[98:99], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[82:83], v[102:103], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[82:83], v[106:107], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[86:87], v[94:95], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[86:87], v[98:99], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[86:87], v[102:103], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[86:87], v[106:107], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[90:91], v[94:95], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[90:91], v[98:99], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[90:91], v[102:103], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[90:91], v[106:107], v[14:17]
	s_waitcnt vmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[30:33], v[108:109], v[124:125], v[30:33]
	s_waitcnt vmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[42:45], v[108:109], v[128:129], v[42:45]
	s_waitcnt vmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[54:57], v[108:109], v[132:133], v[54:57]
	s_waitcnt vmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[62:65], v[108:109], v[72:73], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[112:113], v[124:125], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[112:113], v[128:129], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[112:113], v[132:133], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[112:113], v[72:73], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[116:117], v[124:125], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[116:117], v[128:129], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[116:117], v[132:133], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[116:117], v[72:73], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[120:121], v[124:125], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[120:121], v[128:129], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[120:121], v[132:133], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[120:121], v[72:73], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[110:111], v[126:127], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[110:111], v[130:131], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[110:111], v[134:135], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[110:111], v[74:75], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[114:115], v[126:127], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[114:115], v[130:131], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[114:115], v[134:135], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[114:115], v[74:75], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[118:119], v[126:127], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[118:119], v[130:131], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[118:119], v[134:135], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[118:119], v[74:75], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[122:123], v[126:127], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[122:123], v[130:131], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[122:123], v[134:135], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[122:123], v[74:75], v[14:17]
.Ltmp7:
	.loc	1 164 8
	s_cbranch_scc1 .LBB0_1
.Ltmp8:
	.loc	1 120 19
	v_lshlrev_b32_e32 v0, 3, v0
	.loc	1 186 25
	v_lshlrev_b32_e32 v66, 6, v67
	v_lshlrev_b32_e32 v70, 2, v1
	v_and_or_b32 v66, v70, 12, v66
	v_and_b32_e32 v69, 0xc00, v69
	.loc	1 187 4
	v_and_b32_e32 v0, 56, v0
	.loc	1 186 25
	v_or_b32_e32 v70, v66, v69
	.loc	1 187 4
	v_xor_b32_e32 v66, v66, v0
	v_or_b32_e32 v66, v66, v69
	v_lshlrev_b32_e32 v66, 2, v66
	ds_write_b128 v66, v[30:33]
	ds_write_b128 v66, v[42:45] offset:16384
	ds_write_b128 v66, v[54:57] offset:32768
	ds_write_b128 v66, v[62:65] offset:49152
	v_or_b32_e32 v30, 16, v70
	v_xor_b32_e32 v30, v30, v0
	v_lshlrev_b32_e32 v30, 2, v30
	ds_write_b128 v30, v[18:21]
	v_or_b32_e32 v18, 0x1010, v70
	v_xor_b32_e32 v18, v18, v0
	v_lshlrev_b32_e32 v18, 2, v18
	ds_write_b128 v18, v[34:37]
	v_or_b32_e32 v18, 0x2010, v70
	v_xor_b32_e32 v18, v18, v0
	v_lshlrev_b32_e32 v18, 2, v18
	ds_write_b128 v18, v[46:49]
	v_or_b32_e32 v18, 0x3010, v70
	v_xor_b32_e32 v18, v18, v0
	v_lshlrev_b32_e32 v18, 2, v18
	ds_write_b128 v18, v[58:61]
	v_or_b32_e32 v18, 32, v70
	v_xor_b32_e32 v18, v18, v0
	v_lshlrev_b32_e32 v18, 2, v18
	ds_write_b128 v18, v[6:9]
	v_or_b32_e32 v6, 0x1020, v70
	v_xor_b32_e32 v6, v6, v0
	v_lshlrev_b32_e32 v6, 2, v6
	ds_write_b128 v6, v[22:25]
	v_or_b32_e32 v6, 0x2020, v70
	v_xor_b32_e32 v6, v6, v0
	v_lshlrev_b32_e32 v6, 2, v6
	ds_write_b128 v6, v[38:41]
	v_or_b32_e32 v6, 0x3020, v70
	v_xor_b32_e32 v6, v6, v0
	v_lshlrev_b32_e32 v6, 2, v6
	ds_write_b128 v6, v[50:53]
	v_or_b32_e32 v6, 48, v70
	v_xor_b32_e32 v6, v6, v0
	v_lshlrev_b32_e32 v6, 2, v6
	ds_write_b128 v6, v[2:5]
	v_or_b32_e32 v2, 0x1030, v70
	v_xor_b32_e32 v2, v2, v0
	v_lshlrev_b32_e32 v2, 2, v2
	ds_write_b128 v2, v[10:13]
	v_or_b32_e32 v2, 0x2030, v70
	v_xor_b32_e32 v2, v2, v0
	v_lshlrev_b32_e32 v2, 2, v2
	ds_write_b128 v2, v[26:29]
	v_or_b32_e32 v2, 0x3030, v70
	v_xor_b32_e32 v0, v2, v0
	v_lshlrev_b32_e32 v0, 2, v0
	ds_write_b128 v0, v[14:17]
	.loc	1 198 25
	v_lshlrev_b32_e32 v0, 2, v67
	v_lshl_or_b32 v2, v1, 6, v0
	.loc	1 200 4
	v_and_b32_e32 v3, 56, v68
	v_xor_b32_e32 v2, v2, v3
	v_lshlrev_b32_e32 v62, 2, v2
.Ltmp9:
	.loc	1 77 10
	s_load_dwordx2 s[0:1], s[0:1], 0x0
.Ltmp10:
	.loc	1 188 4
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 200 4
	ds_read_b128 v[2:5], v62
	ds_read_b128 v[6:9], v62 offset:4096
	ds_read_b128 v[10:13], v62 offset:8192
	ds_read_b128 v[14:17], v62 offset:12288
	ds_read_b128 v[18:21], v62 offset:16384
	ds_read_b128 v[22:25], v62 offset:20480
	ds_read_b128 v[26:29], v62 offset:24576
	ds_read_b128 v[30:33], v62 offset:28672
	ds_read_b128 v[34:37], v62 offset:32768
	ds_read_b128 v[38:41], v62 offset:36864
	ds_read_b128 v[42:45], v62 offset:49152
	ds_read_b128 v[46:49], v62 offset:53248
	ds_read_b128 v[50:53], v62 offset:40960
	ds_read_b128 v[54:57], v62 offset:45056
	ds_read_b128 v[58:61], v62 offset:57344
	ds_read_b128 v[62:65], v62 offset:61440
	.file	2 "/usr/lib/python3.12" "functools.py"
	.loc	2 388 19
	s_waitcnt lgkmcnt(4)
	v_pk_add_f32 v[44:45], v[44:45], v[48:49]
	v_pk_add_f32 v[36:37], v[36:37], v[40:41]
	v_pk_add_f32 v[20:21], v[20:21], v[24:25]
	v_pk_add_f32 v[4:5], v[4:5], v[8:9]
	v_pk_add_f32 v[8:9], v[42:43], v[46:47]
	v_pk_add_f32 v[24:25], v[34:35], v[38:39]
	v_pk_add_f32 v[18:19], v[18:19], v[22:23]
	v_pk_add_f32 v[2:3], v[2:3], v[6:7]
	v_pk_add_f32 v[4:5], v[4:5], v[12:13]
	v_pk_add_f32 v[6:7], v[20:21], v[28:29]
	s_waitcnt lgkmcnt(3)
	v_pk_add_f32 v[12:13], v[36:37], v[52:53]
	s_waitcnt lgkmcnt(1)
	v_pk_add_f32 v[20:21], v[44:45], v[60:61]
	v_pk_add_f32 v[2:3], v[2:3], v[10:11]
	v_pk_add_f32 v[10:11], v[18:19], v[26:27]
	v_pk_add_f32 v[18:19], v[24:25], v[50:51]
	v_pk_add_f32 v[8:9], v[8:9], v[58:59]
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[20:21], v[20:21], v[64:65]
	v_pk_add_f32 v[12:13], v[12:13], v[56:57]
	v_pk_add_f32 v[6:7], v[6:7], v[32:33]
	v_pk_add_f32 v[4:5], v[4:5], v[16:17]
	v_pk_add_f32 v[8:9], v[8:9], v[62:63]
	v_pk_add_f32 v[16:17], v[18:19], v[54:55]
	v_pk_add_f32 v[10:11], v[10:11], v[30:31]
	v_pk_add_f32 v[2:3], v[2:3], v[14:15]
	s_mov_b32 s4, 0x8000
	.loc	1 104 13
	s_mul_i32 s5, s2, 0x50000
	s_lshl_b32 s6, s3, 6
	.loc	2 388 19
	v_pk_add_f32 v[4:5], v[4:5], s[4:5] op_sel_hi:[1,0]
	v_pk_add_f32 v[6:7], v[6:7], s[4:5] op_sel_hi:[1,0]
	v_pk_add_f32 v[12:13], v[12:13], s[4:5] op_sel_hi:[1,0]
	v_pk_add_f32 v[14:15], v[20:21], s[4:5] op_sel_hi:[1,0]
	v_pk_add_f32 v[2:3], v[2:3], s[4:5] op_sel_hi:[1,0]
	v_pk_add_f32 v[10:11], v[10:11], s[4:5] op_sel_hi:[1,0]
	v_pk_add_f32 v[16:17], v[16:17], s[4:5] op_sel_hi:[1,0]
	v_pk_add_f32 v[8:9], v[8:9], s[4:5] op_sel_hi:[1,0]
	.loc	1 210 10
	s_mov_b32 s4, 0x7060302
	v_perm_b32 v15, v15, v14, s4
	v_perm_b32 v13, v13, v12, s4
	v_perm_b32 v7, v7, v6, s4
	v_perm_b32 v5, v5, v4, s4
	v_perm_b32 v14, v9, v8, s4
	v_perm_b32 v12, v17, v16, s4
	v_perm_b32 v6, v11, v10, s4
	v_perm_b32 v4, v3, v2, s4
	.loc	1 220 21
	s_add_i32 s4, s5, s6
	s_movk_i32 s5, 0x1400
	.loc	1 104 13
	v_mov_b32_e32 v2, s4
	v_mad_u32_u24 v1, v1, s5, v2
	.loc	1 220 21
	v_or_b32_e32 v0, v1, v0
	.loc	1 98 15
	s_mul_i32 s2, s14, 0x2800
	.loc	1 221 4
	v_lshlrev_b32_e32 v0, 1, v0
	.loc	1 98 15
	s_addk_i32 s2, 0x2800
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x27000
	.loc	1 221 4
	v_add_u32_e32 v1, 0x28000, v0
	buffer_store_dwordx2 v[4:5], v0, s[0:3], 0 offen
	buffer_store_dwordx2 v[6:7], v1, s[0:3], 0 offen
	v_add_u32_e32 v1, 0x50000, v0
	v_add_u32_e32 v0, 0x78000, v0
	buffer_store_dwordx2 v[12:13], v1, s[0:3], 0 offen
	buffer_store_dwordx2 v[14:15], v0, s[0:3], 0 offen
	.loc	1 829 8
	s_endpgm
.Ltmp11:
.Lfunc_end0:
	.size	gemm_splitk_0, .Lfunc_end0-gemm_splitk_0
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_splitk_0
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 48
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 257
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 136
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

	.set .Lgemm_splitk_0.num_vgpr, 136
	.set .Lgemm_splitk_0.num_agpr, 0
	.set .Lgemm_splitk_0.numbered_sgpr, 15
	.set .Lgemm_splitk_0.num_named_barrier, 0
	.set .Lgemm_splitk_0.private_seg_size, 0
	.set .Lgemm_splitk_0.uses_vcc, 0
	.set .Lgemm_splitk_0.uses_flat_scratch, 0
	.set .Lgemm_splitk_0.has_dyn_sized_stack, 0
	.set .Lgemm_splitk_0.has_recursion, 0
	.set .Lgemm_splitk_0.has_indirect_call, 0
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
	.byte	1
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
	.byte	2
	.byte	46
	.byte	0
	.byte	3
	.byte	14
	.byte	32
	.byte	11
	.byte	0
	.byte	0
	.byte	3
	.byte	46
	.byte	1
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	49
	.byte	19
	.byte	0
	.byte	0
	.byte	4
	.byte	29
	.byte	0
	.byte	49
	.byte	19
	.byte	85
	.byte	23
	.byte	88
	.byte	11
	.byte	89
	.byte	5
	.byte	87
	.byte	11
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
	.byte	2
	.long	.Linfo_string3
	.byte	1
	.byte	3
	.quad	.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
	.long	42
	.byte	4
	.long	42
	.long	.Ldebug_ranges0
	.byte	1
	.short	3112
	.byte	27
	.byte	0
	.byte	0
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp0-.Lfunc_begin0
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"MLIR"
.Linfo_string1:
	.asciz	"test_gemm.py"
.Linfo_string2:
	.asciz	"/models/toqiu/ai-framework-labs/gemm"
.Linfo_string3:
	.asciz	"gemm_splitk_0"
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
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 256
    .name:           gemm_splitk_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     21
    .sgpr_spill_count: 0
    .symbol:         gemm_splitk_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     136
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
