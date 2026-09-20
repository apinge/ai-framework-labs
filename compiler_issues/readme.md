### reproduce
```bash
  FLYDSL_RUNTIME_ENABLE_CACHE=0 python /opt/fp8_unpack_mfma_repro.py --impl asm
  FLYDSL_RUNTIME_ENABLE_CACHE=0 python /opt/fp8_unpack_mfma_repro.py --impl api
```
  默认只有一个 64-thread block。在 FlyDSL 0.3.2 and FlyDSL 0.4.0.dev919 / gfx942 上测得：
```
   指标        inline asm    API
  ━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━
   VGPR                20     22
  ──────────  ────────────  ─────
   FP8 转换             8      8
  ──────────  ────────────  ─────
   MFMA                 4      4
  ──────────  ────────────  ─────
   显存写回             2      2
  ──────────  ────────────  ─────
   NOP                  4      1
```