"""Small gfx942 FlyDSL kernel: FP8 unpack -> BF16 packing -> MFMA -> global store.

Run --impl asm / --impl api separately; use normal FlyDSL dump environment vars.
COMPILE_ONLY=1 needs no GPU. FlyDSL 0.3.2 is the reference version.
"""

import argparse
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec


def build(use_api, ntiles=2, ktiles=2):
    @flyc.kernel
    def unpack_mfma(a: fx.Pointer, b: fx.Pointer, out: fx.Pointer):
        lane = fx.gpu.thread_idx.x
        words = fx.make_view(
            b + lane * (ntiles * ktiles), fx.make_layout(ntiles * ktiles, 1)
        ).load()
        decoded = []
        for i in range_constexpr(ntiles * ktiles):
            if const_expr(use_api):
                lo = Vec(rocdl.cvt_pk_f32_fp8(T.f32x2, words[i], word_sel=False))
                hi = Vec(rocdl.cvt_pk_f32_fp8(T.f32x2, words[i], word_sel=True))
            else:
                lo = Vec(
                    llvm.inline_asm(
                        T.f32x2,
                        [words[i].ir_value()],
                        "v_cvt_pk_f32_fp8 $0, $1",
                        "=v,v",
                        has_side_effects=False,
                    )
                )
                hi = Vec(
                    llvm.inline_asm(
                        T.f32x2,
                        [words[i].ir_value()],
                        "v_cvt_pk_f32_fp8_sdwa $0, $1 src0_sel:WORD_1",
                        "=v,v",
                        has_side_effects=False,
                    )
                )
            # Preserve the original high-16-bit BF16 packing, not a numeric cast.
            lo = (lo.bitcast(fx.Uint32) >> 16).to(fx.Uint16)
            hi = (hi.bitcast(fx.Uint32) >> 16).to(fx.Uint16)
            decoded.append(Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Uint16))

        acc = []
        for n in range_constexpr(ntiles):
            acc.append(Vec.filled(4, 0.0, fx.Float32))
        # Decode order is N-major; consumption is K-major, as in the MoE tile.
        for k in range_constexpr(ktiles):
            av = fx.make_view(
                a + lane * (ktiles * 4) + k * 4, fx.make_layout(4, 1)
            ).load()
            for n in range_constexpr(ntiles):
                acc[n] = Vec(
                    rocdl.mfma_f32_16x16x16bf16_1k(
                        T.f32x4,
                        [
                            decoded[n * ktiles + k],
                            av.bitcast(fx.Uint16),
                            acc[n],
                            0,
                            0,
                            0,
                        ],
                    )
                )
        for n in range_constexpr(ntiles):
            fx.make_view(out + lane * (ntiles * 4) + n * 4, fx.make_layout(4, 1)).store(
                acc[n]
            )

    @flyc.jit
    def launch(a: fx.Pointer, b: fx.Pointer, out: fx.Pointer, stream: fx.Stream):
        unpack_mfma(a, b, out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("asm", "api"), default="api")
    parser.add_argument("--n-tiles", type=int, default=2)
    parser.add_argument("--k-tiles", type=int, default=2)
    args = parser.parse_args()
    assert args.n_tiles > 0 and args.k_tiles > 0
    launch = build(args.impl == "api", args.n_tiles, args.k_tiles)
    if os.environ.get("COMPILE_ONLY") == "1":
        pointers = [
            flyc.from_c_void_p(dtype, 0)
            for dtype in (fx.BFloat16, fx.Uint32, fx.Float32)
        ]
        flyc.compile(launch, *pointers, 0)
        print(f"Compiled {args.impl}: ntiles={args.n_tiles}, ktiles={args.k_tiles}")
        return

    import torch

    assert "gfx942" in torch.cuda.get_device_properties(0).gcnArchName
    # Values are loaded from device memory: the compiler cannot constant-fold them.
    a = torch.ones(64, args.k_tiles, 4, dtype=torch.bfloat16, device="cuda")
    b = torch.ones(
        64, args.n_tiles, args.k_tiles, 4, dtype=torch.bfloat16, device="cuda"
    )
    b = b.to(torch.float8_e4m3fnuz).view(torch.int32)
    out = torch.empty(64, args.n_tiles, 4, dtype=torch.float32, device="cuda")
    pointers = [
        flyc.from_c_void_p(dtype, tensor.data_ptr())
        for dtype, tensor in ((fx.BFloat16, a), (fx.Uint32, b), (fx.Float32, out))
    ]
    flyc.compile(launch, *pointers, torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.equal(out, torch.full_like(out, 16 * args.k_tiles))
    print(f"PASS {args.impl}: every result is {16 * args.k_tiles}")


if __name__ == "__main__":
    main()
