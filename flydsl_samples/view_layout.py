# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Generate Typst visualizations for Layout/ComposedLayout, TiledMma, and TiledCopy.
After the Typst source is written, run:

    typst compile layout.typ

to produce the corresponding PDF document.
"""
# 参考 /root/workspace/FlyDSL/examples/utils/print_typst.py

import flydsl.compiler as flyc
import flydsl.expr as fx

OUTPUT_TYPST = "layout.typ"


@flyc.jit
def visualize():
    N = 4096
    K = 2 * 8192  # 16384

    # arg_b layout from test_gemm_cdna4.py L54
    # Full layout:
    #   shape:  ((16, N//16),  (8, 4, K//32))
    #   stride: ((8, 16*K),   (1, 128, 512))
    #
    # Visualize the inner tile: one 16x32 atom block
    # dim0: 16 rows (stride 8)
    # dim1: (8, 4) = 32 cols — 8 contiguous, repeated 4 times with stride 128
    arg_b_inner = fx.make_layout((4, 4 * 4, 2), (1, 8, 4))= fx.make_layout(
        (16, (8, 4)),
        (8, (1, 128))
    )
    # arg_new_inner = fx.make_layout((4, (4 * 4, 2)), (1, (8, 4)))
    fx.utils.print_typst(arg_b_inner, file=OUTPUT_TYPST)

    # Visualize how tiles repeat across N and K dimensions
    # Show 2 groups in N direction and 2 atoms in K direction
    # arg_b_tiled = fx.make_layout(
    #     ((16, 2), (8, 4, 2)),
    #     ((8, 16 * K), (1, 128, 512))
    # )
    # fx.utils.print_typst(arg_b_tiled, file=OUTPUT_TYPST)


visualize()

print(f"Wrote to {OUTPUT_TYPST}")
