#!/usr/bin/env python3
"""Standalone fixed-tiling NPU load-order comparison (copy only this file).

Activate pta_213_cicd and CANN, then run:
    python run.py                 # Compile front/near, multibuffer=False/True
    python run.py --check         # Also launch compilable variants and check eager
    python run.py --show-config   # Display the exact tiling without NPU imports

The two embedded DSL kernels differ only in load placement. No torch.compile,
Inductor codegen, autotune, sibling Python files or captured inputs are needed.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback


RUNTIME = {"x0_numel": 4096, "r1_numel": 2048, "X0BLOCK": 64, "grid": [64, 1, 1]}
TILING = {"X0BLOCK_SUB": 1, "R1BLOCK_SUB": 2048}
TARGET = {"backend": "npu", "arch": "Ascend950PR_9579", "warp_size": 32}
BASE_OPTIONS = {"num_warps": 1, "num_stages": 2, "debug": False, "compile_mode": "simd_simt_template"}


def print_config():
    print(
        "\nFixed tiling (identical for front and near):\n"
        "  Total extent:       x0_numel=4096, r1_numel=2048\n"
        "  Grid:               (64, 1, 1)\n"
        "  Program coverage:   X0BLOCK=64 output columns\n"
        "  DSL subtile:        X0BLOCK_SUB=1, R1BLOCK_SUB=2048\n"
        "  Load tensor shape:  (R1BLOCK_SUB, X0BLOCK_SUB) = (2048, 1), FP32\n"
        "  Loop per program:   64 x-iterations, each reducing all 2048 rows\n"
        "  Matrix layout:      shape=(2048, 4096), stride=(4096, 1)\n"
        "  Compile options:    num_warps=1, num_stages=2, compile_mode=simd_simt_template\n"
        "  Target:             npu / Ascend950PR_9579 / warp_size=32\n"
        "  Comparison:         same tiling; only load placement changes within each pair\n",
        flush=True,
    )


def load_kernels():
    # Lazy imports let --help/--show-config work without an NPU software stack.
    global torch, torch_npu, triton, tl
    import torch
    import torch_npu
    import triton
    import triton.language as tl

    # Original generated DSL is embedded verbatim apart from name/indentation.
    # Its long generated signature is intentionally preserved.
    @triton.jit
    def kernel_front(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
        R1BLOCK_SUB: tl.constexpr = 2048
        x0_offset = tl.program_id(0) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            r1 = base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp1 = tl.load(in_ptr1 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp2 = tl.load(in_ptr2 + (r1), None)
            tmp4 = tl.load(in_ptr3 + (r1), None)
            tmp11 = tl.load(in_ptr4 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp12 = tl.load(in_ptr5 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp14 = tl.load(in_ptr6 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp16 = tl.load(in_ptr7 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp18 = tl.load(in_ptr8 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp19 = tl.load(in_ptr9 + (r1), None)
            tmp21 = tl.load(in_ptr10 + (r1), None)
            tmp28 = tl.load(in_ptr11 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp29 = tl.load(in_ptr12 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp31 = tl.load(in_ptr13 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp33 = tl.load(in_ptr14 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp35 = tl.load(in_ptr15 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp36 = tl.load(in_ptr16 + (r1), None)
            tmp38 = tl.load(in_ptr17 + (r1), None)
            tmp45 = tl.load(in_ptr18 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp46 = tl.load(in_ptr19 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp48 = tl.load(in_ptr20 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp50 = tl.load(in_ptr21 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp52 = tl.load(in_ptr22 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp53 = tl.load(in_ptr23 + (r1), None)
            tmp55 = tl.load(in_ptr24 + (r1), None)
            tmp62 = tl.load(in_ptr25 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp63 = tl.load(in_ptr26 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp65 = tl.load(in_ptr27 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp67 = tl.load(in_ptr28 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp69 = tl.load(in_ptr29 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp70 = tl.load(in_ptr30 + (r1), None)
            tmp72 = tl.load(in_ptr31 + (r1), None)
            tmp79 = tl.load(in_ptr32 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp80 = tl.load(in_ptr33 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp82 = tl.load(in_ptr34 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp84 = tl.load(in_ptr35 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp86 = tl.load(in_ptr36 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp87 = tl.load(in_ptr37 + (r1), None)
            tmp89 = tl.load(in_ptr38 + (r1), None)
            tmp96 = tl.load(in_ptr39 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp97 = tl.load(in_ptr40 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp99 = tl.load(in_ptr41 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp101 = tl.load(in_ptr42 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp103 = tl.load(in_ptr43 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp104 = tl.load(in_ptr44 + (r1), None)
            tmp106 = tl.load(in_ptr45 + (r1), None)
            tmp113 = tl.load(in_ptr46 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp114 = tl.load(in_ptr47 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp116 = tl.load(in_ptr48 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp118 = tl.load(in_ptr49 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp120 = tl.load(in_ptr50 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp121 = tl.load(in_ptr51 + (r1), None)
            tmp123 = tl.load(in_ptr52 + (r1), None)
            tmp130 = tl.load(in_ptr53 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp131 = tl.load(in_ptr54 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp133 = tl.load(in_ptr55 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp135 = tl.load(in_ptr56 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp137 = tl.load(in_ptr57 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp138 = tl.load(in_ptr58 + (r1), None)
            tmp140 = tl.load(in_ptr59 + (r1), None)
            tmp147 = tl.load(in_ptr60 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp148 = tl.load(in_ptr61 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp150 = tl.load(in_ptr62 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp152 = tl.load(in_ptr63 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp154 = tl.load(in_ptr64 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp155 = tl.load(in_ptr65 + (r1), None)
            tmp157 = tl.load(in_ptr66 + (r1), None)
            tmp164 = tl.load(in_ptr67 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp165 = tl.load(in_ptr68 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp167 = tl.load(in_ptr69 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp169 = tl.load(in_ptr70 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp171 = tl.load(in_ptr71 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp172 = tl.load(in_ptr72 + (r1), None)
            tmp174 = tl.load(in_ptr73 + (r1), None)
            tmp181 = tl.load(in_ptr74 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp182 = tl.load(in_ptr75 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp184 = tl.load(in_ptr76 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp186 = tl.load(in_ptr77 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp188 = tl.load(in_ptr78 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp189 = tl.load(in_ptr79 + (r1), None)
            tmp191 = tl.load(in_ptr80 + (r1), None)
            tmp3 = tmp1 - tmp2
            tmp5 = tmp3 * tmp4
            tmp6 = tmp0 * tmp5
            tmp8 = tl.sum(tmp6, 0).reshape(1, X0BLOCK_SUB)
            tmp10 = tl.sum(tmp0, 0).reshape(1, X0BLOCK_SUB)
            tmp13 = tmp11 + tmp12
            tmp15 = tmp13 + tmp14
            tmp17 = tmp15 + tmp16
            tmp20 = tmp18 - tmp19
            tmp22 = tmp20 * tmp21
            tmp23 = tmp17 * tmp22
            tmp25 = tl.sum(tmp23, 0).reshape(1, X0BLOCK_SUB)
            tmp27 = tl.sum(tmp17, 0).reshape(1, X0BLOCK_SUB)
            tmp30 = tmp28 + tmp29
            tmp32 = tmp30 + tmp31
            tmp34 = tmp32 + tmp33
            tmp37 = tmp35 - tmp36
            tmp39 = tmp37 * tmp38
            tmp40 = tmp34 * tmp39
            tmp42 = tl.sum(tmp40, 0).reshape(1, X0BLOCK_SUB)
            tmp44 = tl.sum(tmp34, 0).reshape(1, X0BLOCK_SUB)
            tmp47 = tmp45 + tmp46
            tmp49 = tmp47 + tmp48
            tmp51 = tmp49 + tmp50
            tmp54 = tmp52 - tmp53
            tmp56 = tmp54 * tmp55
            tmp57 = tmp51 * tmp56
            tmp59 = tl.sum(tmp57, 0).reshape(1, X0BLOCK_SUB)
            tmp61 = tl.sum(tmp51, 0).reshape(1, X0BLOCK_SUB)
            tmp64 = tmp62 + tmp63
            tmp66 = tmp64 + tmp65
            tmp68 = tmp66 + tmp67
            tmp71 = tmp69 - tmp70
            tmp73 = tmp71 * tmp72
            tmp74 = tmp68 * tmp73
            tmp76 = tl.sum(tmp74, 0).reshape(1, X0BLOCK_SUB)
            tmp78 = tl.sum(tmp68, 0).reshape(1, X0BLOCK_SUB)
            tmp81 = tmp79 + tmp80
            tmp83 = tmp81 + tmp82
            tmp85 = tmp83 + tmp84
            tmp88 = tmp86 - tmp87
            tmp90 = tmp88 * tmp89
            tmp91 = tmp85 * tmp90
            tmp93 = tl.sum(tmp91, 0).reshape(1, X0BLOCK_SUB)
            tmp95 = tl.sum(tmp85, 0).reshape(1, X0BLOCK_SUB)
            tmp98 = tmp96 + tmp97
            tmp100 = tmp98 + tmp99
            tmp102 = tmp100 + tmp101
            tmp105 = tmp103 - tmp104
            tmp107 = tmp105 * tmp106
            tmp108 = tmp102 * tmp107
            tmp110 = tl.sum(tmp108, 0).reshape(1, X0BLOCK_SUB)
            tmp112 = tl.sum(tmp102, 0).reshape(1, X0BLOCK_SUB)
            tmp115 = tmp113 + tmp114
            tmp117 = tmp115 + tmp116
            tmp119 = tmp117 + tmp118
            tmp122 = tmp120 - tmp121
            tmp124 = tmp122 * tmp123
            tmp125 = tmp119 * tmp124
            tmp127 = tl.sum(tmp125, 0).reshape(1, X0BLOCK_SUB)
            tmp129 = tl.sum(tmp119, 0).reshape(1, X0BLOCK_SUB)
            tmp132 = tmp130 + tmp131
            tmp134 = tmp132 + tmp133
            tmp136 = tmp134 + tmp135
            tmp139 = tmp137 - tmp138
            tmp141 = tmp139 * tmp140
            tmp142 = tmp136 * tmp141
            tmp144 = tl.sum(tmp142, 0).reshape(1, X0BLOCK_SUB)
            tmp146 = tl.sum(tmp136, 0).reshape(1, X0BLOCK_SUB)
            tmp149 = tmp147 + tmp148
            tmp151 = tmp149 + tmp150
            tmp153 = tmp151 + tmp152
            tmp156 = tmp154 - tmp155
            tmp158 = tmp156 * tmp157
            tmp159 = tmp153 * tmp158
            tmp161 = tl.sum(tmp159, 0).reshape(1, X0BLOCK_SUB)
            tmp163 = tl.sum(tmp153, 0).reshape(1, X0BLOCK_SUB)
            tmp166 = tmp164 + tmp165
            tmp168 = tmp166 + tmp167
            tmp170 = tmp168 + tmp169
            tmp173 = tmp171 - tmp172
            tmp175 = tmp173 * tmp174
            tmp176 = tmp170 * tmp175
            tmp178 = tl.sum(tmp176, 0).reshape(1, X0BLOCK_SUB)
            tmp180 = tl.sum(tmp170, 0).reshape(1, X0BLOCK_SUB)
            tmp183 = tmp181 + tmp182
            tmp185 = tmp183 + tmp184
            tmp187 = tmp185 + tmp186
            tmp190 = tmp188 - tmp189
            tmp192 = tmp190 * tmp191
            tmp193 = tmp187 * tmp192
            tmp195 = tl.sum(tmp193, 0).reshape(1, X0BLOCK_SUB)
            tmp197 = tl.sum(tmp187, 0).reshape(1, X0BLOCK_SUB)
            tmp198 = tmp8 + tmp25
            tmp199 = tmp198 + tmp42
            tmp200 = tmp199 + tmp59
            tmp201 = tmp200 + tmp76
            tmp202 = tmp201 + tmp93
            tmp203 = tmp202 + tmp127
            tmp204 = tmp203 + tmp161
            tmp205 = tmp204 + tmp195
            tmp206 = tmp205 + tmp110
            tmp207 = tmp206 + tmp144
            tmp208 = tmp207 + tmp178
            tmp209 = tmp10 + tmp27
            tmp210 = tmp209 + tmp44
            tmp211 = tmp210 + tmp61
            tmp212 = tmp211 + tmp78
            tmp213 = tmp212 + tmp95
            tmp214 = tmp213 + tmp129
            tmp215 = tmp214 + tmp163
            tmp216 = tmp215 + tmp197
            tmp217 = tmp216 + tmp112
            tmp218 = tmp217 + tmp146
            tmp219 = tmp218 + tmp180
            tl.store(in_out_ptr0 + (x0), tmp208, x0_mask)
            tl.store(in_out_ptr1 + (x0), tmp219, x0_mask)

    @triton.jit
    def kernel_near(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
        R1BLOCK_SUB: tl.constexpr = 2048
        x0_offset = tl.program_id(0) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            r1 = base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp1 = tl.load(in_ptr1 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp2 = tl.load(in_ptr2 + (r1), None)
            tmp3 = tmp1 - tmp2
            tmp4 = tl.load(in_ptr3 + (r1), None)
            tmp5 = tmp3 * tmp4
            tmp0 = tl.load(in_ptr0 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp6 = tmp0 * tmp5
            tmp8 = tl.sum(tmp6, 0).reshape(1, X0BLOCK_SUB)
            tmp10 = tl.sum(tmp0, 0).reshape(1, X0BLOCK_SUB)
            tmp11 = tl.load(in_ptr4 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp12 = tl.load(in_ptr5 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp13 = tmp11 + tmp12
            tmp14 = tl.load(in_ptr6 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp15 = tmp13 + tmp14
            tmp16 = tl.load(in_ptr7 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp17 = tmp15 + tmp16
            tmp18 = tl.load(in_ptr8 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp19 = tl.load(in_ptr9 + (r1), None)
            tmp20 = tmp18 - tmp19
            tmp21 = tl.load(in_ptr10 + (r1), None)
            tmp22 = tmp20 * tmp21
            tmp23 = tmp17 * tmp22
            tmp25 = tl.sum(tmp23, 0).reshape(1, X0BLOCK_SUB)
            tmp27 = tl.sum(tmp17, 0).reshape(1, X0BLOCK_SUB)
            tmp28 = tl.load(in_ptr11 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp29 = tl.load(in_ptr12 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp30 = tmp28 + tmp29
            tmp31 = tl.load(in_ptr13 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp32 = tmp30 + tmp31
            tmp33 = tl.load(in_ptr14 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp34 = tmp32 + tmp33
            tmp35 = tl.load(in_ptr15 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp36 = tl.load(in_ptr16 + (r1), None)
            tmp37 = tmp35 - tmp36
            tmp38 = tl.load(in_ptr17 + (r1), None)
            tmp39 = tmp37 * tmp38
            tmp40 = tmp34 * tmp39
            tmp42 = tl.sum(tmp40, 0).reshape(1, X0BLOCK_SUB)
            tmp44 = tl.sum(tmp34, 0).reshape(1, X0BLOCK_SUB)
            tmp45 = tl.load(in_ptr18 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp46 = tl.load(in_ptr19 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp47 = tmp45 + tmp46
            tmp48 = tl.load(in_ptr20 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp49 = tmp47 + tmp48
            tmp50 = tl.load(in_ptr21 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp51 = tmp49 + tmp50
            tmp52 = tl.load(in_ptr22 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp53 = tl.load(in_ptr23 + (r1), None)
            tmp54 = tmp52 - tmp53
            tmp55 = tl.load(in_ptr24 + (r1), None)
            tmp56 = tmp54 * tmp55
            tmp57 = tmp51 * tmp56
            tmp59 = tl.sum(tmp57, 0).reshape(1, X0BLOCK_SUB)
            tmp61 = tl.sum(tmp51, 0).reshape(1, X0BLOCK_SUB)
            tmp62 = tl.load(in_ptr25 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp63 = tl.load(in_ptr26 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp64 = tmp62 + tmp63
            tmp65 = tl.load(in_ptr27 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp66 = tmp64 + tmp65
            tmp67 = tl.load(in_ptr28 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp68 = tmp66 + tmp67
            tmp69 = tl.load(in_ptr29 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp70 = tl.load(in_ptr30 + (r1), None)
            tmp71 = tmp69 - tmp70
            tmp72 = tl.load(in_ptr31 + (r1), None)
            tmp73 = tmp71 * tmp72
            tmp74 = tmp68 * tmp73
            tmp76 = tl.sum(tmp74, 0).reshape(1, X0BLOCK_SUB)
            tmp78 = tl.sum(tmp68, 0).reshape(1, X0BLOCK_SUB)
            tmp79 = tl.load(in_ptr32 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp80 = tl.load(in_ptr33 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp81 = tmp79 + tmp80
            tmp82 = tl.load(in_ptr34 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp83 = tmp81 + tmp82
            tmp84 = tl.load(in_ptr35 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp85 = tmp83 + tmp84
            tmp86 = tl.load(in_ptr36 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp87 = tl.load(in_ptr37 + (r1), None)
            tmp88 = tmp86 - tmp87
            tmp89 = tl.load(in_ptr38 + (r1), None)
            tmp90 = tmp88 * tmp89
            tmp91 = tmp85 * tmp90
            tmp93 = tl.sum(tmp91, 0).reshape(1, X0BLOCK_SUB)
            tmp95 = tl.sum(tmp85, 0).reshape(1, X0BLOCK_SUB)
            tmp96 = tl.load(in_ptr39 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp97 = tl.load(in_ptr40 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp98 = tmp96 + tmp97
            tmp99 = tl.load(in_ptr41 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp100 = tmp98 + tmp99
            tmp101 = tl.load(in_ptr42 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp102 = tmp100 + tmp101
            tmp103 = tl.load(in_ptr43 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp104 = tl.load(in_ptr44 + (r1), None)
            tmp105 = tmp103 - tmp104
            tmp106 = tl.load(in_ptr45 + (r1), None)
            tmp107 = tmp105 * tmp106
            tmp108 = tmp102 * tmp107
            tmp110 = tl.sum(tmp108, 0).reshape(1, X0BLOCK_SUB)
            tmp112 = tl.sum(tmp102, 0).reshape(1, X0BLOCK_SUB)
            tmp113 = tl.load(in_ptr46 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp114 = tl.load(in_ptr47 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp115 = tmp113 + tmp114
            tmp116 = tl.load(in_ptr48 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp117 = tmp115 + tmp116
            tmp118 = tl.load(in_ptr49 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp119 = tmp117 + tmp118
            tmp120 = tl.load(in_ptr50 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp121 = tl.load(in_ptr51 + (r1), None)
            tmp122 = tmp120 - tmp121
            tmp123 = tl.load(in_ptr52 + (r1), None)
            tmp124 = tmp122 * tmp123
            tmp125 = tmp119 * tmp124
            tmp127 = tl.sum(tmp125, 0).reshape(1, X0BLOCK_SUB)
            tmp129 = tl.sum(tmp119, 0).reshape(1, X0BLOCK_SUB)
            tmp130 = tl.load(in_ptr53 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp131 = tl.load(in_ptr54 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp132 = tmp130 + tmp131
            tmp133 = tl.load(in_ptr55 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp134 = tmp132 + tmp133
            tmp135 = tl.load(in_ptr56 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp136 = tmp134 + tmp135
            tmp137 = tl.load(in_ptr57 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp138 = tl.load(in_ptr58 + (r1), None)
            tmp139 = tmp137 - tmp138
            tmp140 = tl.load(in_ptr59 + (r1), None)
            tmp141 = tmp139 * tmp140
            tmp142 = tmp136 * tmp141
            tmp144 = tl.sum(tmp142, 0).reshape(1, X0BLOCK_SUB)
            tmp146 = tl.sum(tmp136, 0).reshape(1, X0BLOCK_SUB)
            tmp147 = tl.load(in_ptr60 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp148 = tl.load(in_ptr61 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp149 = tmp147 + tmp148
            tmp150 = tl.load(in_ptr62 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp151 = tmp149 + tmp150
            tmp152 = tl.load(in_ptr63 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp153 = tmp151 + tmp152
            tmp154 = tl.load(in_ptr64 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp155 = tl.load(in_ptr65 + (r1), None)
            tmp156 = tmp154 - tmp155
            tmp157 = tl.load(in_ptr66 + (r1), None)
            tmp158 = tmp156 * tmp157
            tmp159 = tmp153 * tmp158
            tmp161 = tl.sum(tmp159, 0).reshape(1, X0BLOCK_SUB)
            tmp163 = tl.sum(tmp153, 0).reshape(1, X0BLOCK_SUB)
            tmp164 = tl.load(in_ptr67 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp165 = tl.load(in_ptr68 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp166 = tmp164 + tmp165
            tmp167 = tl.load(in_ptr69 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp168 = tmp166 + tmp167
            tmp169 = tl.load(in_ptr70 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp170 = tmp168 + tmp169
            tmp171 = tl.load(in_ptr71 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp172 = tl.load(in_ptr72 + (r1), None)
            tmp173 = tmp171 - tmp172
            tmp174 = tl.load(in_ptr73 + (r1), None)
            tmp175 = tmp173 * tmp174
            tmp176 = tmp170 * tmp175
            tmp178 = tl.sum(tmp176, 0).reshape(1, X0BLOCK_SUB)
            tmp180 = tl.sum(tmp170, 0).reshape(1, X0BLOCK_SUB)
            tmp181 = tl.load(in_ptr74 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp182 = tl.load(in_ptr75 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp183 = tmp181 + tmp182
            tmp184 = tl.load(in_ptr76 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp185 = tmp183 + tmp184
            tmp186 = tl.load(in_ptr77 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp187 = tmp185 + tmp186
            tmp188 = tl.load(in_ptr78 + (x0 + 4096*r1), x0_mask, other=0.0)
            tmp189 = tl.load(in_ptr79 + (r1), None)
            tmp190 = tmp188 - tmp189
            tmp191 = tl.load(in_ptr80 + (r1), None)
            tmp192 = tmp190 * tmp191
            tmp193 = tmp187 * tmp192
            tmp195 = tl.sum(tmp193, 0).reshape(1, X0BLOCK_SUB)
            tmp197 = tl.sum(tmp187, 0).reshape(1, X0BLOCK_SUB)
            tmp198 = tmp8 + tmp25
            tmp199 = tmp198 + tmp42
            tmp200 = tmp199 + tmp59
            tmp201 = tmp200 + tmp76
            tmp202 = tmp201 + tmp93
            tmp203 = tmp202 + tmp127
            tmp204 = tmp203 + tmp161
            tmp205 = tmp204 + tmp195
            tmp206 = tmp205 + tmp110
            tmp207 = tmp206 + tmp144
            tmp208 = tmp207 + tmp178
            tmp209 = tmp10 + tmp27
            tmp210 = tmp209 + tmp44
            tmp211 = tmp210 + tmp61
            tmp212 = tmp211 + tmp78
            tmp213 = tmp212 + tmp95
            tmp214 = tmp213 + tmp129
            tmp215 = tmp214 + tmp163
            tmp216 = tmp215 + tmp197
            tmp217 = tmp216 + tmp112
            tmp218 = tmp217 + tmp146
            tmp219 = tmp218 + tmp180
            tl.store(in_out_ptr0 + (x0), tmp208, x0_mask)
            tl.store(in_out_ptr1 + (x0), tmp219, x0_mask)

    return {"front": kernel_front, "near": kernel_near}


def reference(inputs):
    weighted = [(inputs[0] * ((inputs[1] - inputs[2][:, None]) * inputs[3][:, None])).sum(0)]
    summed = [inputs[0].sum(0)]
    for start in range(4, 81, 7):
        dy = inputs[start] + inputs[start + 1] + inputs[start + 2] + inputs[start + 3]
        normalized = (inputs[start + 4] - inputs[start + 5][:, None]) * inputs[start + 6][:, None]
        weighted.append((dy * normalized).sum(0))
        summed.append(dy.sum(0))
    out0, out1 = weighted[0], summed[0]
    for index in [1, 2, 3, 4, 5, 7, 9, 11, 6, 8, 10]:
        out0 = out0 + weighted[index]
        out1 = out1 + summed[index]
    return out0, out1


def check(binary, output):
    torch.npu.set_device(0)
    torch.manual_seed(0)
    scalar_inputs = {2, 3}
    for start in range(4, 81, 7):
        scalar_inputs.update((start + 5, start + 6))
    inputs = [
        torch.randn((2048,) if index in scalar_inputs else (2048, 4096), device="npu")
        for index in range(81)
    ]
    expected = reference(inputs)
    actual = [torch.empty(4096, device="npu") for _ in range(2)]
    args = [*actual, *inputs, 4096, 2048, 64]
    stream = torch.npu.current_stream().npu_stream
    binary[(64, 1, 1)](*args, stream=stream)
    torch.npu.synchronize()
    errors = [(left - right).abs().max().item() for left, right in zip(actual, expected)]
    output["max_abs"] = errors
    output["atol"], output["rtol"] = 2e-3, 1e-3
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=2e-3, rtol=1e-3)
    output["numeric_status"] = "PASS"


def compile_case(args):
    args.output.mkdir(parents=True, exist_ok=True)
    options = dict(BASE_OPTIONS, multibuffer=args.multibuffer == "true")
    result = {
        "order": args._case,
        "target": TARGET,
        "tiling": TILING,
        "constants": {"X0BLOCK_SUB": TILING["X0BLOCK_SUB"]},
        "runtime": RUNTIME,
        "options": options,
        "cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "inductor_codegen_called": False,
        "compile_status": "RUNNING",
        "numeric_status": "NOT_RUN",
    }
    result_path = args.output / "result.json"

    def save():
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")

    save()
    start = time.monotonic()
    try:
        kernels = load_kernels()
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import ASTSource

        kernel = kernels[args._case]
        result.update(
            torch=torch.__version__, torch_npu=torch_npu.__version__,
            torch_npu_git=getattr(torch_npu.version, "git_version", None),
            torch_file=torch.__file__, torch_npu_file=torch_npu.__file__,
            triton=triton.__version__, triton_file=triton.__file__,
            kernel_sha256=hashlib.sha256(kernel.src.encode()).hexdigest(),
        )
        signature = {"in_out_ptr0": "*fp32", "in_out_ptr1": "*fp32"}
        signature.update({f"in_ptr{index}": "*fp32" for index in range(81)})
        signature.update({"x0_numel": "i32", "r1_numel": "i32", "X0BLOCK": "i32"})
        save()
        print(json.dumps(result), flush=True)
        binary = triton.compile(
            ASTSource(kernel, signature, result["constants"]),
            target=GPUTarget(TARGET["backend"], TARGET["arch"], TARGET["warp_size"]),
            options=options,
        )
        result["compile_status"] = "PASS"
        result["compile_seconds"] = time.monotonic() - start
        result["compiled_metadata"] = dict(binary.metadata._asdict())
        save()
        for name, text in binary.asm.items():
            if isinstance(text, str):
                (args.output / f"kernel.{name}").write_text(text)
        if args.check:
            result["numeric_status"] = "RUNNING"
            save()
            print("Compile PASS; starting numeric check", flush=True)
            check(binary, result)
    except Exception as error:
        result["exception_type"] = type(error).__name__
        result["error"] = str(error)
        if result["compile_status"] == "RUNNING":
            result["compile_status"] = "FAIL"
            result["compile_seconds"] = time.monotonic() - start
        elif args.check:
            result["numeric_status"] = "FAIL"
        (args.output / "traceback.txt").write_text(traceback.format_exc())
        match = re.search(r"requires (\d+) bits while (\d+) bits available", str(error))
        if match:
            result["ub_required_bits"], result["ub_available_bits"] = map(int, match.groups())
    save()
    print(json.dumps(result, default=str), flush=True)


def run_case(order, multibuffer, args):
    case_dir = args.output / f"{order}_mb{multibuffer}"
    case_dir.mkdir()
    command = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--_case", order, "--multibuffer", multibuffer, "--output", str(case_dir),
    ]
    if args.check:
        command.append("--check")
    env = dict(os.environ, TRITON_CACHE_DIR=str(args.output / "cache"))
    timed_out = False
    with (case_dir / "stdout.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
        try:
            process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    result_path = case_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    result.update(
        order=order, multibuffer=multibuffer, returncode=process.returncode,
        timed_out=timed_out, log=str(case_dir / "stdout.log"),
    )
    if timed_out:
        result["status"] = "TIMEOUT"
    elif process.returncode != 0 or not result_path.exists():
        result["status"] = "ERROR"
    elif (
        result.get("compile_status") == "FAIL"
        and result.get("ub_required_bits", 0) > result.get("ub_available_bits", 0)
    ):
        result["status"] = "UB_OVERFLOW"
    elif result.get("compile_status") == "PASS" and not result.get("error"):
        result["status"] = "PASS" if not args.check or result.get("numeric_status") == "PASS" else "ERROR"
    else:
        result["status"] = "ERROR"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--_case", choices=("front", "near"), help=argparse.SUPPRESS)
    parser.add_argument("--show-config", action="store_true", help="Print the fixed tiling and exit")
    parser.add_argument("--check", action="store_true", help="Launch compilable kernels and compare eager outputs")
    parser.add_argument("--multibuffer", choices=("both", "false", "true"), default="both")
    parser.add_argument("--timeout", type=int, default=180, help="Seconds allowed per compile/check subprocess")
    parser.add_argument("--output", type=Path, help="New result directory; existing directories are rejected")
    args = parser.parse_args()
    if args._case:
        if args.output is None or args.multibuffer == "both":
            parser.error("internal worker needs --output and one multibuffer value")
        compile_case(args)
        return 0
    print_config()
    if args.show_config:
        return 0
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    args.output = (
        args.output or Path.cwd() / datetime.now().strftime("load_order_results_%Y%m%d_%H%M%S_%f")
    ).resolve()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    summary = {
        "tiling": TILING,
        "runtime": RUNTIME,
        "target": TARGET,
        "base_options": BASE_OPTIONS,
        "check_requested": args.check,
        "timeout_seconds": args.timeout,
        "inductor_codegen_called": False,
        "cases": [],
        "comparisons": [],
    }
    print(f"Results: {args.output}", flush=True)
    print("front = loads before compute; near = loads immediately before first use", flush=True)
    print("multibuffer  order  result       compile  numeric", flush=True)
    modes = ("false", "true") if args.multibuffer == "both" else (args.multibuffer,)
    for mode in modes:
        pair = []
        for order in ("front", "near"):
            result = run_case(order, mode, args)
            pair.append(result)
            summary["cases"].append(result)
            print(
                f"{mode:11}  {order:5}  {result['status']:11}  "
                f"{result.get('compile_status', 'UNKNOWN'):7}  {result.get('numeric_status', 'NOT_RUN')}",
                flush=True,
            )
            if result["status"] == "UB_OVERFLOW":
                print(
                    f"  UB required={result['ub_required_bits']:,} bits; "
                    f"available={result['ub_available_bits']:,} bits", flush=True,
                )
            elif result["status"] in ("ERROR", "TIMEOUT"):
                print(f"  See {result['log']}", flush=True)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        front, near = (result["status"] for result in pair)
        if front == "UB_OVERFLOW" and near == "PASS":
            status = "REPRODUCED"
        elif front == near == "PASS":
            status = "NOT_REPRODUCED"
        else:
            status = "ERROR"
        summary["comparisons"].append({"multibuffer": mode, "status": status})
        print(f"multibuffer={mode}: {status}", flush=True)
    first = summary["cases"][0]
    environment = {key: first.get(key) for key in ("torch", "torch_npu", "torch_npu_git", "triton")}
    summary["environment"] = environment
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Environment: {json.dumps(environment)}", flush=True)
    print(f"Summary: {args.output / 'summary.json'}", flush=True)
    if any(result["status"] == "ERROR" for result in summary["comparisons"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
