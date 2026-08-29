"""The DSv4 fp8 e4m3 conversion on its own, byte for byte against torch.

On ROCm there is no hardware float -> e4m3 instruction behind ``pack_fp8``, so it is
a hand-written bit twiddle in ``deepseek_v4/fp8_utils.cuh``. Every fp8 store in the
DSv4 tree goes through it, and none of them can see when it is wrong: the value has
already been divided by a quantization scale, so a bad rounding or saturation
boundary comes back as fp8 being lossier than it should be, not as a failure.

Two bugs found that way, both pinned below: the whole top exponent segment was
saturated to the max normal, and the binade under the min subnormal was flushed to
zero instead of rounding up to it.

The reference is torch's own cast, and the dtype follows the arch, so this covers
e4m3fnuz on gfx94x and e4m3fn on gfx95x.
"""

import math
import unittest

import torch

from sglang.kernels.ops.attention.dsv4.fp8_cvt import cvt_fp8_e4m3
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-a-test-1-gpu-small-amd")

DEVICE = torch.device("cuda")

# start of the top exponent segment, i.e. the largest power of two the format holds
TOP_BINADE = 2.0 ** math.floor(math.log2(fp8_max))


def _representable():
    vals = torch.arange(256, dtype=torch.uint8).view(fp8_dtype).float()
    return vals[vals.isfinite()]


def _min_normal():
    """3 mantissa bits, so the min normal is 8 subnormal steps up -- fn and fnuz both"""
    sub = _representable().abs()
    return 8 * sub[sub > 0].min().item()


def _domain():
    """every representable value, every midpoint between two of them, every bf16, and
    tiny fp32 that is not bf16-shaped"""
    vals = _representable()
    mids = ((vals[:, None] + vals[None, :]) / 2).flatten()
    # small values can't come from bf16 patterns alone: bf16 leaves mant23's low 16
    # bits zero, so sticky never comes out 1 and the subnormal shift guard goes
    # untouched. what reaches the cast in the fused kernels is fp32 that has been
    # divided by a quantization scale, mantissa and all. raw exponent 96..121 brackets
    # the guard from both sides -- it starts biting at 117, and the shifts under it run
    # off the end of a uint32 from 109 down. a cast missing it differs on 23682 of these
    tiny = torch.cat(
        [
            torch.arange(e << 23, (e << 23) + (1 << 23), 8191, dtype=torch.int32)
            for e in range(96, 122)
        ]
    ).view(torch.float32)
    cases = torch.cat(
        [
            vals,
            mids,
            torch.linspace(-fp8_max, fp8_max, 100003),
            torch.arange(1 << 16, dtype=torch.int32).view(torch.bfloat16).float(),
            tiny,
            -tiny,
        ]
    )
    # stay inside the range: past the max the two casts are allowed to disagree on
    # whether to clamp or produce NaN, which is not what this is testing
    cases = cases[cases.isfinite() & (cases.abs() <= fp8_max)].unique()
    if cases.numel() % 2:
        cases = cases[:-1]
    return cases.contiguous()


def _as_bytes(x):
    # the conversion runs two values at a time, so the length has to stay even --
    # e4m3fnuz has an odd number of representable values (only 0x80 is NaN)
    if x.numel() % 2:
        x = x[:-1]
    x = x.contiguous().to(DEVICE)
    return cvt_fp8_e4m3(x), x.to(fp8_dtype).view(torch.uint8)


@unittest.skipUnless(is_hip(), "the software conversion only exists on ROCm")
class TestDsv4Fp8Cast(CustomTestCase):
    def test_matches_torch_over_the_whole_range(self):
        cases = _domain()
        got, want = _as_bytes(cases)
        bad = got != want
        if bool(bad.any()):
            v = cases.to(DEVICE)[bad]
            worst = v.abs().argmax()
            self.fail(
                f"{int(bad.sum())} of {cases.numel()} bytes differ, "
                f"|v| in [{v.abs().min():.4e}, {v.abs().max():.4e}]; e.g. "
                f"{v[worst]:.6g} -> {got[bad][worst].item():#04x} "
                f"(torch {want[bad][worst].item():#04x})"
            )

    def test_top_exponent_segment_is_not_saturated(self):
        # testing the exponent alone here used to write every value from TOP_BINADE up
        # to the max out as the max normal
        vals = _representable()
        top = vals[vals.abs() >= TOP_BINADE]
        self.assertGreater(top.numel(), 2)
        got, want = _as_bytes(top)
        self.assertTrue(torch.equal(got, want))
        # and they really are distinct values, not all the same byte
        self.assertGreater(int(got.unique().numel()), 2)

    def test_the_domain_reaches_the_subnormal_shift(self):
        # a coverage assert, not a value check: shrink the domain back to
        # representables/midpoints/bf16 and the subnormal shift stops being tested at
        # all, silently -- every byte still comes out right
        cases = _domain()
        small = cases.abs() < _min_normal()
        odd_mantissa = (cases.view(torch.int32) & 0xFFFF) != 0
        self.assertGreater(int((small & odd_mantissa).sum()), 10000)

    def test_binade_below_the_min_subnormal_rounds_up(self):
        min_subnormal = _representable().abs()
        min_subnormal = min_subnormal[min_subnormal > 0].min().item()
        # (midpoint, min subnormal): rounds up. the midpoint itself is a tie and goes
        # to even, i.e. to zero
        band = torch.linspace(min_subnormal / 2, min_subnormal, 2049)[1:-1]
        band = torch.cat([band, -band])
        got, want = _as_bytes(band.contiguous())
        self.assertTrue(torch.equal(got, want))
        self.assertTrue(bool((got & 0x7F).all()))


if __name__ == "__main__":
    unittest.main()
