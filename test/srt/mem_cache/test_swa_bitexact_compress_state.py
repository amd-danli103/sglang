"""Unit test for strict bit-exact c4 / c4-indexer compress-state capture.

Exercises ``CompressorHip._capture_compress_state_windows`` (Phase 4''.S.1)
with faked collaborators and real CPU tensors -- no GPU / model required:

  * capture strides by page; one host tile per (rid, B) page boundary
  * the [B-ratio, B) window lands on ring offset (s % swa_ring) % ring_size
  * captured bytes are byte-identical to the kv_and_score_buffer window
  * flag OFF (no host pool wired) is a no-op

Run:
  PYTHONPATH=<worktree>/python python -m pytest \
      test/srt/mem_cache/test_swa_bitexact_compress_state.py -q
"""

import types
import unittest

import torch

from sglang.srt.layers.attention.dsv4.compress_hip import CompressorHip
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.unified_cache_components import swa_component as SC

_CAPTURE = CompressorHip._capture_compress_state_windows
_RESTORE = SC._restore_state_windows
_TRANSLATE = CompressStatePool.translate_from_swa_loc_to_state_loc


def _fake_host_pool(*, ring_size, slot_bytes, num_pages):
    item_bytes = ring_size * slot_bytes
    host_buf = torch.zeros((num_pages, item_bytes), dtype=torch.uint8)
    counter = {"n": 0}

    def alloc(need):
        assert need == ring_size
        page = counter["n"]
        counter["n"] += 1
        if page >= num_pages:
            return None
        return torch.arange(page * ring_size, page * ring_size + ring_size)

    return types.SimpleNamespace(
        slot_page_size=ring_size,
        item_bytes=item_bytes,
        data_refs=[host_buf],
        _capture_staging={},
        alloc=alloc,
    )


def _fake_backend(host_pool, *, page=256, swa_ring=128, is_indexer=False):
    attr = "_c4_indexer_state_host_pool" if is_indexer else "_c4_state_host_pool"
    pool = types.SimpleNamespace(
        unified_swa_ring_size=swa_ring,
        _c4_state_layer_index={0: 0},
    )
    setattr(pool, attr, host_pool)
    return types.SimpleNamespace(token_to_kv_pool=pool, page_size=page)


def _fake_self(*, ratio=4, is_indexer=False, layer_id=0):
    return types.SimpleNamespace(
        ratio=ratio, is_in_indexer=is_indexer, layer_id=layer_id
    )


class TestCompressStateCapture(unittest.TestCase):
    def _run(self, *, prefix_len, extend_len, ratio=4, page=256, swa_ring=128,
             ring_size=8, last_dim=16, dtype=torch.bfloat16, is_indexer=False):
        slot_bytes = last_dim * torch.tensor([], dtype=dtype).element_size()
        hp = _fake_host_pool(ring_size=ring_size, slot_bytes=slot_bytes, num_pages=16)
        backend = _fake_backend(hp, page=page, swa_ring=swa_ring, is_indexer=is_indexer)
        valid_kv_len = prefix_len + extend_len  # empty pre_state for prefix_len%ratio==0
        buf = types.SimpleNamespace(
            kv_score=torch.randint(
                0, 255, (valid_kv_len, last_dim), dtype=torch.int32
            ).to(dtype)
        )
        fs = _fake_self(ratio=ratio, is_indexer=is_indexer)
        _CAPTURE(
            fs,
            kv_and_score_buffer=buf,
            valid_kv_len=valid_kv_len,
            prefix_len=prefix_len,
            extend_len=extend_len,
            rid=7,
            backend=backend,
        )
        return hp, buf, slot_bytes

    def test_capture_windows_byte_exact(self):
        page, ring_size, ratio, last_dim = 256, 8, 4, 16
        hp, buf, slot_bytes = self._run(
            prefix_len=0, extend_len=512, page=page, ring_size=ring_size,
            ratio=ratio, last_dim=last_dim,
        )
        # boundaries 256, 512 -> two tiles keyed (7, B)
        self.assertEqual(set(hp._capture_staging), {(7, 256), (7, 512)})
        for B in (256, 512):
            hidx = hp._capture_staging[(7, B)]
            page_row = int(hidx[0].item()) // ring_size
            off0 = ((B - ratio) % 128) % ring_size
            self.assertEqual(off0, ratio)  # window lands on slots [4, 8)
            got = hp.data_refs[0][page_row][
                off0 * slot_bytes : (off0 + ratio) * slot_bytes
            ]
            want = buf.kv_score[B - ratio : B].contiguous().view(torch.uint8).reshape(-1)
            self.assertTrue(torch.equal(got, want))
            # leading slots [0, 4) never populated by capture
            self.assertTrue(
                torch.equal(
                    hp.data_refs[0][page_row][: off0 * slot_bytes],
                    torch.zeros(off0 * slot_bytes, dtype=torch.uint8),
                )
            )

    def test_no_boundary_no_capture(self):
        # chunk shorter than a page -> no page boundary -> nothing captured
        hp, _, _ = self._run(prefix_len=0, extend_len=100)
        self.assertEqual(hp._capture_staging, {})

    def test_flag_off_noop(self):
        # No host pool wired (bit-exact flag off) -> silent no-op.
        pool = types.SimpleNamespace(
            unified_swa_ring_size=128, _c4_state_layer_index={0: 0}
        )
        backend = types.SimpleNamespace(token_to_kv_pool=pool, page_size=256)
        buf = types.SimpleNamespace(kv_score=torch.zeros((512, 16), dtype=torch.bfloat16))
        _CAPTURE(
            _fake_self(),
            kv_and_score_buffer=buf,
            valid_kv_len=512,
            prefix_len=0,
            extend_len=512,
            rid=1,
            backend=backend,
        )  # must not raise

    def test_ratio_128_skipped(self):
        hp, _, _ = self._run(prefix_len=0, extend_len=512, ratio=128)
        self.assertEqual(hp._capture_staging, {})

    def test_indexer_pool_routing(self):
        hp, buf, slot_bytes = self._run(
            prefix_len=0, extend_len=256, is_indexer=True
        )
        self.assertEqual(set(hp._capture_staging), {(7, 256)})


class TestCaptureRestoreRoundTrip(unittest.TestCase):
    """End-to-end (real capture + real restore + real translate): the state
    restored onto the device ring at the reuse boundary is byte-identical to the
    compressor buffer window captured at prefill, independent of the actual SWA
    slot offsets used by the reusing request."""

    def test_roundtrip_byte_exact(self):
        page, swa_ring, ring_size, ratio = 256, 128, 8, 4
        last_dim, dtype = 16, torch.bfloat16
        slot_bytes = last_dim * torch.tensor([], dtype=dtype).element_size()

        # --- capture (original request, rid=7, prefill [0, 256)) ---
        hp = _fake_host_pool(ring_size=ring_size, slot_bytes=slot_bytes, num_pages=8)
        backend = _fake_backend(hp, page=page, swa_ring=swa_ring)
        valid_kv_len = 256
        buf = types.SimpleNamespace(
            kv_score=torch.randint(0, 255, (valid_kv_len, last_dim), dtype=torch.int32)
            .to(dtype)
        )
        _CAPTURE(
            _fake_self(),
            kv_and_score_buffer=buf,
            valid_kv_len=valid_kv_len,
            prefix_len=0,
            extend_len=256,
            rid=7,
            backend=backend,
        )
        host_indices = hp._capture_staging[(7, 256)]

        # --- restore (reusing request): its window [128, 256) got restored SWA
        # slots swa_chunk; the last `ratio` are [B-ratio, B). Use a swa_base that
        # is a multiple of swa_ring (as real SWA pages are) but different from
        # capture, to prove offset independence. ---
        swa_base = 256
        swa_chunk = torch.arange(swa_base, swa_base + swa_ring, dtype=torch.int64)

        dev = torch.zeros((64, last_dim), dtype=dtype)
        fake_sp = types.SimpleNamespace(
            ring_size=ring_size,
            swa_page_size=swa_ring,
            ratio=ratio,
            kv_score_buffer=types.SimpleNamespace(kv_score=dev),
        )
        fake_sp.translate_from_swa_loc_to_state_loc = types.MethodType(
            _TRANSLATE, fake_sp
        )
        node = types.SimpleNamespace(
            _c4_state_host_value=host_indices,
            _c4_indexer_state_host_value=None,
        )
        restorer = types.SimpleNamespace(
            _c4_state_layer_index={0: 0},
            _c4_state_host_pool=hp,
            _c4_indexer_state_host_pool=None,
            _compress_state_pools=[fake_sp],
            _indexer_compress_state_pools=None,
        )
        _RESTORE(restorer, node, swa_chunk)

        # device ring slots for [B-ratio, B) == captured buffer window
        state_locs = _TRANSLATE(fake_sp, swa_chunk[-ratio:])
        got = dev[state_locs]
        want = buf.kv_score[valid_kv_len - ratio : valid_kv_len]
        self.assertTrue(torch.equal(got, want))

    def test_restore_noop_without_host_value(self):
        dev = torch.zeros((16, 8), dtype=torch.bfloat16)
        fake_sp = types.SimpleNamespace(
            ring_size=8, swa_page_size=128, ratio=4,
            kv_score_buffer=types.SimpleNamespace(kv_score=dev),
        )
        node = types.SimpleNamespace(
            _c4_state_host_value=None, _c4_indexer_state_host_value=None
        )
        restorer = types.SimpleNamespace(
            _c4_state_layer_index={0: 0},
            _c4_state_host_pool=types.SimpleNamespace(slot_page_size=8, data_refs=[]),
            _c4_indexer_state_host_pool=None,
            _compress_state_pools=[fake_sp],
            _indexer_compress_state_pools=None,
        )
        _RESTORE(restorer, node, torch.arange(128))  # must not raise / touch dev
        self.assertTrue(torch.equal(dev, torch.zeros_like(dev)))

    def test_active_rides_empty_when_unwired(self):
        restorer = types.SimpleNamespace(
            _c4_state_layer_index=None,
            _c4_state_host_pool=None,
            _c4_indexer_state_host_pool=None,
            _compress_state_pools=None,
            _indexer_compress_state_pools=None,
        )
        self.assertEqual(SC._state_rides(restorer), [])


if __name__ == "__main__":
    unittest.main()
