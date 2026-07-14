"""Component-level tests for the window-granular strict SWA HiCache capture.

Exercises the real methods with a minimal fake host pool and small CPU bf16
tensors — no GPU / full unified pool needed.

Covers:
  * capture geometry + byte identity: stride == page, one window [B-win, B) per
    page boundary keyed (rid, B); the page's first half is never captured;
    per-request offset + per-layer indexing correct.
  * binding: a node's SWA host_value is the single window at its end boundary
    (len == win), not the node's full value; falls back when the window is
    missing.
  * restore: LOAD_BACK maps only the window's (last n_tokens) full indices to
    the restored SWA slots.
"""

import types
import unittest

import torch

from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache_components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
)

SWA = ComponentType.SWA
FULL = BASE_COMPONENT_TYPE


class _FakeHostPool:
    def __init__(self, *, win, head_dim, num_pages, layers, elem=2):
        self.slot_page_size = win
        self.item_bytes = win * head_dim * elem
        self.data_refs = [
            torch.zeros(num_pages, self.item_bytes, dtype=torch.uint8)
            for _ in range(layers)
        ]
        self._capture_staging = {}
        self._num_pages = num_pages
        self._next_page = 0
        self.freed = []

    def alloc(self, n):
        assert n == self.slot_page_size
        if self._next_page >= self._num_pages:
            return None
        p = self._next_page
        self._next_page += 1
        start = p * self.slot_page_size
        return torch.arange(start, start + n, dtype=torch.int64)

    def free(self, idx):
        self.freed.append(idx)


class _FakePool:
    def __init__(self, host_pool, win, start_layer=0):
        self._swa_host_pool = host_pool
        self.unified_swa_ring_size = win
        self.unified_swa_window = win
        self.start_layer = start_layer


def _fake_fb(ext, seqs, rids):
    return types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_extend=lambda: True),
        extend_seq_lens_cpu=ext,
        seq_lens_cpu=torch.tensor(seqs),
        req_pool_indices=torch.tensor(rids),
    )


def _cd(value=None, host_value=None):
    return types.SimpleNamespace(value=value, host_value=host_value)


class TestSwaCaptureGeometry(unittest.TestCase):
    """Stride == page, one [B-win, B) window per boundary, first half never
    captured, per-request offset + per-layer bytes correct."""

    def _run(self):
        win, page, head_dim, layers = 2, 4, 4, 3
        host = _FakeHostPool(
            win=win, head_dim=head_dim, num_pages=8, layers=layers
        )
        pool = _FakePool(host, win)
        be = types.SimpleNamespace(token_to_kv_pool=pool, page_size=page)
        # batch: req0 = positions [0,8) (cs=0,e=8); req1 = [8,12) (cs=8,e=4)
        ext = [8, 4]
        seqs = [8, 12]
        rids = [5, 7]
        fb = _fake_fb(ext, seqs, rids)
        total_rows = sum(ext)
        # distinct bf16 bytes per (layer, row): base value differs per layer.
        kv_by_layer = [
            (torch.arange(total_rows * head_dim, dtype=torch.float32)
             .reshape(total_rows, head_dim) + L * 10000.0).to(torch.bfloat16)
            for L in range(layers)
        ]
        for L in range(layers):
            DeepseekV4HipRadixBackend.capture_swa_windows(
                be, L, kv_by_layer[L], fb
            )
        return win, head_dim, layers, host, kv_by_layer

    def test_keys_are_second_half_windows_only(self):
        win, head_dim, layers, host, kv = self._run()
        # boundaries: req0 -> 4,8 ; req1 -> 12. windows [2,4),[6,8),[10,12).
        self.assertEqual(
            set(host._capture_staging.keys()), {(5, 4), (5, 8), (7, 12)}
        )
        # exactly 3 host pages allocated (no first-half tiles).
        self.assertEqual(host._next_page, 3)

    def test_bytes_match_flat_kv_window_per_layer(self):
        win, head_dim, layers, host, kv = self._run()
        # (key -> (page_row, batch_row_start)) ; batch rows are absolute in kv.
        expected = {
            (5, 4): (0, 2),   # window [2,4)
            (5, 8): (1, 6),   # window [6,8)
            (7, 12): (2, 10),  # window [10,12)
        }
        for key, (page_row, r0) in expected.items():
            self.assertIn(key, host._capture_staging)
            got_page = int(host._capture_staging[key][0].item()) // win
            self.assertEqual(got_page, page_row)
            for L in range(layers):
                want = (
                    kv[L][r0 : r0 + win]
                    .contiguous()
                    .view(torch.uint8)
                    .reshape(-1)
                )
                got = host.data_refs[L][page_row]
                self.assertTrue(
                    torch.equal(got, want),
                    f"layer {L} key {key}: bytes differ",
                )

    def test_flag_off_is_noop(self):
        # No _swa_host_pool => capture short-circuits.
        pool = types.SimpleNamespace(_swa_host_pool=None)
        be = types.SimpleNamespace(token_to_kv_pool=pool, page_size=4)
        fb = _fake_fb([4], [4], [0])
        # Must not raise / touch anything.
        DeepseekV4HipRadixBackend.capture_swa_windows(
            be, 0, torch.zeros(4, 4, dtype=torch.bfloat16), fb
        )


class TestSwaBindWindow(unittest.TestCase):
    """Bind the single window at node_end; fall back when it is absent."""

    def _fake_self(self, host):
        calls = []
        return types.SimpleNamespace(
            _swa_kv_pool_host=host,
            _capture_rid=5,
            component_type=SWA,
            _attach_swa_host_value=lambda node, hv: calls.append(hv),
        ), calls

    def test_binds_single_window_not_node_length(self):
        win, head_dim = 2, 4
        host = _FakeHostPool(win=win, head_dim=head_dim, num_pages=4, layers=1)
        # stage a window tile at boundary B=4 keyed (rid=5, 4)
        tile = torch.arange(10, 10 + win, dtype=torch.int64)
        host._capture_staging[(5, 4)] = tile
        me, calls = self._fake_self(host)
        # node covers one page [0,4): SWA value length == page == 4 (256-analogue)
        node = types.SimpleNamespace(
            component_data={SWA: _cd(value=torch.arange(4), host_value=None)}
        )
        SWAComponent._bind_captured_swa_host(me, node, swa_start=0)
        # Co-lifetime: bind stashes a PENDING page (attached later, together with
        # Full host_value, via the coordinated BACKUP_HOST), not host_value now.
        pending = getattr(node, "_swa_pending_host", None)
        self.assertIsNotNone(pending)
        self.assertEqual(len(pending), win)  # window (2), not node length (4)
        self.assertTrue(torch.equal(pending, tile.to(torch.int64)))
        self.assertEqual(len(calls), 0)  # _attach deferred, not called at bind
        # tile consumed from staging
        self.assertNotIn((5, 4), host._capture_staging)

    def test_missing_window_is_i6_noop(self):
        win, head_dim = 2, 4
        host = _FakeHostPool(win=win, head_dim=head_dim, num_pages=4, layers=1)
        me, calls = self._fake_self(host)
        node = types.SimpleNamespace(
            component_data={SWA: _cd(value=torch.arange(4), host_value=None)}
        )
        SWAComponent._bind_captured_swa_host(me, node, swa_start=0)
        self.assertEqual(calls, [])  # nothing bound
        self.assertIsNone(getattr(node, "_swa_pending_host", None))


class TestSwaRestoreWindowMapping(unittest.TestCase):
    """infra: LOAD_BACK maps only the window's (last n_tokens) full indices."""

    def test_maps_only_window_full_indices(self):
        win = 2
        mapping_calls = []
        restore_calls = []
        allocator = types.SimpleNamespace(
            set_full_to_swa_mapping=lambda full, swa: mapping_calls.append(
                (full.clone(), swa.clone())
            )
        )
        me = types.SimpleNamespace(
            component_type=SWA,
            cache=types.SimpleNamespace(token_to_kv_pool_allocator=allocator),
            _restore_device_value=lambda n, v: restore_calls.append(v.clone()),
        )
        me._gather_window_full_indices = (
            lambda n, nt: SWAComponent._gather_window_full_indices(me, n, nt)
        )
        # node: full value length 4 (page node), SWA host_value == window (2)
        full_val = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        node = types.SimpleNamespace(
            component_data={
                SWA: _cd(value=None, host_value=torch.tensor([0, 0])),
                FULL: _cd(value=full_val),
            }
        )
        device_indices = torch.tensor([700, 701], dtype=torch.int64)
        xfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.tensor([0, 0]),
            device_indices=device_indices,
            nodes_to_load=[node],
        )
        SWAComponent.commit_hicache_transfer(
            me, node, CacheTransferPhase.LOAD_BACK, transfers=[xfer]
        )
        self.assertEqual(len(mapping_calls), 1)
        mapped_full, mapped_swa = mapping_calls[0]
        # only the LAST win (=2) full indices [102,103] are mapped
        self.assertTrue(torch.equal(mapped_full, full_val[-win:]))
        self.assertTrue(torch.equal(mapped_swa, device_indices))
        self.assertEqual(len(restore_calls), 1)
        self.assertTrue(torch.equal(restore_calls[0], device_indices))


class TestSwaRestoreSplitWindow(unittest.TestCase):
    """R.1 (Phase 4-prime.R): after a node split, a child shorter than the
    sliding window still owns the whole window host_value [B-win, B). Restore
    must gather the window full indices across the child AND its ancestors (in
    token order), not just the child own (shorter) full value. Regression for
    the set_full_to_swa_mapping length-mismatch assert."""

    def test_window_spans_parent_and_child(self):
        mapping_calls = []
        restore_calls = []
        allocator = types.SimpleNamespace(
            set_full_to_swa_mapping=lambda full, swa: mapping_calls.append(
                (full.clone(), swa.clone())
            )
        )
        root = types.SimpleNamespace(component_data={}, parent=None)
        me = types.SimpleNamespace(
            component_type=SWA,
            cache=types.SimpleNamespace(
                token_to_kv_pool_allocator=allocator, root_node=root
            ),
            _restore_device_value=lambda n, v: restore_calls.append(v.clone()),
        )
        me._gather_window_full_indices = (
            lambda n, nt: SWAComponent._gather_window_full_indices(me, n, nt)
        )
        # parent holds full tokens [B-4, B-2); child holds [B-2, B). The child
        # keeps the whole win=4 window host_value (parent.host_value is None
        # after redistribute_on_node_split).
        parent = types.SimpleNamespace(
            parent=root,
            component_data={
                SWA: _cd(value=None, host_value=None),
                FULL: _cd(value=torch.tensor([100, 101], dtype=torch.int64)),
            },
        )
        child = types.SimpleNamespace(
            parent=parent,
            component_data={
                SWA: _cd(value=None, host_value=torch.tensor([0, 0, 0, 0])),
                FULL: _cd(value=torch.tensor([102, 103], dtype=torch.int64)),
            },
        )
        device_indices = torch.tensor([700, 701, 702, 703], dtype=torch.int64)
        xfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.tensor([0, 0, 0, 0]),
            device_indices=device_indices,
            nodes_to_load=[child],
        )
        SWAComponent.commit_hicache_transfer(
            me, child, CacheTransferPhase.LOAD_BACK, transfers=[xfer]
        )
        self.assertEqual(len(mapping_calls), 1)
        mapped_full, mapped_swa = mapping_calls[0]
        # window full indices, token order: parent tail [100,101] ++ child [102,103]
        self.assertTrue(
            torch.equal(
                mapped_full, torch.tensor([100, 101, 102, 103], dtype=torch.int64)
            )
        )
        self.assertTrue(torch.equal(mapped_swa, device_indices))
        self.assertEqual(len(restore_calls), 1)
        self.assertTrue(torch.equal(restore_calls[0], device_indices))


class TestStrictMatchValidatorI2Prime(unittest.TestCase):
    """R.6 / I2-prime: in strict mode a node whose SWA truth lives only in the
    per-request device ring (cd.value present, host_value None) must NOT extend
    a reuse match. The device ring is recycled across requests and is not a
    durable cross-request truth; the match must truncate so reuse restores from
    host or recomputes (I6). Best-effort (non-strict) keeps trusting device."""

    def _validator(self, strict):
        me = types.SimpleNamespace(
            sliding_window_size=4,
            component_type=SWA,
            _swa_kv_pool_host=object(),  # host pool wired => feature on, not device-only
            _strict_bit_exact=strict,
            cache=types.SimpleNamespace(cache_controller=object()),
        )
        return SWAComponent.create_match_validator(me)

    def _node(self, key_len, value, host_value):
        return types.SimpleNamespace(
            key=list(range(key_len)),
            backuped=True,
            evicted=False,
            component_data={SWA: _cd(value=value, host_value=host_value)},
        )

    def test_strict_device_only_node_truncates_match(self):
        v = self._validator(strict=True)
        node = self._node(4, value=[1, 2, 3, 4], host_value=None)
        self.assertFalse(v(node))

    def test_strict_host_node_extends_match(self):
        v = self._validator(strict=True)
        node = self._node(4, value=None, host_value=[0, 0, 0, 0])
        self.assertTrue(v(node))

    def test_strict_device_and_host_extends_match(self):
        v = self._validator(strict=True)
        node = self._node(4, value=[1, 2, 3, 4], host_value=[0, 0, 0, 0])
        self.assertTrue(v(node))

    def test_non_strict_trusts_device_value(self):
        v = self._validator(strict=False)
        node = self._node(4, value=[1, 2, 3, 4], host_value=None)
        self.assertTrue(v(node))


if __name__ == "__main__":
    unittest.main()
