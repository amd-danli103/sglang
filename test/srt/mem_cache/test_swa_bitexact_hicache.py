"""Pure-logic unit tests for the strict bit-exact SWA HiCache feature.

Covers the three commits:
  * sizing: hybrid_pool_assembler._swa_host_num_pages
  * co-eviction observability: UnifiedRadixCache._note_binding_full_coevict
  * strict atomic leaf eviction: UnifiedRadixCache.drive_host_leaf_eviction
    and SWAComponent.drive_host_eviction routing

No GPU / model is required; heavy collaborators are faked so we exercise only
the new logic. Run:
  PYTHONPATH=<worktree>/python python -m pytest test/srt/mem_cache/test_swa_bitexact_hicache.py -q
"""

import math
import types
import unittest

from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as A
from sglang.srt.mem_cache import unified_radix_cache as R
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_cache_components.swa_component import SWAComponent

FULL = R.BASE_COMPONENT_TYPE
SWA = ComponentType.SWA


def _sargs(avg=None):
    return types.SimpleNamespace(hicache_swa_avg_seq_len=avg)


class TestSwaHostSizing(unittest.TestCase):
    def _pages(
        self,
        *,
        avg=None,
        full_host_pages=100_000,
        device_ring_pages=65,
        page_bytes=1,
        page_size=256,
    ):
        return A._swa_host_num_pages(
            server_args=_sargs(avg),
            full_host_pages=full_host_pages,
            device_ring_pages=device_ring_pages,
            page_bytes=page_bytes,
            page_size=page_size,
        )

    def test_default_avg_seq_len(self):
        # avg unset -> 50000 tokens window -> one window per host-cached prefix.
        window = math.ceil(A._SWA_HICACHE_DEFAULT_AVG_SEQ_LEN / 256)
        self.assertEqual(self._pages(), math.ceil(100_000 / window))

    def test_customer_override_wins(self):
        # A shorter assumed prefix means more prefixes fit -> bigger SWA pool.
        smaller_avg = self._pages(avg=8_000)
        default = self._pages()
        self.assertGreater(smaller_avg, default)
        self.assertEqual(smaller_avg, math.ceil(100_000 / math.ceil(8_000 / 256)))

    def test_floor_device_ring(self):
        # Huge avg -> tiny computed size -> floored at the device ring depth.
        self.assertEqual(self._pages(avg=10_000_000, device_ring_pages=65), 65)

    def test_floor_one_page(self):
        self.assertEqual(
            self._pages(avg=10_000_000, device_ring_pages=0, full_host_pages=1), 1
        )

    def test_no_84gb_regression(self):
        # The fix: pool ~ full_host_pages / window, a small fraction of the full
        # host pool -- NOT device_ring * ratio which over-allocated to ~84GB.
        pages = self._pages(full_host_pages=100_000)
        self.assertLess(pages, 100_000 * 0.02)
        self.assertEqual(pages, math.ceil(100_000 / math.ceil(50_000 / 256)))

    def test_warn_above_16gb_but_no_clamp(self):
        # page_bytes chosen so the result exceeds the 16GB slow-launch threshold.
        window = math.ceil(50_000 / 256)
        expected = math.ceil(100_000 / window)  # 511
        page_bytes = int(16e9 / expected) + 1_000_000  # push over 16GB
        with self.assertLogs(A.logger, level="WARNING") as cm:
            pages = self._pages(full_host_pages=100_000, page_bytes=page_bytes)
        self.assertEqual(pages, expected)  # warned, not clamped
        self.assertTrue(any("may slow server launch" in m for m in cm.output))

    def test_no_warn_below_16gb(self):
        # Small page_bytes -> comfortably under threshold -> no warning emitted.
        with self.assertRaises(AssertionError):
            with self.assertLogs(A.logger, level="WARNING"):
                self._pages(full_host_pages=100_000, page_bytes=1)


class TestCoEvictWarning(unittest.TestCase):
    def _fresh(self):
        return types.SimpleNamespace()

    def _note(self, obj, full_tokens, leaves):
        R.UnifiedRadixCache._note_binding_full_coevict(obj, full_tokens, leaves)

    def test_noop_on_nonpositive(self):
        obj = self._fresh()
        self._note(obj, 0, 5)
        self._note(obj, 5, 0)
        self.assertFalse(hasattr(obj, "_binding_full_coevict_tokens"))

    def test_below_threshold_no_warn(self):
        obj = self._fresh()
        # 15 leaves total (< 16) across two calls -> accumulate, do not warn.
        with self.assertRaises(AssertionError):
            with self.assertLogs(R.logger, level="WARNING"):
                self._note(obj, 100, 8)
                self._note(obj, 100, 7)
        self.assertEqual(obj._binding_full_coevict_leaves, 15)
        self.assertFalse(getattr(obj, "_binding_full_coevict_warned", False))

    def test_warns_once_after_threshold(self):
        obj = self._fresh()
        with self.assertLogs(R.logger, level="WARNING") as cm:
            self._note(obj, 8_000, 8)
            self._note(obj, 8_000, 8)  # now 16 leaves -> warn
        self.assertTrue(getattr(obj, "_binding_full_coevict_warned"))
        self.assertEqual(len(cm.output), 1)
        # Further pressure must not warn again.
        with self.assertRaises(AssertionError):
            with self.assertLogs(R.logger, level="WARNING"):
                self._note(obj, 8_000, 8)

    def test_recommended_avg_matches_observed(self):
        obj = self._fresh()
        # 16 leaves, 16*1000 tokens -> avg 1000 tokens/prefix, recommend "1000".
        with self.assertLogs(R.logger, level="WARNING") as cm:
            self._note(obj, 16_000, 16)
        self.assertTrue(any("1000" in m for m in cm.output))


class _Node:
    def __init__(self, name, prio, full, swa, parent=None):
        self.name = name
        self.prio = prio
        self.full = full
        self.swa = swa
        self.parent = parent

    def __repr__(self):
        return f"_Node({self.name})"


class _FakeCacheForLeafEvict:
    """Minimal stand-in exercising drive_host_leaf_eviction's traversal and
    accounting. _evict_host_leaf models atomic Full+SWA drop and exposes the
    parent as a new host leaf (walk-up)."""

    def __init__(self, leaves):
        self.evictable_host_leaves = set(leaves)
        self.eviction_strategy = types.SimpleNamespace(
            get_priority=lambda n: n.prio
        )
        self.evicted = []
        self.coevict_calls = []

    def _evict_host_leaf(self, x, tracker):
        self.evictable_host_leaves.discard(x)
        tracker[FULL] = tracker.get(FULL, 0) + x.full
        tracker[SWA] = tracker.get(SWA, 0) + x.swa
        self.evicted.append(x)
        if x.parent is not None:
            # Parent becomes a host leaf now that its child is gone.
            self.evictable_host_leaves.add(x.parent)

    def _note_binding_full_coevict(self, full_tokens, leaves):
        self.coevict_calls.append((full_tokens, leaves))


class TestDriveHostLeafEviction(unittest.TestCase):
    def _drive(self, cache, num_tokens, key, tracker):
        R.UnifiedRadixCache.drive_host_leaf_eviction(cache, num_tokens, key, tracker)

    def test_priority_order_and_stop(self):
        # Two independent leaves; lower priority (LRU) evicted first, stop once
        # the key component target is met -> the colder leaf is spared.
        a = _Node("a", prio=1, full=10, swa=10)  # colder -> evicted first
        b = _Node("b", prio=5, full=10, swa=10)
        cache = _FakeCacheForLeafEvict([a, b])
        tracker = {FULL: 0, SWA: 0}
        self._drive(cache, num_tokens=10, key=SWA, tracker=tracker)
        self.assertEqual(cache.evicted, [a])
        self.assertIn(b, cache.evictable_host_leaves)

    def test_walk_up_parents(self):
        # Chain c <- b <- a (a is the only initial leaf); freeing pulls up the
        # whole branch as each parent becomes a leaf.
        c = _Node("c", prio=3, full=5, swa=5)
        b = _Node("b", prio=2, full=5, swa=5, parent=c)
        a = _Node("a", prio=1, full=5, swa=5, parent=b)
        cache = _FakeCacheForLeafEvict([a])
        tracker = {FULL: 0, SWA: 0}
        self._drive(cache, num_tokens=15, key=SWA, tracker=tracker)
        self.assertEqual(cache.evicted, [a, b, c])
        self.assertEqual(tracker[SWA], 15)

    def test_stale_entries_skipped(self):
        # Evicting a also removes sibling b from the evictable set (collapsed);
        # b is then popped-but-stale and skipped, so it is not counted.
        a = _Node("a", prio=1, full=10, swa=10)
        b = _Node("b", prio=2, full=10, swa=10)
        cache = _FakeCacheForLeafEvict([a, b])
        orig = cache._evict_host_leaf

        def evict_and_collapse(x, tracker):
            orig(x, tracker)
            cache.evictable_host_leaves.discard(b)

        cache._evict_host_leaf = evict_and_collapse
        tracker = {FULL: 0, SWA: 0}
        self._drive(cache, num_tokens=100, key=SWA, tracker=tracker)
        self.assertEqual(cache.evicted, [a])

    def test_coevict_recorded_for_aux_component(self):
        a = _Node("a", prio=1, full=7, swa=7)
        cache = _FakeCacheForLeafEvict([a])
        tracker = {FULL: 0, SWA: 0}
        self._drive(cache, num_tokens=7, key=SWA, tracker=tracker)
        self.assertEqual(cache.coevict_calls, [(7, 1)])  # full freed, 1 leaf

    def test_no_coevict_note_for_full_key(self):
        # When Full itself is the driver there is no auxiliary binding pressure.
        a = _Node("a", prio=1, full=7, swa=7)
        cache = _FakeCacheForLeafEvict([a])
        tracker = {FULL: 0, SWA: 0}
        self._drive(cache, num_tokens=7, key=FULL, tracker=tracker)
        self.assertEqual(cache.coevict_calls, [])


class TestSwaComponentRouting(unittest.TestCase):
    """drive_host_eviction must route to atomic leaf eviction iff strict."""

    def _fake_component(self, strict):
        comp = types.SimpleNamespace()
        comp._strict_bit_exact = strict
        comp.component_type = SWA
        calls = {"leaf": [], "lru_get": 0}

        class _Cache:
            def drive_host_leaf_eviction(self, num_tokens, ct, tracker):
                calls["leaf"].append((num_tokens, ct))

            @property
            def host_lru_lists(self):
                calls["lru_get"] += 1

                class _L:
                    def get_lru_no_host_lock(_s):
                        return None

                return {SWA: _L()}

        comp.cache = _Cache()
        return comp, calls

    def _drive(self, comp, tracker):
        SWAComponent.drive_host_eviction(comp, 100, tracker)

    def test_strict_routes_to_leaf_eviction(self):
        comp, calls = self._fake_component(strict=True)
        self._drive(comp, {SWA: 0})
        self.assertEqual(calls["leaf"], [(100, SWA)])
        self.assertEqual(calls["lru_get"], 0)  # never touches the tombstone path

    def test_non_strict_uses_lru_path(self):
        comp, calls = self._fake_component(strict=False)
        self._drive(comp, {SWA: 0})
        self.assertEqual(calls["leaf"], [])
        self.assertGreaterEqual(calls["lru_get"], 1)


if __name__ == "__main__":
    unittest.main()


class TestWriteBackGuard(unittest.TestCase):
    """Strict bit-exact must fail fast at build() entry if write policy is not
    write_through (write_back can leave the SWA ring un-offloaded -> silent
    non-bit-exact reuse)."""

    import unittest.mock as _mock

    def _build(self, *, unified, write_policy, flag):
        strat = A._DeepSeekV4Strategy()
        kv = types.SimpleNamespace(_unified_kv=unified)
        sa = types.SimpleNamespace(hicache_write_policy=write_policy)
        flag_obj = types.SimpleNamespace(get=lambda: flag)
        with self._mock.patch.object(
            A.envs, "SGLANG_UNIFIED_KV_SWA_BIT_EXACT_HICACHE", flag_obj
        ):
            strat.build(
                cache=None,
                kvcache=kv,
                params=None,
                server_args=sa,
                load_cache_event=None,
            )

    def test_write_back_trips_guard(self):
        with self.assertRaises(ValueError) as ctx:
            self._build(unified=True, write_policy="write_back", flag=True)
        self.assertIn("write_through", str(ctx.exception))

    def test_write_through_passes_guard(self):
        # Passes the guard, then fails later on the None collaborators -- we only
        # assert it is NOT the guard ValueError.
        with self.assertRaises(Exception) as ctx:
            self._build(unified=True, write_policy="write_through", flag=True)
        self.assertNotIn(
            "requires --hicache-write-policy", str(ctx.exception)
        )

    def test_flag_off_no_guard(self):
        with self.assertRaises(Exception) as ctx:
            self._build(unified=True, write_policy="write_back", flag=False)
        self.assertNotIn(
            "requires --hicache-write-policy", str(ctx.exception)
        )

    def test_non_unified_no_guard(self):
        with self.assertRaises(Exception) as ctx:
            self._build(unified=False, write_policy="write_back", flag=True)
        self.assertNotIn(
            "requires --hicache-write-policy", str(ctx.exception)
        )
