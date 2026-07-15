from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

logger = logging.getLogger(__name__)
_SWA_DBG_CHECKSUM = os.environ.get("SGLANG_SWA_DBG_CHECKSUM") == "1"

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    LRURefreshPhase,
    TreeComponent,
    next_component_uuid,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


def _state_rides(comp):
    """(host_pool, device_state_pools, pending_attr, host_value_attr) for each
    enabled compress-state pool that rides an SWA node. Empty unless strict
    bit-exact wired the c4 state pools (non-DSv4 / flag-off / test fake)."""
    if getattr(comp, "_c4_state_layer_index", None) is None:
        return []
    rides = []
    if getattr(comp, "_c4_state_host_pool", None) is not None:
        rides.append(
            (
                comp._c4_state_host_pool,
                comp._compress_state_pools,
                "_c4_state_pending_host",
                "_c4_state_host_value",
            )
        )
    if getattr(comp, "_c4_indexer_state_host_pool", None) is not None:
        rides.append(
            (
                comp._c4_indexer_state_host_pool,
                comp._indexer_compress_state_pools,
                "_c4_indexer_state_pending_host",
                "_c4_indexer_state_host_value",
            )
        )
    return rides


def _free_state_bindings(comp, node, which: str) -> None:
    """Free a node's ridden state pages. which in {pending, host, both}."""
    for hp, _dev, pending_attr, hv_attr in _state_rides(comp):
        if which in ("pending", "both"):
            pend = getattr(node, pending_attr, None)
            if pend is not None:
                hp.free(pend)
                setattr(node, pending_attr, None)
        if which in ("host", "both"):
            hv = getattr(node, hv_attr, None)
            if hv is not None:
                hp.free(hv)
                setattr(node, hv_attr, None)


def _promote_state_pending(comp, node) -> None:
    """Adopt each ridden state pending page as its host_value, coupled to the
    SWA/Full coordinated BACKUP_HOST (co-lifetime)."""
    for _hp, _dev, pending_attr, hv_attr in _state_rides(comp):
        pend = getattr(node, pending_attr, None)
        if pend is not None:
            if getattr(node, hv_attr, None) is None:
                setattr(node, hv_attr, pend)
            setattr(node, pending_attr, None)


def _restore_state_windows(comp, node, swa_chunk) -> None:
    """LOAD_BACK: write the ridden c4 / c4-indexer overlap state back into the
    device state ring at translate_from_swa_loc_to_state_loc(restored swa slots).
    No-op unless the node carries ridden state host pages.

    Each captured window token sits at host ring offset ``swa_loc % ring``
    (deterministic from position; see the compressor capture), so gather by the
    restored slots' offsets and scatter to their state locs -- config
    independent (no dependence on the 4..7 layout)."""
    rides = _state_rides(comp)
    if not rides:
        return
    layer_index = comp._c4_state_layer_index
    for hp, dev_pools, _pending_attr, hv_attr in rides:
        hv = getattr(node, hv_attr, None)
        if hv is None:
            continue
        slot_page = hp.slot_page_size
        page_row = int(hv[0].item()) // slot_page
        first_pool = dev_pools[next(iter(layer_index))]
        ratio = first_pool.ratio
        if swa_chunk.numel() < ratio:
            continue
        swa_win = swa_chunk[-ratio:]
        offsets = (swa_win % slot_page).to("cpu")
        for gl, li in layer_index.items():
            sp = dev_pools[gl]
            dev = sp.kv_score_buffer.kv_score
            state_locs = sp.translate_from_swa_loc_to_state_loc(swa_win)
            host_view = (
                hp.data_refs[li][page_row]
                .view(dev.dtype)
                .reshape(slot_page, dev.shape[1])
            )
            window = host_view[offsets].to(dev.device)
            dev[state_locs.to(dev.device)] = window


class SWAComponent(TreeComponent):
    """Sliding window attention component.

    Each SWA node stores translated SWA pool indices as its component
    value, independent of the full attention indices on the same tree node.
    When SWA data is evicted from an internal node the node is tombstoned
    — its SWA component value becomes None while the full attention
    value stays intact.
    """

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

        assert isinstance(
            cache.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        ), f"SWAComponent requires SWATokenToKVPoolAllocator, got {type(cache.token_to_kv_pool_allocator)}"
        super().__init__(cache, params)
        self.sliding_window_size = params.sliding_window_size
        # HiCache state: set to host SWA pool when HiCache enabled
        self._swa_kv_pool_host = None
        # Strict bit-exact SWA HiCache (unified_kv only): when True, SWA host
        # eviction must never drop a node's SWA copy while keeping its Full
        # copy on host (that "Full-host without SWA-host" orphan would force a
        # non-bit-exact tail reprefill on reuse). Wired at pool-attach time.
        self._strict_bit_exact = False
        # req_pool_idx of the request currently being cached; used to look up
        # its prefill-captured SWA host pages during insert.
        self._capture_rid = None
        # Strict bit-exact: c4 / c4-indexer compress-state pages ride the SWA
        # node (captured at prefill, restored on reuse). Wired at pool-attach
        # time; None keeps all state logic a no-op (non-DSv4 / flag off).
        self._c4_state_host_pool = None
        self._c4_indexer_state_host_pool = None
        self._c4_state_layer_index = None
        self._compress_state_pools = None
        self._indexer_compress_state_pools = None

    component_type = ComponentType.SWA

    def _translate_full_to_swa(self, full_indices: torch.Tensor) -> torch.Tensor:
        return self.cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            full_indices
        )

    def refresh_lru(
        self,
        phase: LRURefreshPhase,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
    ) -> None:
        match phase:
            case LRURefreshPhase.WALKDOWN:
                # Walk-down would refresh every visited ancestor to MRU,
                # but most are outside the active sliding window and must
                # stay evictable. Window-bounded refresh runs at
                # MATCH_END / INSERT_END instead.
                return
            case LRURefreshPhase.MATCH_END | LRURefreshPhase.INSERT_END:
                self.cache.lru_lists[
                    self.component_type
                ].reset_node_and_window_ancestors_mru(
                    node,
                    root_node,
                    self.sliding_window_size + self.cache.page_size,
                    self.node_has_component_data,
                )
            case _:
                raise ValueError(f"Unknown LRURefreshPhase: {phase}")

    def _restore_device_value(self, node: UnifiedTreeNode, value: torch.Tensor) -> None:
        ct = self.component_type
        node.component_data[ct].value = value
        # A freshly (re)assigned device SWA value is live for the current
        # holder; drop any stale deferred owner-release intent from a prior life.
        if getattr(node, "_swa_release_pending", False):
            node._swa_release_pending = False
        host_lru = self.cache.host_lru_lists[ct]
        if host_lru.in_list(node):
            host_lru.remove_node(node)
        self.cache.lru_lists[ct].insert_mru(node)
        self.cache.component_evictable_size_[ct] += len(value)

    def _restore_device_value_with_locked_full(
        self,
        node: UnifiedTreeNode,
        full_value: torch.Tensor,
        incoming_full_value: torch.Tensor,
    ) -> None:
        allocator = self.cache.token_to_kv_pool_allocator
        swa_value = self._translate_full_to_swa(incoming_full_value)
        allocator.set_full_to_swa_mapping(full_value, swa_value)
        allocator.full_to_swa_index_mapping[incoming_full_value.to(torch.int64)] = 0
        allocator.full_attn_allocator.free(incoming_full_value)
        self._restore_device_value(node, swa_value)

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        sliding_window_size = self.sliding_window_size
        ct = self.component_type
        strict_bit_exact = self._strict_bit_exact
        state = {"len": float("inf")}

        # unified_kv never caches the SWA ring (per-request, not content-stable),
        # so SWA bookkeeping must not gate the match here.
        swa_device_only_hicache = (
            self._swa_kv_pool_host is None and self.cache.cache_controller is not None
        )

        def validator(node: UnifiedTreeNode) -> bool:
            cd = node.component_data[ct]
            # HiCache: a host-only tombstone is a valid match boundary too
            # — load_back will restore SWA from host before use.
            if cd.value is None and (match_device_only or cd.host_value is None):
                state["len"] = 0
                if swa_device_only_hicache and (node.backuped or not node.evicted):
                    return True
                return False
            # I2-prime: strict bit-exact never trusts the per-request device SWA
            # ring as a cross-request truth source for the REUSE boundary (the
            # device ring is recycled when the owner's req_pool_idx is reused).
            # Only a durable host copy counts for the device-or-host reuse match;
            # a node with device value but no host copy truncates the reuse match
            # so it restores from host (LOAD_BACK) or recomputes (I6) instead of
            # serving a stale device ring. This is scoped to the reuse match
            # (``not match_device_only``): the device-only match must still report
            # a request's own freshly-computed, not-yet-backed-up nodes as device
            # resident, else cache_unfinished_req's self-match returns empty
            # device indices (new_prefix_len > len(new_indices)). Stale device
            # residency across requests is instead closed by the deferred
            # owner-release tombstone, which nulls the device value once the host
            # copy is durable.
            if (
                strict_bit_exact
                and not match_device_only
                and cd.host_value is None
            ):
                state["len"] = 0
                return False
            state["len"] += len(node.key)
            return state["len"] >= sliding_window_size

        return validator

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        ct = self.component_type
        n_swa = 0
        swa_host_hit = 0
        node = result.best_match_node
        root = self.cache.root_node
        # Mine 2 (warm reuse): on the reuse path in strict mode, the per-request
        # device SWA ring is not a durable cross-request truth (I2'), so a node
        # that is BOTH device-resident and host-backed must still be counted as
        # a host hit -- otherwise swa_host_hit_length stays 0 and the load_back
        # gate never opens for it. This uses the SAME host-backed predicate as
        # build_hicache_transfers(LOAD_BACK) below. Self-match (for_reuse=False)
        # keeps the OLD behavior: cd.value is trusted first, since the request's
        # own freshly-computed nodes aren't host-backed yet and
        # cache_unfinished_req relies on this not falsely opening the gate.
        strict_reuse = self._strict_bit_exact and params.for_reuse
        while node is not root and n_swa < self.sliding_window_size:
            cd = node.component_data[ct]
            if strict_reuse and cd.host_value is not None:
                swa_host_hit += len(cd.host_value)
                n_swa += len(cd.host_value)
            elif cd.value is not None:
                n_swa += len(cd.value)
            elif cd.host_value is not None:
                # TODO(hzh): load_back may currently restore a full host-tombstone
                # segment whose length exceeds sliding_window_size. Once
                # load_back is constrained to fetch only one sliding window
                # worth of pages, cap swa_host_hit at sliding_window_size
                # here so the scheduler budget matches the actual device-pool
                # consumption.
                swa_host_hit += len(cd.host_value)
                n_swa += len(cd.host_value)
            else:
                break
            node = node.parent
        if swa_host_hit > 0:
            return result._replace(
                swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit)
            )
        return result

    def update_component_on_insert_overlap(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> int:
        if params.prev_prefix_len >= total_prefix_len + prefix_len:
            return prefix_len

        is_tombstone = node.component_data[self.component_type].value is None
        if not is_tombstone:
            return prefix_len

        full_cd = node.component_data[BASE_COMPONENT_TYPE]
        swa_evicted_seqlen = params.swa_evicted_seqlen
        assert (
            node.component_data[self.component_type].lock_ref == 0
        ), f"tombstone {self.component_type} lock_ref should be 0, node {node.id}"
        assert (
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{self.component_type}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        if swa_evicted_seqlen <= total_prefix_len:
            # Branch 1: entire value_slice is within SWA window — recover
            if full_cd.lock_ref > 0:
                self._restore_device_value_with_locked_full(
                    node, full_cd.value, value_slice
                )
                return 0
            self.cache.token_to_kv_pool_allocator.free(full_cd.value)
            full_cd.value = value_slice.clone()
            swa_value = self._translate_full_to_swa(full_cd.value)
            self._restore_device_value(node, swa_value)
            return 0
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            # Branch 2: value_slice[start_idx:] is within SWA window — partial recover
            start_idx = swa_evicted_seqlen - total_prefix_len
            if full_cd.lock_ref > 0:
                self.cache._split_node(node.key, node, start_idx)
                full_cd = node.component_data[BASE_COMPONENT_TYPE]
                self._restore_device_value_with_locked_full(
                    node, full_cd.value, value_slice[start_idx:]
                )
                return start_idx
            self.cache.token_to_kv_pool_allocator.free(full_cd.value[start_idx:])
            self.cache._split_node(node.key, node, start_idx)
            node.component_data[BASE_COMPONENT_TYPE].value = value_slice[
                start_idx:
            ].clone()
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            self._restore_device_value(node, swa_value)
            return start_idx
        else:
            # Branch 3: entire value_slice is outside SWA window — not consumed
            return prefix_len

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
    ) -> None:
        # _unevict_node_on_insert already wrote the request's fresh KV slice
        # into the base value. We just need to rebuild SWA from that slice for
        # the in-window portion. There is no old SWA slot to free here.
        ct = self.component_type
        if node.component_data[ct].value is not None:
            return
        assert (
            node.component_data[ct].lock_ref == 0
        ), f"tombstone {ct} lock_ref should be 0 on unevict, node {node.id}"
        swa_evicted_seqlen = params.swa_evicted_seqlen
        assert (
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{ct}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        full_value = node.component_data[BASE_COMPONENT_TYPE].value
        if swa_evicted_seqlen <= total_prefix_len:
            swa_value = self._translate_full_to_swa(full_value)
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            start_idx = swa_evicted_seqlen - total_prefix_len
            self.cache._split_node(node.key, node, start_idx)
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            swa_value = self._translate_full_to_swa(full_value)
        else:
            return
        self._restore_device_value(node, swa_value)

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        if not is_new_leaf:
            return

        node_start = result.prefix_len
        split_pos = params.swa_evicted_seqlen - node_start

        if split_pos <= 0:
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[self.component_type].value = swa_value
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(swa_value)
        elif split_pos < len(node.key):
            # Node straddles the SWA eviction boundary
            # Split into parent (tombstone, no SWA) and child (with SWA)
            # After _split_node, `node` becomes the child
            self.cache._split_node(node.key, node, split_pos)
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[self.component_type].value = swa_value
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(swa_value)
        else:
            # Entire leaf is outside the SWA window — left as a tombstone.
            return

        # Bind the prefill-captured host window to this SWA node. Both branches
        # above leave `node` covering [swa_start, swa_start + len(value)) with
        # swa_start == max(node_start, swa_evicted_seqlen).
        self._bind_captured_swa_host(
            node, max(node_start, params.swa_evicted_seqlen)
        )
        self._maybe_split_leaf_for_swa_lock(node)

    def _maybe_split_leaf_for_swa_lock(self, leaf: UnifiedTreeNode) -> None:
        """Cap a fresh SWA leaf at one page-aligned window so locking it pins
        only one window of SWA pool, not the whole (long chunked-prefill) leaf.
        """
        ct = self.component_type
        cd = leaf.component_data[ct]
        if leaf is self.cache.root_node or cd.value is None or cd.lock_ref > 0:
            return

        page_size = self.cache.page_size
        # Smallest page-aligned size that still covers the sliding window.
        tail_size = (self.sliding_window_size + page_size - 1) // page_size * page_size
        leaf_len = len(leaf.key)
        if leaf_len <= tail_size:
            return
        split_at = leaf_len - tail_size
        if page_size > 1 and (split_at % page_size != 0 or leaf_len % page_size != 0):
            return

        self.cache._split_node(leaf.key, leaf, split_at)

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        new_parent.component_data[self.component_type].lock_ref = child.component_data[
            self.component_type
        ].lock_ref

        child_swa_value = child.component_data[self.component_type].value
        if child_swa_value is not None:
            split_len = len(new_parent.key)
            new_parent.component_data[self.component_type].value = child_swa_value[
                :split_len
            ].clone()
            child.component_data[self.component_type].value = child_swa_value[
                split_len:
            ].clone()
        else:
            new_parent.component_data[self.component_type].value = None

        child_swa_host_value = child.component_data[self.component_type].host_value
        if child_swa_host_value is not None:
            split_len = len(new_parent.key)
            full_span = split_len + len(child.key)
            host_lru = self.cache.host_lru_lists[self.component_type]
            if len(child_swa_host_value) == full_span:
                # Common case: host_value spans the whole node; split by key len.
                new_parent.component_data[self.component_type].host_value = (
                    child_swa_host_value[:split_len].clone()
                )
                child.component_data[self.component_type].host_value = (
                    child_swa_host_value[split_len:].clone()
                )
            else:
                # host_value holds only the sliding window at the child's end
                # boundary, so it belongs entirely to the child. The parent's own
                # boundary window (if any) is stored separately, not here.
                new_parent.component_data[self.component_type].host_value = None
                # child keeps child_swa_host_value unchanged
            if (
                new_parent.component_data[self.component_type].value is None
                and new_parent.component_data[self.component_type].host_value
                is not None
            ):
                host_lru.insert_mru(new_parent)
            if child.component_data[
                self.component_type
            ].value is None and not host_lru.in_list(child):
                host_lru.insert_mru(child)

        # parent inherits the swa_uuid from child for swa lock ref
        new_parent.component_data[self.component_type].metadata["uuid"] = (
            child.component_data[self.component_type].metadata.get("uuid")
        )
        child.component_data[self.component_type].metadata.pop("uuid", None)

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        ct = self.component_type
        cd = node.component_data[ct]
        freed = 0
        host_freed = 0

        # Device layer
        if EvictLayer.DEVICE in target and cd.value is not None:
            # Pass full indices to free_swa so slots with no SWA pair are
            # skipped. Freeing swa_value directly would double free those
            # entries since they all map to the same sentinel slot.
            self.cache.token_to_kv_pool_allocator.free_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            freed = len(cd.value)
            self.cache.component_evictable_size_[ct] -= freed
            cd.value = None
            # Co-lifetime: a captured page not yet promoted to host_value must
            # not outlive its device SWA; free it (node degrades to recompute).
            pending = getattr(node, "_swa_pending_host", None)
            if pending is not None:
                if self._swa_kv_pool_host is not None:
                    self._swa_kv_pool_host.free(pending)
                node._swa_pending_host = None
            # Ridden c4-state pending co-lives with SWA pending: free together.
            _free_state_bindings(self, node, "pending")

        # Host layer
        host_lru = self.cache.host_lru_lists[ct]
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            if self._swa_kv_pool_host is not None:
                self._swa_kv_pool_host.free(cd.host_value)
            cd.host_value = None
            if host_lru.in_list(node):
                host_lru.remove_node(node)
            # Ridden c4-state host_value co-lives with SWA host_value.
            _free_state_bindings(self, node, "host")

        # After device tombstone: if host_value remains, move into host LRU
        if (
            target is EvictLayer.DEVICE
            and cd.value is None
            and cd.host_value is not None
        ):
            if not host_lru.in_list(node):
                host_lru.insert_mru(node)

        return freed, host_freed

    def evict_device_on_owner_release(self, node: UnifiedTreeNode) -> None:
        """Strict bit-exact: drop a node's per-request device SWA ring value
        once its owning request has finished and no other request holds the
        SWA lock, so cross-request reuse restores the true window from host
        (I1) instead of trusting the device ring.

        The device SWA lives in a per-request ring (``req_slot*ring +
        pos%ring``) that is overwritten as the owner decodes and is recycled
        when the owner's ``req_pool_idx`` is reused. Its bytes are therefore
        only valid for the owning request's live window, never for a later
        cross-request reuse. Called from ``cache_finished_req`` after the owner
        released its lock: at that point the ring slots still belong to the
        finishing request (safe to free, no aliasing with any live window),
        and the host copy is the durable truth source.

        Gates (all required for safety / sanity_check):
          - strict mode + SWA host pool wired (feature on);
          - device value present;
          - ``host_value`` committed (keep the host copy so reuse restores it;
            a pending-only page is left until its coordinated BACKUP_HOST
            commits, avoiding co-lifetime races);
          - SWA ``lock_ref == 0`` (no other active holder — required, else
            sanity_check flags "evicted but lock_ref>0").
        """
        if not self._strict_bit_exact or self._swa_kv_pool_host is None:
            return
        cd = node.component_data[self.component_type]
        if cd.value is None:
            return
        if cd.host_value is None or cd.lock_ref > 0:
            # Host copy not durable yet (async write_through backup still in
            # flight) or another request still holds the SWA lock, so we cannot
            # free the device ring value right now. But once this owner is gone
            # the per-request ring slot is recycled and its bytes become stale,
            # so the value MUST NOT be trusted for cross-request reuse. Defer:
            # mark the node so the coordinated BACKUP_HOST commit drops the
            # device value the instant the host copy becomes durable and no
            # holder remains. Without this the device ring is never invalidated
            # and reuse would keep a stale device slot alive (I1/I2 violation).
            node._swa_release_pending = True
            return
        self.cache._evict_component_and_detach_lru(
            node, self, target=EvictLayer.DEVICE
        )

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 1

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        request = params.swa_num_tokens
        ct = self.component_type
        lru = self.cache.lru_lists[ct]
        x = lru.get_lru_no_lock()
        while tracker[ct] < request and x is not None and lru.in_list(x):
            assert x.component_data[ct].value is not None
            if x in self.cache.evictable_device_leaves:
                # D-leaf: atomic eviction of all components
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_device_leaf(x, tracker)
                if not lru.in_list(x_next):
                    x_next = lru.get_lru_no_lock()
                x = x_next
            else:
                # Internal: tombstone SWA + cascade
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.DEVICE, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = x_next

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type
        root = self.cache.root_node
        sliding_window_size = self.sliding_window_size
        swa_lock_size = 0
        swa_uuid = None
        uuid_key = "host_uuid" if lock_host else "uuid"
        lru = self.cache.host_lru_lists[ct] if lock_host else self.cache.lru_lists[ct]

        # Tombstoned nodes (cd.value is None) have no SWA chunk to protect
        # skip them and keep walking up. This path is hit when HiCache
        # backs up a FULL present internal node whose SWA was already evicted.
        cur = node
        while cur != root and swa_lock_size < sliding_window_size:
            comp = cur.component_data[ct]
            value = comp.host_value if lock_host else comp.value
            if value is None:
                result.skip_lock_node_ids.setdefault(ct, set()).add(cur.id)
                cur = cur.parent
                continue

            ref = comp.host_lock_ref if lock_host else comp.lock_ref
            if ref == 0:
                if lock_host:
                    if lru.in_list(cur):
                        lru.remove_node(cur)
                else:
                    key_len = len(cur.key)
                    self.cache.component_evictable_size_[ct] -= key_len
                    self.cache.component_protected_size_[ct] += key_len
            if lock_host:
                comp.host_lock_ref = ref + 1
            else:
                comp.lock_ref = ref + 1
            swa_lock_size += len(value)
            if swa_lock_size >= sliding_window_size:
                if comp.metadata.get(uuid_key) is None:
                    comp.metadata[uuid_key] = next_component_uuid()
                swa_uuid = comp.metadata[uuid_key]
            cur = cur.parent

        if lock_host:
            result.swa_uuid_for_host_lock = swa_uuid
        else:
            result.swa_uuid_for_lock = swa_uuid
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        ct = self.component_type
        root = self.cache.root_node
        swa_uuid_for_lock = (
            (params.swa_uuid_for_host_lock if lock_host else params.swa_uuid_for_lock)
            if params
            else None
        )
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()
        dec_swa = True
        uuid_key = "host_uuid" if lock_host else "uuid"

        # A node in skip_lock_node_ids was a tombstone when this lock was acquired.
        cur = node
        while cur != root and dec_swa:
            comp = cur.component_data[ct]
            if cur.id in skip_lock_node_ids:
                cur = cur.parent
                continue
            ref = comp.host_lock_ref if lock_host else comp.lock_ref
            if ref == 0:
                cur = cur.parent
                continue
            if ref == 1:
                if lock_host:
                    if comp.value is None and comp.host_value is not None:
                        host_lru = self.cache.host_lru_lists[ct]
                        if not host_lru.in_list(cur):
                            host_lru.insert_mru(cur)
                else:
                    key_len = len(comp.value)
                    self.cache.component_evictable_size_[ct] += key_len
                    self.cache.component_protected_size_[ct] -= key_len
            if lock_host:
                comp.host_lock_ref = ref - 1
            else:
                comp.lock_ref = ref - 1
            if swa_uuid_for_lock and comp.metadata.get(uuid_key) == swa_uuid_for_lock:
                dec_swa = False
            cur = cur.parent

    def release_window_lock(
        self,
        node: UnifiedTreeNode,
        swa_uuid_for_lock: Optional[int] = None,
    ) -> None:
        """Early-release the SWA lock along [node, swa_uuid_for_lock] while
        leaving Full and Mamba locks intact.

        Called when a request's decode position has advanced past the sliding
        window — the SWA portion of the tree lock is no longer needed but the
        Full lock must stay so the request's prefix is protected.

        Caller (UnifiedRadixCache.dec_swa_lock_only) must ensure this is
        invoked at most once per (node, swa_uuid_for_lock) pair.
        """
        ct = self.component_type
        root = self.cache.root_node

        cur = node
        while cur is not root:
            cd = cur.component_data[ct]
            # Acquire skips tombstoned nodes; release must skip them too. Same
            # for nodes with lock_ref == 0 — acquire never credited them.
            if cd.value is None or cd.lock_ref == 0:
                if swa_uuid_for_lock and cd.metadata.get("uuid") == swa_uuid_for_lock:
                    break
                cur = cur.parent
                continue

            cd.lock_ref -= 1
            if cd.lock_ref == 0:
                key_len = len(cur.key)
                self.cache.component_protected_size_[ct] -= key_len
                self.cache.component_evictable_size_[ct] += key_len
                if self.cache._is_device_leaf(cur):
                    self.cache._evict_component_and_detach_lru(
                        cur, self, target=EvictLayer.DEVICE
                    )

            if swa_uuid_for_lock and cd.metadata.get("uuid") == swa_uuid_for_lock:
                break
            cur = cur.parent

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        # Unfinished requests can already have an SWA-evicted prefix; preserve
        # that boundary so insertion creates a tombstone instead of live SWA KV.
        insert_params.swa_evicted_seqlen = req.swa_evicted_seqlen
        self._capture_rid = req.req_pool_idx
        return None

    def free_out_of_window_slots(
        self, req: Req, pre_len: int, insert_params: InsertParams
    ) -> None:
        if self.sliding_window_size is not None:
            free_swa_out_of_window_slots(
                req,
                pre_len,
                sliding_window_size=self.sliding_window_size,
                page_size=self.cache.page_size,
                req_to_token_pool=self.cache.req_to_token_pool,
                token_to_kv_pool_allocator=self.cache.token_to_kv_pool_allocator,
            )
        insert_params.swa_evicted_seqlen = req.swa_evicted_seqlen

    # ---- HiCache Hooks ----

    def _bind_captured_swa_host(
        self, node: UnifiedTreeNode, swa_start: int
    ) -> None:
        """Stash the prefill-captured host page as a PENDING ref on the node.

        Co-lifetime (I3): the SWA host_value must not exist before the node's
        Full host_value. So we do NOT set host_value here; we attach it later
        through the coordinated BACKUP_HOST commit (which runs together with the
        Full host backup). Until then the page is held in ``node._swa_pending_host``
        and is freed on device eviction if the node is never backed up.

        The node ends at page boundary B; its captured window is [B-win, B) keyed
        (rid, B). If it was not captured (host pool full / outside this chunk),
        leave the node to the normal backup / recompute path.
        """
        hp = self._swa_kv_pool_host
        if hp is None:
            return
        staging = getattr(hp, "_capture_staging", None)
        rid = self._capture_rid
        if not staging or rid is None:
            return
        cd = node.component_data[self.component_type]
        if cd.value is None or cd.host_value is not None:
            return
        win = hp.slot_page_size
        # The node ends at page boundary B = swa_start + len(value); its host
        # copy is the single captured window keyed (rid, B). The earlier,
        # out-of-window part of the node is never attended and is not stored.
        node_end = swa_start + len(cd.value)
        h = staging.pop((rid, int(node_end)), None)
        if h is None:
            # Window not captured -> fall back to normal backup / recompute.
            return
        host_value = h.to(torch.int64)
        if len(host_value) != win:
            hp.free(host_value)
            return
        # Atomic co-lifetime across {SWA, c4-state, indexer-state}: bind only if
        # every enabled ridden state window for this boundary was also captured.
        # If any is missing, free everything so the node degrades to recompute --
        # a reused SWA window without its exact c4 pre-state would read stale
        # state and break bit-exactness.
        rides = _state_rides(self)
        state_tiles = []
        atomic_ok = True
        for shp, _dev, pending_attr, _hv in rides:
            sh = shp._capture_staging.pop((rid, int(node_end)), None)
            state_tiles.append((shp, pending_attr, sh))
            if sh is None:
                atomic_ok = False
        if not atomic_ok:
            for shp, _pa, sh in state_tiles:
                if sh is not None:
                    shp.free(sh)
            hp.free(host_value)
            return
        # Defer attach to the coordinated BACKUP_HOST (co-lifetime with Full host).
        node._swa_pending_host = host_value
        for _shp, pending_attr, sh in state_tiles:
            setattr(node, pending_attr, sh)
        if _SWA_DBG_CHECKSUM:
            crc_map = getattr(hp, "_capture_crc", None)
            if crc_map:
                keys = [
                    k for k in crc_map if k[0] == rid and k[1] == int(node_end)
                ]
                if keys:
                    cd.metadata["dbg_swa_crc"] = {k[2]: crc_map.pop(k) for k in keys}

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: Optional[InsertResult] = None,
        insert_params: Optional[InsertParams] = None,
    ) -> None:
        # Release any capture staging owned by this request that no node claimed
        # (interior / out-of-window windows), then drop the stashed rid.
        hp = self._swa_kv_pool_host
        rid = self._capture_rid
        self._capture_rid = None
        if hp is None or rid is None:
            return
        staging = getattr(hp, "_capture_staging", None)
        if not staging:
            return
        leftover = [k for k in staging if k[0] == rid]
        for k in leftover:
            hp.free(staging.pop(k))
        # Release any ridden state capture staging this request never claimed.
        for shp, _dev, _pa, _hv in _state_rides(self):
            sstage = getattr(shp, "_capture_staging", None)
            if not sstage:
                continue
            for k in [k for k in sstage if k[0] == rid]:
                shp.free(sstage.pop(k))
        if _SWA_DBG_CHECKSUM:
            crc_map = getattr(hp, "_capture_crc", None)
            if crc_map:
                for k in [k for k in crc_map if k[0] == rid]:
                    crc_map.pop(k, None)

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        req: Optional[Req] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        ct = self.component_type

        # unified_kv keeps SWA as a device-only ring.
        if self._swa_kv_pool_host is None and self.cache.cache_controller is not None:
            return None

        if phase == CacheTransferPhase.BACKUP_HOST:
            cd = node.component_data[ct]
            if cd.value is None:
                return None
            if cd.host_value is not None:
                # Already populated from a prior backup; do not re-copy.
                return None
            pending = getattr(node, "_swa_pending_host", None)
            if pending is not None:
                # Co-lifetime: adopt the prefill-captured host page (already on
                # host) through the coordinated backup, so SWA host_value is set
                # together with Full host_value (never before). device_indices is
                # None -> write_backup skips the (redundant) device->host copy.
                return [
                    PoolTransfer(
                        name=PoolName.SWA,
                        host_indices=pending,
                        device_indices=None,
                    )
                ]
            if self._strict_bit_exact:
                # Strict: SWA host pages are allocated only at prefill capture
                # time. With no captured page (host pool full / window missed),
                # emit no SWA host_value; the node falls back to recompute on
                # reuse (I6). Never back up the device ring here -- it holds only
                # the latest window per slot (older windows byte-stale) and
                # allocating host at backup can exhaust the small SWA pool.
                return None
            # Best-effort: back up the device ring.
            # cd.value already holds SWA-pool indices (translated at insert time).
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=cd.value.to(torch.int64),
                )
            ]

        if phase == CacheTransferPhase.LOAD_BACK:
            # `node` is best_match_node; the SWA validator guarantees every
            # ancestor within `sliding_window_size` has value or host_value.
            n_swa = 0
            backed_up: list[torch.Tensor] = []
            nodes: list = []
            cur = node
            while cur is not self.cache.root_node and n_swa < self.sliding_window_size:
                cd = cur.component_data[ct]
                assert cd.host_value is not None or cd.value is not None
                if self._strict_bit_exact and cd.host_value is not None:
                    # Mine 2 (warm reuse): the per-request device SWA ring is
                    # not a durable cross-request truth in strict mode, even
                    # when `cd.value` is still set (stale, recycled slot from
                    # a prior request). Collect the host copy so it is loaded
                    # and commit_hicache_transfer's _restore_device_value
                    # overwrites the stale slot with host truth. Same
                    # host-backed predicate as finalize_match_result's
                    # for_reuse=True gate above.
                    backed_up.append(cd.host_value)
                    nodes.append(cur)
                    n_swa += len(cd.host_value)
                elif cd.value is not None:
                    # device exists (best-effort mode, or strict with no
                    # durable host copy), skip it
                    n_swa += len(cd.value)
                else:
                    # host only, collect it
                    backed_up.append(cd.host_value)
                    nodes.append(cur)
                    n_swa += len(cd.host_value)
                cur = cur.parent

            if not backed_up:
                return None

            backed_up.reverse()
            nodes.reverse()

            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=torch.cat(backed_up),
                    device_indices=None,
                    nodes_to_load=nodes,
                )
            ]

        if phase == CacheTransferPhase.BACKUP_STORAGE:
            # I4 (L2-only): strict bit-exact SWA HiCache keeps SWA values on the
            # host pool only. Persisting them to the L3 storage backend would let
            # a reused prefix restore an SWA window that is no longer coupled to
            # its Full-host lifetime, breaking bit-exactness. Never emit an L3
            # transfer in strict mode; this holds regardless of L3 config.
            if self._strict_bit_exact:
                return None
            cd = node.component_data[ct]
            if cd.host_value is None or not node.hash_value:
                return None
            num_pages = len(cd.host_value) // self.cache.page_size
            if num_pages == 0:
                return None
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=cd.host_value[-num_pages * self.cache.page_size :],
                    keys=node.hash_value[-num_pages:],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        if phase == CacheTransferPhase.PREFETCH:
            # Require a full sliding window.
            sw_pages = (
                self.sliding_window_size + self.cache.page_size - 1
            ) // self.cache.page_size
            if sw_pages == 0 or prefetch_tokens // self.cache.page_size < sw_pages:
                return None
            num_tokens = sw_pages * self.cache.page_size
            host_indices = self._swa_kv_pool_host.alloc(num_tokens)
            if host_indices is None:
                self.cache.evict_host(num_tokens, ComponentType.SWA)
                host_indices = self._swa_kv_pool_host.alloc(num_tokens)
            if host_indices is None:
                return []
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=host_indices,
                    keys=["__placeholder__"] * sw_pages,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        *,
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfers and transfers[0].host_indices is not None:
                cd = node.component_data[ct]
                if cd.host_value is None:
                    # Same bookkeeping the eager insert path used (host_value +
                    # evictable-leaf sets); host-LRU insert is deferred to the
                    # device tombstone (cd.value is still set here).
                    self._attach_swa_host_value(node, transfers[0].host_indices)
                if transfers[0].device_indices is None:
                    # Adopted the pre-staged capture page; ownership now held by
                    # host_value (same page) -> drop the pending ref.
                    node._swa_pending_host = None
                    # Adopt the ridden c4-state pages together (co-lifetime).
                    _promote_state_pending(self, node)
            # Deferred owner-release tombstone: if the owning request finished
            # while this host backup was still in flight (host_value was None at
            # cache_finished_req, so evict_device_on_owner_release deferred the
            # device free), drop the now-recycled per-request device SWA value
            # now that the host copy is durable and no holder remains. This
            # closes the async write_through race where the device ring would
            # otherwise stay alive and be trusted on cross-request reuse.
            if getattr(node, "_swa_release_pending", False):
                cd = node.component_data[ct]
                if (
                    self._strict_bit_exact
                    and self._swa_kv_pool_host is not None
                    and cd.value is not None
                    and cd.host_value is not None
                    and cd.lock_ref == 0
                ):
                    node._swa_release_pending = False
                    self.cache._evict_component_and_detach_lru(
                        node, self, target=EvictLayer.DEVICE
                    )
            return

        if phase == CacheTransferPhase.LOAD_BACK:
            assert transfers and transfers[0].device_indices is not None
            xfer = transfers[0]
            device_indices = xfer.device_indices
            allocator = self.cache.token_to_kv_pool_allocator

            offset = 0
            for n in xfer.nodes_to_load or []:
                cd_n = n.component_data[ct]
                n_tokens = len(cd_n.host_value)
                swa_chunk = device_indices[offset : offset + n_tokens].clone()
                self._restore_device_value(n, swa_chunk)
                # host_value holds the sliding window [B-n_tokens, B). Map its
                # full indices to the restored SWA slots (out-of-window full
                # tokens keep sentinel 0, never read under the SWA mask). The
                # window may extend before this node own start when the node was
                # split shorter than the window (its host_value still spans the
                # whole window); gather the window full indices across the node
                # and its ancestors, in token order, so full<->swa lengths
                # match. In the common (unsplit) case the node own full value
                # already has >= n_tokens and no ancestor is touched.
                window_full = self._gather_window_full_indices(n, n_tokens)
                allocator.set_full_to_swa_mapping(
                    window_full, swa_chunk[-window_full.numel() :]
                )
                # Restore the ridden c4 / c4-indexer overlap state onto the
                # device state ring at the freshly restored SWA slots.
                _restore_state_windows(self, n, swa_chunk)
                if _SWA_DBG_CHECKSUM and hasattr(self, "_dbg_verify_restore"):
                    self._dbg_verify_restore(cd_n)
                offset += n_tokens
            assert offset == len(xfer.host_indices)
            return

        if phase == CacheTransferPhase.PREFETCH:
            self._commit_prefetch(
                node,
                transfers,
                insert_result=insert_result,
                pool_storage_result=pool_storage_result,
            )
            return

    def _gather_window_full_indices(
        self, node: UnifiedTreeNode, n_tokens: int
    ) -> torch.Tensor:
        """Collect the last n_tokens FULL indices ending at node boundary, in
        token order, walking into ancestors when the node own full value is
        shorter than the sliding window (post-split case). In the common case
        the node own full value already has >= n_tokens, so this returns
        full.value[-n_tokens:] without touching any ancestor."""
        parts = []
        need = n_tokens
        cur = node
        root = getattr(self.cache, "root_node", None)
        while need > 0 and cur is not None and cur is not root:
            fv = cur.component_data[BASE_COMPONENT_TYPE].value
            if fv is None or len(fv) == 0:
                break
            take = min(need, len(fv))
            parts.append(fv[len(fv) - take :])
            need -= take
            if need <= 0:
                break
            cur = getattr(cur, "parent", None)
        assert parts, "no FULL indices available to restore SWA window"
        return torch.cat(list(reversed(parts)))

    def _dbg_verify_restore(self, cd_n) -> None:
        """TEMP (SGLANG_SWA_DBG_CHECKSUM): assert the bound host page still
        matches the checksum captured at prefill, proving the restore path
        served byte-exact windows. Immune to model non-determinism."""
        hp = self._swa_kv_pool_host
        crcs = (cd_n.metadata or {}).get("dbg_swa_crc")
        if hp is None or not crcs or cd_n.host_value is None:
            return
        slot_page = hp.slot_page_size
        page_row = int(cd_n.host_value[0].item()) // slot_page
        for layer, expected in crcs.items():
            b = hp.data_refs[layer][page_row].view(torch.uint8).reshape(-1)
            idx = torch.arange(b.numel(), device=b.device, dtype=torch.int64) + 1
            got = int((b.to(torch.int64) * idx).sum().item())
            assert got == expected, (
                f"[SWA-DBG] restore checksum mismatch layer={layer} "
                f"page_row={page_row} expected={expected} got={got}"
            )
        hp._dbg_restore_verified = getattr(hp, "_dbg_restore_verified", 0) + 1
        n = hp._dbg_restore_verified
        if n <= 5 or n % 50 == 0:
            logger.warning(
                "[SWA-DBG] restore verified bit-exact: %d windows (layers/window=%d)",
                n,
                len(crcs),
            )

    def _release_swa_host(self, host_indices: torch.Tensor) -> None:
        if host_indices is not None and host_indices.numel() > 0:
            self.cache.cache_controller.append_host_mem_release(
                extra_pools=[PoolTransfer(name=PoolName.SWA, host_indices=host_indices)]
            )

    def _attach_swa_host_value(
        self, node: UnifiedTreeNode, host_indices: torch.Tensor
    ) -> None:
        """Write host_indices into node's SWA host_value and refresh tree state."""
        ct = self.component_type
        cd = node.component_data[ct]
        cd.host_value = host_indices.clone()
        host_lru = self.cache.host_lru_lists[ct]
        if cd.value is None and not host_lru.in_list(node):
            host_lru.insert_mru(node)
        self.cache._update_evictable_leaf_sets(node)
        if node.parent:
            self.cache._update_evictable_leaf_sets(node.parent)

    def _commit_prefetch(
        self,
        anchor,
        transfers: list[PoolTransfer],
        *,
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        """Fill the prefetched SWA window onto the leaf→anchor path.

        All-or-nothing over one full window: ``loaded_pages`` is the cross-rank
        MIN, so ``loaded_pages < window_pages`` drops the whole window (keeps the
        tree identical across TP ranks). Otherwise map the buffer to token range
        ``[loaded_start, total_len)`` and walk leaf→anchor, filling SWA
        tombstones and releasing slices that already have host_value.
        """
        if not transfers:
            return
        ct = self.component_type
        page_size = self.cache.page_size
        host_indices = transfers[0].host_indices
        window_require_pages = (
            host_indices.numel() // page_size if host_indices is not None else 0
        )
        loaded_pages = (
            pool_storage_result.extra_pool_hit_pages.get(PoolName.SWA, 0)
            if pool_storage_result
            else 0
        )
        target = insert_result.inserted_host_node if insert_result else None
        if (
            target is None
            or window_require_pages == 0
            or loaded_pages < window_require_pages
        ):
            self._release_swa_host(host_indices)
            return

        # Buffer covers token range [loaded_start, total_len).
        loaded_start = insert_result.total_len - window_require_pages * page_size

        # Walk leaf → anchor; ``pos`` is the right edge of ``cur`` in tokens.
        pos, cur = insert_result.total_len, target
        while cur is not anchor and pos > loaded_start:
            node_start = pos - len(cur.key)
            # Intersection of cur's range and the buffer.
            fill_start = max(node_start, loaded_start)
            fill_len = pos - fill_start
            buf_off = fill_start - loaded_start
            slice_ = host_indices[buf_off : buf_off + fill_len]

            cd = cur.component_data[ct]
            if cd.host_value is None and fill_len > 0:
                # Tombstone: split off the in-buffer tail if needed, then fill.
                if fill_start > node_start:
                    self.cache._split_node(cur.key, cur, fill_start - node_start)
                self._attach_swa_host_value(cur, slice_)
            else:
                # Already has SWA (or empty overlap): drop this slice.
                self._release_swa_host(slice_)

            pos = node_start
            cur = cur.parent

        # Buffer prefix that fell outside the anchor→leaf path.
        if pos > loaded_start:
            self._release_swa_host(host_indices[: pos - loaded_start])

    def drive_host_eviction(
        self, num_tokens: int, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict SWA host resources.
        Internal nodes: private tombstone (free SWA host only).
        Host leaves: atomic eviction via _evict_host_leaf."""
        ct = self.component_type
        if self._strict_bit_exact:
            # Bit-exact: free SWA host space only by evicting whole host leaves
            # (atomic Full+SWA), never by tombstoning an internal node's SWA
            # alone. This keeps the invariant "Full-host copy => SWA-host copy",
            # so any Full-host hit can restore its true sliding window instead
            # of reprefilling the tail. Sizing then only affects hit rate.
            self.cache.drive_host_leaf_eviction(num_tokens, ct, tracker)
            return
        host_lru = self.cache.host_lru_lists[ct]
        x = host_lru.get_lru_no_host_lock()
        while tracker[ct] < num_tokens and x is not None and host_lru.in_list(x):
            x_next = host_lru.get_prev_no_host_lock(x)
            cd = x.component_data[ct]
            if x in self.cache.evictable_host_leaves:
                self.cache._evict_host_leaf(x, tracker)
            else:
                assert cd.host_value is not None
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.HOST, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker, target=EvictLayer.HOST)
            x = x_next
