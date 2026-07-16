from __future__ import annotations

import logging
import os

_STATECAP_DBG_LOGGER = logging.getLogger(__name__)
_STATECAP_DBG_SEEN = set()


def _statecap_dbg(msg):
    if msg not in _STATECAP_DBG_SEEN:
        _STATECAP_DBG_SEEN.add(msg)
        _STATECAP_DBG_LOGGER.warning("[STATECAP-DBG] %s", msg)
from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation
from sglang.srt.layers.attention.dsv4.compressor import Compressor as _CompressorBase
from sglang.srt.layers.attention.dsv4.fused_compress_triton import (
    fused_ape_pool_norm_rope,
)
from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation
from sglang.srt.layers.deepseek_v4_rope import (
    apply_rotary_emb_triton,
    fused_norm_rope_inplace_triton,
    fused_softmax_pool_triton,
)

try:
    from sglang.srt.layers.deepseek_v4_rope import fused_softmax_pool_triton
except ImportError:
    fused_softmax_pool_triton = None
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    KVAndScore,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DeepseekV4HipRadixBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@triton.jit
def _rms_normalize_kernel(
    x_ptr,
    weight_ptr,
    eps,
    stride_row,
    dim,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim
    base = pid * stride_row
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / dim
    rms_inv = tl.rsqrt(mean_sq + eps)
    out = x * rms_inv
    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        out = out * weight
    tl.store(x_ptr + base + offs, out, mask=mask)


def rms_normalize_triton(
    x: torch.Tensor, eps: float, weight: torch.Tensor = None
) -> torch.Tensor:
    dim = x.shape[-1]
    x_flat = x.view(-1, dim)
    num_rows = x_flat.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(dim)
    grid = (num_rows,)
    _rms_normalize_kernel[grid](
        x_flat,
        weight,
        eps,
        x_flat.stride(0),
        dim,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=(weight is not None),
    )
    return x


class DeepseekRefRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return rms_normalize_triton(x, self.eps, self.weight)


class CompressorHip(_CompressorBase):
    """HIP (ROCm) specific Compressor implementation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm = DeepseekRefRMSNorm(self.head_dim, eps=self.norm.variance_epsilon)
        self._freqs_cis_real: torch.Tensor | None = None

    @cached_property
    def use_fused_compress(self) -> bool:
        return envs.SGLANG_OPT_USE_FUSED_COMPRESS.get()

    @cached_property
    def use_hip_fused_compress(self) -> bool:
        return envs.SGLANG_OPT_USE_FUSED_COMPRESS.get()

    @cached_property
    def use_fused_compress_triton(self) -> bool:
        # The fused Triton kernel only benefits non-overlap (HCA, ratio=128)
        # but HCA's K=128 loop is too sequential to outperform batched ops.
        # CSA (overlap=True) has a reshape/overlap-transform semantic mismatch.
        # Disabled until a tiled kernel for CSA overlap is implemented.
        return False

    def _get_states(
        self,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> KVAndScore:
        token_to_kv_pool = attn_backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            return token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            return token_to_kv_pool.get_attention_compress_states(self.layer_id)

    def _get_state_pool(self, attn_backend: AttentionBackend) -> CompressStatePool:
        token_to_kv_pool = attn_backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            ret = token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            ret = token_to_kv_pool.get_attention_compress_states(self.layer_id)
        assert isinstance(ret, CompressStatePool)
        return ret

    def overlap_transform(self, tensor: torch.Tensor, fill_value: Any) -> torch.Tensor:
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (self.ratio, 2 * self.head_dim)

        s, r, d = tensor.size(0), self.ratio, self.head_dim
        new_tensor = tensor.new_full((s, 2 * r, d), fill_value)
        new_tensor[:, r:] = tensor[:, :, d:]
        new_tensor[1:, :r] = tensor[:-1, :, :d]
        return new_tensor

    def overlap_transform_decode(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (2 * self.ratio, 2 * self.head_dim)
        r, d = self.ratio, self.head_dim
        ret = torch.cat((tensor[:, :r, :d], tensor[:, r:, d:]), dim=1)
        return ret

    @staticmethod
    def compute_state_len(seq_len: int, ratio: int):
        return seq_len % ratio + (ratio == 4) * ratio

    @staticmethod
    def compute_state_len_indices(seq_len: int, ratio: int):
        state_len = seq_len % ratio + (ratio == 4) * ratio
        return torch.arange(seq_len - state_len, seq_len).clamp(min=-1)

    def print_tensor(self, y: torch.Tensor, name: str):
        enable = int(os.environ.get("SGLANG_ENABLE_PRINT_TENSOR", 0))
        if enable:
            print(f"[sgl] {name}: shape={y.shape}, dtype={y.dtype}, device={y.device}")
            print(f"{y.flatten()[:10]}...{y.flatten()[-10:]}")

    def _capture_compress_state_windows(
        self,
        kv_and_score_buffer,
        valid_kv_len: int,
        prefix_len: int,
        extend_len: int,
        rid: int,
        backend,
    ) -> None:
        """Capture the c4 / c4-indexer overlap state ``[B-ratio, B)`` at each
        page boundary ``B`` in this chunk into the strict-mode host state pool.

        The device state ring is a small ``ring_size`` rolling buffer, so interior
        boundary states are overwritten during chunked prefill; snapshot them
        from ``kv_and_score_buffer`` (same object written to the ring) instead.
        Each window token at sequence position ``s`` maps to ring offset
        ``(s % swa_ring_size) % ring_size`` -- deterministic from ``s`` alone
        because SWA pages align to ``ring_size`` -- so the reusing request lands
        the restored state on the same slots. No-op unless the bit-exact host
        state pool has been wired onto the device pool.
        """
        idx = self.is_in_indexer
        if self.ratio != 4:
            _statecap_dbg(f"indexer={idx} ret: ratio={self.ratio}!=4")
            return
        token_to_kv_pool = backend.token_to_kv_pool
        attr = (
            "_c4_indexer_state_host_pool"
            if self.is_in_indexer
            else "_c4_state_host_pool"
        )
        hp = getattr(token_to_kv_pool, attr, None)
        if hp is None:
            _statecap_dbg(f"indexer={idx} ret: hp None ({attr})")
            return
        layer_index = getattr(token_to_kv_pool, "_c4_state_layer_index", None)
        if layer_index is None:
            _statecap_dbg(f"indexer={idx} ret: layer_index None")
            return
        li = layer_index.get(self.layer_id)
        if li is None:
            _statecap_dbg(
                f"indexer={idx} ret: li None layer_id={self.layer_id} "
                f"keys={sorted(layer_index.keys())[:8]}"
            )
            return
        if extend_len <= 0:
            _statecap_dbg(f"indexer={idx} ret: extend_len<=0")
            return
        _statecap_dbg(f"indexer={idx} PASS gating layer_id={self.layer_id} li={li}")

        page = backend.page_size
        swa_ring = token_to_kv_pool.unified_swa_ring_size
        slot_page = hp.slot_page_size  # == ring_size
        if swa_ring % slot_page != 0 or page % swa_ring != 0:
            raise AssertionError(
                f"state capture geometry: page={page} swa_ring={swa_ring} "
                f"ring_size={slot_page}"
            )
        win = self.ratio  # compute_state_len(B, 4) == 4 at a page boundary
        slot_bytes = hp.item_bytes // slot_page
        staging = hp._capture_staging
        state_buf = kv_and_score_buffer.kv_score
        host_layer_buf = hp.data_refs[li]
        pre_len = valid_kv_len - extend_len

        cs = prefix_len
        total = prefix_len + extend_len
        boundary = (total // page) * page
        B = ((cs // page) + 1) * page
        while B <= boundary:
            off0 = ((B - win) % swa_ring) % slot_page
            if B - win < cs or off0 + win > slot_page:
                raise AssertionError(
                    f"state window out of range: B={B} win={win} cs={cs} "
                    f"off0={off0} ring_size={slot_page}"
                )
            key = (rid, int(B))
            hidx = staging.get(key)
            if hidx is None:
                hidx = hp.alloc(slot_page)
                if hidx is None:
                    _statecap_dbg(f"indexer={idx} ALLOC FULL key={key}")
                    B += page
                    continue
                staging[key] = hidx
                _statecap_dbg(f"indexer={idx} STAGED key={key}")
            buf_lo = pre_len + (B - win - prefix_len)
            buf_hi = pre_len + (B - prefix_len)
            win_slice = state_buf[buf_lo:buf_hi]
            flat = win_slice.contiguous().view(torch.uint8).reshape(-1)
            if flat.numel() != win * slot_bytes:
                raise AssertionError(
                    f"state window bytes {flat.numel()} != {win * slot_bytes}"
                )
            page_row = int(hidx[0].item()) // slot_page
            dst = host_layer_buf[page_row]
            dst[off0 * slot_bytes : off0 * slot_bytes + flat.numel()].copy_(
                flat, non_blocking=True
            )
            B += page

    def compress_extend_paged(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ):
        backend = attn_backend
        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4HipRadixBackend)
        token_to_kv_pool = backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        state_pool = self._get_state_pool(backend)
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        extend_lens = forward_batch.extend_seq_lens_cpu
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = backend.req_to_token_pool.req_to_token
        assert not self.forward_mode.is_target_verify()

        assert extend_lens is not None and prefix_lens is not None
        device = kv_and_scores.kv.device

        assert kv_and_scores.kv.shape[-1] == self.head_dim * self.coff
        compressed_kv_output = torch.full(
            (kv_and_scores.kv.size(0), self.head_dim),
            fill_value=10000.0,
            dtype=kv_and_scores.kv.dtype,
            device=device,
        )

        bs = forward_batch.batch_size
        pt = 0
        for i in range(bs):
            kv_and_score = kv_and_scores[pt : pt + extend_lens[i]]
            pre_state_indices = self.compute_state_len_indices(
                seq_len=prefix_lens[i], ratio=self.ratio
            ).to(device)
            if self.ratio == 128:
                state_loc = state_pool.translate_from_req_position_to_state_loc(
                    req_pool_indices[i], pre_state_indices
                )
            else:
                raw_loc = torch.where(
                    pre_state_indices < 0,
                    -1,
                    req_to_token[req_pool_indices[i], pre_state_indices],
                )
                swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(raw_loc)
                state_loc = state_pool.translate_from_swa_loc_to_state_loc(swa_loc)
            pre_kv_state = state_pool.get_state_by_state_loc(state_loc)
            kv_and_score_buffer = KVAndScore.cat([pre_kv_state, kv_and_score], dim=0)
            valid_kv_len = kv_and_score_buffer.kv.size(0)

            # Strict SWA HiCache: snapshot the c4 / c4-indexer overlap state at
            # each page boundary into the host state pool (no-op unless the pool
            # is wired, i.e. the bit-exact flag is on). Captured before the
            # in-place ape.add_ / overlap transform below; the [B-ratio, B)
            # window lives in the uncompressed tail, byte-identical to what
            # set_state_by_state_loc would persist.
            self._capture_compress_state_windows(
                kv_and_score_buffer=kv_and_score_buffer,
                valid_kv_len=valid_kv_len,
                prefix_len=int(prefix_lens[i]),
                extend_len=int(extend_lens[i]),
                rid=int(req_pool_indices[i]),
                backend=backend,
            )

            post_state_indices = self.compute_state_len_indices(
                seq_len=prefix_lens[i] + extend_lens[i], ratio=self.ratio
            ).to(device)
            post_state_len = post_state_indices.size(0)

            assert post_state_len <= valid_kv_len
            if self.ratio == 128:
                post_state_loc = state_pool.translate_from_req_position_to_state_loc(
                    req_pool_indices[i], post_state_indices
                )
            else:
                post_raw_loc = torch.where(
                    post_state_indices < 0,
                    -1,
                    req_to_token[req_pool_indices[i], post_state_indices],
                )
                post_swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(
                    post_raw_loc
                )
                post_state_loc = state_pool.translate_from_swa_loc_to_state_loc(
                    post_swa_loc
                )
            post_state_to_set = kv_and_score_buffer[valid_kv_len - post_state_len :]
            state_pool.set_state_by_state_loc(post_state_loc, post_state_to_set)

            compress_len = valid_kv_len // self.ratio * self.ratio
            if compress_len == 0:
                pt += extend_lens[i]
                continue

            kv_and_score_to_compress = kv_and_score_buffer[:compress_len].view(
                compress_len // self.ratio, self.ratio, -1
            )
            kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

            if self.overlap:
                new_kv = self.overlap_transform(
                    kv_and_score_to_compress.kv, fill_value=0
                )
                new_score = self.overlap_transform(
                    kv_and_score_to_compress.score, fill_value=float("-inf")
                )
                kv_and_score_to_compress = KVAndScore.from_kv_score(
                    kv=new_kv, score=new_score
                )
                del new_kv, new_score
                kv_and_score_to_compress = kv_and_score_to_compress[1:]

                if kv_and_score_to_compress.kv.size(0) == 0:
                    pt += extend_lens[i]
                    continue

            beg_idx = prefix_lens[i] // self.ratio * self.ratio
            end_idx = (prefix_lens[i] + extend_lens[i]) // self.ratio * self.ratio

            if self.use_hip_fused_compress:
                kv_compressed = fused_softmax_pool_triton(
                    kv_and_score_to_compress.kv_score,
                    kv_and_score_to_compress._item_size,
                )
            else:
                kv_compressed = (
                    kv_and_score_to_compress.kv
                    * kv_and_score_to_compress.score.softmax(dim=1)
                ).sum(dim=1)

            assert kv_compressed.dtype == torch.float32

            freqs_cis = self.freqs_cis[beg_idx : end_idx : self.ratio]
            assert freqs_cis.size(0) == kv_compressed.size(
                0
            ), f"{freqs_cis.shape=} {kv_compressed.shape=}"
            if self.use_hip_fused_compress:
                fused_norm_rope_inplace_triton(
                    kv_compressed, self.norm.weight, self.norm.eps, freqs_cis
                )
            else:
                kv_compressed = self.norm(kv_compressed)
                apply_rotary_emb_triton(
                    kv_compressed[..., -self.rope_head_dim :], freqs_cis
                )
            del beg_idx, end_idx

            if self.rotate:
                kv_compressed = rotate_activation(kv_compressed)

            start = prefix_lens[i]
            start = start + self.ratio - 1 - start % self.ratio
            indices_in_seq = torch.arange(
                start,
                prefix_lens[i] + extend_lens[i],
                self.ratio,
                device=kv_and_scores.kv.device,
            )
            assert indices_in_seq.size(0) == kv_compressed.size(0)
            compressed_kv_output[indices_in_seq - prefix_lens[i] + pt] = kv_compressed

            pt += extend_lens[i]

        return compressed_kv_output

    def compress_decode_paged(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ):
        """Paged and cudagraph compatible version of compress_decode"""
        assert self.ape_converted
        state_pool = self._get_state_pool(attn_backend)
        token_to_kv_pool = attn_backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = attn_backend.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens

        if forward_batch.forward_mode.is_target_verify():
            draft_tokens = attn_backend.speculative_num_draft_tokens
            offsets = torch.arange(1, draft_tokens + 1, device=seq_lens.device)
            seq_lens_2d = seq_lens[:, None] + offsets[None, :]
            seq_lens = seq_lens_2d.view(-1)
            req_pool_indices = req_pool_indices.repeat_interleave(draft_tokens)

        if self.ratio == 128:
            state_locs = state_pool.translate_from_req_position_to_state_loc(
                req_pool_indices, seq_lens - 1
            )
        else:
            raw_locs = req_to_token[req_pool_indices, seq_lens - 1]
            swa_locs = token_to_kv_pool.translate_loc_from_full_to_swa(raw_locs)
            state_locs = state_pool.translate_from_swa_loc_to_state_loc(swa_locs)
        state_pool.set_state_by_state_loc(state_locs, kv_and_scores)

        compress_bulk_len = self.ratio * self.coff
        compress_indices = seq_lens[:, None] + torch.arange(
            -compress_bulk_len, 0, device=seq_lens.device
        )
        compress_indices.clamp_(min=-1)
        if self.ratio == 128:
            compress_indices_state = (
                state_pool.translate_from_req_position_to_state_loc(
                    req_pool_indices[:, None], compress_indices
                )
            )
        else:
            compress_indices_raw = torch.where(
                compress_indices < 0,
                -1,
                req_to_token[req_pool_indices[:, None], compress_indices],
            )
            compress_indices_swa = token_to_kv_pool.translate_loc_from_full_to_swa(
                compress_indices_raw
            )
            compress_indices_state = state_pool.translate_from_swa_loc_to_state_loc(
                compress_indices_swa
            )
        kv_and_score_to_compress = state_pool.get_state_by_state_loc(
            compress_indices_state.view(-1)
        ).view(-1, self.ratio, self.coff * self.head_dim)
        bs = seq_lens.size(0)

        if self.use_fused_compress_triton and not self.overlap:
            # Fused path for non-overlap (HCA, ratio=128, coff=1):
            # APE + softmax-pool + norm + RoPE in one kernel.
            # Overlap (CSA) is excluded because the overlap_transform_decode
            # rearranges A/B halves across the coff dimension in a way
            # that simple reshape cannot replicate correctly.
            raw = kv_and_score_to_compress.kv_score
            gathered = raw.reshape(bs, self.ratio, raw.shape[-1]).contiguous()

            comp_positions = (seq_lens - 1) // self.ratio * self.ratio
            freqs_real_table = self._get_freqs_cis_real()
            freqs_batch = freqs_real_table[comp_positions]

            kv_compressed = fused_ape_pool_norm_rope(
                kv_score_gathered=gathered,
                ape=self.ape,
                rms_weight=self.norm.weight,
                rms_eps=self.norm.eps,
                freqs_cis_real=freqs_batch,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                ratio=self.ratio,
                overlap=self.overlap,
            )
            if self.rotate:
                kv_compressed = rotate_activation(kv_compressed)
            return kv_compressed

        # Unfused reference path
        kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

        if self.overlap:
            kv_and_score_to_compress = kv_and_score_to_compress.view(
                bs, self.coff * self.ratio, self.coff * self.head_dim
            )
            kv_and_score_to_compress = KVAndScore.from_kv_score(
                kv=self.overlap_transform_decode(kv_and_score_to_compress.kv),
                score=self.overlap_transform_decode(kv_and_score_to_compress.score),
            )

        kv_and_score_to_compress = kv_and_score_to_compress.view(
            bs, self.ratio * self.coff, self.head_dim
        )

        if self.use_hip_fused_compress:
            kv_compressed = fused_softmax_pool_triton(
                kv_and_score_to_compress.kv_score,
                kv_and_score_to_compress._item_size,
            )
        else:
            kv_compressed = (
                kv_and_score_to_compress.kv
                * kv_and_score_to_compress.score.softmax(dim=1)
            ).sum(dim=1)
        if self.use_hip_fused_compress:
            freqs_cis = self._init_freqs_cis_per_decode_step(forward_batch, seq_lens)
            fused_norm_rope_inplace_triton(
                kv_compressed, self.norm.weight, self.norm.eps, freqs_cis
            )
        else:
            kv_compressed = self.norm(kv_compressed)
            freqs_cis = self.freqs_cis[(seq_lens - 1) // self.ratio * self.ratio]
            apply_rotary_emb_triton(
                kv_compressed[..., -self.rope_head_dim :], freqs_cis
            )
        if self.rotate:
            kv_compressed = rotate_activation(kv_compressed)

        return kv_compressed

    def compress_fused(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        backend = attn_backend
        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4HipRadixBackend)
        kv_score_buffer = self._get_state_pool(backend)
        kv_score_buffer = kv_score_buffer.kv_score_buffer.kv_score

        return backend.forward_compress(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score,
            ape=self.ape.view(-1, self.head_dim),
            head_dim=self.head_dim,
            norm=self.norm,
            freqs_cis_cache=self.freqs_cis,
            rotate=self.rotate,
            compress_ratio=self.ratio,
            forward_batch=forward_batch,
            is_paged=True,
        )

    def _get_freqs_cis_real(self) -> torch.Tensor:
        """Cache the float32 view of freqs_cis (complex64 -> real interleaved)."""
        if self._freqs_cis_real is None:
            if self.freqs_cis.is_complex():
                self._freqs_cis_real = (
                    torch.view_as_real(self.freqs_cis).flatten(-2).contiguous()
                )
            else:
                self._freqs_cis_real = self.freqs_cis.contiguous()
        return self._freqs_cis_real

    def compress_dispatch(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        # Strict bit-exact SWA HiCache captures the c4 / c4-indexer overlap
        # state at each page boundary, but that capture hook only lives in the
        # paged extend path (`compress_extend_paged` -> `_capture_compress_
        # state_windows`). The fused path bypasses it, so when the flag is on we
        # must route the ratio==4 (overlap) compressors through the paged path
        # or the state is never staged and cross-request reuse silently falls
        # back to full recompute. ratio==128 (c128) needs no capture (I8), so it
        # keeps the fused fast path. Flag OFF -> behavior unchanged.
        strict_capture_needs_paged = (
            self.ratio == 4
            and envs.SGLANG_UNIFIED_KV_SWA_BIT_EXACT_HICACHE.get()
        )
        _statecap_dbg(
            f"GATE ratio={self.ratio} fused={self.use_fused_compress} "
            f"fused_triton={self.use_fused_compress_triton} overlap={self.overlap} "
            f"envflag_os={os.getenv(chr(83)+chr(71)+chr(76)+chr(65)+chr(78)+chr(71)+chr(95)+chr(85)+chr(78)+chr(73)+chr(70)+chr(73)+chr(69)+chr(68)+chr(95)+chr(75)+chr(86)+chr(95)+chr(83)+chr(87)+chr(65)+chr(95)+chr(66)+chr(73)+chr(84)+chr(95)+chr(69)+chr(88)+chr(65)+chr(67)+chr(84)+chr(95)+chr(72)+chr(73)+chr(67)+chr(65)+chr(67)+chr(72)+chr(69))} "
            f"envflag_get={envs.SGLANG_UNIFIED_KV_SWA_BIT_EXACT_HICACHE.get()} "
            f"needs_paged={strict_capture_needs_paged} mode={forward_batch.forward_mode}"
        )
        if (
            self.use_fused_compress
            and not strict_capture_needs_paged
            and (
                envs.SGLANG_OPT_DPSK_V4_RADIX.get()
                and (
                    forward_batch.forward_mode.is_decode()
                    or forward_batch.forward_mode.is_extend_without_speculative()
                )
            )
        ):
            return self.compress_fused(
                kv_score, forward_batch, attn_backend=attn_backend
            )

        self.compress_decode = self.compress_decode_paged
        self.compress_extend = self.compress_extend_paged
        kv_and_scores = KVAndScore(kv_score)

        if TYPE_CHECKING:
            assert isinstance(kv_and_scores, KVAndScore)

        if (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        ):
            result = self.compress_decode(
                kv_and_scores=kv_and_scores,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
            )
        elif forward_batch.forward_mode.is_extend():
            result = self.compress_extend(
                kv_and_scores=kv_and_scores,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
            )
        else:
            msg = f"Forward mode {forward_batch.forward_mode} not supported in Compressor."
            raise NotImplementedError(msg)

        return result

    def _init_freqs_cis_per_decode_step(
        self,
        forward_batch: ForwardBatch,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        attr = f"freqs_cis_c{self.ratio}"
        cached = getattr(forward_batch, attr, None)
        if cached is not None:
            return cached
        decoded = self.freqs_cis[(seq_lens - 1) // self.ratio * self.ratio]
        setattr(forward_batch, attr, decoded)
        return decoded

    def forward(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)
        kv_score = self.compute_kv_score(x, forward_batch)
        self.forward_mode = forward_batch.forward_mode
        return self.compress_dispatch(
            kv_score, forward_batch, attn_backend=attn_backend
        )
