from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from minisgl.distributed import get_tp_info
from minisgl.env import ENV
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData, make_positions

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from minisgl.core import Batch
    from minisgl.kvcache import BaseKVCache
    from minisgl.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)


@dataclass
class FICaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class FIMetadata(BaseAttnMetadata):
    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu - contains page-based indptr
    cu_seqlens_q_gpu:   torch.Tensor  # on gpu
    indices:            torch.Tensor  # on gpu - page indices
    last_page_len_cpu:  torch.Tensor  # on cpu - length of last page for each seq
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          int  # tokens per page (e.g., 16)
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # on cpu - token lengths
    q_dtype:            torch.dtype  # dtype for query tensor (computation dtype)
    kv_dtype:           torch.dtype  # dtype for kv cache (storage dtype, may be FP8)
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False
    window_left:        int = -1  # -1 means infinite context window (no sliding window)
    # fmt: on

    @property
    def dtype(self) -> torch.dtype:
        """Backward compatibility: returns kv_dtype."""
        return self.kv_dtype

    def __post_init__(self) -> None:
        assert self.page_size > 0, f"page_size must be positive, got {self.page_size}"
        assert (
            self.positions.is_cuda
            and self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    def __init__(
        self,
        config: ModelConfig,
        kvcache: BaseKVCache,
        page_table: torch.Tensor,
        page_size: int = 16,
        sliding_window: Optional[int] = None,
        computation_dtype: Optional[torch.dtype] = None,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = kvcache
        self.device = kvcache.device
        self.page_size = page_size
        # sliding_window: -1 means infinite context window, positive value enables sliding window
        self.window_left = sliding_window if sliding_window and sliding_window > 0 else -1
        # computation_dtype: dtype for query tensor (model computation dtype)
        # kv_cache_dtype: dtype for kv cache (may be FP8 for storage efficiency)
        self.computation_dtype = computation_dtype if computation_dtype is not None else kvcache.dtype
        
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        # initialize some data members
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size)

        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        self.cached_last_page_lens: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        
        # for cuda graph
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: FICaptureData | None = None
        self.page_table = page_table

    @staticmethod
    def _initialize_metadata_once(metadata: FIMetadata) -> None:
        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        # Use separate dtypes for q and kv:
        # - q_dtype: computation dtype (e.g., bfloat16)
        # - kv_dtype: storage dtype (e.g., fp8_e4m3fn)
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.q_dtype,  # Use q_dtype for output
                q_data_type=metadata.q_dtype,
                kv_data_type=metadata.kv_dtype,
                window_left=metadata.window_left,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.q_dtype,
                kv_data_type=metadata.kv_dtype,
                window_left=metadata.window_left,
                non_blocking=True,
                causal=True,
            )

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]
    
    def _compute_last_page_lens(self, seqlens: List[int]) -> torch.Tensor:
        """
        Compute the length of the last page for each sequence.
        For page_size > 1, the last page may be partially filled.
        Formula: ((seqlen - 1) % page_size) + 1
        """
        if self.page_size == 1:
            return self._get_ones_cpu(len(seqlens))
        
        bs = len(seqlens)
        last_lens = [((s - 1) % self.page_size) + 1 for s in seqlens]
        
        if bs > len(self.cached_last_page_lens):
            next_len = _next_power_of_2(bs)
            self.cached_last_page_lens = torch.empty(next_len, dtype=torch.int32, pin_memory=True)
        
        result = self.cached_last_page_lens[:bs]
        for i, val in enumerate(last_lens):
            result[i] = val
        return result
    
    def _compute_num_pages(self, seqlen: int) -> int:
        """Compute number of pages needed for a sequence."""
        return (seqlen + self.page_size - 1) // self.page_size

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.device
        seq_len_cpu = torch.tensor(seqlens_k, **cpu_kwargs)
        
        # Convert token-based seqlens to page-based seqlens for indptr
        # cu_seqlens_k_cpu contains cumulative page counts
        seqlens_k_pages = [self._compute_num_pages(s) for s in seqlens_k]
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k_pages, **cpu_kwargs).cumsum_(dim=0)
        
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **cpu_kwargs)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            # For prefill without cache, qo_indptr uses token counts
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(dim=0)
        
        # Compute page indices: for each request, we need the page indices it uses
        # page_table[table_idx, :num_pages] gives the physical page indices
        indices_list = []
        for req in reqs:
            num_pages = self._compute_num_pages(req.device_len)
            indices_list.append(self.page_table[req.table_idx, :num_pages])
        
        batch.attn_metadata = FIMetadata(
            positions=make_positions(device, reqs),
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            indices=torch.cat(indices_list),
            last_page_len_cpu=self._compute_last_page_lens(seqlens_k),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=self.page_size,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            q_dtype=self.computation_dtype,  # Query uses computation dtype
            kv_dtype=self.kvcache.dtype,     # KV uses cache dtype (may be FP8)
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,
            window_left=self.window_left,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        # Calculate max pages based on page_size
        max_pages = self._compute_num_pages(max_seq_len)
        capture = FICaptureData.create(max_bs, max_pages, self.kvcache.device)
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    @cached_property
    def use_tensor_cores(self) -> bool:
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        batch.padded_reqs = batch.reqs
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        self.graph_wrappers[bs]._backend = "fa2"
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = self.graph_wrappers[bs]
        metadata.positions = capture.positions[:bs]
        batch.input_ids = capture.input_ids[:bs]
        batch.out_loc = capture.out_loc[:bs]
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        
        # Copy basic data
        self.capture.input_ids[:bs].copy_(batch.input_ids)
        self.capture.out_loc[:bs].copy_(batch.out_loc)
        self.capture.positions[:bs].copy_(metadata.positions)
        self.capture.seq_lens[:bs].copy_(metadata.seq_lens_cpu)
        
        # Update page-based indptr
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k_cpu)
        
        # Update last page lengths (computed from seq_lens_cpu)
        last_page_lens = [((metadata.seq_lens_cpu[i].item() - 1) % self.page_size) + 1 
                         for i in range(bs)]
        for i, val in enumerate(last_page_lens):
            self.capture.one_tensor[i] = val
        
        # Update page indices for attention - this is critical for correct KV cache access
        # metadata.indices contains the physical page indices for all sequences in the batch
        num_indices = metadata.indices.shape[0]
        self.capture.page_table[:num_indices].copy_(metadata.indices)
        
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)
