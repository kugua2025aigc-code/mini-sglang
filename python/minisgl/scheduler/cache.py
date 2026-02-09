from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.kvcache import BaseCacheHandle, create_cache_manager

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    def __init__(self, device: torch.device, num_pages: int, type: str, page_size: int = 1):
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        self.manager = create_cache_manager(device=device, type=type)
        self.num_pages = num_pages
        self.page_size = page_size

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices])

    def match_req(self, req: PendingReq):
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.manager.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        """Return available token capacity (pages * page_size)."""
        return (self.manager.size_info.evictable_size + len(self._free_slots)) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=True)

    def allocate(self, needed_len: int) -> torch.Tensor:
        """
        Allocate slots for tokens in the KV cache.
        
        Args:
            needed_len: Number of tokens to allocate space for.
            
        Returns:
            Tensor of flat indices into the paged KV cache.
            Each index = page_idx * page_size + offset_in_page
        """
        # Convert token count to number of pages needed
        num_pages_needed = (needed_len + self.page_size - 1) // self.page_size
        
        if num_pages_needed <= (free_len := len(self._free_slots)):
            allocated_pages = self._free_slots[:num_pages_needed]
            self._free_slots = self._free_slots[num_pages_needed:]
        else:
            # NOTE: len(evicted) + free_len >= num_pages_needed
            evicted = self.manager.evict(num_pages_needed - free_len)
            merged = torch.cat([self._free_slots, evicted])
            assert len(merged) >= num_pages_needed, "Eviction did not free enough space."
            
            allocated_pages = merged[:num_pages_needed]
            self._free_slots = merged[num_pages_needed:]
        
        # Convert page indices to flat indices
        # For each page, generate indices: page_idx * page_size + [0, 1, ..., page_size-1]
        if self.page_size == 1:
            return allocated_pages[:needed_len]
        
        # Generate flat indices: [page_idx * page_size + offset for each page, for each offset]
        num_allocated = len(allocated_pages)
        # Create offset tensor [0, 1, ..., page_size-1] repeated for each page
        offsets = torch.arange(self.page_size, dtype=torch.int32, device=self.device)
        # Expand pages: [[p0, p0, ...], [p1, p1, ...], ...] with page_size repeats
        expanded_pages = allocated_pages.unsqueeze(1).expand(num_allocated, self.page_size)
        # Calculate flat indices: page * page_size + offset
        flat_indices = (expanded_pages * self.page_size + offsets).view(-1)
        
        # Return only the needed number of indices
        return flat_indices[:needed_len]

    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        self._free(indices[old_handle.cached_len : in_cache_len])
        self.unlock(old_handle)

    def check_integrity(self) -> None:
        self.manager.check_integrity()
        if len(self._free_slots) + self.manager.size_info.total_size != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_slots({len(self._free_slots)}) +"
                f" total_size({self.manager.size_info.total_size}) != num_pages({self.num_pages})"
            )
