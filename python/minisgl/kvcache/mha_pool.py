from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCache, KVCacheLayout


class MHAKVCache(BaseKVCache):
    """
    Multi-Head Attention KV Cache with paged memory layout.
    
    This implementation uses a paged layout where each page contains
    `page_size` tokens, enabling efficient memory management and
    compatibility with FlashInfer's paged attention kernels.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        kv_layout: KVCacheLayout,
        device: torch.device,
    ):
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size)
        
        self._page_size = page_size
        self._num_pages = num_pages
        
        match kv_layout:
            case KVCacheLayout.PageFirst:
                # Shape: (2, num_pages, page_size, num_layers, local_kv_heads, head_dim)
                kv_buffer = torch.empty(
                    (2, num_pages, page_size, num_layers, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                ).permute(0, 3, 1, 2, 4, 5)  # -> (2, num_layers, num_pages, page_size, local_kv_heads, head_dim)
            case KVCacheLayout.LayerFirst:
                # Shape: (2, num_layers, num_pages, page_size, local_kv_heads, head_dim)
                kv_buffer = torch.empty(
                    (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                )
            case _:
                raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        
        self._kv_buffer = kv_buffer
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]  # (num_layers, num_pages, page_size, local_kv_heads, head_dim)
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        # Storage shape for store_cache: (num_pages, page_size, local_kv_heads, head_dim)
        self._storage_shape = (num_pages, page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        """
        Get K cache for a specific layer.
        Returns shape: (num_pages, page_size, local_kv_heads, head_dim)
        """
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        """
        Get V cache for a specific layer.
        Returns shape: (num_pages, page_size, local_kv_heads, head_dim)
        """
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """
        Store K and V tensors to the cache.
        
        Args:
            k: Key tensor of shape (num_tokens, local_kv_heads, head_dim)
            v: Value tensor of shape (num_tokens, local_kv_heads, head_dim)
            out_loc: Output locations as flat indices into the paged cache
                    Shape: (num_tokens,). Each index = page_idx * page_size + offset_in_page
            layer_id: Layer index
        """
        from minisgl.kernel import store_cache

        # Convert k, v to match cache dtype if needed (e.g., bfloat16 -> fp8)
        cache_dtype = self._kv_buffer.dtype
        if k.dtype != cache_dtype:
            k = k.to(cache_dtype)
        if v.dtype != cache_dtype:
            v = v.to(cache_dtype)

        # The store_cache kernel expects flat indices into the flattened (num_pages * page_size, ...) view
        store_cache(
            k_cache=self._k_buffer[layer_id].view(-1, self._storage_shape[2], self._storage_shape[3]),
            v_cache=self._v_buffer[layer_id].view(-1, self._storage_shape[2], self._storage_shape[3]),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
    
    @property
    def page_size(self) -> int:
        return self._page_size
    
    @property
    def num_pages(self) -> int:
        return self._num_pages
