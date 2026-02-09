"""
Scheduler configuration module for mini-sglang.

This module defines the SchedulerConfig dataclass which extends EngineConfig
with scheduling-specific parameters for KV cache management, networking, and
inference optimization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    """
    Generate a unique suffix based on process ID for IPC isolation.
    
    Returns:
        str: Process ID suffix string (e.g., ".pid=12345")
    """
    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    """
    Configuration for the request scheduler and KV cache manager.
    
    Extends EngineConfig with parameters for request scheduling, 
    KV cache optimization, and inter-process communication.
    
    Attributes:
        max_extend_tokens: Maximum number of tokens to process in a single 
            forward pass during prefill phase. Default: 8192.
        cache_type: Type of KV cache implementation. Currently only "radix" 
            is supported for radix attention caching.
        page_size: Number of tokens per KV cache page. Default is 16, which
            is optimal for FlashAttention on modern GPUs (Ampere/Hopper).
        offline_mode: If True, disables online features for batch benchmarking.
        max_kv_pages: Hard limit on KV cache pages. None for automatic sizing
            based on available GPU memory, or integer for explicit control 
            (e.g., 32768 for 256 seqs * 2048 len / 16 page_size).
        kv_cache_dtype: Data type for KV cache storage. Options:
            - "fp16": Half precision (2 bytes), compatible with all GPUs
            - "bf16": BFloat16 (2 bytes), better numeric range than fp16
            - "fp8": 8-bit floating point (1 byte), requires Hopper/Ada GPU
        sliding_window: Sliding window attention size. None for full attention,
            or integer (e.g., 4096) to limit attention to recent tokens only,
            reducing memory usage for long sequences.
    """
    
    # Scheduling parameters
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    page_size: int = 16
    offline_mode: bool = False
    
    # KV Cache optimization parameters
    max_kv_pages: Optional[int] = None
    """Hard limit on number of KV cache pages. None for auto, int for explicit limit."""
    
    kv_cache_dtype: str = "fp16"
    """KV cache data type: 'fp16', 'bf16', or 'fp8' (requires Hopper/Ada)."""
    
    sliding_window: Optional[int] = None
    """Sliding window size for constrained attention. None for full attention."""
    
    # Internal networking configuration
    _unique_suffix: str = field(default_factory=_get_pid_suffix)
    """Internal unique identifier for IPC socket isolation."""
    
    @property
    def zmq_backend_addr(self) -> str:
        """
        ZeroMQ IPC address for backend communication.
        
        Uses Unix domain sockets for low-latency inter-process communication
        between the scheduler and model backend.
        
        Returns:
            str: IPC socket path (e.g., "ipc:///tmp/minisgl_0.pid=12345")
        """
        return f"ipc:///tmp/minisgl_0{self._unique_suffix}"
    
    @property
    def zmq_detokenizer_addr(self) -> str:
        """
        ZeroMQ IPC address for detokenization service.
        
        Separate socket for token-to-text conversion to offload from 
        main compute thread.
        
        Returns:
            str: IPC socket path (e.g., "ipc:///tmp/minisgl_1.pid=12345")
        """
        return f"ipc:///tmp/minisgl_1{self._unique_suffix}"
    
    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        """
        ZeroMQ IPC address for scheduler broadcast channel.
        
        Used for broadcasting batch scheduling decisions to all workers
        in tensor-parallel configurations.
        
        Returns:
            str: IPC socket path (e.g., "ipc:///tmp/minisgl_2.pid=12345")
        """
        return f"ipc:///tmp/minisgl_2{self._unique_suffix}"
    
    @property
    def max_forward_len(self) -> int:
        """
        Maximum forward pass length property.
        
        Currently aliases max_extend_tokens for backward compatibility.
        
        Returns:
            int: Maximum tokens per forward iteration.
        """
        return self.max_extend_tokens
    
    @property
    def backend_create_detokenizer_link(self) -> bool:
        """
        Flag indicating whether backend should create detokenizer link.
        
        Always returns True to ensure detokenization service is available.
        
        Returns:
            bool: True (always enabled).
        """
        return True
