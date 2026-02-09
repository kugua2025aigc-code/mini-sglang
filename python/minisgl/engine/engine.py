from __future__ import annotations

from datetime import timedelta
from typing import Dict, NamedTuple, Tuple

import torch
import torch.cuda as cuda
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_hf_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


def create_page_table(shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.int32, device=device)


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def get_gpu_architecture() -> str:
    """
    Detect GPU compute capability to determine supported features.
    
    Returns:
        str: One of ["blackwell", "hopper", "ada", "ampere", "turing", "volta", "legacy"]
    """
    if not cuda.is_available():
        return "cpu"
    
    major, minor = cuda.get_device_capability()
    capability = major * 10 + minor
    
    if capability >= 100:
        return "blackwell"  # RTX 50-series, B200 (sm_100+)
    elif capability >= 90:
        return "hopper"     # H100, H800 (sm_90)
    elif capability >= 89:
        return "ada"        # RTX 4090, 4080, L40S (sm_89) 
    elif capability >= 80:
        return "ampere"     # A100, A800, RTX 3090 (sm_80, sm_86)
    elif capability >= 75:
        return "turing"     # RTX 20-series, T4 (sm_75)
    elif capability >= 70:
        return "volta"      # V100 (sm_70)
    else:
        return "legacy"
    


def get_optimal_page_size(gpu_arch: str) -> int:
    """
    Determine optimal KV cache page size based on GPU architecture.
    
    Args:
        gpu_arch: GPU architecture identifier from get_gpu_architecture()
        
    Returns:
        int: Optimal page size (16 for modern GPUs, larger for older ones)
    """
    # Modern GPUs (Ampere/Hopper): 16 tokens/page optimal for FlashAttention
    # Older GPUs: Larger pages reduce page table overhead
    optimal_sizes = {
        "hopper": 16,
        "ampere": 16,
        "turing": 32,    # Better memory coalescing on Turing
        "volta": 32,
        "legacy": 64     # Reduce management overhead on old GPUs
    }
    return optimal_sizes.get(gpu_arch, 16)


def supports_fp8(gpu_arch: str) -> bool:
    """
    Check if GPU has native FP8 (E4M3) support in hardware.
    
    Note: Ampere (A100) can emulate FP8 but it's slower than FP16.
    Only Hopper (H100) and Ada (RTX 40-series) have native FP8 Tensor Cores.
    
    Args:
        gpu_arch: GPU architecture identifier
        
    Returns:
        bool: True if native FP8 is supported and faster than FP16
    """
    return gpu_arch in ["hopper", "ada"]


def check_fp8_usability() -> bool:
    """
    Runtime check for FP8 availability for KV cache usage.
    
    FP8 in KV cache requires:
    1. Tensor creation (storage)
    2. View/reshape operations (paged attention)
    3. FP8 <-> FP16 conversion (attention computation)
    
    Note: Direct FP8 arithmetic (+, -, *, /) is NOT supported by PyTorch.
    This is expected - FP8 is for storage and scaled matmul only.
    """
    try:
        # 1. Check dtype exists
        if not hasattr(torch, 'float8_e4m3fn'):
            return False
            
        # 2. Test tensor creation (core requirement)
        x = torch.zeros((16, 64), dtype=torch.float8_e4m3fn, device="cuda")
        
        # 3. Test reshape (paged attention needs this)
        x_reshaped = x.view(4, 256)
        
        # 4. Test FP8 <-> FP16 conversion (attention casts to FP16 for compute)
        x_fp16 = x.to(torch.float16)
        x_back = x_fp16.to(torch.float8_e4m3fn)
        
        # 5. Test slicing (sequence management)
        _ = x[0:8, :]
        
        # 6. Test copy_ (KV cache updates)
        y = torch.zeros_like(x)
        y.copy_(x)
        
        del x, y, x_fp16, x_back
        torch.cuda.synchronize()
        
        return True
        
    except Exception as e:
        logger.debug(f"FP8 usability check failed: {e}")
        return False


class Engine:
    def __init__(self, config: EngineConfig):
        self.model_config = config.model_config
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        assert not torch.cuda.is_initialized()
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        
        # Detect GPU capabilities for cross-platform compatibility
        self.gpu_arch = get_gpu_architecture()
        logger.info_rank0(f"Detected GPU architecture: {self.gpu_arch}")
        
        # Determine KV cache dtype
        kv_dtype = getattr(config, 'kv_cache_dtype', 'auto')
        if kv_dtype == "fp8":
            self.kv_cache_dtype = torch.float8_e4m3fn if supports_fp8(self.gpu_arch) else config.dtype
        else:
            self.kv_cache_dtype = config.dtype

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # Load model and determine number of KV cache pages
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # Determine KV cache capacity with configurable hard limit
        self.num_pages = self.dummy_page = self._determine_num_pages(init_free_memory, config)

        # Calculate maximum sequence length based on page capacity
        page_size = getattr(config, 'page_size', None) or get_optimal_page_size(self.gpu_arch)
        
        self.kv_cache = create_kvcache(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=page_size,
            device=self.device,
            dtype=self.kv_cache_dtype,
        )
        
        # Check if sliding window attention is enabled
        sliding_window = getattr(config, 'sliding_window', None)
        
        if sliding_window and sliding_window > 0:
            # Sliding window mode: max_seq_len should support full sequence length,
            # not just the window size. The num_pages is limited to window size 
            # for memory efficiency, but we must preserve full length capability.
            max_kv_pages = getattr(config, 'max_kv_pages', None)
            if max_kv_pages and max_kv_pages > 0:
                # Use explicit max_kv_pages to calculate total capacity
                full_capacity = max_kv_pages * page_size
            else:
                # No explicit limit, use configured max_seq_len
                full_capacity = config.max_seq_len
            
            self.max_seq_len = _align_up_32(min(config.max_seq_len, full_capacity))
            logger.info_rank0(
                f"Sliding window mode: max_seq_len={self.max_seq_len}, "
                f"window_size={sliding_window}, cache_pages={self.num_pages}"
            )
        else:
            # Standard mode: max_seq_len limited by allocated KV cache pages
            effective_max_len = self.num_pages * page_size
            self.max_seq_len = _align_up_32(min(config.max_seq_len, effective_max_len))
        
        logger.info_rank0(
            f"Effective max_seq_len: {self.max_seq_len} "
            f"(pages={self.num_pages}, page_size={page_size})"
        )
        
        # Page table: (max_running_requests + 1) x max_seq_len
        self.page_table = create_page_table(
            (config.max_running_req + 1, self.max_seq_len),
            device=self.device,
        )
        # Get sliding window config if enabled
        sliding_window = getattr(config, 'sliding_window', None)
        
        self.attn_backend = create_attention_backend(
            config.attention_backend,
            config.model_config,
            self.kv_cache,
            self.page_table,
            page_size=page_size,
            sliding_window=sliding_window,
            computation_dtype=self.dtype,
        )
        self.moe_backend = (
            create_moe_backend(config.moe_backend)
            if "moe" in config.model_config.model_type
            else None
        )
        # Initialize context with correct page size
        self.ctx = Context(page_size=page_size, attn_backend=self.attn_backend)
        if self.moe_backend:
            self.ctx.moe_backend = self.moe_backend
        set_global_ctx(self.ctx)
        self.sampler = Sampler(self.device, self.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # Initialize dummy request for CUDA graph capture
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(self.dummy_page)
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=self.max_seq_len,
            vocab_size=self.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        """Initialize distributed communication backend."""
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        """Load model weights from HuggingFace format."""
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {
                k: v.to(self.dtype)
                for k, v in load_hf_weight(config.model_path, self.device).items()
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        """
        Calculate number of KV cache pages with hardware-aware optimizations.
        
        Considers:
        - GPU architecture (page size, FP8 support)
        - User constraints (max_kv_pages, sliding_window)
        - Available memory
        
        Returns:
            int: Number of pages to allocate
        """
        new_free_memory = self._sync_get_memory()[1]

        # Determine optimal page size based on GPU architecture
        # Can be overridden via config.page_size
        page_size = getattr(config, 'page_size', None) or get_optimal_page_size(self.gpu_arch)
        
        # Determine KV cache dtype with automatic fallback
        kv_dtype = getattr(config, 'kv_cache_dtype', 'auto')
        
        if kv_dtype == "auto":
            # Auto-select best dtype for hardware
            if supports_fp8(self.gpu_arch) and check_fp8_usability():
                dtype_size = 1  # FP8: 1 byte
                logger.info(f"Auto-selected FP8 (E4M3) for {self.gpu_arch}")
            else:
                dtype_size = 2  # FP16/BF16: 2 bytes
                logger.info(f"Auto-selected FP16 for {self.gpu_arch} (FP8 not available)")
        elif kv_dtype == "fp8":
            # User requested FP8, but verify hardware support
            if not (supports_fp8(self.gpu_arch) and check_fp8_usability()):
                logger.warning(
                    f"FP8 requested but not supported on {self.gpu_arch}. "
                    f"Falling back to FP16."
                )
                dtype_size = 2
            else:
                dtype_size = 1
                logger.info("Using FP8 (E4M3) as requested")
        elif kv_dtype in ["fp16", "bf16"]:
            dtype_size = 2
            logger.info(f"Using {kv_dtype.upper()}")
        else:
            # Unknown dtype, fallback to model dtype
            dtype_size = self.dtype.itemsize
            logger.warning(f"Unknown kv_cache_dtype={kv_dtype}, using default")

        # Calculate memory per page (memory for one page of KV cache)
        # Note: The actual allocation is 2x this value (for K and V tensors),
        # which is handled by MHAKVCache's buffer shape (2, num_pages, ...)
        # Formula: head_dim * num_kv_heads * page_size * dtype_size * num_layers
        cache_per_page = (
            self.model_config.head_dim
            * div_even(self.model_config.num_kv_heads, config.tp_info.size)
            * page_size
            * dtype_size
            * self.model_config.num_layers
        )
        
        num_pages = config.num_page_override
        if num_pages is None:
            # Check if user explicitly set max_kv_pages
            max_kv_pages = getattr(config, 'max_kv_pages', None)
            if max_kv_pages and max_kv_pages > 0:
                # User explicitly set max_kv_pages, use it directly
                num_pages = max_kv_pages
                logger.info(f"Using user-specified max_kv_pages: {num_pages}")
            else:
                # Calculate from available memory
                # Note: Actual allocation is 2x cache_per_page (for K and V tensors)
                model_memory = old_free_memory - new_free_memory
                available_memory = int(config.memory_ratio * old_free_memory) - model_memory
                num_pages = available_memory // (2 * cache_per_page)

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"

        # Apply hard limit from max_kv_pages config (prevents OOM if num_page_override was set)
        max_kv_pages = getattr(config, 'max_kv_pages', None)
        if max_kv_pages and max_kv_pages > 0:
            if num_pages > max_kv_pages:
                logger.info(
                    f"Limiting KV cache from {num_pages} to {max_kv_pages} pages "
                    f"(max_kv_pages config)"
                )
                num_pages = min(num_pages, max_kv_pages)

        # Apply sliding window constraint for long sequences
        # Note: sliding window attention limits the attention range during computation,
        # but we still need to allocate enough KV cache to support the full max_seq_len.
        # The KV cache size is determined by max_seq_len, not window_size.
        sliding_window = getattr(config, 'sliding_window', None)
        if sliding_window and sliding_window > 0:
            # Ensure we have enough pages for at least one full sequence
            min_pages_needed = config.max_seq_len // page_size
            if num_pages < min_pages_needed:
                logger.warning(
                    f"Sliding window enabled but KV cache too small: "
                    f"{num_pages} pages < {min_pages_needed} needed for max_seq_len={config.max_seq_len}. "
                    f"Consider increasing max_kv_pages."
                )

        real_kv_size = num_pages * cache_per_page
        logger.info(
            f"Allocating {num_pages} pages for KV cache, "
            f"per-page={mem_GB(cache_per_page)}, total K+V={mem_GB(2 * real_kv_size)} "
            f"(page_size={page_size}, dtype={'FP8' if dtype_size == 1 else 'FP16'})"
        )
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Synchronize and get free memory across all tensor parallel ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor(
            [free_memory, -free_memory], device="cpu", dtype=torch.int64
        )
        torch.distributed.all_reduce(
            free_mem_tensor, 
            op=torch.distributed.ReduceOp.MIN, 
            group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        
        # Check for memory imbalance across GPUs (indicates error)
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced: "
                f"min={mem_GB(min_free_memory)}, max={mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory imbalance detected across GPUs")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """Execute forward pass for a batch of requests."""
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        # Update request progress
        for req in batch.reqs:
            req.complete_one()

        # Sample next tokens
        next_tokens_gpu = self.sampler.sample(
            logits[: batch.size], args
        ).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        """Clean up resources."""
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()
