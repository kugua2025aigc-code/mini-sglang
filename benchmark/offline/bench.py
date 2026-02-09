"""
Benchmark script for mini-sglang inference engine with flexible configuration.

Supports command-line customization of batch size, sequence length, and precision.

Usage Examples:

  # 1. Use Preset Mode
  python bench.py --mode high_throughput

  # 2. Custom Batch Size
  python bench.py --mode high_throughput --batch 128

  # 3. Custom Sequence Length (Full Override)
  python bench.py --batch 64 --seq-len 4096 --input-len 2048 --output-len 2048

  # 4. Long Context with Sliding Window
  python bench.py --batch 32 --seq-len 16384 --window 4096 --kv-dtype fp8

  # 5. Auto-Calculate KV Pages
  python bench.py --batch 256 --seq-len 4096 --auto-pages
  # Calculates: 256 * 4096 / 16 = 65536 pages

  # 6. Mixed Preset + Custom
  python bench.py --mode long_context_balanced --batch 32 --kv-dtype fp16

  # Help
  python bench.py --help
"""

import argparse
import sys
import time
from dataclasses import dataclass, replace
from random import randint, seed
from typing import Optional

from minisgl.core import SamplingParams
from minisgl.llm import LLM
import torch


@dataclass
class BenchmarkConfig:
    """
    Configuration container for benchmark scenarios.
    
    Attributes:
        name: Identifier for this configuration mode.
        num_seqs: Number of concurrent sequences (batch size).
        max_seq_len: Maximum total sequence length in tokens.
        input_len: Input prompt length in tokens.
        output_len: Target generation length in tokens.
        max_kv_pages: Hard limit on KV cache pages.
        kv_cache_dtype: Data type for KV cache storage ('fp8', 'fp16', 'bf16').
        sliding_window: Optional sliding window size for constrained attention.
    """
    name: str
    num_seqs: int
    max_seq_len: int
    input_len: int
    output_len: int
    max_kv_pages: int
    kv_cache_dtype: str = "fp16"
    sliding_window: Optional[int] = None
    
    def validate(self) -> None:
        """Validate configuration consistency."""
        total_len = self.input_len + self.output_len
        if total_len > self.max_seq_len:
            raise ValueError(
                f"input_len({self.input_len}) + output_len({self.output_len}) "
                f"= {total_len} exceeds max_seq_len({self.max_seq_len})"
            )
        if self.num_seqs <= 0:
            raise ValueError("num_seqs must be positive")
        if self.max_kv_pages <= 0:
            raise ValueError("max_kv_pages must be positive")
        if self.input_len <= 0 or self.output_len <= 0:
            raise ValueError("input_len and output_len must be positive")


# Predefined benchmark modes for common scenarios
BENCHMARK_MODES = {
    "high_throughput": BenchmarkConfig(
        name="high_throughput",
        num_seqs=256,
        max_seq_len=2048,
        input_len=1024,
        output_len=1024,
        max_kv_pages=32768,      # 256 * 2048 / 16
        kv_cache_dtype="fp8",
        sliding_window=None
    ),
    "long_context_balanced": BenchmarkConfig(
        name="long_context_balanced",
        num_seqs=64,
        max_seq_len=8192,
        input_len=4096,
        output_len=4096,
        max_kv_pages=32768,      # 64 * 8192 / 16
        kv_cache_dtype="fp8",
        sliding_window=4096
    ),
    "long_context_fast": BenchmarkConfig(
        name="long_context_fast",
        num_seqs=128,
        max_seq_len=8192,
        input_len=4096,
        output_len=4096,
        max_kv_pages=65536,      # 128 * 8192 / 16
        kv_cache_dtype="fp8",
        sliding_window=2048
    ),
    "extreme_context": BenchmarkConfig(
        name="extreme_context",
        num_seqs=16,
        max_seq_len=32768,
        input_len=16384,
        output_len=16384,
        max_kv_pages=32768,      # 16 * 32768 / 16
        kv_cache_dtype="fp8",
        sliding_window=4096
    ),
    "legacy_safe": BenchmarkConfig(
        name="legacy_safe",
        num_seqs=64,
        max_seq_len=2048,
        input_len=1024,
        output_len=1024,
        max_kv_pages=8192,
        kv_cache_dtype="fp16",
        sliding_window=None
    )
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for flexible benchmark configuration.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark mini-sglang inference engine with custom configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Use Preset Mode
  python bench.py --mode high_throughput

  # 2. Custom Batch Size
  python bench.py --mode high_throughput --batch 128

  # 3. Custom Sequence Length (Full Override)
  python bench.py --batch 64 --seq-len 4096 --input-len 2048 --output-len 2048

  # 4. Long Context with Sliding Window
  python bench.py --batch 32 --seq-len 16384 --window 4096 --kv-dtype fp8

  # 5. Auto-Calculate KV Pages
  python bench.py --batch 256 --seq-len 4096 --auto-pages
  # Calculates: 256 * 4096 / 16 = 65536 pages

  # 6. Mixed Preset + Custom
  python bench.py --mode long_context_balanced --batch 32 --kv-dtype fp16

Preset modes:
  high_throughput       - 256 seqs, 2048 max_len, fp8 (optimized for throughput)
  long_context_balanced - 64 seqs, 8192 max_len, fp8, window=4096
  long_context_fast     - 128 seqs, 8192 max_len, fp8, window=2048
  extreme_context       - 16 seqs, 32768 max_len, fp8, window=4096
  legacy_safe           - 64 seqs, 2048 max_len, fp16 (safest compatibility)
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=list(BENCHMARK_MODES.keys()),
        default=None,
        help="Preset benchmark mode (default: None, use custom config or long_context_balanced fallback)"
    )
    
    # Override parameters
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=None,
        help="Number of concurrent sequences (overrides preset)"
    )
    parser.add_argument(
        "--seq-len", "-s",
        type=int,
        default=None,
        help="Maximum sequence length (overrides preset)"
    )
    parser.add_argument(
        "--input-len", "-i",
        type=int,
        default=None,
        help="Input prompt length (overrides preset)"
    )
    parser.add_argument(
        "--output-len", "-o",
        type=int,
        default=None,
        help="Output generation length (overrides preset)"
    )
    parser.add_argument(
        "--kv-dtype",
        type=str,
        choices=["fp8", "fp16", "bf16"],
        default=None,
        help="KV cache data type (overrides preset)"
    )
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=None,
        help="Sliding window size, 0 for full attention (overrides preset)"
    )
    parser.add_argument(
        "--max-kv-pages",
        type=int,
        default=None,
        help="Hard limit on KV cache pages (overrides preset or auto-calculation)"
    )
    parser.add_argument(
        "--auto-pages",
        action="store_true",
        help="Automatically calculate max_kv_pages from batch and seq_len (page_size=16)"
    )
    
    # Other options
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model path or HuggingFace repo (default: Qwen/Qwen3-0.6B)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """
    Create BenchmarkConfig from command-line arguments.
    
    Supports two modes:
    1. Preset mode: Use --mode to select a predefined configuration
    2. Full override mode: Provide batch, seq-len, input-len, output-len without mode
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        BenchmarkConfig with applied overrides.
    """
    # Check if we have full override parameters (batch + seq-len + input-len + output-len)
    has_full_override = (
        args.batch is not None and 
        args.seq_len is not None and 
        args.input_len is not None and 
        args.output_len is not None
    )
    
    # Check if we have partial custom parameters
    has_partial_custom = (
        args.batch is not None or 
        args.seq_len is not None or 
        args.input_len is not None or 
        args.output_len is not None or
        args.kv_dtype is not None or
        args.window is not None or
        args.max_kv_pages is not None or
        args.auto_pages
    )
    
    if args.mode is None:
        if has_full_override:
            # Full override mode: create config from scratch
            batch = args.batch
            seq_len = args.seq_len
            input_len = args.input_len
            output_len = args.output_len
            kv_dtype = args.kv_dtype or "fp16"
            window = args.window if args.window is not None and args.window > 0 else None
            
            # Calculate max_kv_pages
            if args.auto_pages:
                max_kv_pages = (batch * seq_len) // 16
                print(f"[Auto] Calculated max_kv_pages: {max_kv_pages} "
                      f"({batch} * {seq_len} / 16)")
            elif args.max_kv_pages is not None:
                max_kv_pages = args.max_kv_pages
            else:
                # Default: batch * seq_len / 16
                max_kv_pages = (batch * seq_len) // 16
            
            config = BenchmarkConfig(
                name="custom_full_override",
                num_seqs=batch,
                max_seq_len=seq_len,
                input_len=input_len,
                output_len=output_len,
                max_kv_pages=max_kv_pages,
                kv_cache_dtype=kv_dtype,
                sliding_window=window
            )
            return config
        else:
            # No mode specified and not full override, use default mode
            args.mode = "long_context_balanced"
    
    # Preset mode (with optional overrides)
    if args.mode not in BENCHMARK_MODES:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    base_config = BENCHMARK_MODES[args.mode]
    
    # Prepare override dictionary
    overrides = {}
    
    if args.batch is not None:
        overrides["num_seqs"] = args.batch
    if args.seq_len is not None:
        overrides["max_seq_len"] = args.seq_len
    if args.input_len is not None:
        overrides["input_len"] = args.input_len
    if args.output_len is not None:
        overrides["output_len"] = args.output_len
    if args.kv_dtype is not None:
        overrides["kv_cache_dtype"] = args.kv_dtype
    if args.window is not None:
        overrides["sliding_window"] = args.window if args.window > 0 else None
    
    # Handle max_kv_pages: auto-calculate or use provided value
    if args.auto_pages:
        # Calculate: batch * seq_len / page_size (16)
        seq_len = args.seq_len or base_config.max_seq_len
        batch = args.batch or base_config.num_seqs
        overrides["max_kv_pages"] = (batch * seq_len) // 16
        print(f"[Auto] Calculated max_kv_pages: {overrides['max_kv_pages']} "
              f"({batch} * {seq_len} / 16)")
    elif args.max_kv_pages is not None:
        overrides["max_kv_pages"] = args.max_kv_pages
    elif args.batch is not None:
        # Auto-adjust max_kv_pages when batch is overridden (but not explicitly set)
        # Calculate based on the new batch size and original seq_len ratio
        seq_len = args.seq_len or base_config.max_seq_len
        batch = args.batch
        overrides["max_kv_pages"] = (batch * seq_len) // 16
        print(f"[Auto] Adjusted max_kv_pages for batch={batch}: {overrides['max_kv_pages']} "
              f"({batch} * {seq_len} / 16)")
    
    # Apply overrides
    if overrides:
        config = replace(base_config, **overrides)
        config.name = f"{base_config.name}_custom"
    else:
        config = base_config
    
    return config


def create_prompts(num_seqs: int, input_len: int, vocab_range: int = 10000) -> list:
    """
    Generate random prompt token IDs for benchmarking.
    
    Args:
        num_seqs: Number of sequences to generate.
        input_len: Length of each prompt in tokens.
        vocab_range: Upper bound for random token IDs.
        
    Returns:
        List of token ID lists.
    """
    return [
        [randint(0, vocab_range) for _ in range(input_len)]
        for _ in range(num_seqs)
    ]


def create_sampling_params(num_seqs: int, output_len: int) -> list:
    """
    Create sampling parameters for all sequences.
    
    Args:
        num_seqs: Number of sequences.
        output_len: Target number of tokens to generate.
        
    Returns:
        List of SamplingParams objects.
    """
    return [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=output_len
        )
        for _ in range(num_seqs)
    ]


def run_benchmark(config: BenchmarkConfig, model_path: str, warmup_iters: int) -> dict:
    """
    Execute benchmark with given configuration.
    
    Args:
        config: BenchmarkConfig instance.
        model_path: HuggingFace model path or local directory.
        warmup_iters: Number of warmup iterations.
        
    Returns:
        Dictionary containing benchmark results.
    """
    config.validate()
    
    # Print configuration summary
    print(f"\n{'='*70}")
    print(f"[Config] Mode: {config.name}")
    print(f"[Config] Batch Size: {config.num_seqs}")
    print(f"[Config] Max Sequence Length: {config.max_seq_len}")
    print(f"[Config] Input Length: {config.input_len}")
    print(f"[Config] Output Length: {config.output_len}")
    print(f"[Config] KV Cache Pages: {config.max_kv_pages}")
    print(f"[Config] KV Cache Dtype: {config.kv_cache_dtype}")
    if config.sliding_window:
        print(f"[Config] Sliding Window: {config.sliding_window} tokens")
    else:
        print(f"[Config] Attention: Full (no window)")
    print(f"{'='*70}")
    
    # Initialize LLM
    llm = LLM(
        model_path,
        max_seq_len_override=config.max_seq_len,
        max_extend_tokens=min(2048, config.max_seq_len),
        cuda_graph_max_bs=config.num_seqs,
        max_running_req=config.num_seqs,
        max_kv_pages=config.max_kv_pages,
        kv_cache_dtype=config.kv_cache_dtype,
        sliding_window=config.sliding_window,
    )
    
    # Generate test data
    prompt_token_ids = create_prompts(config.num_seqs, config.input_len)
    sampling_params = create_sampling_params(config.num_seqs, config.output_len)
    
    # Warmup phase
    if warmup_iters > 0:
        print(f"Warming up ({warmup_iters} iterations)...")
        llm.generate(["Warmup"] * warmup_iters, SamplingParams(max_tokens=10))
    
    # Benchmark execution
    print("Running benchmark...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    llm.generate(prompt_token_ids, sampling_params)
    
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    total_tokens = config.num_seqs * config.output_len
    throughput = total_tokens / elapsed_time
    latency_per_token = (elapsed_time / total_tokens) * 1000  # ms
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    
    results = {
        "mode": config.name,
        "batch_size": config.num_seqs,
        "seq_len": config.max_seq_len,
        "throughput_tok_s": throughput,
        "total_tokens": total_tokens,
        "elapsed_time_s": elapsed_time,
        "latency_ms": latency_per_token,
        "peak_memory_gb": peak_memory
    }
    
    return results


def print_results(results: dict) -> None:
    """Pretty print benchmark results."""
    print(f"\n{'='*70}")
    print(f"[Results] Configuration: {results['mode']}")
    print(f"[Results] Batch: {results['batch_size']}, SeqLen: {results['seq_len']}")
    print(f"[Results] Throughput: {results['throughput_tok_s']:.2f} tok/s")
    print(f"[Results] Total Tokens: {results['total_tokens']}")
    print(f"[Results] Time: {results['elapsed_time_s']:.2f}s")
    print(f"[Results] Latency: {results['latency_ms']:.2f} ms/token")
    print(f"[Results] Peak Memory: {results['peak_memory_gb']:.2f} GB")
    print(f"{'='*70}")


def main():
    """Main entry point."""
    args = parse_args()
    seed(args.seed)
    
    try:
        # Create configuration from arguments
        config = create_config_from_args(args)
        
        # Run benchmark
        results = run_benchmark(config, args.model, args.warmup)
        print_results(results)
        
    except ValueError as e:
        print(f"[Error] Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Benchmark failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
