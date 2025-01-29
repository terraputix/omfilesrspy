from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from helpers.stats import BenchmarkStats
from numpy.typing import NDArray
from zarr.core.buffer import NDArrayLike


@dataclass
class WriteBenchmarkResult:
    write_stats: BenchmarkStats


@dataclass
class ReadBenchmarkResult:
    read_stats: BenchmarkStats
    sample_data: NDArray[np.float32]


def print_data_info(data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
    print(
        f"""
Data shape: {data.shape}
Data type: {data.dtype}
Chunk size: {chunk_size}
"""
    )


def print_write_benchmark_results(results: Dict[str, WriteBenchmarkResult]) -> None:
    for fmt, result in results.items():
        print(f"\n{fmt} Write Results:")
        _print_benchmark_stats(result.write_stats)


def print_read_benchmark_results(results: Dict[str, ReadBenchmarkResult], array_size: Tuple[int, ...]) -> None:
    for fmt, result in results.items():
        print(f"\n{fmt} Read Results:")
        _print_benchmark_stats(result.read_stats)

        # Verify data
        read_data = result.sample_data
        if read_data.shape == array_size:
            print(f"{fmt} first five elements: {read_data[0, :5]}")
        else:
            print(f"{fmt} read data shape is {read_data.shape}")
            # print(f"{read_data}")


def _print_benchmark_stats(stats: BenchmarkStats) -> None:
    print(f"  Time: {stats.mean:.5f}s ± {stats.std:.5f}s")
    print(f"  CPU Time: {stats.cpu_mean:.5f}s ± {stats.cpu_std:.5f}s")
    print(f"  Memory Delta: {stats.memory_usage:.2f}MB")
