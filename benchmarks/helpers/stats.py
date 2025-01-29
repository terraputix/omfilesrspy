import os
import statistics
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, NamedTuple, TypeVar

import psutil

T = TypeVar("T")


@dataclass
class BenchmarkStats:
    mean: float
    std: float
    min: float
    max: float
    cpu_mean: float
    cpu_std: float
    memory_usage: float


class MeasurementResult(NamedTuple):
    result: Any
    elapsed: float
    cpu_elapsed: float
    memory_delta: float


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024  # Convert to KB


def measure_execution(func: Callable[..., T]) -> Callable[..., MeasurementResult]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MeasurementResult:
        before_mem = get_memory_usage()
        start_time = time.time()
        cpu_start_time = time.process_time()

        result = func(*args, **kwargs)

        elapsed_time = time.time() - start_time
        cpu_elapsed_time = time.process_time() - cpu_start_time
        after_mem = get_memory_usage()

        # fmt: off
        return MeasurementResult(
            result=result,
            elapsed=elapsed_time,
            cpu_elapsed=cpu_elapsed_time,
            memory_delta=after_mem - before_mem
        )

    return wrapper


def run_multiple_benchmarks(func: Callable[..., MeasurementResult], iterations: int = 5) -> BenchmarkStats:
    times: List[float] = []
    cpu_times: List[float] = []
    memory_usages: List[float] = []

    for _ in range(iterations):
        result = func()
        times.append(result.elapsed)
        cpu_times.append(result.cpu_elapsed)
        memory_usages.append(result.memory_delta)

    return BenchmarkStats(
        mean=statistics.mean(times),
        std=statistics.stdev(times) if len(times) > 1 else 0,
        min=min(times),
        max=max(times),
        cpu_mean=statistics.mean(cpu_times),
        cpu_std=statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0,
        memory_usage=statistics.mean(memory_usages),
    )


def print_write_benchmark_results(results: Dict[str, BenchmarkStats]) -> None:
    for fmt, result in results.items():
        print(f"\n{fmt} Write Results:")
        _print_benchmark_stats(result)


def print_read_benchmark_results(results: Dict[str, BenchmarkStats]) -> None:
    for fmt, result in results.items():
        print(f"\n{fmt} Read Results:")
        _print_benchmark_stats(result)


def _print_benchmark_stats(stats: BenchmarkStats) -> None:
    print(f"  Time: {stats.mean:.6f}s ± {stats.std:.6f}s (min: {stats.min:.6f}s, max: {stats.max:.6f}s)")
    print(f"  CPU Time: {stats.cpu_mean:.6f}s ± {stats.cpu_std:.6f}s")
    print(f"  Memory Delta: {stats.memory_usage:.2f}KB")
