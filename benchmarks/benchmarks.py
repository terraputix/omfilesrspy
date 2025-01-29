from dataclasses import dataclass
from typing import Tuple

import numpy as np
from helpers.args import parse_args
from helpers.formats import BaseFormat, FormatFactory
from helpers.generate_data import generate_test_data
from helpers.stats import BenchmarkStats, measure_execution, run_multiple_benchmarks
from numpy.typing import NDArray
from omfilesrspy.types import BasicSelection
from zarr.core.buffer import NDArrayLike


@dataclass
class FormatBenchmarkResult:
    write_stats: BenchmarkStats
    read_stats: BenchmarkStats
    sample_data: NDArray[np.float32]


def benchmark_format(
    format_handler: BaseFormat,
    data: NDArrayLike,
    chunk_size: Tuple[int, ...],
    index: BasicSelection,
    iterations: int,
) -> FormatBenchmarkResult:
    # Measure write performance
    @measure_execution
    def write():
        format_handler.write(data, chunk_size)

    # Measure read performance
    @measure_execution
    def read():
        return format_handler.read(index)

    write_stats = run_multiple_benchmarks(write, iterations)
    read_result = read()  # Get sample data for verification
    read_stats = run_multiple_benchmarks(read, iterations)

    return FormatBenchmarkResult(write_stats=write_stats, read_stats=read_stats, sample_data=read_result.result)


def main():
    # Defines chunk and array sizes
    args = parse_args()
    data = generate_test_data(args.array_size, noise_level=5, amplitude=20, offset=20)

    print(
        f"""
Data shape: {data.shape}
Data type: {data.dtype}
Chunk size: {args.chunk_size}
"""
    )

    # Measure times
    results = {}
    for format_name in ["h5", "zarr", "nc", "om"]:
        handler = FormatFactory.create(format_name, f"data.{format_name}")

        try:
            results[format_name] = benchmark_format(handler, data, args.chunk_size, args.read_index, args.iterations)

            # Verify data
            read_data = results[format_name].sample_data
            if read_data.shape == args.array_size:
                print(f"{format_name} first five elements: {read_data[0, :5]}")
            else:
                print(f"{format_name} read data shape is {read_data.shape}")
                # print(f"{read_data}")
        except Exception as e:
            print(f"Error with {format_name}: {e}")

    # Print results
    for fmt, result in results.items():
        print(f"\n{fmt} Results:")
        print("Write:")
        print(f"  Time: {result.write_stats.mean:.5f}s ± {result.write_stats.std:.5f}s")
        print(f"  CPU Time: {result.write_stats.cpu_mean:.5f}s ± {result.write_stats.cpu_std:.5f}s")
        print(f"  Memory Delta: {result.write_stats.memory_usage:.2f}MB")
        print("Read:")
        print(f"  Time: {result.read_stats.mean:.5f}s ± {result.read_stats.std:.5f}s")
        print(f"  CPU Time: {result.read_stats.cpu_mean:.5f}s ± {result.read_stats.cpu_std:.5f}s")
        print(f"  Memory Delta: {result.read_stats.memory_usage:.2f}MB")


if __name__ == "__main__":
    main()
