from typing import Tuple

import numpy as np
from helpers.args import parse_args
from helpers.formats import BaseFormat, FormatFactory
from helpers.generate_data import generate_test_data
from helpers.stats import measure_time
from numpy.typing import NDArray
from omfilesrspy.types import BasicIndexType


def run_benchmark(
    format_handler: BaseFormat, data: NDArray[np.float32], chunk_size: Tuple[int, ...], index: BasicIndexType
) -> Tuple[dict[str, float], NDArray[np.float32]]:
    results = {}

    # Measure write performance
    @measure_time
    def write():
        format_handler.write(data, chunk_size)

    # Measure read performance
    @measure_time
    def read():
        return format_handler.read(index)

    _, results["write_time"], results["cpu_write_time"] = write()
    read_data, results["read_time"], results["cpu_read_time"] = read()

    return results, read_data


def main():
    # Defines chunk and array sizes
    args = parse_args()
    data = generate_test_data(args.array_size, noise_level=5, amplitude=20, offset=20)

    print("Data shape:", data.shape)
    print("Data type:", data.dtype)
    print("Chunk size:", args.chunk_size)

    # Measure times
    results = {}
    for format_name in ["h5", "zarr", "nc", "om"]:
        handler = FormatFactory.create(format_name, f"data.{format_name}")

        try:
            format_results, read_data = run_benchmark(handler, data, args.chunk_size, args.read_index)
            results[format_name] = format_results

            # Verify data
            if read_data.shape == args.array_size:
                print(f"{format_name} first five elements: {read_data[0, :5]}")
            else:
                print(f"    {format_name} read data shape is {read_data.shape}")
                print(f"{read_data}")
        except Exception as e:
            print(f"Error with {format_name}: {e}")

    # Print results
    for fmt, times in results.items():
        print(f"{fmt} write time: {times['write_time']:.5f} seconds (CPU: {times['cpu_write_time']:.5f} seconds)")
        print(f"{fmt} read time: {times['read_time']:.5f} seconds (CPU: {times['cpu_read_time']:.5f} seconds)")


if __name__ == "__main__":
    main()
