from argparse import Namespace

from helpers.args import parse_args
from helpers.formats import FormatFactory
from helpers.stats import (
    measure_execution,
    print_read_benchmark_results,
    print_write_benchmark_results,
    run_multiple_benchmarks,
)
from zarr.core.buffer import NDArrayLike


def bm_write_all_formats(args: Namespace, data: NDArrayLike):
    write_results = {}
    for format_name in ["h5", "zarr", "nc", "om"]:
        writer = FormatFactory.create_writer(format_name, f"data.{format_name}")

        @measure_execution
        def write():
            writer.write(data, args.chunk_size)

        try:
            write_stats = run_multiple_benchmarks(write, args.iterations)
            write_results[format_name] = write_stats
        except Exception as e:
            print(f"Error with {format_name}: {e}")

    print_write_benchmark_results(write_results)


def bm_read_all_formats(args: Namespace):
    formats_and_filenames = {
        "h5": "data.h5",
        "h5hidefix": "data.h5",
        "zarr": "data.zarr",
        "nc": "data.nc",
        "om": "data.om",
    }

    read_results = {}
    for format_name, file in formats_and_filenames.items():
        reader = FormatFactory.create_reader(format_name, file)

        @measure_execution
        def read():
            return reader.read(args.read_index)

        try:
            # sample_data = reader.read(args.read_index)  # Get sample data for verification
            read_stats = run_multiple_benchmarks(read, args.iterations)
            read_results[format_name] = read_stats

        except Exception as e:
            print(f"Error with {format_name}: {e}")
        finally:
            reader.close()

    print_read_benchmark_results(read_results)


def main():
    # Defines chunk and array sizes
    args = parse_args()

    # data = generate_test_data(args.array_size, noise_level=5, amplitude=20, offset=20)
    # print_data_info(data, args.chunk_size)
    # bm_write_all_formats(args, data)

    bm_read_all_formats(args)


if __name__ == "__main__":
    main()
