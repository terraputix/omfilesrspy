import time

# from numcodecs import Blosc
from functools import wraps
from typing import Any, Callable

import h5py
import netCDF4 as nc
import numpy as np
import omfilesrspy as om
import xarray as xr
import zarr

# filenames
h5_filename = "data.h5"
zarr_filename = "data.zarr"
nc_filename = "data.nc"
om_filename = "data.om"

# Create a large NumPy array
array_size = (10, 10, 1, 10, 10, 10)
# Generate a sinusoidal curve with noise
x = np.linspace(0, 2 * np.pi * (array_size[1] / 100), array_size[1])
sinusoidal_data = 20 * np.sin(x) + 20  # Sinusoidal curve between -20 and 60
noise = np.random.normal(0, 5, array_size)  # Add some noise
data = (sinusoidal_data + noise).astype(np.float32)

# Define chunk size
chunk_size = (10, 5, 1, 1, 1, 1)

print("Data shape:", data.shape)
print("Data type:", data.dtype)
print("Chunk size:", chunk_size)


# Decorator to measure execution time
def measure_time(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: tuple, **kwargs: dict) -> tuple[Any, float, float]:
        start_time = time.time()
        cpu_start_time = time.process_time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        cpu_elapsed_time = time.process_time() - cpu_start_time
        return result, elapsed_time, cpu_elapsed_time

    return wrapper


@measure_time
def write_hdf5(data: np.typing.NDArray, chunk_size: tuple):
    with h5py.File(h5_filename, "w") as f:
        f.create_dataset(
            "dataset",
            data=data,
            chunks=chunk_size,
            # compression="gzip",
            # compression_opts=9,
        )


@measure_time
def read_hdf5():
    with h5py.File(h5_filename, "r") as f:
        return f["dataset"][0, 0, 0, 0, ...]


@measure_time
def write_zarr(data: np.typing.NDArray, chunk_size: tuple):
    # compressor = Blosc(cname="lz4", clevel=5)
    # _z = zarr.array(
    #     data, chunks=chunk_size, compressor=compressor, chunk_store="data.zarr"
    # )
    zarr.save(zarr_filename, data, chunks=np.array(chunk_size))


@measure_time
def read_zarr():
    z = zarr.open(zarr_filename, mode="r")
    return z["arr_0"][0, 0, 0, 0, ...]


@measure_time
def write_netcdf(data: np.typing.NDArray, chunk_size: tuple):
    with nc.Dataset(nc_filename, "w", format="NETCDF4") as ds:
        dimension_names = ()
        for dim in range(data.ndim):
            ds.createDimension(f"dim{dim}", data.shape[dim])
            dimension_names += (f"dim{dim}",)

        var = ds.createVariable(
            "dataset",
            np.float32,
            dimension_names,
            chunksizes=chunk_size,
            # zlib=True,
            # complevel=9,
        )
        var[:] = data


@measure_time
def read_netcdf():
    with nc.Dataset(nc_filename, "r") as ds:
        return ds.variables["dataset"][0, 0, 0, 0, ...]


@measure_time
def write_om(data: np.typing.NDArray, chunk_size: tuple):
    writer = om.OmFilePyWriter(om_filename)
    writer.write_array(
        data,
        chunk_size,
        100,
        0,
        compression="pfor_delta_2d_int16",
    )


# @measure_time
# def read_om():
#     reader = om.OmFilePyReader("data.om")
#     return reader[0, 0, 0, 0, ...]


@measure_time
def read_om():
    ds = xr.open_dataset(om_filename, engine=om.xarray_backend.OmXarrayEntrypoint)
    return ds["dataset"][0, 0, 0, 0, ...].values


# Measure times
results = {}
formats = {
    "HDF5": (write_hdf5, read_hdf5),
    "Zarr": (write_zarr, read_zarr),
    "NetCDF": (write_netcdf, read_netcdf),
    "OM": (write_om, read_om),
}

for fmt, (write_func, read_func) in formats.items():
    results[fmt] = {}
    _, results[fmt]["write_time"], results[fmt]["cpu_write_time"] = write_func(data, chunk_size)
    read_data, results[fmt]["read_time"], results[fmt]["cpu_read_time"] = read_func()

    if read_data.shape == array_size:
        # Print the first five elements of the read data
        print(f"{fmt} first five elements: {read_data[0, :5]}")
    else:
        print(f"    {fmt} read data shape is {read_data.shape}")
        print(f"{read_data}")

# Print results
for fmt, times in results.items():
    print(f"{fmt} write time: {times['write_time']:.4f} seconds (CPU: {times['cpu_write_time']:.4f} seconds)")
    print(f"{fmt} read time: {times['read_time']:.4f} seconds (CPU: {times['cpu_read_time']:.4f} seconds)")
