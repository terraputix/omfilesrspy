import numpy as np
import h5py
import zarr
import time
import netCDF4 as nc
import omfilesrspy as om

# from numcodecs import Blosc
from functools import wraps

# Create a large NumPy array
array_size = (1000, 10000)
# Generate a sinusoidal curve with noise
x = np.linspace(0, 2 * np.pi * (array_size[1] / 100), array_size[1])
sinusoidal_data = 20 * np.sin(x) + 20  # Sinusoidal curve between -20 and 60
noise = np.random.normal(0, 5, array_size)  # Add some noise
data = (sinusoidal_data + noise).astype(np.float32)

print("Data shape:", data.shape)
print("Data type:", data.dtype)

# Define chunk size
chunk_size = (100, 100)


# Decorator to measure execution time
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        cpu_start_time = time.process_time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        cpu_elapsed_time = time.process_time() - cpu_start_time
        return result, elapsed_time, cpu_elapsed_time

    return wrapper


@measure_time
def write_hdf5(data, chunk_size):
    with h5py.File("data.h5", "w") as f:
        f.create_dataset(
            "dataset",
            data=data,
            chunks=chunk_size,
            # compression="gzip",
            # compression_opts=9,
        )


@measure_time
def read_hdf5():
    with h5py.File("data.h5", "r") as f:
        return f["dataset"][:]


@measure_time
def write_zarr(data, chunk_size):
    # compressor = Blosc(cname="lz4", clevel=5)
    # _z = zarr.array(
    #     data, chunks=chunk_size, compressor=compressor, chunk_store="data.zarr"
    # )
    zarr.save("data.zarr", data, chunks=chunk_size)


@measure_time
def read_zarr():
    data = zarr.load("data.zarr", path="arr_0")
    return data


@measure_time
def write_netcdf(data, chunk_size):
    with nc.Dataset("data.nc", "w", format="NETCDF4") as ds:
        ds.createDimension("dim1", data.shape[0])
        ds.createDimension("dim2", data.shape[1])
        var = ds.createVariable(
            "dataset",
            np.float32,
            ("dim1", "dim2"),
            chunksizes=chunk_size,
            # zlib=True,
            # complevel=9,
        )
        var[:] = data


@measure_time
def read_netcdf():
    with nc.Dataset("data.nc", "r") as ds:
        return ds.variables["dataset"][:]


@measure_time
def write_om(data, chunk_size):
    om.write_om_file(
        "data.om",
        data,
        data.shape[0],
        data.shape[1],
        chunk_size[0],
        chunk_size[1],
        1000,
    )


@measure_time
def read_om():
    return om.read_om_file("data.om", 0, 1000, 0, 10000)


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
    _, results[fmt]["write_time"], results[fmt]["cpu_write_time"] = write_func(
        data, chunk_size
    )
    read_data, results[fmt]["read_time"], results[fmt]["cpu_read_time"] = read_func()

    if read_data.shape == (1000, 10000):
        # Print the first five elements of the read data
        print(f"{fmt} first five elements: {read_data[0, :5]}")
    else:
        print(f"{fmt} read data shape: {read_data[:5]}")

# Print results
for fmt, times in results.items():
    print(
        f"{fmt} write time: {times['write_time']:.4f} seconds (CPU: {times['cpu_write_time']:.4f} seconds)"
    )
    print(
        f"{fmt} read time: {times['read_time']:.4f} seconds (CPU: {times['cpu_read_time']:.4f} seconds)"
    )
