import numpy as np
import h5py
import zarr
import time
import netCDF4 as nc
import omfilesrspy as om

# Create a large NumPy array
array_size = (1000, 10000)
# Generate a sinusoidal curve with noise
x = np.linspace(0, 2 * np.pi * (array_size[1] / 100), array_size[1])
sinusoidal_data = 20 * np.sin(x) + 20  # Sinusoidal curve between -20 and 60
noise = np.random.normal(0, 5, array_size)  # Add some noise
data = (sinusoidal_data + noise).astype(np.float32)

# Define chunk size
chunk_size = (100, 100)

# Measure HDF5 write time
start_time = time.time()
with h5py.File("data.h5", "w") as f:
    f.create_dataset("dataset", data=data, chunks=chunk_size)
hdf5_write_time = time.time() - start_time

# Measure HDF5 read time
start_time = time.time()
with h5py.File("data.h5", "r") as f:
    data_read = f["dataset"][:]
hdf5_read_time = time.time() - start_time

# Measure Zarr write time
start_time = time.time()
zarr.save("data.zarr", data, chunks=chunk_size)
zarr_write_time = time.time() - start_time

# Measure Zarr read time
start_time = time.time()
data_read = zarr.load("data.zarr")
zarr_read_time = time.time() - start_time

# Measure NetCDF write time
start_time = time.time()
with nc.Dataset("data.nc", "w", format="NETCDF4") as ds:
    ds.createDimension("dim1", array_size[0])
    ds.createDimension("dim2", array_size[1])
    var = ds.createVariable(
        "dataset", np.float32, ("dim1", "dim2"), chunksizes=chunk_size
    )
    var[:] = data
netcdf_write_time = time.time() - start_time

# Measure NetCDF read time
start_time = time.time()
with nc.Dataset("data.nc", "r") as ds:
    data_read = ds.variables["dataset"][:]
netcdf_read_time = time.time() - start_time

# Measure OM write time
start_time = time.time()
om.write_om_file("data.om", data, 1000, 10000, chunk_size[0], chunk_size[1])
om_write_time = time.time() - start_time

# Measure OM read time
start_time = time.time()
data_read = om.read_om_file("data.om", 0, 1000, 0, 10000)
om_read_time = time.time() - start_time

# Print results
print(f"HDF5 write time: {hdf5_write_time:.4f} seconds")
print(f"HDF5 read time: {hdf5_read_time:.4f} seconds")
print(f"Zarr write time: {zarr_write_time:.4f} seconds")
print(f"Zarr read time: {zarr_read_time:.4f} seconds")
print(f"NetCDF write time: {netcdf_write_time:.4f} seconds")
print(f"NetCDF read time: {netcdf_read_time:.4f} seconds")
print(f"OM write time: {om_write_time:.4f} seconds")
print(f"OM read time: {om_read_time:.4f} seconds")
