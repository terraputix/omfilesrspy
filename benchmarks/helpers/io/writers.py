from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import h5py
import netCDF4 as nc
import numpy as np
import omfilesrspy as om
import zarr
from zarr.core.buffer import NDArrayLike


class BaseWriter(ABC):
    def __init__(self, filename: str):
        self.filename = Path(filename)

    @abstractmethod
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        raise NotImplementedError("The write method must be implemented by subclasses")


class HDF5Writer(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        with h5py.File(self.filename, "w") as f:
            f.create_dataset("dataset", data=data, chunks=chunk_size)


class ZarrWriter(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        zarr.save(str(self.filename), data, chunks=np.array(chunk_size))


class NetCDFWriter(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        with nc.Dataset(self.filename, "w", format="NETCDF4") as ds:
            dimension_names = tuple(f"dim{i}" for i in range(data.ndim))
            for dim, size in zip(dimension_names, data.shape):
                ds.createDimension(dim, size)

            var = ds.createVariable("dataset", data.dtype, dimension_names, chunksizes=chunk_size)
            var[:] = data


class OMWriter(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        writer = om.OmFilePyWriter(str(self.filename))
        writer.write_array(data.__array__(), chunk_size, 100, 0)
