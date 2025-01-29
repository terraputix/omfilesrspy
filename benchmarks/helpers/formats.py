from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import h5py
import netCDF4 as nc
import numpy as np
import omfilesrspy as om
import zarr
from omfilesrspy.types import BasicSelection
from zarr.core.buffer import NDArrayLike


class BaseWriter(ABC):
    def __init__(self, filename: str):
        self.filename = Path(filename)

    @abstractmethod
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        raise NotImplementedError("The write method must be implemented by subclasses")


class BaseReader(ABC):
    def __init__(self, filename: str):
        self.filename = Path(filename)

    @abstractmethod
    def read(self, index: BasicSelection) -> np.ndarray:
        raise NotImplementedError("The read method must be implemented by subclasses")


class HDF5Writer(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        with h5py.File(self.filename, "w") as f:
            f.create_dataset("dataset", data=data, chunks=chunk_size)


class HDF5Reader(BaseReader):
    def read(self, index: BasicSelection) -> np.ndarray:
        with h5py.File(self.filename, "r") as f:
            dataset = f["dataset"]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError("Expected a h5py Dataset")
            return dataset[index]


class ZarrWriter(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        zarr.save(str(self.filename), data, chunks=np.array(chunk_size))


class ZarrReader(BaseReader):
    def read(self, index: BasicSelection) -> np.ndarray:
        z = zarr.open(str(self.filename), mode="r")
        if not isinstance(z, zarr.Group):
            raise TypeError("Expected a zarr Group")
        array = z["arr_0"]
        if not isinstance(array, zarr.Array):
            raise TypeError("Expected a zarr Array")
        return array[index].__array__()


class NetCDFWriter(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        with nc.Dataset(self.filename, "w", format="NETCDF4") as ds:
            dimension_names = tuple(f"dim{i}" for i in range(data.ndim))
            for dim, size in zip(dimension_names, data.shape):
                ds.createDimension(dim, size)

            var = ds.createVariable("dataset", data.dtype, dimension_names, chunksizes=chunk_size)
            var[:] = data


class NetCDFReader(BaseReader):
    def read(self, index: BasicSelection) -> np.ndarray:
        with nc.Dataset(self.filename, "r") as ds:
            return ds.variables["dataset"][index]


class OMWriter(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        writer = om.OmFilePyWriter(str(self.filename))
        writer.write_array(data.__array__(), chunk_size, 100, 0)


class OMReader(BaseReader):
    def read(self, index: BasicSelection) -> np.ndarray:
        reader = om.OmFilePyReader(str(self.filename))
        return reader[index]


# Factory for creating format handlers
class FormatFactory:
    # fmt: off
    writers = {
        "h5": HDF5Writer,
        "zarr": ZarrWriter,
        "nc": NetCDFWriter,
        "om": OMWriter
    }

    readers = {
        "h5": HDF5Reader,
        "zarr": ZarrReader,
        "nc": NetCDFReader,
        "om": OMReader
    }

    @classmethod
    def create_writer(cls, format_name: str, filename: str) -> BaseWriter:
        if format_name not in cls.writers:
            raise ValueError(f"Unknown format: {format_name}")
        return cls.writers[format_name](filename)

    @classmethod
    def create_reader(cls, format_name: str, filename: str) -> BaseReader:
        if format_name not in cls.readers:
            raise ValueError(f"Unknown format: {format_name}")
        return cls.readers[format_name](filename)
