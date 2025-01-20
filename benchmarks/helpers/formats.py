from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import h5py
import netCDF4 as nc
import numpy as np
import omfilesrspy as om
import zarr
from omfilesrspy.types import BasicIndexType


class BaseFormat(ABC):
    def __init__(self, filename: str):
        self.filename = Path(filename)

    @abstractmethod
    def write(self, data: np.ndarray, chunk_size: Tuple[int, ...]) -> None:
        pass

    @abstractmethod
    def read(self, index: BasicIndexType) -> np.ndarray:
        pass


class HDF5Format(BaseFormat):
    def write(self, data: np.ndarray, chunk_size: Tuple[int, ...]) -> None:
        with h5py.File(self.filename, "w") as f:
            f.create_dataset("dataset", data=data, chunks=chunk_size)

    def read(self, index: BasicIndexType) -> np.ndarray:
        with h5py.File(self.filename, "r") as f:
            dataset = f["dataset"]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError("Expected a h5py Dataset")
            return dataset[index]


class ZarrFormat(BaseFormat):
    def write(self, data: np.ndarray, chunk_size: Tuple[int, ...]) -> None:
        zarr.save(str(self.filename), data, chunks=chunk_size)

    def read(self, index: BasicIndexType) -> np.ndarray:
        z = zarr.open(str(self.filename), mode="r")
        if not isinstance(z, zarr.Group):
            raise TypeError("Expected a zarr Group")
        array = z["arr_0"]
        if not isinstance(array, zarr.Array):
            raise TypeError("Expected a zarr Array")
        return array[index]


class NetCDFFormat(BaseFormat):
    def write(self, data: np.ndarray, chunk_size: Tuple[int, ...]) -> None:
        with nc.Dataset(self.filename, "w", format="NETCDF4") as ds:
            dimension_names = tuple(f"dim{i}" for i in range(data.ndim))
            for dim, size in zip(dimension_names, data.shape):
                ds.createDimension(dim, size)

            var = ds.createVariable("dataset", data.dtype, dimension_names, chunksizes=chunk_size)
            var[:] = data

    def read(self, index: BasicIndexType) -> np.ndarray:
        with nc.Dataset(self.filename, "r") as ds:
            return ds.variables["dataset"][index]


class OMFormat(BaseFormat):
    def write(self, data: np.ndarray, chunk_size: Tuple[int, ...]) -> None:
        writer = om.OmFilePyWriter(str(self.filename))
        writer.write_array(data, chunk_size, 100, 0)

    def read(self, index: BasicIndexType) -> np.ndarray:
        reader = om.OmFilePyReader(str(self.filename))
        return reader[index]


# Factory for creating format handlers
class FormatFactory:
    # fmt: off
    formats = {
        "h5": HDF5Format,
        "zarr": ZarrFormat,
        "nc": NetCDFFormat,
        "om": OMFormat
    }

    @classmethod
    def create(cls, format_name: str, filename: str) -> BaseFormat:
        if format_name not in cls.formats:
            raise ValueError(f"Unknown format: {format_name}")
        return cls.formats[format_name](filename)
