from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import netCDF4 as nc
import numpy as np
import omfilesrspy as om
import xarray as xr
import zarr
from omfilesrspy.types import BasicSelection


class BaseReader(ABC):
    filename: Path

    def __init__(self, filename: str):
        self.filename = Path(filename)

    @abstractmethod
    def read(self, index: BasicSelection) -> np.ndarray:
        raise NotImplementedError("The read method must be implemented by subclasses")

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError("The close method must be implemented by subclasses")


class HDF5Reader(BaseReader):
    h5_reader: h5py.Dataset

    def __init__(self, filename: str):
        super().__init__(filename)
        file = h5py.File(self.filename, "r")
        dataset = file["dataset"]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError("Expected a h5py Dataset")
        self.h5_reader = dataset

    def read(self, index: BasicSelection) -> np.ndarray:
        return self.h5_reader[index]

    def close(self) -> None:
        self.h5_reader.file.close()


class HDF5HidefixReader(BaseReader):
    h5_reader: xr.Dataset

    def __init__(self, filename: str):
        super().__init__(filename)
        self.h5_reader = xr.open_dataset(self.filename, engine="hidefix")

    def read(self, index: BasicSelection) -> np.ndarray:
        return self.h5_reader["dataset"][index].values

    def close(self) -> None:
        self.h5_reader.close()


class ZarrReader(BaseReader):
    zarr_reader: zarr.Array

    def __init__(self, filename: str):
        super().__init__(filename)
        z = zarr.open(str(self.filename), mode="r")
        if not isinstance(z, zarr.Group):
            raise TypeError("Expected a zarr Group")
        array = z["arr_0"]
        if not isinstance(array, zarr.Array):
            raise TypeError("Expected a zarr Array")

        self.zarr_reader = array

    def read(self, index: BasicSelection) -> np.ndarray:
        return self.zarr_reader[index].__array__()

    def close(self) -> None:
        self.zarr_reader.store.close()


class NetCDFReader(BaseReader):
    nc_reader: nc.Dataset

    def __init__(self, filename: str):
        super().__init__(filename)
        self.nc_reader = nc.Dataset(self.filename, "r")

    def read(self, index: BasicSelection) -> np.ndarray:
        return self.nc_reader.variables["dataset"][index]

    def close(self) -> None:
        self.nc_reader.close()


class OMReader(BaseReader):
    om_reader: om.OmFilePyReader

    def __init__(self, filename: str):
        super().__init__(filename)
        self.om_reader = om.OmFilePyReader(str(self.filename))

    def read(self, index: BasicSelection) -> np.ndarray:
        return self.om_reader[index]

    def close(self) -> None:
        pass
        # TODO: Implement close method
        # self.om_reader.close()
