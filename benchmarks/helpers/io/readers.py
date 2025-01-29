from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import netCDF4 as nc
import numpy as np
import omfilesrspy as om
import zarr
from omfilesrspy.types import BasicSelection


class BaseReader(ABC):
    def __init__(self, filename: str):
        self.filename = Path(filename)

    @abstractmethod
    def read(self, index: BasicSelection) -> np.ndarray:
        raise NotImplementedError("The read method must be implemented by subclasses")


class HDF5Reader(BaseReader):
    def read(self, index: BasicSelection) -> np.ndarray:
        with h5py.File(self.filename, "r") as f:
            dataset = f["dataset"]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError("Expected a h5py Dataset")
            return dataset[index]


class ZarrReader(BaseReader):
    def read(self, index: BasicSelection) -> np.ndarray:
        z = zarr.open(str(self.filename), mode="r")
        if not isinstance(z, zarr.Group):
            raise TypeError("Expected a zarr Group")
        array = z["arr_0"]
        if not isinstance(array, zarr.Array):
            raise TypeError("Expected a zarr Array")
        return array[index].__array__()



class NetCDFReader(BaseReader):
    def read(self, index: BasicSelection) -> np.ndarray:
        with nc.Dataset(self.filename, "r") as ds:
            return ds.variables["dataset"][index]



class OMReader(BaseReader):
    def read(self, index: BasicSelection) -> np.ndarray:
        reader = om.OmFilePyReader(str(self.filename))
        return reader[index]
