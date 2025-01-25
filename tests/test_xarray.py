import os

import numpy as np
import xarray as xr
from omfilesrspy.omfilesrspy import OmFilePyReader
from omfilesrspy.xarray_backend import OmBackendArray
from xarray.core import indexing

from .test_utils import create_test_om_file


def test_om_backend_xarray_dtype():
    temp_file = "test_file.om"

    for dtype in [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
    ]:
        try:
            create_test_om_file(temp_file, dtype=dtype)

            reader = OmFilePyReader(temp_file)
            backend_array = OmBackendArray(reader, "dataset")

            assert isinstance(backend_array.dtype, np.dtype)
            assert backend_array.dtype == dtype

            data = xr.Variable(dims=["x", "y"], data=indexing.LazilyIndexedArray(backend_array))
            assert data.dtype == dtype

            del data, backend_array, reader
        finally:
            os.remove(temp_file)
