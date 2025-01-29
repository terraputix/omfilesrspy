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
            backend_array = OmBackendArray(reader=reader)

            assert isinstance(backend_array.dtype, np.dtype)
            assert backend_array.dtype == dtype

            data = xr.Variable(dims=["x", "y"], data=indexing.LazilyIndexedArray(backend_array))
            assert data.dtype == dtype

            del data, backend_array, reader
        finally:
            os.remove(temp_file)


def test_xarray_backend():
    temp_file = "test_file.om"

    try:
        create_test_om_file(temp_file)

        ds = xr.open_dataset(temp_file, engine="om")
        data = ds["data"][:].values
        del ds

        assert data.shape == (5, 5)
        assert data.dtype == np.float32
        np.testing.assert_array_equal(
            data,
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ],
        )

    finally:
        os.remove(temp_file)
