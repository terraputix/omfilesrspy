import os

# for some reason xr.open_dataset triggers a warning:
# "RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility. Expected 16 from C header, got 96 from PyObject"
# We will just filter it out for now...
# https://github.com/pydata/xarray/issues/7259
import warnings

import numpy as np
import omfiles.omfiles as om
import omfiles.xarray_backend as om_xarray
import xarray as xr
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

            reader = om.OmFilePyReader(temp_file)
            backend_array = om_xarray.OmBackendArray(reader=reader)

            assert isinstance(backend_array.dtype, np.dtype)
            assert backend_array.dtype == dtype

            data = xr.Variable(dims=["x", "y"], data=indexing.LazilyIndexedArray(backend_array))
            assert data.dtype == dtype

            reader.close()
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


def test_xarray_backend():
    temp_file = "test_file.om"

    try:
        create_test_om_file(temp_file)

        warnings.filterwarnings("ignore", message="numpy.ndarray size changed", category=RuntimeWarning)
        ds = xr.open_dataset(temp_file, engine="om")
        variable = ds["data"]

        data = variable.values
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
        if os.path.exists(temp_file):
            os.remove(temp_file)

def test_xarray_hierarchical_file():
    temp_file = "test_hierarchical_xarray.om"

    try:
        # Create test data
        # temperature: lat, lon, alt, time
        temperature_data = np.random.rand(5, 5, 5, 10).astype(np.float32)
        # precipitation: lat, lon, time
        precipitation_data = np.random.rand(5, 5, 10).astype(np.float32)

        # Write hierarchical structure
        writer = om.OmFilePyWriter(temp_file)

        # dimensionality metadata
        temperature_dimension_var = writer.write_scalar("LATITUDE,LONGITUDE,ALTITUDE,TIME", name="_ARRAY_DIMENSIONS")
        temp_units = writer.write_scalar("celsius", name="units")
        temp_metadata = writer.write_scalar("Surface temperature", name="description")

        # Write child2 array
        temperature_var = writer.write_array(
            temperature_data,
            chunks=[2, 2, 1, 10],
            name="temperature",
            scale_factor=100000.0,
            children=[temperature_dimension_var, temp_units, temp_metadata]
        )

        # dimensionality metadata
        precipitation_dimension_var = writer.write_scalar("LATITUDE,LONGITUDE,TIME", name="_ARRAY_DIMENSIONS")
        precip_units = writer.write_scalar("mm", name="units")
        precip_metadata = writer.write_scalar("Precipitation", name="description")

        # Write child1 array with attribute children
        precipitation_var = writer.write_array(
            precipitation_data,
            chunks=[2, 2, 10],
            name="precipitation",
            scale_factor=100000.0,
            children=[precipitation_dimension_var, precip_units, precip_metadata]
        )

        # Write dimensions
        lat = writer.write_array(name="LATITUDE", data=np.arange(5).astype(np.float32), chunks=[5])
        lon = writer.write_array(name="LONGITUDE", data=np.arange(5).astype(np.float32), chunks=[5])
        alt = writer.write_array(name="ALTITUDE", data=np.arange(5).astype(np.float32), chunks=[5])
        time = writer.write_array(name="TIME", data=np.arange(10).astype(np.float32), chunks=[10])

        # Write root array with children
        root_var = writer.write_group(
            name="",
            children=[temperature_var, precipitation_var, lat, lon, alt, time]
        )

        # Finalize the file
        writer.close(root_var)

        warnings.filterwarnings("ignore", message="numpy.ndarray size changed", category=RuntimeWarning)
        ds = xr.open_dataset(temp_file, engine="om")

        # Check temperature data
        temp = ds["temperature"]
        np.testing.assert_array_almost_equal(temp.values, temperature_data, decimal=4)
        assert temp.shape == (5, 5, 5, 10)
        assert temp.dtype == np.float32
        assert temp.dims == ("LATITUDE", "LONGITUDE", "ALTITUDE", "TIME")
        # Check attributes
        assert temp.attrs["description"] == "Surface temperature"
        assert temp.attrs["units"] == "celsius"

        # Check precipitation data
        precip = ds["precipitation"]
        np.testing.assert_array_almost_equal(precip.values, precipitation_data, decimal=4)
        assert precip.shape == (5, 5, 10)
        assert precip.dtype == np.float32
        assert precip.dims == ("LATITUDE", "LONGITUDE", "TIME")
        # Check attributes
        assert precip.attrs["description"] == "Precipitation"
        assert precip.attrs["units"] == "mm"

        # Check dimensions
        assert ds["LATITUDE"].values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert ds["LONGITUDE"].values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert ds["ALTITUDE"].values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert ds["TIME"].values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

        # Test some xarray operations to ensure everything works as expected
        # Try selecting a subset
        subset = ds.sel(TIME=slice(0, 5))
        assert subset["temperature"].shape == (5, 5, 5, 6)
        assert subset["precipitation"].shape == (5, 5, 6)

        # Try computing mean over a dimension
        mean_temp = ds["temperature"].mean(dim="TIME")
        assert mean_temp.shape == (5, 5, 5)
        assert mean_temp.dims == ("LATITUDE", "LONGITUDE", "ALTITUDE")

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
