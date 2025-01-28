import os

import fsspec
import numpy as np
import omfilesrspy
import xarray as xr

from .test_utils import create_test_om_file


def test_write_om_roundtrip():
    temp_file = "test_file.om"

    try:
        create_test_om_file(temp_file)

        reader = omfilesrspy.OmFilePyReader(temp_file)
        data = reader[0:5, 0:5]
        del reader

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


def test_round_trip_array_datatypes():
    shape = (5, 5, 5, 2)
    chunks = [2, 2, 2, 1]
    test_cases = [
        (np.random.rand(*shape).astype(np.float32), "float32"),
        (np.random.rand(*shape).astype(np.float64), "float64"),
        (np.random.randint(-128, 127, size=shape, dtype=np.int8), "int8"),
        (np.random.randint(-32768, 32767, size=shape, dtype=np.int16), "int16"),
        (np.random.randint(-2147483648, 2147483647, size=shape, dtype=np.int32), "int32"),
        (np.random.randint(-9223372036854775808, 9223372036854775807, size=shape, dtype=np.int64), "int64"),
        (np.random.randint(0, 255, size=shape, dtype=np.uint8), "uint8"),
        (np.random.randint(0, 65535, size=shape, dtype=np.uint16), "uint16"),
        (np.random.randint(0, 4294967295, size=shape, dtype=np.uint32), "uint32"),
        (np.random.randint(0, 18446744073709551615, size=shape, dtype=np.uint64), "uint64"),
    ]

    for test_data, dtype in test_cases:
        temp_file = f"test_file_{dtype}.om"

        try:
            # Write data
            writer = omfilesrspy.OmFilePyWriter(temp_file)
            writer.write_array(test_data, chunks=chunks, scale_factor=10000.0, add_offset=0.0)
            del writer

            # Read data back
            reader = omfilesrspy.OmFilePyReader(temp_file)
            read_data = reader[:]
            del reader

            # Verify data
            assert read_data.dtype == test_data.dtype
            assert read_data.shape == test_data.shape
            # use assert_array_almost_equal since our floating point values are compressed lossy
            np.testing.assert_array_almost_equal(read_data, test_data, decimal=4)

        finally:
            # Always try to remove the temp file
            os.remove(temp_file)


# def test_fsspec_backend():
#     fsspec_object = fsspec.open("test_files/read_test.om", "rb")

#     file = omfilesrspy.FsSpecBackend(fsspec_object)
#     assert file.file_size == 144


def test_s3_reader():
    file_path = "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om"
    fs = fsspec.filesystem("s3", anon=True)
    backend = fs.open(file_path, mode="rb")

    # Create reader over fs spec backend
    reader = omfilesrspy.OmFilePyReader(backend)
    data = reader[57812:57813, 0:100]

    # Verify the data
    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[:10], expected)


def test_s3_reader_with_cache():
    file_path = "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om"
    fs = fsspec.filesystem(protocol="s3", anon=True)
    backend = fs.open(file_path, mode="rb", cache_type="mmap", block_size=1024, cache_options={"location": "cache"})

    # Create reader over fs spec backend
    reader = omfilesrspy.OmFilePyReader(backend)
    data = reader[57812:57813, 0:100]

    # Verify the data
    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[:10], expected)
