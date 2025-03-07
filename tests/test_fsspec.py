import fsspec
import numpy as np
import pyomfiles

# def test_fsspec_backend():
#     fsspec_object = fsspec.open("test_files/read_test.om", "rb")

#     file = pyomfiles.FsSpecBackend(fsspec_object)
#     assert file.file_size == 144


def test_s3_reader():
    file_path = "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om"
    fs = fsspec.filesystem("s3", anon=True)
    backend = fs.open(file_path, mode="rb")

    # Create reader over fs spec backend
    reader = pyomfiles.OmFilePyReader(backend)
    data = reader[57812:57813, 0:100]

    # Verify the data
    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[:10], expected)


def test_s3_reader_with_cache():
    file_path = "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om"
    fs = fsspec.filesystem(protocol="s3", anon=True)
    backend = fs.open(file_path, mode="rb", cache_type="mmap", block_size=1024, cache_options={"location": "cache"})

    # Create reader over fs spec backend
    reader = pyomfiles.OmFilePyReader(backend)
    data = reader[57812:57813, 0:100]

    # Verify the data
    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[:10], expected)
