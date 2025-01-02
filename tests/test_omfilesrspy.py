import fsspec
import numpy as np
import omfilesrspy


def test_write_om_file():
    # Create test data
    test_data = np.random.rand(10, 10).astype(np.float32)
    temp_file = "test_file.om"
    file_writer = omfilesrspy.OmFilePyWriter(temp_file)

    # Write data
    file_writer.write_array(test_data, chunks=[5, 5], scale_factor=1.0, add_offset=0.0)


# def test_read_om_file():
#     # To run this test you need to execute cargo test --no-default-features once to create the test data...
#     # Read data
#     temp_file = "test_files/read_test.om"
#     reader = omfilesrspy.OmFilePyReader(temp_file)
#     data = reader[0:5, 0:5]

#     # Check data
#     assert data.shape == (5, 5)
#     assert data.dtype == np.float32
#     np.testing.assert_array_equal(
#         data,
#         [
#             [0.0, 1.0, 2.0, 3.0, 4.0],
#             [5.0, 6.0, 7.0, 8.0, 9.0],
#             [10.0, 11.0, 12.0, 13.0, 14.0],
#             [15.0, 16.0, 17.0, 18.0, 19.0],
#             [20.0, 21.0, 22.0, 23.0, 24.0],
#         ],
#     )


def test_fsspec_backend():
    fsspec_object = fsspec.open("test_files/read_test.om", "rb")

    file = omfilesrspy.FsSpecBackend(fsspec_object)
    assert file.file_size == 144


def test_s3_reader():
    file_path = "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om"
    fs = fsspec.filesystem("s3", anon=True)
    backend = fs.open(file_path, mode="rb")

    # Create reader over fs spec backend
    reader = omfilesrspy.OmFilePyFsSpecReader(backend)
    data = reader[57812:57813, 0:100]

    # Verify the data
    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[:10], expected)


def test_s3_reader_with_cache():
    file_path = "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om"
    fs = fsspec.filesystem(protocol="s3", anon=True)
    backend = fs.open(file_path, mode="rb", cache_type="mmap", block_size=1024, cache_options={"location": "cache"})

    # Create reader over fs spec backend
    reader = omfilesrspy.OmFilePyFsSpecReader(backend)
    data = reader[57812:57813, 0:100]

    # Verify the data
    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[:10], expected)
