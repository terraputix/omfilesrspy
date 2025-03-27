
import fsspec
import numpy as np
import omfiles
import pytest
import xarray as xr


@pytest.fixture
def s3_backend():
    s3_test_file = "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om"
    fs = fsspec.filesystem("s3", anon=True)
    yield fs.open(s3_test_file, mode="rb")

@pytest.fixture
def s3_backend_with_cache():
    s3_test_file = "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om"
    fs = fsspec.filesystem(protocol="s3", anon=True)
    yield fs.open(s3_test_file, mode="rb", cache_type="mmap", block_size=1024, cache_options={"location": "cache"})


def test_s3_reader(s3_backend):
    reader = omfiles.OmFilePyReader(s3_backend)
    data = reader[57812:57813, 0:100]

    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[:10], expected)


def test_s3_reader_with_cache(s3_backend_with_cache):
    reader = omfiles.OmFilePyReader(s3_backend_with_cache)
    data = reader[57812:57813, 0:100]

    expected = [18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
    np.testing.assert_array_almost_equal(data[:10], expected)


@pytest.mark.xfail(reason="Om Files on S3 currently have no names assigned for the variables")
def test_s3_xarray(s3_backend_with_cache):
    ds = xr.open_dataset(s3_backend_with_cache, engine="om")
    assert any(ds.variables.keys())


def test_fsspec_reader_close(temp_om_file):
    """Test that closing a reader with fsspec file object works correctly."""
    fs = fsspec.filesystem("file")

    # Test explicit closure
    with fs.open(temp_om_file, "rb") as f:
        reader = omfiles.OmFilePyReader(f)

        # Check properties before closing
        assert reader.shape == [5, 5]
        assert not reader.closed

        # Get data and verify
        data = reader[0:4, 0:4]
        assert data.dtype == np.float32
        assert data.shape == (4, 4)

        # Close and verify
        reader.close()
        assert reader.closed

        # Operations should fail after close
        try:
            _ = reader[0:4, 0:4]
            assert False, "Should fail on closed reader"
        except ValueError:
            pass

    # Test context manager
    with fs.open(temp_om_file, "rb") as f:
        with omfiles.OmFilePyReader(f) as reader:
            ctx_data = reader[0:4, 0:4]
            np.testing.assert_array_equal(ctx_data, data)

        # Should be closed after context
        assert reader.closed

    # Data obtained before closing should still be valid
    expected = [
        [0.0, 1.0, 2.0, 3.0],
        [5.0, 6.0, 7.0, 8.0],
        [10.0, 11.0, 12.0, 13.0],
        [15.0, 16.0, 17.0, 18.0],
    ]
    np.testing.assert_array_equal(data, expected)


def test_fsspec_file_actually_closes(temp_om_file):
    """Test that the underlying fsspec file is actually closed."""
    fs = fsspec.filesystem("file")
    f = fs.open(temp_om_file, "rb")

    # Create, verify and close reader
    reader = omfiles.OmFilePyReader(f)
    assert reader.shape == [5, 5]
    dtype = reader.dtype
    assert dtype == np.float32
    reader.close()

    # File should be closed - verify by trying to read from it
    try:
        f.read(1)
        assert False, "File should be closed"
    except (ValueError, OSError):
        pass
