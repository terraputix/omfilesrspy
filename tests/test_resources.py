import os

import fsspec
import omfiles
import psutil

from .test_utils import create_test_om_file


def test_no_file_handle_leaks():
    """Test that no file handles are leaked when opening and closing readers."""
    temp_file = "test_leak.om"


    try:
        create_test_om_file(temp_file)

        # Get current process
        process = psutil.Process(os.getpid())
        fd_count_before = len(process.open_files())

        # Create and use multiple readers
        for _ in range(10):  # Open and close multiple times
            reader = omfiles.OmFilePyReader(temp_file)
            _ = reader[0:5, 0:5]
            reader.close()

        # Also test with fsspec
        fs = fsspec.filesystem("file")
        for _ in range(10):
            with fs.open(temp_file, "rb") as f:
                reader = omfiles.OmFilePyReader(f)
                _ = reader[0:5, 0:5]
                reader.close()

        # Clean up potentially lingering objects
        import gc
        gc.collect()

        fd_count_after = len(process.open_files())

        # Verify no file handles were leaked
        assert fd_count_after <= fd_count_before, f"File descriptor leak: {fd_count_before} before, {fd_count_after} after"

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
