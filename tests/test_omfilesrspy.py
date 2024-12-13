import numpy as np
import pytest
import omfilesrspy

def test_write_om_file():
    # Create test data
    test_data = np.random.rand(10, 10).astype(np.float32)
    temp_file = "test_file.om"

    # Write data
    omfilesrspy.write_om_file(
        temp_file,
        test_data,
        dim0=10,
        dim1=10,
        chunk0=5,
        chunk1=5,
        scale_factor=1.0,
        add_offset=0.0
    )
