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

# def test_read_om_file():
#     # To run this test you need to execute cargo test --no-default-features once to create the test data...
#     # Read data
#     temp_file = "test_files/read_test.om"
#     data = omfilesrspy.read_om_file(temp_file, 0, 5, 0, 5)

#     # Check data
#     np.testing.assert_array_equal(data, [
#         0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
#         15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
#     ])

#     # TODO: Check shape and dtype
#     # assert data.shape == (10, 10)
#     # assert data.dtype == np.float32
