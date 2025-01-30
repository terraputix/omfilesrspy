import numpy as np
from omfilesrspy import OmFilePyWriter


def create_test_om_file(filename: str = "test_file.om", shape=(5, 5), dtype=np.float32) -> tuple[str, np.ndarray]:
    test_data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

    with OmFilePyWriter(filename) as writer:
        writer.write_array(test_data, chunks=[5, 5])

    return filename, test_data
