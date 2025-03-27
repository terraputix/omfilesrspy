from __future__ import annotations

import gc
import os

import numpy as np
import numpy.typing as npt
import psutil
from omfiles import OmFilePyWriter


def create_test_om_file(filename: str = "test_file.om", shape=(5, 5), dtype: npt.DTypeLike =np.float32) -> tuple[str, np.ndarray]:
    test_data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

    writer = OmFilePyWriter(filename)
    variable = writer.write_array(test_data, chunks=[5, 5])
    writer.close(variable)

    return filename, test_data


class FileDescriptorCounter:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.fd_count_before = 0

    def count_before(self):
        self.fd_count_before = len(self.process.open_files())
        return self.fd_count_before

    def count_after(self):
        # Clean up potentially lingering objects
        gc.collect()
        return len(self.process.open_files())

    def assert_no_leaks(self):
        fd_count_after = self.count_after()
        assert fd_count_after <= self.fd_count_before, (
            f"File descriptor leak: {self.fd_count_before} before, {fd_count_after} after"
        )
