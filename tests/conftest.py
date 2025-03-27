import tempfile

import numpy as np
import numpy.typing as npt
import pytest

from .test_utils import create_test_om_file


@pytest.fixture
def temp_om_file():
    """
    Fixture that creates a temporary OM file.
    Returns a path to the temporary file.
    """

    dtype: npt.DTypeLike = np.float32
    shape: tuple = (5, 5)

    with tempfile.NamedTemporaryFile(suffix=".om") as temp_file:
        create_test_om_file(temp_file.name, shape=shape, dtype=dtype)
        yield temp_file.name
