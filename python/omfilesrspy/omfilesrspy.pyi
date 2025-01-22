from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

# Define types for indexing
DimIndex = Union[
    slice,  # e.g., :, 1:10, 1:10:2
    int,  # e.g., 5
    None,  # e.g., None
    type(...),  # Ellipsis
]
# This represents pythons basic indexing types
# https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing
BasicIndexType = Union[
    DimIndex,
    Tuple[DimIndex, ...],  # e.g., (1, :, ..., 2:10)
]

class OmFilePyWriter:
    """A Python wrapper for the Rust OmFileWriter implementation."""

    def __init__(self, file_path: str) -> None:
        """
        Initialize an OmFilePyWriter.

        Args:
            file_path: Path where the .om file will be created

        Raises:
            OSError: If the file cannot be created
        """
        ...

    def write_array(
        self,
        data: npt.NDArray[
            Union[
                np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64, np.int8, np.uint8, np.int16, np.uint16
            ]
        ],
        chunks: list[int] | tuple[int, ...],
        scale_factor: float = 1.0,
        add_offset: float = 0.0,
        compression: str = "p4nzdec256",
    ) -> None:
        """
        Write a numpy array to the .om file with specified chunking and scaling parameters.

        Args:
            data: Input array to be written. Supported dtypes are:
                 float32, float64, int32, int64, uint32, uint64, int8, uint8, int16, uint16
            chunks: Chunk sizes for each dimension of the array
            scale_factor: Scale factor for data compression (default: 1.0)
            add_offset: Offset value for data compression (default: 0.0)
            compression: Compression algorithm to use (default: "p4nzdec256")
                       Supported values: "p4nzdec256", "fpxdec32", "p4nzdec256logarithmic"

        Raises:
            PyValueError: If the data type is unsupported or if parameters are invalid
            OSError: If there's an error writing to the file
        """
        ...

class OmFilePyReader:
    """A Python wrapper for the Rust OmFileReader implementation."""

    def __init__(self, file_path: str) -> None:
        """
        Initialize an OmFilePyReader.

        Args:
            file_path: Path to the .om file to read

        Raises:
            PyValueError: If the file cannot be opened or is invalid
        """
        ...

    def __getitem__(
        self, ranges: BasicIndexType
    ) -> npt.NDArray[
        Union[np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64]
    ]:
        """
        Read data from the .om file using numpy-style indexing.
        Currently only slices with step 1 are supported.

        The returned array will have singleton dimensions removed (squeezed).
        For example, if you index a 3D array with [1,:,2], the result will
        be a 1D array since dimensions 0 and 2 have size 1.

        Args:
            ranges: Index expression that can be either a single slice/integer
                   or a tuple of slices/integers for multi-dimensional access.
                   Supports NumPy basic indexing including:
                   - Integers (e.g., a[1,2])
                   - Slices (e.g., a[1:10])
                   - Ellipsis (...)
                   - None/newaxis

        Returns:
            NDArray containing the requested data with squeezed singleton dimensions.
            The data type of the array matches the data type stored in the file
            (int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, or float64).

        Raises:
            PyValueError: If the requested ranges are invalid or if there's an error reading the data
        """
        ...

class OmFilePyFsSpecReader:
    """A Python wrapper for the Rust OmFileFsSpecReader implementation."""

    def __init__(self, file_obj: object) -> None:
        """
        Initialize an OmFilePyFsSpecReader.

        Args:
            file_obj: A fsspec file object with read, seek methods and fs attribute

        Raises:
            TypeError: If the provided file_obj is not a valid fsspec file object
            PyValueError: If the file cannot be opened or is invalid
        """
        ...

    def __getitem__(
        self, ranges: BasicIndexType
    ) -> npt.NDArray[
        Union[np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64]
    ]:
        """
        Read data from the .om file using numpy-style indexing.
        Currently only slices with step 1 are supported.

        The returned array will have singleton dimensions removed (squeezed).
        For example, if you index a 3D array with [1,:,2], the result will
        be a 1D array since dimensions 0 and 2 have size 1.

        Args:
            ranges: Index expression that can be either a single slice/integer
                   or a tuple of slices/integers for multi-dimensional access.
                   Supports NumPy basic indexing including:
                   - Integers (e.g., a[1,2])
                   - Slices (e.g., a[1:10])
                   - Ellipsis (...)
                   - None/newaxis

        Returns:
            NDArray containing the requested data with squeezed singleton dimensions.
            The data type of the array matches the data type stored in the file
            (int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, or float64).

        Raises:
            PyValueError: If the requested ranges are invalid or if there's an error reading the data
        """
        ...
