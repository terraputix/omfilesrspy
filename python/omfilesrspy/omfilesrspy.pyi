import numpy as np
import numpy.typing as npt

from .types import BasicIndexType

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
        data: npt.NDArray[np.float32],
        chunks: list[int] | tuple[int, ...],
        scale_factor: float,
        add_offset: float,
    ) -> None:
        """
        Write a numpy array to the .om file with specified chunking and scaling parameters.

        Args:
            data: Input array of float32 values to be written
            chunks: Chunk sizes for each dimension of the array
            scale_factor: Scale factor for data compression
            add_offset: Offset value for data compression

        Raises:
            PyValueError: If the data or parameters are invalid
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

    def __getitem__(self, ranges: BasicIndexType) -> npt.NDArray[np.float32]:
        """
        Read data from the .om file using numpy-style indexing.
        Currently only slices with step 1 are supported.

        Args:
            ranges: Index expression that can be either a single slice/integer
                   or a tuple of slices/integers for multi-dimensional access

        Returns:
            NDArray containing the requested data with squeezed singleton dimensions

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

    def __getitem__(self, ranges: BasicIndexType) -> npt.NDArray[np.float32]:
        """
        Read data from the .om file using numpy-style indexing.
        Currently only slices with step 1 are supported.

        Args:
            ranges: Index expression that can be either a single slice/integer
                   or a tuple of slices/integers for multi-dimensional access

        Returns:
            NDArray containing the requested data with squeezed singleton dimensions

        Raises:
            PyValueError: If the requested ranges are invalid or if there's an error reading the data
        """
        ...
