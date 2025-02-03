from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from .types import BasicSelection

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

    def __enter__(self) -> "OmFilePyWriter":
        """Enter the context manager."""
        ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Exit the context manager."""
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
        compression: str = "pfor_delta_2d",
        name: str = "data",
        children: list[str] = [],
    ) -> None:
        """
        Write a numpy array to the .om file with specified chunking and scaling parameters.

        Args:
            data: Input array to be written. Supported dtypes are:
                 float32, float64, int32, int64, uint32, uint64, int8, uint8, int16, uint16
            chunks: Chunk sizes for each dimension of the array
            scale_factor: Scale factor for data compression (default: 1.0)
            add_offset: Offset value for data compression (default: 0.0)
            compression: Compression algorithm to use (default: "pfor_delta_2d")
                       Supported values: "pfor_delta_2d", "fpx_xor_2d", "pfor_delta_2d_int16", "pfor_delta_2d_int16_logarithmic"
            name: Name of the variable to be written (default: "data")
            children: List of names of child variables (default: [])

        Raises:
            PyValueError: If the data type is unsupported or if parameters are invalid
            OSError: If there's an error writing to the file
        """
        ...

    def write_attribute(self, name: str, value: Union[str, int, float, bool], children: list[str] = []) -> None:
        """
        Write an attribute to the .om file.

        Args:
            name: The attribute name
            value: The attribute value
            children: List of names of child attributes

        Raises:
            ValueError: If the key is invalid
        """
        ...

class OmFilePyReader:
    """
    An OmFilePyReader class for reading .om files.

    A reader object can have an arbitrary number of child readers, each representing
    a multidimensional variable or a scalar variable (an attribute). Thus, this class
    implements a tree-like structure for multi-dimensional data access.

    Variables in OM-Files do not have named dimensions! That means you have to know
    what the dimensions represent in advance or you need to explicitly encode them as
    some kind of attribute.

    Most likely we will adopt the xarray convention which is implemented for zarr
    which requires multi-dimensional variables to have an attribute called
    _ARRAY_DIMENSIONS that contains a list of dimension names.
    These dimension names should be encoded somewhere in the .om file hierarchy
    as attributes.

    Therefore, it might be useful to differentiate in some way between
    hdf5-like groups and datasets/n-dim arrays in an om-file.

    Group: Can contain datasets/arrays, attributes, and other groups.
    Dataset: Data-array, might have associated attributes.
    Attribute: A named data value associated with a group or dataset.

    E.g. in netcdf4 the dimensions are special datasets that are associated with
    a group. Each additional dataset that is not a dimension is a variable and
    needs to have dimensions associated with it which belong to the group.
    """

    def __init__(self, file: Union[str, object]) -> None:
        """
        Initialize an OmFilePyReader from a file path or fsspec file object.

        Args:
            file: Path to the .om file to read or a fsspec file object

        Raises:
            PyValueError: If the file cannot be opened or is invalid
        """
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the data stored in the .om file.

        Returns:
            Tuple containing the dimensions of the data
        """
        ...

    @property
    def dtype(self) -> np.dtype:
        """
        Get the data type of the data stored in the .om file.

        Returns:
            Numpy data type of the data
        """

    @property
    def name(self) -> str:
        """
        Get the name of the variable stored in the .om file.

        Returns:
            Name of the variable or an empty string if not available
        """

    @classmethod
    def from_path(cls, path: str) -> "OmFilePyReader":
        """
        Create an OmFilePyReader from a file path.

        Args:
            path: Path to the .om file to read

        Returns:
            OmFilePyReader instance
        """

    @classmethod
    def from_fsspec(cls, file_obj: object) -> "OmFilePyReader":
        """
        Create an OmFilePyReader from a fsspec file object.

        Args:
            file_obj: fsspec file object with read, seek methods and fs attribute

        Returns:
            OmFilePyReader instance
        """

    def __getitem__(
        self, ranges: BasicSelection
    ) -> npt.NDArray[
        Union[np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64]
    ]:
        """
        Read data from the open variable.om file using numpy-style indexing.
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

    def get_scalar_value(self) -> Union[str, int, float, bool]:
        """Get the scalar value of a scalar variable."""

    def init_from_offset_size(self, offset: int, size: int) -> "OmFilePyReader":
        """Initialize a new OmFilePyReader from an offset and size in an existing file."""

    def get_flat_variable_metadata(self) -> dict[str, tuple[int, int, bool]]:
        """Get a mapping of variable names to their file offsets and sizes."""
