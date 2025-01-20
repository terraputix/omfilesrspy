from types import EllipsisType
from typing import Tuple, TypeAlias, Union

# Define types for indexing
DimIndex: TypeAlias = Union[
    slice,  # e.g., :, 1:10, 1:10:2
    int,  # e.g., 5
    None,  # e.g., None
    EllipsisType,  # Ellipsis
]

# This represents pythons basic indexing types
# https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing
BasicIndexType: TypeAlias = Union[
    DimIndex,
    Tuple[DimIndex, ...],  # e.g., (1, :, ..., 2:10)
]
