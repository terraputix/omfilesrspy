from __future__ import annotations

try:
    from types import EllipsisType
except ImportError:
    EllipsisType = type(Ellipsis)

# This is similar to https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/core/indexing.py#L38C1-L40C87
BasicSelector = int | slice | EllipsisType
BasicSelection = BasicSelector | tuple[BasicSelector, ...]  # also used for BlockIndex
