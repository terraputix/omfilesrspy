from types import EllipsisType

# This is from https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/core/indexing.py#L38C1-L40C87
BasicSelector = int | slice | EllipsisType
BasicSelection = BasicSelector | tuple[BasicSelector, ...]  # also used for BlockIndex
