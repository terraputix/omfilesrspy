import numpy as np
from xarray.backends.common import BackendArray, BackendEntrypoint, WritableCFDataStore, _normalize_path
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import FrozenDict
from xarray.core.variable import Variable

from .omfilesrspy import OmFilePyReader


class OmXarrayEntrypoint(BackendEntrypoint):
    def guess_can_open(self, filename_or_obj):
        return isinstance(filename_or_obj, str) and filename_or_obj.endswith(".om")

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
    ):
        filename_or_obj = _normalize_path(filename_or_obj)
        root_variable = OmFilePyReader(filename_or_obj)
        store = OmDataStore(root_variable)
        store_entrypoint = StoreBackendEntrypoint()
        return store_entrypoint.open_dataset(
            store,
            drop_variables=drop_variables,
        )

    description = "Use .om files in Xarray"

    url = "https://github.com/open-meteo/om-file-format/"


class OmDataStore(WritableCFDataStore):
    root_variable: OmFilePyReader
    variables_offset_store: dict[str, tuple[int, int]]

    def __init__(self, root_variable: OmFilePyReader):
        self.root_variable = root_variable
        self.variables_offset_store = self._build_variables_offset_store()

    def _build_variables_offset_store(self) -> dict[str, tuple[int, int]]:
        return self.root_variable.get_flat_variable_metadata()

    def get_variables(self):
        return FrozenDict((k, self.open_store_variable(k)) for k in self.variables_offset_store)

    def get_attrs(self):
        # TODO: Currently no attributes are supported!
        return FrozenDict()

    def open_store_variable(self, k):
        if k not in self.variables_offset_store:
            raise KeyError(f"Variable {k} not found in the store")

        # Create a new reader for the specific variable
        offset, size = self.variables_offset_store[k]
        reader = self.root_variable.init_from_offset_size(offset, size)
        if reader is None:
            raise ValueError(f"Failed to read variable {k} at offset {offset}")

        backend_array = OmBackendArray(reader=reader)
        shape = backend_array.reader.shape

        # In om-files dimensions are not named, so we just use dim0, dim1, ...
        dim_names = [f"dim{i}" for i in range(len(shape))]
        data = indexing.LazilyIndexedArray(backend_array)
        return Variable(dims=dim_names, data=data, attrs=None, encoding=None, fastpath=True)


class OmBackendArray(BackendArray):
    def __init__(self, reader: OmFilePyReader):
        self.reader = reader

    @property
    def shape(self):
        return self.reader.shape

    @property
    def dtype(self):
        return np.dtype(self.reader.dtype)

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self.reader.__getitem__,
        )
