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
    variables_store: dict[str, tuple[int, int, bool]]

    def __init__(self, root_variable: OmFilePyReader):
        self.root_variable = root_variable
        self.variables_store = self.root_variable.get_flat_variable_metadata()

    def get_variables(self):
        all_variables = {}
        for k, v in self.variables_store.items():
            if not v[2]:  # It's not a scalar variable
                all_variables[k] = self.open_store_variable(k)
        return FrozenDict(all_variables)

    def get_attrs(self):
        # Global attributes are attributes directly under the root variable.
        return FrozenDict(self._get_attributes_for_variable(self.root_variable, self.root_variable.variable_name()))

    def _find_direct_children_in_store(self, path: str):
        children = {}
        for key, value in self.variables_store.items():
            # Skip paths that don't start with the target path
            if not key.startswith(path + "/"):
                continue

            # Split into parts after the base path
            remaining_path = key[len(path) + 1 :]
            parts = remaining_path.split("/")

            # Only include direct children (one level deep)
            if len(parts) == 1:
                children[parts[0]] = value

        return children

    def _get_attributes_for_variable(self, reader: OmFilePyReader, path: str):
        attrs = {}
        direct_children = self._find_direct_children_in_store(path)
        for k, v in direct_children.items():
            offset, size, is_scalar = v
            if is_scalar:
                child_reader = reader.init_from_offset_size(offset, size)
                if child_reader:
                    attrs[k] = child_reader.get_scalar_value()
        return attrs

    def open_store_variable(self, path):
        offset, size, is_scalar = self.variables_store[path]  # Get metadata from nested dict

        if is_scalar:
            raise ValueError(f"{path} is a scalar variable, not an array variable.")

        reader = self.root_variable.init_from_offset_size(offset, size)
        if reader is None:
            raise ValueError(f"Failed to read variable {path} at offset {offset}")

        backend_array = OmBackendArray(reader=reader)
        shape = backend_array.reader.shape
        # In om-files dimensions are not named, so we just use dim0, dim1, ...
        dim_names = [f"{path}_dim{i}" for i in range(len(shape))]
        data = indexing.LazilyIndexedArray(backend_array)

        attrs = self._get_attributes_for_variable(reader, path)

        return Variable(dims=dim_names, data=data, attrs=attrs, encoding=None, fastpath=True)


class OmBackendArray(BackendArray):
    def __init__(self, reader: OmFilePyReader):
        self.reader = reader

    @property
    def shape(self):
        return self.reader.shape

    @property
    def dtype(self):
        return np.dtype(self.reader.dtype())

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self.reader.__getitem__,
        )
