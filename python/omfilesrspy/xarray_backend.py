import numpy as np
import xarray as xr
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.core import indexing

from .omfilesrspy import OmFilePyReader


class OmBackendArray(BackendArray):
    def __init__(self, om_reader: OmFilePyReader, variable_name):
        self.reader = om_reader
        self.variable_name = variable_name

    @property
    def shape(self):
        return self.reader.shape

    @property
    def dtype(self):
        return self.reader.dtype()

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self.reader.__getitem__,
        )


class OmXarrayEntrypoint(BackendEntrypoint):
    def guess_can_open(self, filename_or_obj):
        return isinstance(filename_or_obj, str) and filename_or_obj.endswith(".om")

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
    ):
        reader = OmFilePyReader(filename_or_obj)

        backend_array = OmBackendArray(om_reader=reader, variable_name="dataset")
        shape = backend_array.reader.shape
        dim_names = [f"dim{i}" for i in range(len(shape))]
        data = indexing.LazilyIndexedArray(backend_array)
        var = xr.Variable(dims=dim_names, data=data, attrs=None, encoding=None, fastpath=True)
        # FIXME: this is not really correct and so far only supports one variable
        return xr.Dataset(data_vars=dict(dataset=(dim_names, var)))

    # This is optional and could also be tracked automatically
    # If we specify it, we need to ensure that it is correct!
    open_dataset_parameters = ("filename_or_obj",)

    description = "Use .om files in Xarray"

    url = "https://github.com/open-meteo/om-file-format/"


# TODO: Register the backend in pyproject.toml
