# import the contents of the Rust library into the Python extension
from . import xarray_backend

from .omfilesrspy import OmFilePyReader, OmFilePyWriter

__all__ = ["OmFilePyReader", "OmFilePyWriter", "xarray_backend"]
