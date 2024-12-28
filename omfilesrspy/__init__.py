# import the contents of the Rust library into the Python extension
from .omfilesrspy import OmFilePyReader, OmFilePyWriter

__all__ = ["OmFilePyReader", "OmFilePyWriter"]
