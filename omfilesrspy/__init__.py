# import the contents of the Rust library into the Python extension
from .omfilesrspy import *

# optional: include the documentation from the Rust module
from .omfilesrspy import (
    __all__,
    __doc__,  # noqa: F401
)

__all__ = __all__
