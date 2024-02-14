from . import _version
from ._core import SparseTensor
from .sparsetype import DC, C, S, compressed, doubly_compressed, sparse

__version__ = _version.get_versions()["version"]
