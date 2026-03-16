"""Top-level package for MimicPlay."""

from importlib.metadata import version, PackageNotFoundError

__all__ = ["__version__"]

try:
    __version__ = version("mimicplay")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

