"""Atagia package."""

from atagia.client import AtagiaClient, HttpAtagiaClient, LocalAtagiaClient, connect_atagia
from atagia.engine import Atagia

__all__ = [
    "Atagia",
    "AtagiaClient",
    "HttpAtagiaClient",
    "LocalAtagiaClient",
    "connect_atagia",
    "__version__",
]

__version__ = "0.1.0"
