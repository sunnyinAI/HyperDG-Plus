"""Algorithm interfaces and SFT implementation."""

from .registry import register_algorithm, build_algorithm
# Import algorithms so they self-register on package import.
from . import sft, flatland, hyperdg, hyperdg_meru  # noqa: F401

__all__ = ["register_algorithm", "build_algorithm"]
