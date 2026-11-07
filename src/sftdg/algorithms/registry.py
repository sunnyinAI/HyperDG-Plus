"""Simple registry for algorithm implementations."""

from collections.abc import Callable
from typing import Dict, Type

from domainbed import algorithms as db_algorithms

_ALGORITHM_REGISTRY: Dict[str, Callable] = {}


def register_algorithm(name: str) -> Callable:
    """Decorator to register a new algorithm class."""

    def decorator(cls: Type) -> Type:
        key = name.lower()
        if key in _ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm '{name}' already registered.")
        _ALGORITHM_REGISTRY[key] = cls
        return cls

    return decorator


def build_algorithm(name: str, *args, **kwargs):
    """Instantiate a registered algorithm."""
    key = name.lower()
    if key not in _ALGORITHM_REGISTRY:
        if name in db_algorithms.ALGORITHMS:
            algo_cls = db_algorithms.get_algorithm_class(name)
            return algo_cls(*args, **kwargs)
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: {list(_ALGORITHM_REGISTRY)} "
            f"+ DomainBed algorithms {db_algorithms.ALGORITHMS}"
        )
    return _ALGORITHM_REGISTRY[key](*args, **kwargs)
