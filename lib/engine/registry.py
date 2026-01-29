from __future__ import annotations

from typing import Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, name: str) -> None:
        self._name = name
        self._items: Dict[str, T] = {}

    def register(self, name: Optional[str] = None) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            key = name or item.__name__
            if key in self._items:
                raise KeyError(f"{key} already registered in {self._name}")
            self._items[key] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        if name not in self._items:
            available = ", ".join(sorted(self._items.keys()))
            raise KeyError(f"{name} not found in {self._name}. Available: {available}")
        return self._items[name]

    def has(self, name: str) -> bool:
        return name in self._items


MODELS = Registry("models")
LOSSES = Registry("losses")
DATASETS = Registry("datasets")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
