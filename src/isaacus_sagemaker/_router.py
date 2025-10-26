from __future__ import annotations

from typing import Any, Dict, List, Iterable, Optional, Sequence
from itertools import cycle
from threading import Lock
from collections import defaultdict

from .types import IsaacusSageMakerRuntimeEndpoint


class _RoundRobin:
    def __init__(self, items: Iterable[Any]) -> None:
        self._items = list(items)
        self._it = cycle(self._items) if self._items else None

    def next(self) -> Optional[Any]:
        if self._it is None:
            return None

        return next(self._it)


class Router:
    def __init__(self, endpoints: Sequence[IsaacusSageMakerRuntimeEndpoint]) -> None:
        assert endpoints, "At least one endpoint must be provided."
        assert all(isinstance(e, IsaacusSageMakerRuntimeEndpoint) for e in endpoints), (
            "All provided endpoints must be of the type `IsaacusSageMakerRuntimeEndpoint`."
        )
        assert all(e.name for e in endpoints), "All provided endpoints must have a non-empty name."

        self._lock = Lock()
        self._all_rr = _RoundRobin([e for e in endpoints if e.models is None])

        by_model: Dict[str, List[IsaacusSageMakerRuntimeEndpoint]] = defaultdict(list)

        for e in endpoints:
            if e.models is not None:
                for model in e.models:
                    by_model[model].append(e)

        for e in endpoints:
            if e.models is None:
                for model in by_model.keys():
                    by_model[model].append(e)

        self._by_model_rr: Dict[str, _RoundRobin] = {
            model: _RoundRobin(endpoints) for model, endpoints in by_model.items()
        }

    def pick(self, model: Optional[str]) -> Optional[IsaacusSageMakerRuntimeEndpoint]:
        with self._lock:
            if model and model in self._by_model_rr:
                n = self._by_model_rr[model].next()

                if n is not None:
                    return n

            return self._all_rr.next()
