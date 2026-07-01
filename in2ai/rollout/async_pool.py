"""Asynchronous rollout pool for self-play data collection."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypeVar

T = TypeVar("T")


class AsyncRolloutPool:
    """Run many episodes concurrently; model access is serialized via lock."""

    def __init__(self, n_workers: int = 4):
        self.n_workers = max(1, n_workers)
        self._model_lock = threading.Lock()

    def _locked_generate(self, generate_fn: Callable[[str], str], prompt: str) -> str:
        with self._model_lock:
            return generate_fn(prompt)

    def run(
        self,
        tasks: list[tuple[Callable[..., T], tuple, dict]],
    ) -> list[T]:
        results: list[T] = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = [pool.submit(fn, *args, **kwargs) for fn, args, kwargs in tasks]
            for fut in as_completed(futures):
                results.append(fut.result())
        return results

    def wrap_generator(self, generate_fn: Callable[[str], str]) -> Callable[[str], str]:
        return lambda prompt: self._locked_generate(generate_fn, prompt)
