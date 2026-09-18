"""Toolkit-neutral single-instance socket transport."""

from __future__ import annotations

from typing import Callable, List

from genx.core.single_instance import SingleInstanceClient
from genx.core.single_instance import SingleInstanceServer as _SingleInstanceServer


class SingleInstanceServer(_SingleInstanceServer):
    """Socket server that delegates GUI-thread dispatch to its caller."""

    def __init__(
        self,
        on_message: Callable[[List[str]], None] | None = None,
        *,
        run_on_main_thread: Callable[[Callable[[List[str]], None], List[str]], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(on_message=on_message, **kwargs)
        self._run_on_main_thread = run_on_main_thread or (lambda callback, messages: callback(messages))

    def dispatch(self, callback: Callable[[List[str]], None], messages: List[str]) -> None:
        self._run_on_main_thread(callback, messages)


__all__ = ["SingleInstanceClient", "SingleInstanceServer"]
