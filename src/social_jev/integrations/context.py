"""Contracts for selecting a remote inference engine within a Python context."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Protocol, Self

if TYPE_CHECKING:
    from types import TracebackType


class RemoteEngine(Protocol):
    """Score token sequences on rented compute."""

    def process(self, inputs: list[list[int]]) -> list[float]: ...

    async def aprocess(self, inputs: list[list[int]]) -> list[float]: ...

    def stop(self) -> None: ...


class Provider(Protocol):
    """Own remote inference resources for the duration of a with block."""

    def __enter__(self) -> Self: ...

    def start(self, repo: str, **settings: Any) -> RemoteEngine: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


active_provider: ContextVar[Provider | None] = ContextVar("active_provider", default=None)
