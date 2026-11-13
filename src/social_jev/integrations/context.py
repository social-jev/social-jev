"""Contracts for selecting a remote query client within a Python context."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Protocol, Self

if TYPE_CHECKING:
    from types import TracebackType

    from ..types import Request, Result


class RemoteClient(Protocol):
    """Send complete queries to rented compute."""

    def submit(self, requests: Request | list[Request]) -> list[list[Result]]: ...

    async def asubmit(self, requests: Request | list[Request]) -> list[list[Result]]: ...

    def close(self) -> None: ...


class Provider(Protocol):
    """Own remote inference resources for the duration of a with block."""

    def __enter__(self) -> Self: ...

    def start(self, repo: str, *, backend: str, serve: bool, **settings: Any) -> RemoteClient: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


active_provider: ContextVar[Provider | None] = ContextVar("active_provider", default=None)
