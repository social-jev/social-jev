"""Run SocialJev queries on an ephemeral Modal GPU worker."""

from __future__ import annotations

import sys
from contextlib import ExitStack
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Self, cast

from .context import active_provider

if TYPE_CHECKING:
    from contextvars import Token
    from types import TracebackType

    from ..types import Request, Result
    from .context import Provider


class ModalClient:
    """Forward complete queries to a Modal worker and own its session."""

    def __init__(self, worker: Any, session: ExitStack):
        self.worker = worker
        self.session = session

    def submit(self, requests: Request | list[Request]) -> list[list[Result]]:
        return self.worker.run_query.remote(requests)

    async def asubmit(self, requests: Request | list[Request]) -> list[list[Result]]:
        return await self.worker.run_query.remote.aio(requests)

    def close(self) -> None:
        self.session.close()


class Modal:
    """Run SocialJev inference on Modal within a with block.

    Args:
        gpu: Modal GPU type, defaulting to "H100". Add a count with "H100:2".

    Examples:
        >>> with Modal():  # doctest: +SKIP
        ...     model = SocialJev.from_pretrained(repo)
        ...     results = model.query(requests)
    """

    def __init__(self, *, gpu: str = "H100"):
        self.gpu = gpu
        self.token: Token[Provider | None] | None = None
        self.session: ExitStack | None = None

    def __enter__(self) -> Self:
        if self.token is not None:
            message = "Use a separate Modal instance for each nested context"
            raise RuntimeError(message)
        self.token = active_provider.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self.token is not None
        try:
            if self.session is not None:
                self.session.__exit__(exc_type, exc, traceback)
        finally:
            active_provider.reset(self.token)
            self.token = None
            self.session = None

    def start(self, repo: str, **settings: Any) -> ModalClient:
        """Start one worker and wait for the checkpoint to load."""
        if self.token is None or self.session is not None:
            message = "Load one model per Modal context"
            raise RuntimeError(message)

        import modal  # noqa: PLC0415  # ty: ignore[unresolved-import] -- Optional SDK.

        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        image = modal.Image.debian_slim(python_version=python_version).pip_install(
            f"social-jev[vllm]=={version('social-jev')}"
        )
        app = modal.App("social-jev")

        @app.cls(
            image=image,
            gpu=self.gpu,
            max_containers=1,
            serialized=True,
            include_source=False,
        )
        @modal.concurrent(max_inputs=16)
        class Worker:
            @modal.enter()
            def load(self):
                from social_jev import SocialJev  # noqa: PLC0415

                self.model = SocialJev.from_pretrained(repo, backend="vllm", **settings)

            @modal.method()
            def ready(self):
                """Wait for Modal to finish running the load hook."""

            @modal.method()
            async def run_query(self, requests):
                return await self.model.aquery(requests)

            @modal.exit()
            def stop(self):
                self.model.stop()

        with ExitStack() as startup:
            startup.enter_context(app.run())
            worker = cast("modal.Cls", Worker)()
            worker.ready.remote()
            session = startup.pop_all()

        self.session = session
        return ModalClient(worker, session)
