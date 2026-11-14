"""Run SocialJev queries on an ephemeral Modal GPU worker."""

from __future__ import annotations

import sys
from contextlib import ExitStack
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Self, cast
from warnings import warn

from .context import active_provider

if TYPE_CHECKING:
    from contextvars import Token
    from types import TracebackType

    from ..types import Request, Result
    from .context import Provider


class ModalClient:
    """Forward complete queries to a Modal worker and own its session."""

    def __init__(self, worker: Any, session: ExitStack, *, backend: str, max_inputs: int | None):
        self.worker = worker
        self.session = session
        self.backend = backend
        self.max_inputs = 1 if max_inputs is None else max_inputs
        self.in_flight = 0
        self.warned_about_overlap = False

    def submit(self, requests: Request | list[Request]) -> list[list[Result]]:
        return self.worker.run_query.remote(requests)

    async def asubmit(self, requests: Request | list[Request]) -> list[list[Result]]:
        if self.backend == "transformers":
            message = "aquery requires backend='vllm'"
            raise RuntimeError(message)
        if self.max_inputs == 1 and self.in_flight and not self.warned_about_overlap:
            warn(
                "Multiple aquery() calls are in flight, but this Modal worker "
                "handles one call at a time. Set Modal(max_inputs=2) or higher "
                "to let calls run concurrently on the worker.",
                stacklevel=3,
            )
            self.warned_about_overlap = True
        self.in_flight += 1
        try:
            return await self.worker.run_query.remote.aio(requests)
        finally:
            self.in_flight -= 1

    def close(self) -> None:
        self.session.close()


class Modal:
    """Run SocialJev inference on Modal within a with block.

    Args:
        gpu: Modal GPU type, defaulting to "H100!". Add a count with "H100:2".
        max_inputs: Concurrent calls per container, defaulting to one.
        **kwargs: Modal worker options such as image, cpu, memory, and timeout.
            A custom image must include SocialJev and the selected backend.

    Examples:
        >>> with Modal():  # doctest: +SKIP
        ...     model = SocialJev.from_pretrained(repo)
        ...     results = model.query(requests)
    """

    def __init__(self, **kwargs: Any):
        self.options = {"gpu": "H100!", "max_inputs": 1, "max_containers": 1} | kwargs
        self.context_token: Token[Provider | None] | None = None
        self.session: ExitStack | None = None

    def __enter__(self) -> Self:
        if self.context_token is not None:
            message = "Use a separate Modal instance for each nested context"
            raise RuntimeError(message)
        self.context_token = active_provider.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self.context_token is not None
        try:
            if self.session is not None:
                self.session.__exit__(exc_type, exc, traceback)
        finally:
            active_provider.reset(self.context_token)
            self.context_token = None
            self.session = None

    def start(self, repo: str, *, backend: str, serve: bool, **settings: Any) -> ModalClient:
        """Start one worker and wait for the checkpoint to load."""
        if self.context_token is None or self.session is not None:
            message = "Load one model per Modal context"
            raise RuntimeError(message)

        options = self.options.copy()
        gpu = options["gpu"]
        max_inputs = options.pop("max_inputs")
        if backend == "transformers" and int(gpu.partition(":")[2] or "1") > 1:
            message = (
                "Transformers on Modal uses one GPU. Request one GPU or choose "
                'backend="vllm" and configure parallelism for your workload.'
            )
            raise ValueError(message)

        import modal  # noqa: PLC0415  # ty: ignore[unresolved-import] -- Optional SDK.

        if "image" not in options:
            package = "social-jev[vllm]" if backend == "vllm" else "social-jev"
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            options["image"] = modal.Image.debian_slim(python_version=python_version).pip_install(
                f"{package}=={version('social-jev')}"
            )
        app = modal.App("social-jev")

        worker_class = make_worker(modal, repo, backend, serve, settings)
        if max_inputs is not None:
            worker_class = modal.concurrent(max_inputs=max_inputs)(worker_class)
        remote_worker = app.cls(serialized=True, include_source=False, **options)(worker_class)

        with ExitStack() as startup:
            startup.enter_context(app.run())
            worker = cast("modal.Cls", remote_worker)()
            worker.ready.remote()
            session = startup.pop_all()

        self.session = session
        return ModalClient(worker, session, backend=backend, max_inputs=max_inputs)


def make_worker(modal: Any, repo: str, backend: str, serve: bool, settings: dict):  # noqa: C901 -- Register both handler variants.
    """Define the entry points that run inside the rented container."""

    class Worker:
        @modal.enter()
        def load(self):
            from social_jev import SocialJev  # noqa: PLC0415

            self.model = SocialJev.from_pretrained(repo, backend=backend, serve=serve, **settings)
            if backend == "transformers":
                self.model.engine.model.to("cuda")

        @modal.method()
        def ready(self):
            """Wait for Modal to finish running the load hook."""

        if backend == "transformers":

            @modal.method()
            def run_query(self, requests):
                return self.model.query(requests)

        else:

            @modal.method()
            async def run_query(self, requests):
                return await self.model.aquery(requests)

        @modal.exit()
        def stop(self):
            self.model.stop()

    return Worker
