"""Run vLLM pooling on an ephemeral Modal GPU worker."""

from __future__ import annotations

import sys
from contextlib import ExitStack
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Self, cast

from .context import active_provider

if TYPE_CHECKING:
    from contextvars import Token
    from types import TracebackType

    from .context import Provider


class ModalEngine:
    """Forward token batches to a Modal worker and own its session."""

    def __init__(self, worker: Any, session: ExitStack):
        self.worker = worker
        self.session = session

    def process(self, inputs: list[list[int]]) -> list[float]:
        return self.worker.score.remote(inputs)

    async def aprocess(self, inputs: list[list[int]]) -> list[float]:
        return await self.worker.score.remote.aio(inputs)

    def stop(self) -> None:
        self.session.close()


class Modal:
    """Run SocialJev inference on Modal within a with block.

    Args:
        gpu: Modal GPU type, with an optional count such as "H100:2".

    Examples:
        >>> with Modal(gpu="L40S"):  # doctest: +SKIP
        ...     model = SocialJev.from_pretrained(repo)
        ...     results = model.query(requests)
    """

    def __init__(self, *, gpu: str):
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

    def start(self, repo: str, **settings: Any) -> ModalEngine:
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
                from social_jev.vllm.offline import VllmEngine  # noqa: PLC0415

                self.engine = VllmEngine(repo, **settings)

            @modal.method()
            def ready(self):
                """Wait for Modal to finish running the load hook."""

            @modal.method()
            async def score(self, inputs: list[list[int]]) -> list[float]:
                return await self.engine.aprocess(inputs)

            @modal.exit()
            def stop(self):
                self.engine.stop()

        with ExitStack() as startup:
            startup.enter_context(app.run())
            worker = cast("modal.Cls", Worker)()
            worker.ready.remote()
            session = startup.pop_all()

        self.session = session
        return ModalEngine(worker, session)
