"""Run SocialJev on a temporary Hugging Face Job."""

from __future__ import annotations

import json
import shlex
from contextlib import ExitStack, suppress
from importlib.metadata import version
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Self

from .client import HttpClient
from .context import active_provider

if TYPE_CHECKING:
    from contextvars import Token
    from types import TracebackType

    from .context import Provider


class HuggingFaceJobs:
    """Run SocialJev inference on a Hugging Face Job within a with block.

    Args:
        **kwargs: Hugging Face job options such as flavor, timeout, image, and secrets.
            The local Hub login authenticates job management and query requests.

    Examples:
        >>> with HuggingFaceJobs(flavor="h200", timeout="1h"):  # doctest: +SKIP
        ...     model = SocialJev.from_pretrained(repo, backend="vllm")
        ...     results = model.query(requests)
    """

    def __init__(self, **kwargs: Any):
        if kwargs.get("expose_public"):
            message = "HuggingFaceJobs requires authenticated access to its worker"
            raise ValueError(message)
        self.options: dict[str, Any] = {"image": "python:3.12", "flavor": "h200"} | kwargs
        self.context_token: Token[Provider | None] | None = None
        self.session: ExitStack | None = None

    def __enter__(self) -> Self:
        if self.context_token is not None:
            message = "Use a separate HuggingFaceJobs instance for each nested context"
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

    def start(self, repo: str, *, backend: str, serve: bool, **settings: Any) -> HttpClient:
        """Start a job and connect to the worker through the authenticated Jobs proxy."""
        if self.context_token is None or self.session is not None:
            message = "Load one model per HuggingFaceJobs context"
            raise RuntimeError(message)

        import httpx  # noqa: PLC0415 -- Optional compute dependency.
        from huggingface_hub import HfApi  # noqa: PLC0415 -- Optional compute dependency.
        from huggingface_hub.utils import build_hf_headers  # noqa: PLC0415

        options = self.options.copy()
        hf_token = options.pop("token", True)
        headers = build_hf_headers(token=hf_token)
        api = HfApi(token=hf_token)
        config = json.dumps(
            {"repo": repo, "backend": backend, "serve": serve, "settings": settings}
        )
        options["secrets"] = {**(options.get("secrets") or {}), "MODEL_CONFIG": config}

        with ExitStack() as startup:
            job = api.run_job(command=worker_command(backend), expose=[8000], **options)
            namespace = job.owner.name
            startup.callback(api.cancel_job, job_id=job.id, namespace=namespace)
            url = f"https://{job.id}--8000.hf.jobs"
            http = startup.enter_context(httpx.Client(base_url=url, headers=headers, timeout=None))
            wait_for_worker(api, job.id, namespace, http)

            def async_http():
                return httpx.AsyncClient(base_url=url, headers=headers, timeout=None)

            session = startup.pop_all()

        self.session = session
        return HttpClient(http, async_http, session, backend=backend)


def worker_command(backend: str) -> list[str]:
    """Install the matching release and run the shared worker inside the job."""
    package = "social-jev[vllm]" if backend == "vllm" else "social-jev"
    requirement = f"{package}=={version('social-jev')}"
    install = shlex.join(["python", "-m", "pip", "install", requirement, "fastapi", "uvicorn"])
    source = Path(__file__).with_name("worker.py").read_text()
    run = shlex.join(["python", "-c", source, "--job"])
    return ["sh", "-c", f"{install} && exec {run}"]


def wait_for_worker(api: Any, job_id: str, namespace: str, http: Any) -> None:
    """Wait for the model to load or report the job's terminal status."""
    import httpx  # noqa: PLC0415 -- Optional compute dependency.

    while True:
        status = api.inspect_job(job_id=job_id, namespace=namespace).status
        if status.stage in {"ERROR", "COMPLETED", "CANCELED", "DELETED"}:
            message = (
                f"Hugging Face Job {job_id} ended during startup: {status.stage} {status.message}"
            )
            raise RuntimeError(message)
        if status.stage == "RUNNING":
            with suppress(httpx.TransportError):
                response = http.get("/health", timeout=5)
                if response.status_code in {401, 403}:
                    response.raise_for_status()
                if response.is_success:
                    return
        sleep(1)
