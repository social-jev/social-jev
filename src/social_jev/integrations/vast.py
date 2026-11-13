"""Rent a Vast.ai instance and submit SocialJev queries through SSH."""

from __future__ import annotations

import json
import math
import shlex
import subprocess
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any, Self

from pydantic import TypeAdapter

from ..types import Request, Result
from .context import active_provider

if TYPE_CHECKING:
    from contextvars import Token
    from types import TracebackType

    from .context import Provider

IMAGE = "vastai/base-image:cuda-13.2-mini-py312-2026-10-23"
RESULTS = TypeAdapter(list[list[Result]])


class VastClient:
    """Submit complete requests over a private tunnel to the rented worker."""

    def __init__(self, http: Any, async_http: Any, session: ExitStack, *, backend: str):
        self.http = http
        self.async_http = async_http
        self.session = session
        self.backend = backend

    def submit(self, requests: Request | list[Request]) -> list[list[Result]]:
        response = self.http.post("/query", json=request_payload(requests))
        return RESULTS.validate_json(response.raise_for_status().content)

    async def asubmit(self, requests: Request | list[Request]) -> list[list[Result]]:
        if self.backend == "transformers":
            message = "aquery requires backend='vllm'"
            raise RuntimeError(message)
        # Scope connections to this call so repeated asyncio.run() calls are safe.
        async with self.async_http() as client:
            response = await client.post("/query", json=request_payload(requests))
        return RESULTS.validate_json(response.raise_for_status().content)

    def close(self) -> None:
        self.session.close()


class VastAI:
    """Run SocialJev on an on-demand Vast.ai rental for the duration of a with block.

    Register an SSH public key with Vast.ai and configure trusted SSH host keys
    in OpenSSH before connecting. The Vast SDK reads the local API key.

    Args:
        gpu: Vast.ai GPU name, such as "H100_SXM".
        num_gpus: Number of GPUs on one instance.
        max_hourly_price: Maximum advertised instance rate in USD per hour, if supplied.
        startup_timeout: Seconds allowed for the instance and model to start.
        **kwargs: Vast instance options such as image and disk. Disk defaults
            to 10 GB and is also used to price offers.

    Examples:
        >>> with VastAI():  # doctest: +SKIP
        ...     model = SocialJev.from_pretrained(repo, backend="vllm")
        ...     results = model.query(requests)
    """

    def __init__(
        self,
        *,
        max_hourly_price: float | None = None,
        startup_timeout: float = 600,
        **kwargs: Any,
    ):
        if max_hourly_price is not None and (
            not math.isfinite(max_hourly_price) or max_hourly_price <= 0
        ):
            message = "max_hourly_price must be finite and positive"
            raise ValueError(message)
        self.gpu = kwargs.pop("gpu", "H100_SXM")
        self.num_gpus = kwargs.pop("num_gpus", 1)
        if self.num_gpus < 1:
            message = "num_gpus must be positive"
            raise ValueError(message)
        self.rental_options = {"image": IMAGE, "disk": 10, "cancel_unavail": True} | kwargs
        self.max_hourly_price = max_hourly_price
        self.startup_timeout = startup_timeout
        self.context_token: Token[Provider | None] | None = None
        self.session: ExitStack | None = None

    def __enter__(self) -> Self:
        if self.context_token is not None:
            message = "Use a separate VastAI instance for each nested context"
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

    def start(self, repo: str, *, backend: str, serve: bool, **settings: Any) -> VastClient:
        """Rent one instance and wait for its SocialJev worker to load."""
        if self.context_token is None or self.session is not None:
            message = "Load one model per VastAI context"
            raise RuntimeError(message)
        if backend == "transformers" and self.num_gpus != 1:
            message = (
                "Transformers on VastAI uses one GPU. Request one GPU or choose "
                'backend="vllm" and configure parallelism for your workload.'
            )
            raise ValueError(message)

        from vastai import VastAI as VastSDK  # noqa: PLC0415  # ty: ignore[unresolved-import]

        config = json.dumps(
            {"repo": repo, "backend": backend, "serve": serve, "settings": settings}
        )
        vast = VastSDK()
        offer = self.select_offer(vast)
        deadline = monotonic() + self.startup_timeout
        with ExitStack() as startup:
            rental = vast.create_instance(
                id=offer["id"],
                ssh=True,
                direct=True,
                **self.rental_options,
            )
            instance_id = rental["new_contract"]
            startup.callback(destroy_instance, vast, instance_id)

            host, port = wait_for_ssh(vast, instance_id, deadline)
            http, async_http = connect_worker(startup, (host, port), config, backend, deadline)
            session = startup.pop_all()

        self.session = session
        return VastClient(http, async_http, session, backend=backend)

    def select_offer(self, vast: Any) -> dict:
        """Select the cheapest available offer inside the configured ceiling."""
        query = f"gpu_name={self.gpu} num_gpus={self.num_gpus} cuda_max_good>=13.2"
        ceiling = self.max_hourly_price
        if ceiling is not None:
            query += f" dph_total<={ceiling}"
        offers = vast.search_offers(
            query=query,
            order="dph_total",
            limit=1,
            storage=self.rental_options["disk"],
        )
        if not offers:
            message = "No Vast.ai offer matches the requested configuration"
            raise ValueError(message)
        if ceiling is not None and offers[0]["dph_total"] > ceiling:
            message = "No Vast.ai offer matches the requested hourly price ceiling"
            raise ValueError(message)
        return offers[0]


def request_payload(requests: Request | list[Request]) -> list[dict]:
    """Send the same wire format for a single request and a batch."""
    items = [requests] if isinstance(requests, Request) else requests
    return [item.model_dump(mode="json") for item in items]


def destroy_instance(vast: Any, instance_id: int) -> None:
    """Destroy the owned rental, preserving its ID if cleanup fails."""
    message = f"Could not destroy Vast.ai instance {instance_id}; check the Vast.ai console"
    try:
        response = vast.destroy_instance(id=instance_id)
    except Exception as error:
        raise RuntimeError(message) from error
    if not response["success"]:
        raise RuntimeError(message)


def wait_for_ssh(vast: Any, instance_id: int, deadline: float) -> tuple[str, int]:
    """Wait for Vast to publish the running instance's SSH address."""
    while monotonic() < deadline:
        instance = vast.show_instance(id=instance_id)
        if instance and instance["actual_status"] == "running":
            direct = instance.get("ports", {}).get("22/tcp")
            if direct and instance.get("public_ipaddr"):
                return instance["public_ipaddr"], int(direct[0]["HostPort"])
            if instance.get("ssh_host") and instance.get("ssh_port"):
                return instance["ssh_host"], int(instance["ssh_port"])
        sleep(1)
    message = f"Vast.ai instance {instance_id} did not become ready for SSH"
    raise TimeoutError(message)


def connect_worker(startup: ExitStack, address, config, backend, deadline):
    """Run a worker through OpenSSH and wait for its private HTTP endpoint."""
    import httpx  # noqa: PLC0415 -- Optional compute dependency.

    host, port = address
    directory = Path(
        startup.enter_context(TemporaryDirectory(dir="/tmp", prefix="socialjev-vast-"))
    )
    socket = str(directory / "query.sock")
    log = directory / "worker.log"
    package = "social-jev[vllm]" if backend == "vllm" else "social-jev"
    install = shlex.join(
        [
            "/tmp/socialjev/bin/pip",
            "install",
            f"{package}=={version('social-jev')}",
            "fastapi",
            "uvicorn",
        ]
    )
    source = Path(__file__).with_name("vast_worker.py").read_text()
    run = shlex.join(["/tmp/socialjev/bin/python", "-c", source])
    command = f"python -m venv /tmp/socialjev && {install} && exec {run}"
    with log.open("wb") as output:
        worker = subprocess.Popen(
            [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ConnectTimeout=10",
                "-a",
                "-L",
                f"{socket}:127.0.0.1:8000",
                "-p",
                str(port),
                "--",
                f"root@{host}",
                command,
            ],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
    startup.callback(stop_ssh, worker)
    assert worker.stdin is not None
    with worker.stdin:
        worker.stdin.write(config)

    http = startup.enter_context(
        httpx.Client(
            base_url="http://localhost",
            transport=httpx.HTTPTransport(uds=socket),
            timeout=None,
            trust_env=False,
        )
    )

    def async_http():
        return httpx.AsyncClient(
            base_url="http://localhost",
            transport=httpx.AsyncHTTPTransport(uds=socket),
            timeout=None,
            trust_env=False,
        )

    wait_for_worker(http, worker, log, deadline)
    return http, async_http


def wait_for_worker(http, worker, log: Path, deadline: float) -> None:
    """Report SSH or model startup errors instead of waiting for an inference call."""
    import httpx  # noqa: PLC0415 -- Optional compute dependency.

    while monotonic() < deadline and worker.poll() is None:
        try:
            if http.get("/health", timeout=1).is_success:
                return
        except httpx.TransportError:
            pass  # The worker has not opened its listening socket yet.
        sleep(1)
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 4000))
        details = stream.read().decode(errors="replace").strip()
    message = f"Vast.ai worker did not start:\n{details}"
    raise RuntimeError(message)


def stop_ssh(worker: subprocess.Popen) -> None:
    """Close the query tunnel before destroying its instance."""
    if worker.poll() is None:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
