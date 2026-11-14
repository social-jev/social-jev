"""Send typed SocialJev requests to an HTTP worker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from ..types import Request, Result

if TYPE_CHECKING:
    from contextlib import ExitStack

RESULTS = TypeAdapter(list[list[Result]])


class HttpClient:
    """Submit complete requests to the rented HTTP worker."""

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


def request_payload(requests: Request | list[Request]) -> list[dict]:
    """Send the same wire format for a single request and a batch."""
    items = [requests] if isinstance(requests, Request) else requests
    return [item.model_dump(mode="json") for item in items]
