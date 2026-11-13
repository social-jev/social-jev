"""Run the existing SocialJev query pipeline behind an SSH-forwarded HTTP socket."""

import json
import sys
from contextlib import asynccontextmanager

import uvicorn  # ty: ignore[unresolved-import] -- Optional compute dependency.
from fastapi import FastAPI  # ty: ignore[unresolved-import] -- Optional compute dependency.

from social_jev import Request, Result, SocialJev


def create_app(repo: str, *, backend: str, serve: bool, settings: dict):  # noqa: C901
    """Load a model once and expose its query method to the tunnel."""
    model = SocialJev.from_pretrained(repo, backend=backend, serve=serve, **settings)
    if backend == "transformers":
        model.engine.model.to("cuda")

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            model.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ready": True}

    if backend == "transformers":

        @app.post("/query", response_model=list[list[Result]])
        def query(requests: list[Request]):
            return model.query(requests)

    else:

        @app.post("/query", response_model=list[list[Result]])
        async def query(requests: list[Request]):
            return await model.aquery(requests)

    return app


if __name__ == "__main__":
    uvicorn.run(create_app(**json.load(sys.stdin)), host="127.0.0.1", port=8000, access_log=False)
