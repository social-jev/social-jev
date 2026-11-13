import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from social_jev import Query, Request, Result, SocialJev
from social_jev.integrations import HuggingFaceJobs
from social_jev.integrations.context import active_provider


@pytest.mark.parametrize("job_stage", ["RUNNING", "ERROR"])
def test_huggingface_job_queries_and_cancels(monkeypatch, job_stage):
    hub = pytest.importorskip("huggingface_hub")
    api = Mock()
    api.run_job.return_value = SimpleNamespace(id="test-job", owner=SimpleNamespace(name="org"))
    api.inspect_job.return_value = SimpleNamespace(
        status=SimpleNamespace(stage=job_stage, message="test status")
    )
    monkeypatch.setattr(hub, "HfApi", Mock(return_value=api))
    token = Mock()
    token.return_value = {"Authorization": "Bearer test-token"}
    monkeypatch.setattr(hub.utils, "build_hf_headers", token)
    request = Request(state="A", queries=[Query(text="Pick a label", options=["A", "B"])])
    result = Result(choice="A", probabilities={"A": 1.0})

    def respond(incoming):
        assert incoming.headers["Authorization"] == "Bearer test-token"
        assert incoming.url.host == "test-job--8000.hf.jobs"
        if incoming.url.path == "/health":
            return httpx.Response(200)
        assert incoming.url.path == "/query"
        assert json.loads(incoming.content) == [request.model_dump(mode="json")]
        return httpx.Response(200, json=[[result.model_dump(mode="json")]])

    sync_client, async_client = httpx.Client, httpx.AsyncClient
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: sync_client(transport=transport, **kwargs)
    )

    @asynccontextmanager
    async def async_http(**kwargs):
        async with async_client(transport=transport, **kwargs) as client:
            yield client

    monkeypatch.setattr(httpx, "AsyncClient", async_http)
    provider = HuggingFaceJobs(flavor="a100-large", timeout="1h", namespace="org")
    if job_stage == "RUNNING":
        with provider:
            model = SocialJev.from_pretrained("test/model", backend="vllm")
            assert model.query(request) == [[result]]
            assert asyncio.run(model.aquery(request)) == [[result]]
    else:
        with provider, pytest.raises(RuntimeError, match="ended during startup"):
            SocialJev.from_pretrained("test/model", backend="vllm")

    options = api.run_job.call_args.kwargs
    assert options["expose"] == [8000]
    assert options["flavor"] == "a100-large"
    assert options["timeout"] == "1h"
    assert options["namespace"] == "org"
    assert json.loads(options["secrets"]["MODEL_CONFIG"])["repo"] == "test/model"
    api.cancel_job.assert_called_once_with(job_id="test-job", namespace="org")
    assert active_provider.get() is None
