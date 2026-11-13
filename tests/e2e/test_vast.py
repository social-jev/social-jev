import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from social_jev import Query, Request, Result, SocialJev
from social_jev.integrations import VastAI, vast


@pytest.mark.parametrize(
    ("backend", "serve"), [("transformers", False), ("vllm", False), ("vllm", True)]
)
def test_vast_runs_queries_and_destroys_rental(monkeypatch, backend, serve):
    pytest.importorskip("uvicorn")
    testclient = pytest.importorskip("fastapi.testclient")
    from social_jev.integrations import vast_worker  # noqa: PLC0415

    sdk = fake_rental(monkeypatch)
    active = 0
    both_started = asyncio.Event()

    def predict(requests):
        return [
            [Result(choice=request.state, probabilities={request.state: 1.0})]
            for request in requests
        ]

    async def apredict(requests):
        nonlocal active
        active += 1
        if active == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2)
        return predict(requests)

    remote = SimpleNamespace(
        query=predict, aquery=apredict, stop=Mock(), engine=SimpleNamespace(model=Mock())
    )
    load = Mock(return_value=remote)

    def connect(startup, _address, config, _backend, _deadline):
        with monkeypatch.context() as patch:
            patch.setattr(vast_worker.SocialJev, "from_pretrained", load)
            app = vast_worker.create_app(**json.loads(config))
        http = startup.enter_context(testclient.TestClient(app))

        def async_http():
            return httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://worker"
            )

        return http, async_http

    monkeypatch.setattr(vast, "connect_worker", connect)
    query = Query(text="Pick a topic", options=["Transport", "Education"])
    requests = [Request(state=topic, queries=[query]) for topic in query.options]

    with VastAI(max_hourly_price=1.0):
        model = SocialJev.from_pretrained("test/model", backend=backend, serve=serve)
        check_queries(model, requests, backend, query.options)

    load.assert_called_once_with("test/model", backend=backend, serve=serve)
    remote.stop.assert_called_once_with()
    sdk.destroy_instance.assert_called_once_with(id=12)


def test_vast_destroys_rental_when_ssh_fails(monkeypatch):
    sdk = fake_rental(monkeypatch)
    connect = Mock(side_effect=RuntimeError("Host key verification failed"))
    monkeypatch.setattr(vast, "connect_worker", connect)
    with VastAI(max_hourly_price=1.0), pytest.raises(RuntimeError, match="Host key verification"):
        SocialJev.from_pretrained("test/model", backend="vllm")
    sdk.destroy_instance.assert_called_once_with(id=12)


def fake_rental(monkeypatch):
    sdk = Mock()
    sdk.search_offers.return_value = [{"id": 5, "dph_total": 0.5}]
    sdk.create_instance.return_value = {"new_contract": 12}
    sdk.show_instance.return_value = {
        "actual_status": "running",
        "ssh_host": "test.invalid",
        "ssh_port": 2200,
    }
    sdk.destroy_instance.return_value = {"success": True}
    monkeypatch.setitem(sys.modules, "vastai", SimpleNamespace(VastAI=Mock(return_value=sdk)))
    return sdk


def check_queries(model, requests, backend, options):
    if backend == "transformers":
        assert [row[0].choice for row in model.query(requests)] == options
        with pytest.raises(RuntimeError, match="aquery requires"):
            asyncio.run(model.aquery(requests))
    else:

        async def overlap():
            return await asyncio.gather(*(model.aquery(item) for item in requests))

        assert [batch[0][0].choice for batch in asyncio.run(overlap())] == options


def test_vast_destroys_rental_on_keyboard_interrupt(monkeypatch):
    sdk = fake_rental(monkeypatch)
    monkeypatch.setattr(vast, "connect_worker", Mock(return_value=(Mock(), Mock())))
    with pytest.raises(KeyboardInterrupt):
        interrupt_query()

    sdk.destroy_instance.assert_called_once_with(id=12)


def interrupt_query():
    with VastAI(max_hourly_price=1.0):
        SocialJev.from_pretrained("test/model", backend="vllm")
        raise KeyboardInterrupt
